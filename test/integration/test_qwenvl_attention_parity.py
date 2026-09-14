"""QwenVL PR1 / WS-A — paged attention parity (P1-G5).

Proves that the attention backend the engine hands to ``QwenVLAttention``
produces the same numbers as a dense fp32 SDPA reference for QwenVL's layout:
GQA (``num_attention_heads > num_key_value_heads``), causal masking, packed
variable-length batches, and page-boundary sequence lengths.

Every test enters through ``KVCacheEngine.execute_batch`` so the page tables,
``seq_lens`` side-channel, ``position_advance`` and KV writes are the
production ones, not a fake cache.

Evidence labels
---------------
* ``cpu-dense-fp32`` rows exercise the harness and the paged-store bookkeeping
  (``DenseReferenceCacheManager`` shares ``PagedAllocationManager`` with the
  FlashInfer backend) but are **not** PR1 acceptance evidence.
* ``cuda-flashinfer-bf16`` rows (``-m cuda``) are the P1-G5 evidence: the real
  ``FlashInferCacheManager`` against the dense reference on the same device
  with identical weights.
"""

from __future__ import annotations

import pytest
import qwenvl_harness as H
import torch

pytestmark = pytest.mark.p1_gate("P1-G5")

PAGE_SIZE = 128  # matches configs/qwenvl.yaml so 127/128/129 straddle a real page edge
BOUNDARY_LENGTHS = (PAGE_SIZE - 1, PAGE_SIZE, PAGE_SIZE + 1)


@pytest.fixture(params=H.TARGET_PARAMS)
def target(request, monkeypatch) -> H.Target:
    return H.prepare_target(request.param, monkeypatch)


Prompts = dict[str, H.TextPrompt | H.VisionPrompt]


def _prefill_last_logits(handle: H.LLMEngine, prompts: Prompts) -> dict[str, torch.Tensor]:
    return H.run_prefill(handle, prompts).logits


def _decode_logits(handle: H.LLMEngine, tokens: dict[str, int]) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    result = H.run_decode(handle, tokens)
    return result.logits, result.tokens


# ---------------------------------------------------------------------------
# Paged store correctness: incremental == one-shot (causal + page boundary)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", [*BOUNDARY_LENGTHS, 2 * PAGE_SIZE + 1])
def test_incremental_decode_matches_one_shot_prefill_across_page_boundary(target: H.Target, length: int) -> None:
    """Prefill ``length`` tokens then decode one more must equal prefilling all
    ``length + 1`` tokens at once. Any error in how K/V are appended to or
    read back from pages (especially across the page edge) shows up here."""
    prompt = H.text_prompt(length + 1, seed=length)
    handle = H.build_llm_engine(target, page_size=PAGE_SIZE, max_num_pages=16)

    handle.add_request("one_shot")
    one_shot = _prefill_last_logits(handle, {"one_shot": prompt})["one_shot"]
    handle.remove_request("one_shot")

    handle.add_request("incremental")
    _prefill_last_logits(handle, {"incremental": H.TextPrompt(ids=prompt.ids[:-1])})
    assert handle.seq_len("incremental") == length
    assert len(handle.page_indices("incremental")) == -(-length // PAGE_SIZE)
    step_logits, _ = _decode_logits(handle, {"incremental": int(prompt.ids[-1])})
    assert handle.seq_len("incremental") == length + 1
    assert len(handle.page_indices("incremental")) == -(-(length + 1) // PAGE_SIZE)

    H.assert_logits_close(step_logits["incremental"], one_shot, target, f"decode after {length}-token prefill")


def test_causal_mask_future_tokens_do_not_change_past_hidden_states(target: H.Target) -> None:
    """Per-position hidden states of a prefix must be identical whether or not
    a (different) suffix follows it in the same launch: the causal mask must
    hide future K/V, and a second request packed after it must not leak."""
    prefix = H.text_prompt(40, seed=1)
    long_a = H.TextPrompt(ids=torch.cat([prefix.ids, H.text_prompt(25, seed=2).ids]))
    long_b = H.TextPrompt(ids=torch.cat([prefix.ids, H.text_prompt(25, seed=3).ids]))
    handle = H.build_llm_engine(target, page_size=16, max_num_pages=32)

    handle.add_request("prefix")
    with handle.capture_hidden() as hidden_prefix:
        handle.run(H.prefill_batch(handle, {"prefix": prefix}))
    handle.remove_request("prefix")

    handle.add_request("a")
    handle.add_request("b")
    with handle.capture_hidden() as hidden_ab:
        handle.run(H.prefill_batch(handle, {"a": long_a, "b": long_b}))
    assert len(hidden_prefix) == 1 and len(hidden_ab) == 1
    packed = hidden_ab[0]
    assert packed.shape[0] == long_a.length + long_b.length

    prefix_alone = hidden_prefix[0]
    prefix_in_a = packed[: prefix.length]
    prefix_in_b = packed[long_a.length : long_a.length + prefix.length]
    H.assert_logits_close(prefix_in_a, prefix_alone, target, "prefix hidden states with a suffix appended")
    H.assert_logits_close(prefix_in_b, prefix_alone, target, "prefix hidden states packed second in the batch")
    # The two suffixes differ, so their hidden states must too (otherwise the
    # comparison above is vacuous).
    suffix_a = packed[prefix.length : long_a.length]
    suffix_b = packed[long_a.length + prefix.length :]
    assert not torch.allclose(suffix_a, suffix_b, atol=1e-3)


@pytest.mark.parametrize("lengths", [(5, 9), (17, 3, 33, 1)], ids=["B2-unequal", "B4-unequal"])
def test_packed_variable_length_batch_matches_isolated_requests(target: H.Target, lengths: tuple[int, ...]) -> None:
    """Unequal-length requests packed into one prefill launch must reproduce
    the logits each one gets alone (packed ``seq_lens`` → per-request page
    tables and causal blocks)."""
    prompts = {f"r{i}": H.text_prompt(length, seed=100 + i) for i, length in enumerate(lengths)}
    handle = H.build_llm_engine(target, page_size=16, max_num_pages=32)

    isolated = {}
    for rid, prompt in prompts.items():
        handle.add_request(rid)
        isolated[rid] = _prefill_last_logits(handle, {rid: prompt})[rid]
        handle.remove_request(rid)
    assert handle.free_pages == handle.total_pages

    for rid in prompts:
        handle.add_request(rid)
    packed = _prefill_last_logits(handle, prompts)
    for rid, prompt in prompts.items():
        assert handle.seq_len(rid) == prompt.length
        H.assert_logits_close(packed[rid], isolated[rid], target, f"{rid} (len {prompt.length}) packed vs alone")


# ---------------------------------------------------------------------------
# FlashInfer vs dense reference on the same device (the P1-G5 core claim)
# ---------------------------------------------------------------------------


@pytest.fixture
def cuda_pair(monkeypatch) -> tuple[H.LLMEngine, H.LLMEngine]:
    """Two engines with identical weights: FlashInfer and the dense reference."""
    flashinfer = H.prepare_target(H.cuda_flashinfer_target(), monkeypatch)
    reference = H.cuda_reference_target()
    config = H.make_tiny_config()
    weights = H.build_language_model(config, flashinfer, seed=0)
    fi = H.build_llm_engine(flashinfer, config=config, page_size=PAGE_SIZE, max_num_pages=32, language_model=weights)
    ref = H.build_llm_engine(reference, config=config, page_size=PAGE_SIZE, max_num_pages=32, language_model=weights)
    return fi, ref


def test_tiny_config_exercises_gqa() -> None:
    """Precondition for every parity row above: the model under test must be
    grouped-query, or the GQA head-mapping claim would be vacuous."""
    text = H.make_tiny_config().text_config
    assert text.num_attention_heads > text.num_key_value_heads, "parity would not cover GQA"
    assert text.num_attention_heads % text.num_key_value_heads == 0


def test_flashinfer_target_uses_sm87_valid_head_geometry() -> None:
    """The compact CPU dry-run uses head_dim=8; the CUDA target must widen
    before FlashInfer JIT is invoked on SM87."""
    config = H.make_tiny_config()
    H.configure_flashinfer_geometry(config, H.cuda_flashinfer_target())
    assert config.text_config.head_dim == 64
    assert config.text_config.hidden_size == 128
    assert config.text_config.rope_scaling["mrope_section"] == [8, 12, 12]
    assert config.vision_config.out_hidden_size == 128


@pytest.mark.cuda
@pytest.mark.parametrize(
    "lengths",
    [BOUNDARY_LENGTHS, (16, 64, 100, PAGE_SIZE), (3, 127, 129, 40, 8, 128, 257, 1)],
    ids=["page-boundary-B3", "varied-B4", "mixed-B8"],
)
def test_flashinfer_prefill_and_decode_match_dense_reference(cuda_pair, lengths: tuple[int, ...]) -> None:
    fi, ref = cuda_pair
    prompts = {f"r{i}": H.text_prompt(length, seed=500 + i) for i, length in enumerate(lengths)}
    for rid in prompts:
        fi.add_request(rid)
        ref.add_request(rid)

    fi_logits = _prefill_last_logits(fi, prompts)
    ref_logits = _prefill_last_logits(ref, prompts)
    for rid, prompt in prompts.items():
        assert fi.page_indices(rid) == ref.page_indices(rid), "both backends share the allocator policy"
        H.assert_logits_close(fi_logits[rid], ref_logits[rid], fi.target, f"prefill {rid} len {prompt.length}")

    # Feed the *reference* argmax to both so the trajectories stay comparable.
    tokens = {rid: int(torch.argmax(ref_logits[rid])) for rid in prompts}
    for step in range(3):
        fi_step, _ = _decode_logits(fi, tokens)
        ref_step, _ = _decode_logits(ref, tokens)
        for rid, prompt in prompts.items():
            H.assert_logits_close(fi_step[rid], ref_step[rid], fi.target, f"decode step {step} {rid}")
            assert fi.seq_len(rid) == ref.seq_len(rid) == prompt.length + step + 1
        tokens = {rid: int(torch.argmax(ref_step[rid])) for rid in prompts}


@pytest.mark.cuda
def test_flashinfer_vision_prefill_matches_dense_reference(cuda_pair) -> None:
    """The ``prefill_vision`` LLM step (3-D MRoPE positions + DeepStack)
    through FlashInfer vs the dense reference, two different image grids."""
    fi, ref = cuda_pair
    prompts = {"img_a": H.vision_prompt((1, 4, 4), seed=11), "img_b": H.vision_prompt((1, 2, 8), seed=12, prefix=5)}
    for rid in prompts:
        fi.add_request(rid)
        ref.add_request(rid)
    fi_logits = _prefill_last_logits(fi, prompts)
    ref_logits = _prefill_last_logits(ref, prompts)
    for rid, prompt in prompts.items():
        H.assert_logits_close(fi_logits[rid], ref_logits[rid], fi.target, f"prefill_vision {rid}")
        # MRoPE advance is the 3-D span, not the token count.
        assert fi.position_start(rid) == ref.position_start(rid) == prompt.position_span(fi.config)
        assert fi.seq_len(rid) == ref.seq_len(rid) == prompt.length
    tokens = {rid: int(torch.argmax(ref_logits[rid])) for rid in prompts}
    fi_step, _ = _decode_logits(fi, tokens)
    ref_step, _ = _decode_logits(ref, tokens)
    for rid in prompts:
        H.assert_logits_close(fi_step[rid], ref_step[rid], fi.target, f"decode after image {rid}")
