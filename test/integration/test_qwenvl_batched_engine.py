"""QwenVL PR1 / WS-B — ``KVCacheEngine`` batched vs isolated parity (P1-G2).

Requests enter through ``KVCacheEngine.execute_batch`` (``prepare_batch`` →
``execute_forward``) and are compared between the two production execution
paths of the same engine:

* ``_execute_batched``  — one packed launch through ``forward_batched`` with a
  batch-wide ``BatchedCacheManager`` (the continuous-batching path);
* ``_execute_sequential`` — the per-request reference path (each request gets
  its own cache manager and ``forward`` call), which is what "isolated" means
  for PR1's required tests.

Both paths run on engines holding the *same* weights, so any difference is
attributable to packing/batching, not initialisation.

Coverage: text ``prefill`` at B∈{2,4,8}; ``prefill_vision`` at B∈{2,4} with
different image grids (vision embeds + DeepStack per request); shared
``decode`` at B∈{2,4,8} including mixed text-/image-origin requests. Prefill
compares last-token logits; decode compares logits and greedy token streams
every step.

Tolerance: fp32 dense reference rows must match exactly; bf16 FlashInfer rows
use ``H.BF16_LOGITS_RTOL/ATOL`` (see ``qwenvl_harness``), matching the
Qwen3-Omni CUDA-graph precedent. Sampling is greedy throughout so FlashInfer's
batch-position-dependent RNG cannot masquerade as a batching bug.
"""

from __future__ import annotations

import pytest
import qwenvl_harness as H

pytestmark = pytest.mark.p1_gate("P1-G2")

DECODE_STEPS = 4


@pytest.fixture(params=H.TARGET_PARAMS)
def target(request, monkeypatch) -> H.Target:
    return H.prepare_target(request.param, monkeypatch)


@pytest.fixture
def engines(target: H.Target) -> tuple[H.LLMEngine, H.LLMEngine]:
    """(batched, sequential) engines sharing one set of weights."""
    config = H.make_tiny_config()
    weights = H.build_language_model(config, target, seed=0)
    batched = H.build_llm_engine(target, config=config, page_size=16, max_num_pages=128, language_model=weights)
    sequential = H.build_llm_engine(target, config=config, page_size=16, max_num_pages=128, language_model=weights)
    return batched, sequential


def _add(engines: tuple[H.LLMEngine, H.LLMEngine], rids) -> None:
    for handle in engines:
        for rid in rids:
            handle.add_request(rid)


def _assert_steps_match(batched: H.StepResult, sequential: H.StepResult, target: H.Target, what: str) -> None:
    assert batched.tokens.keys() == sequential.tokens.keys()
    for rid in sequential.tokens:
        H.assert_logits_close(batched.logits[rid], sequential.logits[rid], target, f"{what} {rid}")
        H.assert_greedy_streams_match(
            [batched.tokens[rid]], [sequential.tokens[rid]], [sequential.margin(rid)], target, f"{what} {rid} token"
        )


def _decode_lockstep(
    engines: tuple[H.LLMEngine, H.LLMEngine],
    target: H.Target,
    first_tokens: dict[str, int],
    steps: int,
    what: str,
) -> dict[str, list[int]]:
    """Decode ``steps`` tokens on both paths, each feeding its own greedy
    token, and compare logits + token streams every step."""
    batched, sequential = engines
    tokens_b = dict(first_tokens)
    tokens_s = dict(first_tokens)
    streams_b = {rid: [t] for rid, t in first_tokens.items()}
    streams_s = {rid: [t] for rid, t in first_tokens.items()}
    margins_s = {rid: [] for rid in first_tokens}
    for step in range(steps):
        result_b = H.run_decode(batched, tokens_b)
        result_s = H.run_decode(sequential, tokens_s, sequential=True)
        for rid in first_tokens:
            label = f"{what} decode step {step} {rid}"
            H.assert_logits_close(result_b.logits[rid], result_s.logits[rid], target, label)
            streams_b[rid].append(result_b.tokens[rid])
            streams_s[rid].append(result_s.tokens[rid])
            margins_s[rid].append(result_s.margin(rid))
            assert batched.seq_len(rid) == sequential.seq_len(rid)
            assert batched.position_start(rid) == sequential.position_start(rid)
        tokens_b, tokens_s = result_b.tokens, result_s.tokens
    for rid in first_tokens:
        H.assert_greedy_streams_match(
            streams_b[rid][1:], streams_s[rid][1:], margins_s[rid], target, f"{what} greedy stream {rid}"
        )
    return streams_b


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_text_prefill_batched_matches_sequential(engines, target, batch_size: int) -> None:
    lengths = [16, 64, 100, 5, 33, 1, 48, 17][:batch_size]
    prompts = {f"t{i}": H.text_prompt(length, seed=i) for i, length in enumerate(lengths)}
    _add(engines, prompts)
    batched, sequential = engines

    result_b = H.run_prefill(batched, prompts)
    result_s = H.run_prefill(sequential, prompts, sequential=True)
    _assert_steps_match(result_b, result_s, target, f"text prefill B={batch_size}")
    for rid, prompt in prompts.items():
        assert batched.seq_len(rid) == sequential.seq_len(rid) == prompt.length
        assert batched.position_start(rid) == sequential.position_start(rid) == prompt.length


@pytest.mark.parametrize("batch_size", [2, 4])
def test_vision_prefill_batched_matches_sequential(engines, target, batch_size: int) -> None:
    grids = [(1, 4, 4), (1, 6, 4), (1, 2, 8), (2, 2, 2)][:batch_size]
    prompts = {
        f"v{i}": H.vision_prompt(grid, seed=20 + i, prefix=2 + i, suffix=1 + (i % 2)) for i, grid in enumerate(grids)
    }
    assert len({p.num_visual_tokens for p in prompts.values()}) > 1, "grids must differ in visual token count"
    _add(engines, prompts)
    batched, sequential = engines

    result_b = H.run_prefill(batched, prompts)
    result_s = H.run_prefill(sequential, prompts, sequential=True)
    _assert_steps_match(result_b, result_s, target, f"prefill_vision B={batch_size}")
    for rid, prompt in prompts.items():
        # KV length is the token count; MRoPE advance is the 3-D position span.
        assert batched.seq_len(rid) == sequential.seq_len(rid) == prompt.length
        assert batched.position_start(rid) == sequential.position_start(rid) == prompt.position_span(batched.config)
    # A 2-D image grid compresses positions, so at least one request in the
    # batch must carry an MRoPE offset that differs from its KV length.
    assert any(p.position_span(batched.config) != p.length for p in prompts.values())


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_decode_batched_matches_sequential(engines, target, batch_size: int) -> None:
    lengths = [12, 40, 7, 65, 3, 21, 90, 1][:batch_size]
    prompts = {f"t{i}": H.text_prompt(length, seed=100 + i) for i, length in enumerate(lengths)}
    _add(engines, prompts)
    batched, sequential = engines

    # Prefill through the same path on both engines so decode starts from an
    # identical KV state; only decode is under test here.
    first_b = H.run_prefill(batched, prompts).tokens
    first_s = H.run_prefill(sequential, prompts).tokens
    assert first_b == first_s
    _decode_lockstep(engines, target, first_b, DECODE_STEPS, f"B={batch_size}")


def test_mixed_origin_decode_batch_matches_sequential(engines, target) -> None:
    """One text-origin and one image-origin request share a decode batch
    (PR1 required test #4). Decode never receives vision tensors, so the only
    per-request state that may differ is KV history and MRoPE position."""
    batched, sequential = engines
    text = {"text": H.text_prompt(30, seed=7)}
    image = {"image": H.vision_prompt((1, 4, 4), seed=8, prefix=4, suffix=3)}
    _add(engines, [*text, *image])

    first: dict[str, int] = {}
    first.update(H.run_prefill(batched, text).tokens)
    first.update(H.run_prefill(batched, image).tokens)
    first_s: dict[str, int] = {}
    first_s.update(H.run_prefill(sequential, text).tokens)
    first_s.update(H.run_prefill(sequential, image).tokens)
    assert first == first_s
    assert batched.position_start("image") != batched.seq_len("image"), "image request must carry a 3-D MRoPE offset"

    streams = _decode_lockstep(engines, target, first, DECODE_STEPS, "mixed-origin")

    # Cross-check "isolated" literally: each request alone on a fresh engine
    # with the same weights must reproduce the co-batched stream.
    for rid, prompt in {**text, **image}.items():
        alone = H.build_llm_engine(
            target,
            config=batched.config,
            page_size=16,
            max_num_pages=128,
            language_model=batched.submodule.language_model,
        )
        alone.add_request(rid)
        result = H.run_prefill(alone, {rid: prompt})
        stream = [result.tokens[rid]]
        margins = []
        for _ in range(DECODE_STEPS):
            result = H.run_decode(alone, result.tokens)
            stream.append(result.tokens[rid])
            margins.append(result.margin(rid))
        assert stream[0] == streams[rid][0]
        H.assert_greedy_streams_match(streams[rid][1:], stream[1:], margins, target, f"{rid} co-batched vs alone")


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_decode_batch_of_image_origin_requests_matches_sequential(engines, target, batch_size: int) -> None:
    grids = [(1, 4, 4), (1, 2, 8), (1, 6, 4), (2, 2, 2), (1, 8, 2), (1, 4, 2), (1, 2, 2), (1, 6, 2)][:batch_size]
    prompts = {f"v{i}": H.vision_prompt(grid, seed=300 + i, prefix=1 + i % 3) for i, grid in enumerate(grids)}
    _add(engines, prompts)
    batched, sequential = engines
    first_b = H.run_prefill(batched, prompts).tokens
    first_s = H.run_prefill(sequential, prompts).tokens
    assert first_b == first_s
    _decode_lockstep(engines, target, first_b, DECODE_STEPS, f"image-origin decode B={batch_size}")
