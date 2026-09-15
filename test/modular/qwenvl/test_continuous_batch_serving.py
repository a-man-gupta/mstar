"""Component contract for the QwenVL LLM node's packed (same-walk) batches.

Evidence label: **Component** (fake cache, CPU). These tests prove the
submodule-level packing contract — same-walk homogeneity, per-request MRoPE
advance, last-token selection, decode state — with a fake ``Cache``. They do
*not* prove that the scheduler co-batches requests or that FlashInfer paged
attention is correct; that is the job of ``test/integration/test_qwenvl_*.py``
(PR1 gates P1-G1..G6).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources import SamplingReqConfig
from mstar.model.qwenvl.components import QwenVLForCausalLM
from mstar.model.qwenvl.submodules import (
    QwenVLLLMSubmodule,
    QwenVLVisionSubmodule,
    qwen_vl_position_ids,
    qwenvl_batch_is_homogeneous,
)

from ._helpers import Cache, FixedLanguageModel, tiny_config


def _text_request(submodule, ids: list[int], start: int = 0):
    return submodule.prepare_inputs(
        "prefill",
        None,
        {
            "text_inputs": [torch.tensor(ids)],
            "position_ids": [(torch.arange(len(ids)) + start).expand(3, -1)],
        },
        pos_info={"main": SimpleNamespace(position_id_start=start)},
    )


def _image_request(submodule, config, ids: torch.Tensor, grid, fill: float = 1.0, deepstack_fill: float = 1.0):
    count = int((ids == config.image_token_id).sum())
    return submodule.prepare_inputs(
        "prefill_vision",
        None,
        {
            "text_inputs": [ids],
            "position_ids": [qwen_vl_position_ids(ids, torch.tensor([grid]), config)],
            "vision_embeds": [torch.full((count, 16), fill)],
            "deepstack_visual_embeds": [torch.full((count, 16), deepstack_fill) for _ in range(3)],
        },
    )


def test_visual_chat_uses_post_merge_features_and_preserves_request_context():
    pooled = torch.ones(2, 16)

    class Vision(torch.nn.Module):
        def forward(self, pixels, grid_thw):
            self.grid = grid_thw
            return pooled, [pooled, pooled, pooled]

    vision = Vision()
    submodule = QwenVLVisionSubmodule(vision)
    inputs = submodule.prepare_inputs(
        "prefill_vision",
        None,
        {
            "pixel_values": [torch.ones(4, 12)],
            "image_grid_thw": [torch.tensor([[1, 2, 2]])],
            "text_inputs": [torch.tensor([1, 2])],
            "position_ids": [torch.tensor([[0, 1], [0, 1], [0, 1]])],
        },
    )
    output = submodule.forward("prefill_vision", None, **inputs.tensor_inputs)
    assert torch.equal(vision.grid, inputs.tensor_inputs["image_grid_thw"])
    assert torch.equal(output["vision_embeds"][0], pooled)
    assert len(output["deepstack_visual_embeds"]) == 3


def test_image_prefill_preserves_mrope_position_for_follow_up_decode():
    config = tiny_config()
    submodule = QwenVLLLMSubmodule(QwenVLForCausalLM(config), config)
    cache = Cache()
    tok = config.image_token_id
    # Two image prompts with different grids packed on the same walk. With
    # merge=2 the 4x4 grid occupies four language slots but spans two MRoPE
    # positions (advance 4 over 6 tokens); the 2x4 grid occupies two slots
    # and also spans two positions (advance 5 over 5 tokens). The deltas are
    # per request, not derived from the packed token count.
    image_a = _image_request(submodule, config, torch.tensor([4, tok, tok, tok, tok, 5]), [1, 4, 4])
    image_b = _image_request(submodule, config, torch.tensor([4, tok, tok, 5, 6]), [1, 2, 4])
    engine = SimpleNamespace(cache_manager=cache, request_ids=["a", "b"], sampler=None)
    packed = submodule.preprocess("prefill_vision", engine, [image_a, image_b])
    assert packed["seq_lens"] == [6, 5]
    assert packed["position_advance"] == [4, 5]
    assert cache.custom_pos_advance == [4, 5]
    assert packed["vision_embeds"].shape == (6, 16)
    expected_mask = [False, True, True, True, True, False, False, True, True, False, False]
    assert packed["visual_token_mask"].tolist() == expected_mask
    assert all(layer.shape == (6, 16) for layer in packed["deepstack_visual_embeds"])

    # Text-only prefill packs on its own walk; its advance is the token count.
    text = _text_request(submodule, [1, 2, 3])
    text_packed = submodule.preprocess("prefill", engine, [text])
    assert text_packed["seq_lens"] == [3]
    assert text_packed["position_advance"] == [3]
    assert "vision_embeds" not in text_packed and "deepstack_visual_embeds" not in text_packed
    resumed = submodule.prepare_inputs(
        "prefill",
        None,
        {
            "text_inputs": [torch.tensor([6, 7])],
            "position_ids": [torch.tensor([[4, 5], [4, 5], [4, 5]])],
        },
        pos_info={"main": SimpleNamespace(position_id_start=4)},
    )
    assert resumed.kwargs["position_advance"] == 2
    decode = submodule.prepare_inputs(
        "decode", None, {"text_inputs": [torch.tensor([7])]}, pos_info={"main": SimpleNamespace(position_id_start=4)}
    )
    assert decode.custom_pos_ids.tolist() == [[4], [4], [4]]
    with pytest.raises(ValueError, match="Unknown"):
        submodule.prepare_inputs("unknown", None, {"text_inputs": [torch.tensor([7])]})


def test_packed_prefill_vision_batch_keeps_tenant_visual_features_isolated():
    """Two image requests packed on ``prefill_vision`` each see only their own features.

    Component contract only: the fake cache does no attention, so this proves
    embedding/DeepStack placement, not KV isolation (see integration tests).
    """
    config = tiny_config(image_token_id=30)
    model = FixedLanguageModel(config)
    submodule = QwenVLLLMSubmodule(model, config)
    cache = Cache()
    engine = SimpleNamespace(cache_manager=cache, request_ids=["image_a", "image_b"], sampler=None)
    image_a = _image_request(submodule, config, torch.tensor([4, 30, 30, 30, 30, 5]), [1, 4, 4], fill=3.0)
    image_b = _image_request(submodule, config, torch.tensor([6, 30, 30, 7]), [1, 2, 4], fill=-2.0)
    packed = submodule.preprocess("prefill_vision", engine, [image_a, image_b])
    result = submodule.forward_batched("prefill_vision", engine, **packed)
    assert set(result) == {"image_a", "image_b"}
    merged = model.calls[-1][0]
    assert torch.equal(merged[1:5], torch.full((4, 16), 3.0))
    assert torch.equal(merged[7:9], torch.full((2, 16), -2.0))
    assert torch.equal(merged[0], model.model.embed_tokens.weight[4])
    assert torch.equal(merged[6], model.model.embed_tokens.weight[6])

    class Sampler:
        def sample(self, request_ids, logits, apply_penalty):
            assert (request_ids, logits.shape, apply_penalty) == (
                ["image_a", "image_b"],
                (2, config.text_config.vocab_size),
                True,
            )
            return torch.tensor([8, 9])

    engine.sampler = Sampler()
    sampled = submodule.forward_batched(
        "decode",
        engine,
        text_inputs=torch.tensor([1, 2]),
        position_ids=torch.tensor([[3, 4], [3, 4], [3, 4]]),
        position_advance=[1, 1],
        seq_lens=[1, 1],
        cos_3d=torch.ones(2, 8),
        sin_3d=torch.zeros(2, 8),
    )
    assert sampled.keys() == {"image_a", "image_b"}
    submodule.postprocess("image_a", None, sampled["image_a"])
    assert torch.equal(sampled["image_a"]["text_inputs"][0], torch.tensor([8]))


def test_same_walk_homogeneity_is_enforced_by_can_batch_and_preprocess():
    """PR1 batching boundary: prefill<->prefill, prefill_vision<->prefill_vision, decode<->decode.

    The scheduler groups by ``(node, graph_walk)``; the submodule additionally
    refuses to pack requests whose payload does not match the walk, so a text
    request can never share a launch with a vision request.
    """
    config = tiny_config(image_token_id=30)
    submodule = QwenVLLLMSubmodule(FixedLanguageModel(config), config)
    engine = SimpleNamespace(cache_manager=Cache(), request_ids=["x", "y"], sampler=None)
    text_a = _text_request(submodule, [1, 2, 3])
    text_b = _text_request(submodule, [4, 5])
    image_a = _image_request(submodule, config, torch.tensor([4, 30, 30, 30, 30, 5]), [1, 4, 4])
    image_b = _image_request(submodule, config, torch.tensor([6, 30, 30, 7]), [1, 2, 4])
    decode_a = submodule.prepare_inputs("decode", None, {"text_inputs": [torch.tensor([7])]})
    decode_b = submodule.prepare_inputs("decode", None, {"text_inputs": [torch.tensor([8])]})

    def batch(walk):
        return SimpleNamespace(graph_walk=walk)

    # Same-walk, same-payload groups batch (including B=1).
    assert submodule.can_batch(batch("prefill"), [text_a, text_b])
    assert submodule.can_batch(batch("prefill"), [text_a])
    assert submodule.can_batch(batch("prefill_vision"), [image_a, image_b])
    assert submodule.can_batch(batch("decode"), [decode_a, decode_b])
    # Cross-payload groups never reach the packed launch.
    assert not submodule.can_batch(batch("prefill"), [text_a, image_a])
    assert not submodule.can_batch(batch("prefill_vision"), [image_a, text_a])
    assert not submodule.can_batch(batch("decode"), [decode_a, image_a])
    assert not submodule.can_batch(batch("decode"), [decode_a, text_a])
    assert not submodule.can_batch(batch("prefill"), [])
    assert not qwenvl_batch_is_homogeneous("unknown_walk", [text_a])

    # preprocess is the last line of defence and rejects loudly.
    with pytest.raises(ValueError, match="non-homogeneous"):
        submodule.preprocess("prefill", engine, [text_a, image_a])
    with pytest.raises(ValueError, match="non-homogeneous"):
        submodule.preprocess("prefill_vision", engine, [image_a, text_a])
    with pytest.raises(ValueError, match="non-homogeneous"):
        submodule.preprocess("decode", engine, [decode_a, image_a])
    # A prefill_vision request missing its DeepStack features is rejected too.
    half = submodule.prepare_inputs(
        "prefill_vision",
        None,
        {
            "text_inputs": [torch.tensor([6, 30, 30, 7])],
            "position_ids": [qwen_vl_position_ids(torch.tensor([6, 30, 30, 7]), torch.tensor([[1, 2, 4]]), config)],
            "vision_embeds": [torch.ones(2, 16)],
            "deepstack_visual_embeds": None,
        },
    )
    with pytest.raises(ValueError, match="both 'vision_embeds' and 'deepstack_visual_embeds'"):
        submodule.preprocess("prefill_vision", engine, [half])


def test_decode_and_text_prefill_forward_refuse_vision_tensors():
    """Visual state belongs to the ``prefill_vision`` step only (no bleed into decode)."""
    config = tiny_config(image_token_id=30)
    submodule = QwenVLLLMSubmodule(FixedLanguageModel(config), config)
    engine = SimpleNamespace(cache_manager=Cache(), request_ids=["a"], sampler=None)
    common = dict(
        text_inputs=torch.tensor([30]),
        position_ids=torch.zeros(3, 1, dtype=torch.long),
        position_advance=[1],
        seq_lens=[1],
        cos_3d=torch.ones(1, 8),
        sin_3d=torch.zeros(1, 8),
    )
    for walk in ("decode", "prefill"):
        with pytest.raises(ValueError, match="must not receive vision"):
            submodule.forward_batched(walk, engine, vision_embeds=torch.ones(1, 16), **common)
        with pytest.raises(ValueError, match="must not receive vision"):
            submodule.forward(walk, engine, deepstack_visual_embeds=[torch.ones(1, 16)] * 3, **common)
        with pytest.raises(ValueError, match="must not receive vision"):
            submodule.forward(walk, engine, visual_token_mask=torch.tensor([True]), **common)
    for walk in ("prefill", "prefill_vision", "decode"):
        assert submodule.get_needed_cache_labels(walk, {}) == ["main"]


def test_decode_serving_honors_per_request_last_token_and_stop_conditions():
    config = tiny_config()
    submodule = QwenVLLLMSubmodule(FixedLanguageModel(config), config)
    engine = SimpleNamespace(cache_manager=Cache(), request_ids=["a", "b"], sampler=None)
    output = submodule.forward_batched(
        "decode",
        engine,
        text_inputs=torch.tensor([1, 2]),
        position_ids=torch.tensor([[0, 1], [0, 1], [0, 1]]),
        position_advance=[1, 1],
        cos_3d=torch.ones(2, 8),
        sin_3d=torch.zeros(2, 8),
    )
    assert set(output) == {"a", "b"}
    direct = submodule.forward(
        "prefill",
        engine,
        text_inputs=torch.tensor([1, 2]),
        position_ids=torch.tensor([[0, 1], [0, 1], [0, 1]]),
        position_advance=[1, 1],
        seq_lens=[1, 1],
        cos_3d=torch.ones(2, 8),
        sin_3d=torch.zeros(2, 8),
    )
    assert direct["logits"][0].shape == (2, config.text_config.vocab_size)
    assert submodule.forward(
        "decode",
        engine,
        text_inputs=torch.tensor([1]),
        position_ids=torch.zeros(3, 1, dtype=torch.long),
        position_advance=1,
        cos_3d=torch.ones(1, 8),
        sin_3d=torch.zeros(1, 8),
    )["logits"][0].shape == (1, config.text_config.vocab_size)
    request = SimpleNamespace(
        graph_walk="decode",
        resource_configs={"sampler": SamplingReqConfig(ignore_eos=False)},
        dynamic_loop_iter_counts={"decode_loop": 0},
        max_tokens=3,
    )
    assert submodule.check_stop("a", request, {"new_token": [torch.tensor([config.text_config.eos_token_id])]}) == {
        "decode_loop"
    }
    assert submodule.check_stop("a", request, {}) == set()
    request.graph_walk = "prefill"
    assert submodule.check_stop(
        "a", request, {"new_token": [torch.tensor([config.text_config.eos_token_id])]}
    ) == set()
    assert not submodule.can_batch(SimpleNamespace(graph_walk="decode"), [])
    with pytest.raises(ValueError, match="placeholder count"):
        submodule._merge_embeddings(torch.tensor([1, 2]), torch.ones(1, 16))


@pytest.mark.p1_gate("P1-G6")
def test_pr1_scope_is_eager_only_no_cuda_graph_capture():
    """P1-G6 option A: QwenVL declares no CUDA-graph configs, so every walk
    runs the eager batched path. Opting a walk into capture must come with
    eager-vs-replay parity evidence and a docs update (see
    docs/qwenvl/PR_1_CONTINUOUS_BATCHING.md)."""
    config = tiny_config()
    submodule = QwenVLLLMSubmodule(FixedLanguageModel(config), config)
    assert submodule.get_cuda_graph_configs(torch.device("cpu")) == []
    assert submodule.get_piecewise_cuda_graph_configs(torch.device("cpu"), torch.float32) == {}
