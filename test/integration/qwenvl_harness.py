"""Shared harness for the PR1 QwenVL continuous-batching integration tests.

Everything here drives the *real* M* serving stack -- ``KVCacheEngine``
(``prepare_batch`` -> ``execute_forward`` -> ``_execute_batched`` /
``_execute_sequential``), ``PagedAllocationManager`` page tables, the
production ``MultiSampler`` and the ``QwenVLLLMSubmodule`` -- with a tiny,
deterministically initialised Qwen3-VL-MoE backbone so the tests run in
seconds.

Two attention backends are exposed through ``KVCacheConfig.attention_backend``:

``flashinfer``
    The production paged FlashInfer path (CUDA only). This is the backend
    that PR1's acceptance gates are claimed against.

``qwenvl_dense_reference``
    ``DenseReferenceCacheManager`` below: a paged KV *store* (it uses the same
    ``PagedAllocationManager`` page tables and writes K/V into the same
    ``[layers, pages, 2, page_size, kv_heads, head_dim]`` cache tensor) with a
    dense fp32 SDPA *kernel* that gathers each request's pages back into a
    contiguous sequence. It is the correctness oracle for the FlashInfer path
    (P1-G5) and lets every engine/scheduler-level test in this directory also
    run on CPU, where it validates the batching/lifecycle logic rather than
    the kernel.

The CPU-only shims (``install_cpu_shims``) replace exactly two CUDA-only
leaves -- the CUDA-IPC KV transfer engine and FlashInfer's sampling kernel --
with functionally equivalent torch implementations. Nothing above the engine
is mocked.
"""

from __future__ import annotations

import contextlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pytest
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]

pytest.importorskip("safetensors", reason="QwenVL tests require the qwenvl model extra")


def _load_modular_helpers():
    path = REPO_ROOT / "test" / "modular" / "qwenvl" / "_helpers.py"
    spec = importlib.util.spec_from_file_location("qwenvl_modular_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_modular_helpers = _load_modular_helpers()
tiny_config = _modular_helpers.tiny_config

from mstar.communication.tensors import LocalTransferEngine  # noqa: E402
from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.distributed.communication import WorkerParallelGroups  # noqa: E402
from mstar.engine.engine import Engine, ExecutingBatch  # noqa: E402
from mstar.engine.resources import (  # noqa: E402
    AttentionConfig,
    AttentionSpec,
    KVConfig,
    KVSpec,
    NodeResourceSpec,
    PositionConfig,
    PositionSpec,
    SamplerSpec,
    SamplingReqConfig,
    StepContext,
)
from mstar.engine.resources.attn.base import AttentionManager  # noqa: E402
from mstar.engine.resources.base import EngineResourceInfo  # noqa: E402
from mstar.engine.resources.kv import manager as kv_manager_module  # noqa: E402
from mstar.engine.resources.kv.plan import KVPlanOutputs  # noqa: E402
from mstar.engine.resources.kv.transfer import TransferEngineInfo  # noqa: E402
from mstar.engine.resources.sampler import utils as sampling_module  # noqa: E402
from mstar.model.components import norm as norm_module  # noqa: E402
from mstar.model.qwenvl.components import QwenVLForCausalLM  # noqa: E402
from mstar.model.qwenvl.config import ATTN, KV_CACHE, POS, SAMPLER  # noqa: E402
from mstar.model.qwenvl.submodules import QwenVLLLMSubmodule, qwen_vl_position_ids  # noqa: E402

DENSE_REFERENCE_BACKEND = "qwenvl_dense_reference"
FLASHINFER_BACKEND = "flashinfer"
LLM_NODE = "LLM"

# Placeholder id must be a real row of the tiny 32-entry vocabulary (the
# production id 151655 only exists in the 30B tokenizer) and must not collide
# with EOS (31) or BOS (0).
TINY_IMAGE_TOKEN_ID = 30

# bf16 logits tolerance for the FlashInfer path, matching the Qwen3-Omni
# integration precedent (test_prefill_cuda_graph.py: <= 1e-2 relative).
BF16_LOGITS_RTOL = 1e-2
BF16_LOGITS_ATOL = 2e-2
# fp32 dense-reference path: batched and isolated must agree to float noise.
FP32_LOGITS_RTOL = 1e-5
FP32_LOGITS_ATOL = 1e-5


# ---------------------------------------------------------------------------
# Dense reference attention resource
# ---------------------------------------------------------------------------


class DenseReferenceAttentionManager(AttentionManager):
    """Resource-pool-native dense SDPA oracle for the FlashInfer path.

    The production ``KVManager`` owns page allocation and writes. This test
    resource consumes its plan views and layer page tensor, gathers each
    request back to a contiguous sequence, expands GQA explicitly, and runs
    fp32 SDPA. It is deliberately a test-only resource instead of a legacy
    ``BatchedCacheManager`` compatibility path.
    """

    def __init__(self, kv_cache: str):
        self._kv_cache_name = kv_cache
        self._plans: KVPlanOutputs | None = None
        self._causal = True

    @classmethod
    def build(cls, spec: "DenseReferenceAttentionSpec", info: EngineResourceInfo):
        del info
        return cls(spec.config.kv_cache)

    def depends_on(self) -> set[str]:
        return {self._kv_cache_name}

    def plan(self, step, ctx: StepContext):
        self.reset_default_cursors()
        plans = ctx.plan_results.get(self._kv_cache_name)
        assert plans is not None, f"dense reference expected KV plan {self._kv_cache_name!r}"
        self._plans = plans
        self._causal = bool(step.causal)

    def run(
        self,
        q: torch.Tensor,
        label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        assert self._plans is not None and kv_cache_layer is not None
        label = label or self.default_label
        plan = self._plans[label]
        page_size = kv_cache_layer.shape[2]
        outputs: list[torch.Tensor] = []
        offset = 0
        for view in plan.views:
            new = view.to_compute
            prior = view.length - new
            q_new = q[offset : offset + new]
            offset += new
            all_pos = torch.arange(view.length, device=q.device)
            pages = torch.tensor(view.page_idxs, dtype=torch.long, device=q.device)
            keys = kv_cache_layer[pages[all_pos // page_size], 0, all_pos % page_size]
            values = kv_cache_layer[pages[all_pos // page_size], 1, all_pos % page_size]
            groups = q_new.shape[1] // keys.shape[1]
            keys = keys.repeat_interleave(groups, dim=1)
            values = values.repeat_interleave(groups, dim=1)
            mask = None
            if self._causal:
                query_pos = torch.arange(prior, prior + new, device=q.device)[:, None]
                mask = all_pos[None, :] <= query_pos
            out = F.scaled_dot_product_attention(
                q_new.float().permute(1, 0, 2),
                keys.float().permute(1, 0, 2),
                values.float().permute(1, 0, 2),
                attn_mask=mask,
            )
            outputs.append(out.permute(1, 0, 2).to(q.dtype))
        assert offset == q.shape[0], f"planned {offset} tokens but received {q.shape[0]} queries"
        return torch.cat(outputs, dim=0)


@dataclass
class DenseReferenceAttentionSpec(NodeResourceSpec):
    config: AttentionConfig

    def depends_on(self) -> set[str]:
        return {self.config.kv_cache}

    @property
    def resource_class(self):
        return DenseReferenceAttentionManager


# ---------------------------------------------------------------------------
# CPU shims
# ---------------------------------------------------------------------------


class _NoopKVTransferManager:
    """CPU test replacement for the KV resource's CUDA/IPC transport."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def get_kv_transfer_info(self):
        return None

    def cleanup(self):
        return None


def torch_sample_tokens(
    logits: torch.Tensor,
    temperature=0.6,
    top_k=0,
    top_p=1.0,
    repetition_penalty=1.0,
    seen_token_mask=None,
    any_greedy=None,
    any_top_k_zero=None,
    all_top_k_zero=None,
    seed=None,
    rand_offset=None,
) -> torch.Tensor:
    """Pure-torch equivalent of ``mstar.utils.sampling.sample_tokens``.

    Greedy rows use argmax (bit-identical semantics to the production
    kernel); stochastic rows draw from a per-(seed, offset) generator so two
    runs with the same request seed produce the same stream.
    """
    batch, _vocab = logits.shape
    scores = logits.detach().float()
    temperature_t = sampling_module._to_tensor(temperature, batch, scores.device)
    top_k_t = sampling_module._to_tensor(top_k, batch, scores.device, dtype=torch.int32)
    top_p_t = sampling_module._to_tensor(top_p, batch, scores.device)
    if seen_token_mask is not None:
        penalty = sampling_module._to_tensor(repetition_penalty, batch, scores.device)[:, None]
        penalised = torch.where(scores > 0, scores / penalty, scores * penalty)
        scores = torch.where(seen_token_mask.bool(), penalised, scores)
    tokens = torch.empty(batch, dtype=torch.long)
    for i in range(batch):
        row = scores[i]
        if float(temperature_t[i]) == 0.0:
            tokens[i] = int(row.argmax())
            continue
        probs = torch.softmax(row / temperature_t[i], dim=-1)
        k = int(top_k_t[i])
        if 0 < k < probs.numel():
            threshold = torch.topk(probs, k).values[-1]
            probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
        p = float(top_p_t[i])
        if p < 1.0:
            sorted_probs, order = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative - sorted_probs < p
            filtered = torch.zeros_like(probs)
            filtered[order[keep]] = probs[order[keep]]
            probs = filtered
        probs = probs / probs.sum()
        generator = torch.Generator(device="cpu")
        seed_i = int(seed[i]) if seed is not None else 0
        offset_i = int(rand_offset[i]) if rand_offset is not None else 0
        generator.manual_seed((seed_i * 1_000_003 + offset_i) & 0x7FFF_FFFF_FFFF_FFFF)
        tokens[i] = int(torch.multinomial(probs.cpu(), 1, generator=generator))
    return tokens.to(logits.device)


def torch_rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = input.float() * torch.rsqrt(input.float().square().mean(-1, keepdim=True) + eps)
    return (normalized * weight.float()).to(input.dtype)


def install_cpu_shims(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the three CUDA-only leaves the engine touches on CPU: the
    CUDA-IPC KV transfer engine, FlashInfer's sampling kernel and
    FlashInfer's RMSNorm kernel."""
    monkeypatch.setattr(kv_manager_module, "KVTransferManager", _NoopKVTransferManager)
    monkeypatch.setattr(sampling_module, "sample_tokens", torch_sample_tokens)
    monkeypatch.setattr(norm_module, "run_rms_norm", torch_rms_norm)
    # FlashInfer's workspace buffer is irrelevant to the dense reference, but
    # the engine still allocates it; keep it tiny on CPU.
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "4")


# ---------------------------------------------------------------------------
# Device / backend selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    device: torch.device
    backend: str
    dtype: torch.dtype

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def is_flashinfer(self) -> bool:
        return self.backend == FLASHINFER_BACKEND

    @property
    def logits_tolerance(self) -> tuple[float, float]:
        if self.dtype == torch.float32:
            return FP32_LOGITS_RTOL, FP32_LOGITS_ATOL
        return BF16_LOGITS_RTOL, BF16_LOGITS_ATOL


def cpu_reference_target() -> Target:
    return Target(device=torch.device("cpu"), backend=DENSE_REFERENCE_BACKEND, dtype=torch.float32)


def cuda_flashinfer_target() -> Target:
    return Target(device=torch.device("cuda"), backend=FLASHINFER_BACKEND, dtype=torch.bfloat16)


def cuda_reference_target() -> Target:
    return Target(device=torch.device("cuda"), backend=DENSE_REFERENCE_BACKEND, dtype=torch.bfloat16)


def flashinfer_available() -> bool:
    if not torch.cuda.is_available():
        return False
    return importlib.util.find_spec("flashinfer") is not None


def require_target(target: Target) -> None:
    if target.is_cuda and not torch.cuda.is_available():
        pytest.skip("CUDA is required for this target")
    if target.is_flashinfer and not flashinfer_available():
        pytest.skip("FlashInfer is required for the paged attention target")


def prepare_target(target: Target, monkeypatch: pytest.MonkeyPatch) -> Target:
    require_target(target)
    if not target.is_cuda:
        install_cpu_shims(monkeypatch)
    return target


# Every engine-level PR1 test runs against both targets: the CPU fp32 dense
# reference is a dry run of the harness and engine plumbing; only the CUDA
# FlashInfer row is acceptance evidence (hence the ``cuda`` marker).
TARGET_PARAMS = [
    pytest.param(cpu_reference_target(), id="cpu-dense-fp32"),
    pytest.param(cuda_flashinfer_target(), id="cuda-flashinfer-bf16", marks=pytest.mark.cuda),
]


# ---------------------------------------------------------------------------
# Tiny model construction
# ---------------------------------------------------------------------------


def make_tiny_config(**overrides):
    config = tiny_config(image_token_id=TINY_IMAGE_TOKEN_ID)
    for key, value in overrides.items():
        setattr(config.text_config, key, value)
    return config


def randomize_parameters(module: torch.nn.Module, seed: int, std: float = 0.2) -> None:
    """Deterministically initialise every parameter (M*'s parallel layers
    allocate with ``torch.empty`` and expect a checkpoint to fill them)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, param in sorted(module.named_parameters(), key=lambda item: item[0]):
            values = torch.randn(param.shape, generator=generator, dtype=torch.float32) * std
            if name.endswith("norm.weight") or name.endswith(".q_norm.weight") or name.endswith(".k_norm.weight"):
                values = 1.0 + values * 0.1
            param.copy_(values.to(device=param.device, dtype=param.dtype))


def build_language_model(config, target: Target, seed: int = 0) -> QwenVLForCausalLM:
    model = QwenVLForCausalLM(config)
    randomize_parameters(model, seed)
    return model.to(device=target.device, dtype=target.dtype).eval()


def greedy_sampling(config, ignore_eos: bool = True) -> SamplingReqConfig:
    del config
    return SamplingReqConfig(temperature=0.0, ignore_eos=ignore_eos)


def stochastic_sampling(config, seed: int, temperature: float = 1.0, ignore_eos: bool = True) -> SamplingReqConfig:
    del config
    return SamplingReqConfig(temperature=temperature, ignore_eos=ignore_eos, _seed=seed)


# ---------------------------------------------------------------------------
# Engine handle
# ---------------------------------------------------------------------------


@dataclass
class LLMEngine:
    """A loaded resource-pool ``Engine`` hosting the tiny QwenVL LLM node."""

    engine: Engine
    submodule: QwenVLLLMSubmodule
    config: object
    target: Target
    kv_config: KVConfig
    node: str = LLM_NODE
    infos: dict[str, CurrentForwardPassInfo] = field(default_factory=dict)

    @property
    def kv(self):
        return self.engine._resources[KV_CACHE]

    @property
    def alloc_manager(self):
        """Test-facing view of resource-owned request streams.

        The production cache has no split allocator object after the resource
        migration. Tests use this only to assert request membership; page
        ownership stays on ``KVManager``.
        """
        from types import SimpleNamespace

        return SimpleNamespace(request_states=self.kv._streams)

    @property
    def kv_cache(self) -> torch.Tensor:
        return self.kv.kv_cache.tensor

    @property
    def sampler(self):
        return self.engine._resources[SAMPLER]._sampler

    @property
    def free_pages(self) -> int:
        return self.kv._arena.num_free

    @property
    def total_pages(self) -> int:
        # Resource KV reserves sink page zero for captured padding. The harness
        # exposes the caller-requested usable capacity, matching its tests.
        return self.kv.config.max_num_pages - 1

    def page_indices(self, rid: str, label: str = "main") -> list[int]:
        return list(self.kv._streams[rid][label].page_indices)

    def seq_len(self, rid: str, label: str = "main") -> int:
        return self.kv._streams[rid][label].stored_len

    def position_start(self, rid: str, label: str = "main") -> int:
        return self.engine._resources[POS].position(rid, label)

    def add_request(self, rid: str, sampling: SamplingReqConfig | None = None, max_tokens: int = 64) -> None:
        sampling = sampling or greedy_sampling(self.config)
        self.engine.add_request(rid, {SAMPLER: sampling})
        self.infos[rid] = CurrentForwardPassInfo(
            request_id=rid,
            graph_walk="prefill",
            fwd_index=0,
            random_seed=0,
            max_tokens=max_tokens,
            resource_configs={SAMPLER: sampling},
        )

    def remove_request(self, rid: str) -> None:
        self.engine.remove_request(rid)
        self.infos.pop(rid, None)

    def has_request(self, rid: str) -> bool:
        return rid in self.kv._streams

    def batch(self, graph_walk: str, tensors: dict[str, dict[str, list[torch.Tensor]]]) -> ExecutingBatch:
        rids = list(tensors)
        per_request_info = {}
        for rid in rids:
            info = self.infos[rid]
            info.graph_walk = graph_walk
            per_request_info[rid] = info
        return ExecutingBatch(
            node_name=self.node,
            step_context=StepContext(
                request_ids=rids,
                graph_walk=graph_walk,
                slot=0,
                capture=False,
            ),
            per_request_input_tensors={
                rid: {name: [t.to(self.target.device) for t in values] for name, values in tensors[rid].items()}
                for rid in rids
            },
            per_request_info=per_request_info,
        )

    def run(self, batch: ExecutingBatch, *, force_sequential: bool = False, allow_alloc_failure: bool = False):
        """Execute through the production entrypoints.

        ``force_sequential`` drives the resource engine's per-request fallback
        by making ``can_batch`` decline; otherwise a homogeneous batch uses one
        packed ``forward_batched`` launch.
        """
        submodule = self.submodule
        original = submodule.can_batch
        if force_sequential:
            submodule.can_batch = lambda *_args, **_kwargs: False  # type: ignore[method-assign]
        try:
            self.engine.prepare_inputs(batch)
            output = self.engine.exec_and_postprocess(batch)
        finally:
            if force_sequential:
                submodule.can_batch = original  # type: ignore[method-assign]
            self.engine.finalize_batch(batch)
        if batch.failed_requests:
            raise AssertionError(f"engine reported failed requests: {batch.failed_requests}")
        if batch.admit_error is not None and not allow_alloc_failure:
            raise AssertionError(f"unexpected resource admission failure: {batch.admit_error}")
        return output

    def tokens(self, output: dict[str, dict[str, list[torch.Tensor]]]) -> dict[str, int]:
        tokens: dict[str, int] = {}
        sampler = self.engine._resources[SAMPLER]
        for rid, out in output.items():
            if "new_token" in out:
                tokens[rid] = int(out["new_token"][0].reshape(-1)[0])
            else:
                # The resource engine's unbatched path calls ``forward`` and
                # returns logits; production worker postprocessing samples
                # those rows. Mirror that outer sampling seam here so the
                # sequential reference remains semantically equivalent to the
                # packed ``forward_batched`` path.
                tokens[rid] = int(sampler.sample([rid], out["logits"][0])[0])
        return tokens

    @contextlib.contextmanager
    def capture_logits(self) -> Iterator[list[torch.Tensor]]:
        """Record every ``lm_head`` output produced while the context is open.

        Batched launches emit one ``[B, V]`` tensor (rows in
        ``batch.request_ids`` order); the sequential path emits one ``[1, V]``
        tensor per request in the same order.
        """
        captured: list[torch.Tensor] = []
        handle = self.submodule.lm_head.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output.detach().float().cpu())
        )
        try:
            yield captured
        finally:
            handle.remove()

    @contextlib.contextmanager
    def capture_hidden(self) -> Iterator[list[torch.Tensor]]:
        """Record the final-norm hidden states for every token of each launch
        (``[total_tokens, hidden]`` in packed request order), which exposes
        per-position attention output rather than only the last token."""
        captured: list[torch.Tensor] = []
        handle = self.submodule.language_model.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output.detach().float().cpu())
        )
        try:
            yield captured
        finally:
            handle.remove()


def build_llm_engine(
    target: Target,
    *,
    config=None,
    seed: int = 0,
    max_num_pages: int = 64,
    page_size: int = 16,
    max_seq_len: int = 4096,
    language_model: QwenVLForCausalLM | None = None,
) -> LLMEngine:
    config = config or make_tiny_config()
    text = config.text_config
    if language_model is None:
        language_model = build_language_model(config, target, seed=seed)
    else:
        # Resource binding is mutable: each attention layer holds the KV and
        # attention resources of the Engine that owns it. A batched/reference
        # comparison therefore needs equal *weights* in separate module
        # instances, just as two real workers do; sharing one module would
        # silently rebind the first engine to the second engine's resources.
        source_state = language_model.state_dict()
        language_model = QwenVLForCausalLM(config)
        language_model.load_state_dict(source_state)
        language_model = language_model.to(device=target.device, dtype=target.dtype).eval()
    submodule = QwenVLLLMSubmodule(language_model, config)
    kv_config = KVConfig(
        num_layers=text.num_hidden_layers,
        num_kv_heads=text.num_key_value_heads,
        head_dim=text.head_dim,
        max_seq_len=max_seq_len,
        # Sink page zero is owned by the resource engine, so allocate one
        # additional physical page to preserve the caller's usable budget.
        max_num_pages=max_num_pages + 1,
        page_size=page_size,
        num_qo_heads=text.num_attention_heads,
    )
    attn_spec: NodeResourceSpec
    if target.backend == DENSE_REFERENCE_BACKEND:
        attn_spec = DenseReferenceAttentionSpec(
            resource_key=ATTN,
            nodes={LLM_NODE},
            config=AttentionConfig(kv_cache=KV_CACHE),
        )
    else:
        attn_spec = AttentionSpec(
            resource_key=ATTN,
            nodes={LLM_NODE},
            config=AttentionConfig(kv_cache=KV_CACHE),
        )
    engine = Engine(autocast_dtype=target.dtype)
    engine.load_model(
        {LLM_NODE: submodule},
        specs=[
            KVSpec(resource_key=KV_CACHE, nodes={LLM_NODE}, config=kv_config),
            attn_spec,
            PositionSpec(resource_key=POS, nodes={LLM_NODE}, config=PositionConfig(kv_cache=KV_CACHE)),
            SamplerSpec(
                resource_key=SAMPLER,
                nodes={LLM_NODE},
                vocab_size=text.vocab_size,
                enable_repetion_penalty=True,
            ),
        ],
        parallel_groups=WorkerParallelGroups(global_rank=0, num_workers=1),
        device=target.device,
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="worker0",
            my_session_id="local",
            transfer_engine=LocalTransferEngine("localhost"),
        ),
        kv_cache_type=target.dtype,
    )
    return LLMEngine(engine=engine, submodule=submodule, config=config, target=target, kv_config=kv_config)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


@dataclass
class TextPrompt:
    ids: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.ids.numel())

    def tensors(self, config) -> dict[str, list[torch.Tensor]]:
        positions = torch.arange(self.length, dtype=torch.long).unsqueeze(0).expand(3, -1).contiguous()
        return {"text_inputs": [self.ids], "position_ids": [positions]}


@dataclass
class VisionPrompt:
    ids: torch.Tensor
    grid: tuple[int, int, int]
    vision_embeds: torch.Tensor
    deepstack: list[torch.Tensor]

    @property
    def length(self) -> int:
        return int(self.ids.numel())

    @property
    def num_visual_tokens(self) -> int:
        return int(self.vision_embeds.shape[0])

    def position_ids(self, config) -> torch.Tensor:
        return qwen_vl_position_ids(self.ids, torch.tensor([list(self.grid)]), config)

    def position_span(self, config) -> int:
        return int(self.position_ids(config).max()) + 1

    def tensors(self, config) -> dict[str, list[torch.Tensor]]:
        return {
            "text_inputs": [self.ids],
            "position_ids": [self.position_ids(config)],
            "vision_embeds": [self.vision_embeds],
            "deepstack_visual_embeds": list(self.deepstack),
        }


def text_prompt(length: int, seed: int, config=None) -> TextPrompt:
    config = config or make_tiny_config()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vocab = config.text_config.vocab_size
    # Exclude BOS/EOS and the image placeholder so a text prompt is pure text.
    excluded = (TINY_IMAGE_TOKEN_ID, config.text_config.eos_token_id)
    candidates = torch.tensor([t for t in range(1, vocab) if t not in excluded])
    picks = torch.randint(0, candidates.numel(), (length,), generator=generator)
    return TextPrompt(ids=candidates[picks].to(torch.long))


def vision_prompt(
    grid: tuple[int, int, int],
    seed: int,
    *,
    prefix: int = 3,
    suffix: int = 2,
    config=None,
    feature_scale: float = 1.0,
) -> VisionPrompt:
    config = config or make_tiny_config()
    merge = config.vision_config.spatial_merge_size
    t, h, w = grid
    count = t * (h // merge) * (w // merge)
    hidden = config.text_config.hidden_size
    generator = torch.Generator(device="cpu").manual_seed(seed)
    text_ids = text_prompt(prefix + suffix, seed + 7_919, config).ids
    ids = torch.cat([text_ids[:prefix], torch.full((count,), TINY_IMAGE_TOKEN_ID, dtype=torch.long), text_ids[prefix:]])
    vision = torch.randn(count, hidden, generator=generator) * feature_scale
    deepstack = [
        torch.randn(count, hidden, generator=generator) * feature_scale
        for _ in config.vision_config.deepstack_visual_indexes
    ]
    return VisionPrompt(ids=ids, grid=grid, vision_embeds=vision, deepstack=deepstack)


def prefill_batch(handle: LLMEngine, prompts: dict[str, TextPrompt | VisionPrompt]) -> ExecutingBatch:
    walks = {"prefill_vision" if isinstance(p, VisionPrompt) else "prefill" for p in prompts.values()}
    assert len(walks) == 1, f"a prefill batch must be single-walk; got {sorted(walks)}"
    return handle.batch(walks.pop(), {rid: prompt.tensors(handle.config) for rid, prompt in prompts.items()})


def decode_batch(handle: LLMEngine, tokens: dict[str, int]) -> ExecutingBatch:
    return handle.batch(
        "decode",
        {rid: {"text_inputs": [torch.tensor([token], dtype=torch.long)]} for rid, token in tokens.items()},
    )


@dataclass
class StepResult:
    """Last-token logits and sampled token per request for one engine launch."""

    logits: dict[str, torch.Tensor]
    tokens: dict[str, int]

    def margin(self, rid: str) -> float:
        return top2_margin(self.logits[rid])


def _run_step(handle: LLMEngine, batch: ExecutingBatch, *, sequential: bool) -> StepResult:
    rids = list(batch.request_ids)
    with handle.capture_logits() as captured:
        output = handle.run(batch, force_sequential=sequential)
    if sequential:
        assert len(captured) == len(rids), (
            f"sequential path must launch once per request ({len(captured)} vs {len(rids)})"
        )
        logits = {rid: captured[i].reshape(-1) for i, rid in enumerate(rids)}
    else:
        assert len(captured) == 1, f"batched path must launch exactly once ({len(captured)} launches)"
        assert captured[0].shape[0] == len(rids)
        logits = {rid: captured[0][i] for i, rid in enumerate(rids)}
    return StepResult(logits=logits, tokens=handle.tokens(output))


def run_prefill(
    handle: LLMEngine, prompts: dict[str, TextPrompt | VisionPrompt], *, sequential: bool = False
) -> StepResult:
    """One prefill launch for ``prompts``: ``_execute_batched`` by default,
    or the per-request ``_execute_sequential`` reference path."""
    return _run_step(handle, prefill_batch(handle, prompts), sequential=sequential)


def run_decode(handle: LLMEngine, tokens: dict[str, int], *, sequential: bool = False) -> StepResult:
    return _run_step(handle, decode_batch(handle, tokens), sequential=sequential)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def assert_logits_close(actual: torch.Tensor, expected: torch.Tensor, target: Target, what: str) -> None:
    rtol, atol = target.logits_tolerance
    actual = actual.float().cpu().reshape(-1)
    expected = expected.float().cpu().reshape(-1)
    diff = (actual - expected).abs()
    scale = expected.abs().max().clamp_min(1e-6)
    rel = float(diff.max() / scale)
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"{what}: logits mismatch (max abs diff {float(diff.max()):.3e}, max rel-to-scale {rel:.3e}, "
        f"tolerance rtol={rtol}, atol={atol})\nactual={actual.tolist()}\nexpected={expected.tolist()}"
    )


def top2_margin(logits: torch.Tensor) -> float:
    values = torch.topk(logits.float().reshape(-1), 2).values
    return float(values[0] - values[1])


def assert_greedy_streams_match(
    actual: list[int],
    expected: list[int],
    expected_margins: list[float] | None,
    target: Target,
    what: str,
) -> None:
    """Compare two greedy token streams.

    On the fp32 dense path the streams must be identical. On the bf16
    FlashInfer path a divergence is only tolerated when the reference's
    top-2 logit margin at that step is within bf16 noise of the logits scale
    (a pure close-call swap); everything before the divergence must match,
    and the first divergence ends the comparison because the trajectories
    are no longer comparable afterwards.
    """
    if target.dtype == torch.float32:
        assert actual == expected, f"{what}: token streams differ\nactual={actual}\nexpected={expected}"
        return
    assert expected_margins is not None and len(expected_margins) == len(expected)
    rtol, atol = target.logits_tolerance
    assert len(actual) == len(expected), f"{what}: stream lengths differ ({len(actual)} vs {len(expected)})"
    for step, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if a == e:
            continue
        assert expected_margins[step] <= atol + rtol * 4.0, (
            f"{what}: token streams diverge at step {step} ({a} vs {e}) with a clear reference "
            f"margin of {expected_margins[step]:.3e}\nactual={actual}\nexpected={expected}"
        )
        return


def sync(target: Target) -> None:
    if target.is_cuda:
        torch.cuda.synchronize(target.device)
