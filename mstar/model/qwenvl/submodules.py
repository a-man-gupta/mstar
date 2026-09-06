"""Engine adapters for the Qwen3-VL vision and MoE language nodes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.model.qwenvl.components import compute_mrope_cos_sin
from mstar.model.qwenvl.config import ATTN, KV_CACHE, POS, SAMPLER
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule, ModelInputsFromEngine, NodeInputs, NodeSubmodule


def qwen_vl_position_ids(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    config,
) -> torch.Tensor:
    """Construct Qwen3-VL's three-axis positions for one image/text prompt.

    The processor expands each image placeholder to exactly the post-merge
    token count.  We replace every such contiguous run with the image's
    temporal/height/width grid and keep ordinary text positions identical on
    all three axes.  This is the HF ``get_rope_index`` contract for the
    image+text subset.
    """
    input_ids = input_ids.reshape(-1)
    positions: list[torch.Tensor] = []
    cursor = 0
    text_position = 0
    grids = [] if image_grid_thw is None else image_grid_thw.reshape(-1, 3).tolist()
    grid_index = 0
    while cursor < input_ids.numel():
        if input_ids[cursor].item() != config.image_token_id:
            positions.append(torch.full((3, 1), text_position, device=input_ids.device, dtype=torch.long))
            cursor += 1
            text_position += 1
            continue
        if grid_index >= len(grids):
            raise ValueError("QwenVL prompt has image tokens but no corresponding image_grid_thw entry.")
        temporal, height, width = (int(value) for value in grids[grid_index])
        merge = config.vision_config.spatial_merge_size
        if height % merge or width % merge:
            raise ValueError(f"Image grid {(temporal, height, width)} is not divisible by spatial_merge_size={merge}.")
        merged_h, merged_w = height // merge, width // merge
        count = temporal * merged_h * merged_w
        run = input_ids[cursor : cursor + count]
        if run.numel() != count or not torch.equal(run, torch.full_like(run, config.image_token_id)):
            raise ValueError("QwenVL image-token run does not match the processor's post-merge grid size.")
        t = torch.arange(temporal, device=input_ids.device).repeat_interleave(merged_h * merged_w)
        h = torch.arange(merged_h, device=input_ids.device).repeat_interleave(merged_w).repeat(temporal)
        w = torch.arange(merged_w, device=input_ids.device).repeat(temporal * merged_h)
        positions.append(torch.stack((t, h, w)) + text_position)
        cursor += count
        # Qwen's next text position follows the largest coordinate in the
        # merged T/H/W grid, not the number of image embeddings.  A 2x2 image
        # consumes four language slots but only spans two spatial positions.
        text_position += max(temporal, merged_h, merged_w)
        grid_index += 1
    if grid_index != len(grids):
        raise ValueError("QwenVL processor emitted image grids without matching image-token placeholders.")
    return torch.cat(positions, dim=1)


def _last_token_indices(seq_lens: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(seq_lens, device=device, dtype=torch.long).cumsum(0) - 1


# The only graph walk on which the LLM node may receive vision tensors. Text
# ``prefill`` and ``decode`` requests must never carry visual state into a
# batch: same-walk batching (PR1 boundary) relies on every request in a
# launch sharing one embedding/DeepStack code path.
VISION_WALK = "prefill_vision"
_VISION_TENSOR_KEYS = ("vision_embeds", "deepstack_visual_embeds")


def _has_vision_inputs(request: NodeInputs) -> bool:
    return any(request.tensor_inputs.get(key) is not None for key in _VISION_TENSOR_KEYS)


def qwenvl_batch_is_homogeneous(graph_walk: str, inputs: list[NodeInputs]) -> bool:
    """PR1 same-walk batching contract for the QwenVL LLM node.

    * ``prefill``: every request is text-only (no vision tensors).
    * ``prefill_vision``: every request carries vision + DeepStack tensors.
    * ``decode``: every request is a single-token step with no vision tensors.

    The scheduler already groups by ``(node, graph_walk)``; this check keeps a
    mislabelled or partially-populated request from silently sharing a packed
    launch with requests on a different embedding path.
    """
    if not inputs:
        return False
    with_vision = [_has_vision_inputs(request) for request in inputs]
    if graph_walk == VISION_WALK:
        return all(with_vision)
    if any(with_vision):
        return False
    if graph_walk == "decode":
        return all(getattr(request, "input_seq_len", 1) == 1 for request in inputs)
    return graph_walk == "prefill"


class QwenVLVisionSubmodule(NodeSubmodule):
    """HF's checkpoint-compatible Qwen vision tower as a stateless node."""

    disable_torch_compile = True

    def __init__(self, vision_model: nn.Module):
        super().__init__()
        self.vision_model = vision_model

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        return NodeInputs(
            tensor_inputs={
                "pixel_values": inputs["pixel_values"][0],
                "image_grid_thw": inputs["image_grid_thw"][0],
                # These edges deliberately pass through the stateless vision
                # node so the following LLM node sees one coherent prompt.
                "text_inputs": inputs["text_inputs"][0],
                "position_ids": inputs["position_ids"][0],
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        text_inputs: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        vision_embeds, deepstack_visual_embeds = self.vision_model(pixel_values, grid_thw=image_grid_thw)
        return {
            "vision_embeds": [vision_embeds],
            "deepstack_visual_embeds": list(deepstack_visual_embeds),
            "text_inputs": [text_inputs],
            "position_ids": [position_ids],
        }


class QwenVLLLMSubmodule(ARNodeSubmodule):
    """MoE text backbone with packed prefill/decode and Qwen3-VL MRoPE."""

    def __init__(self, language_model: nn.Module, config):
        super().__init__()
        self.language_model = language_model
        self.config = config
        self.embed_tokens = language_model.model.embed_tokens
        self.lm_head = language_model.lm_head

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        resources: dict | None = None,
        **kwargs,
    ) -> ARNodeInputs:
        input_ids = inputs["text_inputs"][0]
        position_start = 0
        pos_info = kwargs.get("pos_info") or {}
        if "main" in pos_info:
            position_start = getattr(pos_info["main"], "position_id_start", 0)
        elif resources is not None and POS in resources and fwd_info is not None:
            published = resources[POS].publish(fwd_info.request_id)
            if published is not None:
                position_start = published.counters.get("main", 0)
        if graph_walk == "prefill_vision":
            position_ids = inputs["position_ids"][0]
            return ARNodeInputs(
                input_ids=input_ids,
                input_seq_len=input_ids.numel(),
                custom_pos_ids=position_ids,
                tensor_inputs={
                    "vision_embeds": inputs["vision_embeds"][0],
                    "deepstack_visual_embeds": inputs["deepstack_visual_embeds"],
                },
                kwargs={"position_advance": int(position_ids.max()) + 1 - position_start},
            )
        if graph_walk == "prefill":
            position_ids = inputs["position_ids"][0]
            return ARNodeInputs(
                input_ids=input_ids,
                input_seq_len=input_ids.numel(),
                custom_pos_ids=position_ids,
                kwargs={"position_advance": int(position_ids.max()) + 1 - position_start},
            )
        if graph_walk != "decode":
            raise ValueError(f"Unknown QwenVL graph walk {graph_walk!r}.")
        positions = torch.arange(input_ids.numel(), device=input_ids.device, dtype=torch.long) + position_start
        return ARNodeInputs(
            input_ids=input_ids,
            input_seq_len=input_ids.numel(),
            custom_pos_ids=positions.unsqueeze(0).expand(3, -1),
            kwargs={"position_advance": input_ids.numel()},
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        del graph_walk, slot_lease, piecewise_leases, kwargs
        pos_advance = tuple(
            int(inp.kwargs.get("position_advance", inp.input_seq_len)) for inp in inputs
        )
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                KV_CACHE: KVStep(),
                ATTN: AttentionStep(causal=True),
                SAMPLER: SamplerStep(apply_penalty=True),
                POS: PositionStep(advance=pos_advance),
            },
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        if not qwenvl_batch_is_homogeneous(graph_walk, inputs):
            raise ValueError(
                f"QwenVL LLM received a non-homogeneous {graph_walk!r} batch: vision tensors are "
                f"only allowed (and then required) on the {VISION_WALK!r} walk, and decode steps "
                "must be single-token. Requests with vision: "
                f"{[_has_vision_inputs(request) for request in inputs]}."
            )
        cache = getattr(engine_inputs, "cache_manager", None)
        seq_lens = [request.input_seq_len for request in inputs]
        position_ids = torch.cat([request.custom_pos_ids for request in inputs], dim=1)
        position_advance = [int(request.kwargs["position_advance"]) for request in inputs]
        # Compatibility path for component tests that still inject a fake
        # cache_manager. Production planning lives on declare_step.
        # Keep the MRoPE span on that side channel so advance_seq_lens()
        # moves each request by its 3D-position span (P1-G6 is eager-only).
        if cache is not None:
            cache.set_active_label("main")
            cache.plan_attention(seq_lens=seq_lens, is_causal=True, label="main")
            cache.set_custom_pos_advance(position_advance, label="main")
        packed: dict[str, torch.Tensor | Any] = {
            "text_inputs": torch.cat([request.input_ids for request in inputs]),
            "position_ids": position_ids,
            "cos_3d": None,
            "sin_3d": None,
            "position_advance": position_advance,
            "seq_lens": seq_lens,
        }
        if graph_walk == VISION_WALK:
            vision = [request.tensor_inputs.get("vision_embeds") for request in inputs]
            deepstack = [request.tensor_inputs.get("deepstack_visual_embeds") for request in inputs]
            if any(item is None for item in vision) or any(item is None for item in deepstack):
                raise ValueError(
                    f"Every QwenVL {VISION_WALK!r} request must carry both 'vision_embeds' and "
                    "'deepstack_visual_embeds'."
                )
            packed["vision_embeds"] = torch.cat(vision, dim=0)
            packed["visual_token_mask"] = packed["text_inputs"] == self.config.image_token_id
            expected_layers = len(self.config.vision_config.deepstack_visual_indexes)
            for item in deepstack:
                if len(item) != expected_layers:
                    raise ValueError(
                        "QwenVL vision request has the wrong number of DeepStack feature sets; "
                        f"got {len(item)}, expected {expected_layers}."
                    )
            packed["deepstack_visual_embeds"] = [
                torch.cat([item[layer] for item in deepstack], dim=0) for layer in range(expected_layers)
            ]
        cos, sin = compute_mrope_cos_sin(
            position_ids,
            head_dim=self.config.text_config.head_dim,
            rope_theta=self.config.text_config.rope_theta,
            mrope_section=tuple(self.config.text_config.rope_scaling["mrope_section"]),
            dtype=self.embed_tokens.weight.dtype,
        )
        packed["cos_3d"] = cos
        packed["sin_3d"] = sin
        return packed

    def _merge_embeddings(self, text_inputs: torch.Tensor, vision_embeds: torch.Tensor | None) -> torch.Tensor:
        embeddings = self.embed_tokens(text_inputs)
        if vision_embeds is None:
            return embeddings
        image_mask = text_inputs == self.config.image_token_id
        if int(image_mask.sum()) != vision_embeds.shape[0]:
            raise ValueError(
                "QwenVL image placeholder count must equal vision feature count; "
                f"got {int(image_mask.sum())} placeholders and {vision_embeds.shape[0]} features."
            )
        embeddings = embeddings.clone()
        embeddings[image_mask] = vision_embeds.to(device=embeddings.device, dtype=embeddings.dtype)
        return embeddings

    def _hidden_states(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        text_inputs: torch.Tensor,
        position_ids: torch.Tensor,
        vision_embeds: torch.Tensor | None,
        position_advance,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        visual_token_mask: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if graph_walk != VISION_WALK and (
            vision_embeds is not None or visual_token_mask is not None or deepstack_visual_embeds is not None
        ):
            # Visual state is owned by the prefill_vision step of the request
            # that produced it. Decode (and text prefill) must never re-inject
            # it, or one request's image would bleed into a shared launch.
            raise ValueError(f"QwenVL {graph_walk!r} forward must not receive vision or DeepStack tensors.")
        embeddings = self._merge_embeddings(text_inputs, vision_embeds)
        cache = getattr(engine_inputs, "cache_manager", None)
        return self.language_model(
            embeddings,
            cache,
            position_ids,
            position_advance=position_advance,
            cos=cos_3d,
            sin=sin_3d,
            visual_token_mask=visual_token_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        text_inputs: torch.Tensor,
        position_ids: torch.Tensor,
        vision_embeds: torch.Tensor | None = None,
        position_advance: int | list[int] | None = None,
        seq_lens: list[int] | None = None,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        visual_token_mask: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> NameToTensorList:
        hidden = self._hidden_states(
            graph_walk,
            engine_inputs,
            text_inputs,
            position_ids,
            vision_embeds,
            position_advance,
            cos_3d,
            sin_3d,
            visual_token_mask,
            deepstack_visual_embeds,
        )
        if seq_lens is None:
            last = hidden[-1:]
        else:
            last = hidden.index_select(0, _last_token_indices(seq_lens, hidden.device))
        return {"logits": [self.lm_head(last)]}

    def get_needed_cache_labels(
        self,
        graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> list[str]:
        """QwenVL keeps a single ``main`` KV label on every walk; the vision
        encoder is stateless, so there is nothing else to transfer."""
        return ["main"]

    def get_cuda_graph_configs(self, device: torch.device, tp_world_size: int = 1) -> list:
        """PR1 scope is eager-only (P1-G6 option A).

        ``forward``/``forward_batched`` still contain host syncs and
        data-dependent shapes (``int(image_mask.sum())``, the embedding
        ``clone`` + masked scatter, DeepStack injection), so no walk opts into
        CUDA-graph capture. Enabling capture is a follow-on that must come with
        eager-vs-replay parity evidence; see
        ``docs/qwenvl/PR_1_CONTINUOUS_BATCHING.md``.
        """
        return []

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        """Batch only same-walk, same-embedding-path requests (PR1 boundary).

        A heterogeneous group never reaches a packed launch that mixes text
        and vision paths; ``preprocess`` then rejects a mislabelled request
        loudly instead of silently packing it.
        """
        return qwenvl_batch_is_homogeneous(batch.graph_walk, model_inputs)

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        text_inputs: torch.Tensor,
        position_ids: torch.Tensor,
        vision_embeds: torch.Tensor | None = None,
        position_advance: int | list[int] | None = None,
        seq_lens: list[int] | None = None,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        visual_token_mask: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        hidden = self._hidden_states(
            graph_walk,
            engine_inputs,
            text_inputs,
            position_ids,
            vision_embeds,
            position_advance,
            cos_3d,
            sin_3d,
            visual_token_mask,
            deepstack_visual_embeds,
        )
        if seq_lens is None:
            seq_lens = [1] * len(engine_inputs.request_ids)
        last = hidden.index_select(0, _last_token_indices(seq_lens, hidden.device))
        logits = self.lm_head(last)
        resources = getattr(engine_inputs, "resources", None) or {}
        sampler = resources.get(SAMPLER, getattr(engine_inputs, "sampler", None))
        request_ids = engine_inputs.request_ids
        if sampler is None:
            return {rid: {"logits": [logits[i : i + 1]]} for i, rid in enumerate(request_ids)}
        new_tokens = sampler.sample(request_ids, logits, apply_penalty=True)
        return {rid: {"new_token": [new_tokens[i : i + 1]]} for i, rid in enumerate(request_ids)}

    def postprocess(self, request_id, request_info, outputs, **kwargs):
        # The engine samples logits and the decode graph consumes ``new_token``.
        if "new_token" in outputs:
            outputs["text_inputs"] = outputs["new_token"]

    def check_stop(self, request_id, request_info, outputs):
        # ``decode_loop`` belongs only to the decode graph walk. Prefill also
        # emits the first token, but attempting to stop this loop from a
        # prefill walk is a model-logic error that the worker must ignore.
        if "new_token" not in outputs or request_info.graph_walk != "decode":
            return set()
        token = outputs["new_token"][0].item()
        sampling_config = request_info.resource_configs.get(SAMPLER)
        ignore_eos = False if sampling_config is None else sampling_config.ignore_eos
        reached_eos = not ignore_eos and token == self.config.text_config.eos_token_id
        reached_limit = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1 >= request_info.max_tokens
        return {"decode_loop"} if reached_eos or reached_limit else set()
