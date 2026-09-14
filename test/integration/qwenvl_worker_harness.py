"""In-process worker harness for the PR1 QwenVL scheduler tests (P1-G1).

Assembles the production worker stack for a single-rank deployment --
``EngineManager.build`` (the resource-pool ``Engine``),
``WorkerGraphsManager`` + ``WorkerGraphQueues`` built from the QwenVL model's
own ``get_graph_walk_graphs`` declaration, ``MicroScheduler`` and a
``TensorCommunicationManager`` -- and drives it with the same call sequence
``Worker._postprocess_batch`` / ``Worker._send_outputs`` use. The only
pieces replaced are the transports: tensors live in the local
``TensorStore`` (no RDMA / SHM) and the conductor round-trip
(``WORKER_GRAPHS_DONE`` -> ``get_partition_forward_pass_args`` -> next walk)
is executed inline instead of over ZMQ.

Every ``ScheduledBatch`` the scheduler emits, and every ``ExecutingBatch`` that
reaches an engine's ``execute_batch``, is recorded in ``observed`` so tests
can assert the batch shapes (node, walk, request ids) that actually ran.
"""

from __future__ import annotations

import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field

import qwenvl_harness as H
import torch
from torch import nn

from mstar.communication.tensors import LocalTransferEngine, TensorCommunicationManager
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.model.qwenvl.qwenvl_model import QwenVLModel
from mstar.model.qwenvl.submodules import QwenVLLLMSubmodule, QwenVLVisionSubmodule
from mstar.worker.engine_manager import EngineManager
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch, SchedulingType
from mstar.worker.node_manager_utils import WorkerGraphQueues, WorkerGraphsManager

WORKER_ID = "worker0"
PARTITION = "default"
VISION_NODE = "vision_encoder"
PATCH_DIM = 8


# ---------------------------------------------------------------------------
# Tiny vision tower + model
# ---------------------------------------------------------------------------


class TinyVisionTower(nn.Module):
    """Stands in for the HF Qwen3-VL vision tower: ``(pixel_values, grid_thw)``
    -> ``(merged embeddings, DeepStack feature sets)`` with the same shapes."""

    def __init__(self, config):
        super().__init__()
        self.merge = config.vision_config.spatial_merge_size
        hidden = config.text_config.hidden_size
        self.proj = nn.Linear(PATCH_DIM, hidden, bias=False)
        self.deepstack = nn.ModuleList(
            [nn.Linear(PATCH_DIM, hidden, bias=False) for _ in config.vision_config.deepstack_visual_indexes]
        )

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        merged = pixel_values.reshape(-1, self.merge * self.merge, PATCH_DIM).mean(dim=1)
        return self.proj(merged), [layer(merged) for layer in self.deepstack]


def vision_pixels(grid: tuple[int, int, int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    t, h, w = grid
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(t * h * w, PATCH_DIM, generator=generator), torch.tensor([[t, h, w]], dtype=torch.long)


class TinyQwenVLModel(QwenVLModel):
    """``QwenVLModel`` with the checkpoint/processor replaced by tiny seeded
    modules. Graph declaration, forward-pass args, sampling config and
    sharding config are inherited unchanged."""

    def __init__(self, config, target: H.Target, *, seed: int, max_output_tokens: int, kv_config: H.KVConfig):
        self.config = config
        self.target = target
        self._max_output_tokens = max_output_tokens
        self._kv_config = kv_config
        self._submodule_cache = {}
        self._decode_token_ids = {}
        self._decode_text = {}
        language_model = H.build_language_model(config, target, seed=seed)
        tower = TinyVisionTower(config)
        H.randomize_parameters(tower, seed + 1)
        tower = tower.to(device=target.device, dtype=target.dtype).eval()
        self._submodules = {
            H.LLM_NODE: QwenVLLLMSubmodule(language_model, config),
            VISION_NODE: QwenVLVisionSubmodule(tower),
        }

    def get_max_output_tokens(self, **model_kwargs):
        return self._max_output_tokens

    def get_node_resources(self):
        specs = super().get_node_resources()
        for index, spec in enumerate(specs):
            if spec.resource_key == H.KV_CACHE:
                spec.config = self._kv_config
            elif spec.resource_key == H.ATTN and self.target.backend == H.DENSE_REFERENCE_BACKEND:
                specs[index] = H.DenseReferenceAttentionSpec(
                    resource_key=H.ATTN,
                    nodes={H.LLM_NODE},
                    config=H.AttentionConfig(kv_cache=H.KV_CACHE),
                )
        return specs

    def get_autocast_dtype(self):
        return self.target.dtype

    def get_submodule(self, node_name, device="cpu", tp_group=None, autocast_dtype=None, **kwargs):
        return self._submodules[node_name]


# ---------------------------------------------------------------------------
# Local tensor transport
# ---------------------------------------------------------------------------


class InProcessTensorManager(TensorCommunicationManager):
    """All producers and consumers share one ``TensorStore``; nothing is
    ever read remotely, so ``register_for_send`` / ``start_read_tensors``
    are bookkeeping only."""

    def __init__(self, device: torch.device):
        super().__init__(
            my_entity_id=WORKER_ID,
            my_session_id="local",
            device=str(device),
            communicator=None,
            transfer_engine=LocalTransferEngine("localhost"),
        )

    def register_for_send(self, request_id, tensor_infos, skip_cuda_sync=False):
        for info in tensor_infos:
            self.tensor_store.set_metadata(request_id, info.uuid, mem_registered=True)

    def start_read_tensors(self, request_id, graph_edges, graph_walk=None):
        return []

    def _cleanup_by_uuid(self, request_id, uuid):
        super()._cleanup_by_uuid(request_id, uuid)
        if self.tensor_store.check_uuid_presence(request_id, uuid):
            self.tensor_store.remove_tensor(request_id, uuid)

    def cleanup_request(self, request_id):
        # Base implementation ends by ACKing remote producers over the
        # communicator; there are none here.
        self.pending = [ep for ep in self.pending if ep.request_id != request_id]
        self.read_finished.pop(request_id, None)
        self.buffered_shards.pop(request_id, None)
        self.sharding_configs.pop(request_id, None)
        self.req_rx_info.pop(request_id, None)
        self.req_tx_info.pop(request_id, None)
        for uuid in list(self.tensor_store.get_all_uuids(request_id)):
            self.uuid_to_shard_dim.pop(uuid, None)
            self.tensor_store.set_metadata(request_id, uuid, persist=False)
            self._cleanup_by_uuid(request_id, uuid)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchObservation:
    """One batch as seen at the scheduler and again at engine entry."""

    step: int
    node: str
    graph_walk: str
    request_ids: tuple[str, ...]
    engine_request_ids: tuple[str, ...]
    lm_head_launches: int  # number of lm_head calls during the batch (LLM node only; 0 otherwise)
    lm_head_rows: int | None  # total logits rows across those launches
    allocation_failed: bool = False  # KV page OOM: batch pushed back and held

    @property
    def batch_size(self) -> int:
        return len(self.request_ids)

    @property
    def packed(self) -> bool:
        """True when the LLM node ran the batch as one ``forward_batched``
        launch (``_execute_batched``) rather than per-request forwards."""
        return self.node == H.LLM_NODE and self.lm_head_launches == 1 and self.lm_head_rows == self.batch_size


@dataclass
class RequestRecord:
    prompt: H.TextPrompt | H.VisionPrompt
    tokens: list[int] = field(default_factory=list)
    margins: list[float] = field(default_factory=list)
    top2: list[tuple[list[int], list[float]]] = field(default_factory=list)
    done: bool = False
    # (node, graph_walk) for every batch this request took part in.
    walks: list[tuple[str, str]] = field(default_factory=list)

    def llm_walks(self) -> list[str]:
        return [walk for node, walk in self.walks if node == H.LLM_NODE]


# ---------------------------------------------------------------------------
# Worker harness
# ---------------------------------------------------------------------------


class QwenVLWorker:
    def __init__(
        self,
        target: H.Target,
        *,
        seed: int = 0,
        max_output_tokens: int = 6,
        max_num_pages: int = 64,
        page_size: int = 16,
        config=None,
        sched_type: SchedulingType = SchedulingType.ROUND_ROBIN,
    ):
        self.target = target
        self.config = config or H.make_tiny_config()
        H.configure_flashinfer_geometry(self.config, target)
        text = self.config.text_config
        self.kv_config = H.KVConfig(
            num_layers=text.num_hidden_layers,
            num_kv_heads=text.num_key_value_heads,
            head_dim=text.head_dim,
            max_seq_len=4096,
            # The resource KV manager reserves sink page zero for capture
            # padding, preserving ``max_num_pages`` as the harness's usable
            # capacity requires one additional physical page.
            max_num_pages=max_num_pages + 1,
            page_size=page_size,
            num_qo_heads=text.num_attention_heads,
        )
        self.max_output_tokens = max_output_tokens
        self.model = TinyQwenVLModel(
            self.config, target, seed=seed, max_output_tokens=max_output_tokens, kv_config=self.kv_config
        )
        self.tensor_manager = InProcessTensorManager(target.device)
        self.parallel_groups = WorkerParallelGroups(global_rank=0, num_workers=1)
        self.engine_manager = EngineManager.build(
            node_names={H.LLM_NODE, VISION_NODE},
            device=target.device,
            model_config={"autocast_dtype": target.dtype},
            parallel_groups=self.parallel_groups,
            transfer_engine_info=TransferEngineInfo(
                my_entity_id=WORKER_ID,
                my_session_id="local",
                transfer_engine=LocalTransferEngine("localhost"),
            ),
            model=self.model,
        )
        node_groups = [{"node_names": [VISION_NODE, H.LLM_NODE], "ranks": [0]}]
        worker_graphs = []
        for walk, graph in self.model.get_graph_walk_graphs().items():
            worker_graphs.extend(self.model._get_worker_graphs_for_graph_walk(walk, graph, node_groups))
        self.worker_graphs = {wg.worker_graph_id: wg for wg in worker_graphs}
        queues = {
            wg_id: WorkerGraphQueues(
                worker_graph_id=wg_id,
                graph_walks=set(wg.graph_walks),
                worker_graph=wg,
                per_request_queues={},
                tensor_manager=self.tensor_manager,
            )
            for wg_id, wg in self.worker_graphs.items()
        }
        self.graphs_manager = WorkerGraphsManager(
            queues=queues,
            per_request_info={},
            base_sharding_config=self.model.get_default_sharding_config(),
            worker_id=WORKER_ID,
            all_worker_graph_ids_to_graph_walks={
                wg_id: set(wg.graph_walks) for wg_id, wg in self.worker_graphs.items()
            },
            all_worker_graph_ids_to_nodes={
                wg_id: set(wg.section.get_nodes().keys()) for wg_id, wg in self.worker_graphs.items()
            },
            all_worker_graph_ids_to_dyn_loops={
                wg_id: set(wg.section.get_loops().keys()) for wg_id, wg in self.worker_graphs.items()
            },
            node_to_partition={H.LLM_NODE: PARTITION, VISION_NODE: PARTITION},
        )
        # Production ``Worker`` leaves the scheduler on its default policy
        # (round-robin); tests can opt into PRIORITY explicitly.
        self.scheduler = MicroScheduler(
            self.engine_manager,
            sched_type=sched_type,
            parallel_leader_nodes={H.LLM_NODE, VISION_NODE},
        )
        self.records: dict[str, RequestRecord] = {}
        self.observed: list[BatchObservation] = []
        self.oom_events: list[tuple[int, str, tuple[str, ...]]] = []
        # Conductor-side bookkeeping this in-process harness stands in for.
        self._conductor_metadata: dict[str, object] = {}
        self._conductor_persist: dict[str, dict[str, list[TensorPointerInfo]]] = {}
        self._conductor_num_output_tokens: dict[str, int] = {}
        self._step = 0

    # -- convenience -------------------------------------------------------

    @property
    def llm_engine(self):
        return self.engine_manager.get_engine(H.LLM_NODE)

    @property
    def llm_submodule(self) -> QwenVLLLMSubmodule:
        return self.model._submodules[H.LLM_NODE]

    @property
    def alloc_manager(self):
        from types import SimpleNamespace

        return SimpleNamespace(request_states=self.kv._streams)

    @property
    def kv(self):
        return self.llm_engine._resources[H.KV_CACHE]

    @property
    def free_pages(self) -> int:
        return self.kv._arena.num_free

    @property
    def total_pages(self) -> int:
        return self.kv.config.max_num_pages - 1

    @property
    def sampler(self):
        return self.llm_engine._resources[H.SAMPLER]._sampler

    def page_indices(self, rid: str) -> list[int]:
        return list(self.kv._streams[rid]["main"].page_indices)

    def active_requests(self) -> set[str]:
        return set(self.graphs_manager.per_request_info)

    def tokens(self, rid: str) -> list[int]:
        return list(self.records[rid].tokens)

    def margins(self, rid: str) -> list[float]:
        return list(self.records[rid].margins)

    # -- request admission ---------------------------------------------------

    def submit(
        self,
        rid: str,
        prompt: H.TextPrompt | H.VisionPrompt,
        *,
        sampling=None,
        pixel_seed: int | None = None,
    ) -> None:
        """Admit a request exactly like the conductor's NEW_REQUEST path."""
        if isinstance(prompt, H.VisionPrompt):
            position_ids = prompt.position_ids(self.config)
        else:
            position_ids = prompt.tensors(self.config)["position_ids"][0]
        tensors: dict[str, list[torch.Tensor]] = {
            "text_inputs": [prompt.ids.to(self.target.device)],
            "position_ids": [position_ids.to(self.target.device)],
        }
        modalities = ["text"]
        if isinstance(prompt, H.VisionPrompt):
            if pixel_seed is None:
                pixel_seed = zlib.crc32(rid.encode())
            pixels, grid = vision_pixels(prompt.grid, pixel_seed)
            tensors["pixel_values"] = [pixels.to(device=self.target.device, dtype=self.target.dtype)]
            tensors["image_grid_thw"] = [grid.to(self.target.device)]
            modalities = ["image", "text"]
        infos = self.tensor_manager.store_and_return_tensor_info(rid, tensors)
        for entries in infos.values():
            for info in entries:
                self.tensor_manager.increment_ref(rid, info.uuid)
        fwd_args = self.model.get_initial_forward_pass_args(PARTITION, modalities, ["text"], infos)
        metadata = fwd_args.full_metadata
        resource_configs = self.model.get_request_resource_configs({PARTITION: fwd_args}, {"ignore_eos": True})
        if sampling is not None:
            resource_configs[H.SAMPLER] = sampling
        fwd_info = CurrentForwardPassInfo(
            request_id=rid,
            graph_walk=metadata.graph_walk,
            fwd_index=0,
            random_seed=0,
            max_tokens=self.max_output_tokens,
            resource_configs=resource_configs,
            partition_name=PARTITION,
        )
        self.engine_manager.add_request(rid, resource_configs)
        self.graphs_manager.add_request(
            rid,
            partition_worker_graph_ids=list(self.worker_graphs),
            worker_graph_to_workers={wg_id: [WORKER_ID] for wg_id in self.worker_graphs},
            current_fwd_info=fwd_info,
        )
        self._conductor_metadata[rid] = metadata
        self._conductor_persist[rid] = {}
        self._conductor_num_output_tokens[rid] = 0
        leftover = self.graphs_manager.process_new_inputs(rid, fwd_args.inputs)
        assert not leftover, f"initial inputs for {rid} were not claimed by any worker graph: {leftover}"
        self.records[rid] = RequestRecord(prompt=prompt)

    def cancel(self, rid: str) -> None:
        """REMOVE_REQUEST from the conductor while the request is in flight."""
        self.scheduler.pending_removes.add(rid)
        self._finish_request(rid)
        self.scheduler.pending_removes.discard(rid)

    # -- scheduling loop -----------------------------------------------------

    def step(self) -> ScheduledBatch | None:
        batch = self.scheduler.get_next_batch(self.graphs_manager)
        if batch is None:
            return None
        self._step += 1
        node_batch = self._build_node_batch(batch)
        engine = self.engine_manager.get_engine(batch.node_name)
        captured: list[torch.Tensor] = []
        hook = None
        if batch.node_name == H.LLM_NODE:
            hook = self.llm_submodule.lm_head.register_forward_hook(
                lambda _m, _i, out: captured.append(out.detach().float().cpu())
            )
        try:
            engine.prepare_inputs(node_batch)
            output = engine.exec_and_postprocess(node_batch)
        finally:
            if hook is not None:
                hook.remove()
            engine.finalize_batch(node_batch)
        self.observed.append(
            BatchObservation(
                step=self._step,
                node=batch.node_name,
                graph_walk=batch.graph_walk,
                request_ids=tuple(batch.node_objects.keys()),
                engine_request_ids=tuple(node_batch.request_ids),
                lm_head_launches=len(captured),
                lm_head_rows=(sum(t.shape[0] for t in captured) if captured else None),
                allocation_failed=node_batch.admit_error is not None,
            )
        )
        if node_batch.failed_requests:
            raise AssertionError(
                f"engine failed requests {node_batch.failed_requests} in {batch.node_name}/{batch.graph_walk}"
            )
        if node_batch.admit_error is not None:
            self.oom_events.append((self._step, batch.graph_walk, tuple(batch.node_objects.keys())))
            for rid, node in batch.node_objects.items():
                self.graphs_manager.queues[batch.request_to_worker_graph[rid]].push_back_node(rid, node)
            self.scheduler.hold_requests(list(batch.node_objects))
            return batch
        if captured:
            logits = torch.cat(captured, dim=0)
            assert logits.shape[0] == len(node_batch.request_ids), (
                f"lm_head produced {logits.shape[0]} rows for {len(node_batch.request_ids)} requests"
            )
            for rid, row in zip(node_batch.request_ids, logits, strict=True):
                self.records[rid].margins.append(H.top2_margin(row))
                values, indices = torch.topk(row.float(), 2)
                self.records[rid].top2.append((indices.tolist(), values.tolist()))
        self._postprocess(batch, node_batch, output)
        return batch

    def run_until_idle(self, max_steps: int = 10_000) -> list[BatchObservation]:
        start = len(self.observed)
        for _ in range(max_steps):
            if self.step() is None:
                if self.scheduler.held_until:
                    # OOM back-off: the scheduler re-offers held requests
                    # after HOLD_BACKOFF_SECONDS; wait it out instead of
                    # reporting idle.
                    time.sleep(MicroScheduler.HOLD_BACKOFF_SECONDS)
                    continue
                return self.observed[start:]
        raise AssertionError("worker did not go idle")

    def run_until_done(
        self,
        rids: list[str] | None = None,
        max_steps: int = 10_000,
        after_step: Callable[[QwenVLWorker], None] | None = None,
    ) -> list[BatchObservation]:
        start = len(self.observed)
        for _ in range(max_steps):
            pending = [rid for rid in (rids or list(self.records)) if not self.records[rid].done]
            if not pending:
                return self.observed[start:]
            batch = self.step()
            if after_step is not None and batch is not None:
                after_step(self)
            if batch is None:
                if self.scheduler.held_until:
                    time.sleep(MicroScheduler.HOLD_BACKOFF_SECONDS)
                    continue
                raise AssertionError(f"worker idle with unfinished requests {pending}")
        raise AssertionError("requests did not finish")

    # -- worker internals (mirrors Worker._build_node_batch / _postprocess_batch)

    def _build_node_batch(self, batch: ScheduledBatch) -> ExecutingBatch:
        per_request_inputs = {}
        per_request_info = {}
        for rid, node in batch.node_objects.items():
            tensors = {}
            for input_name, edge in node.ready_signals.ready_inputs.items():
                tensors[input_name] = [self.tensor_manager.get_tensor(rid, info.uuid) for info in edge.tensor_info]
            per_request_inputs[rid] = tensors
            per_request_info[rid] = self.graphs_manager.get_fwd_info(rid, PARTITION)
        return ExecutingBatch(
            node_name=batch.node_name,
            step_context=H.StepContext(
                request_ids=list(batch.node_objects.keys()),
                graph_walk=batch.graph_walk,
                slot=0,
                capture=False,
            ),
            per_request_input_tensors=per_request_inputs,
            per_request_info=per_request_info,
        )

    def _postprocess(self, batch: ScheduledBatch, node_batch: ExecutingBatch, output: dict) -> None:
        for node in batch.node_objects.values():
            node.ready_signals.clear()
        for rid, req_info in node_batch.per_request_info.items():
            req_info.dynamic_loop_iter_counts.update(self.graphs_manager.get_dynamic_loop_iters(rid, PARTITION))
            self.records[rid].walks.append((batch.node_name, batch.graph_walk))

        engine = self.engine_manager.get_engine(batch.node_name)
        stops = engine.check_stop_for_batch(node_batch, output)
        assert not node_batch.failed_requests, node_batch.failed_requests
        for rid, loop_names in stops.items():
            loop_names = {name for name in loop_names if self.graphs_manager.check_dyn_loop(rid, PARTITION, name)}
            if loop_names:
                self.graphs_manager.stop_loops(
                    rid,
                    partition=PARTITION,
                    loop_names=loop_names,
                    req_info=node_batch.per_request_info[rid],
                    last_node_run=batch.node_name,
                )

        for rid, wg_id in batch.request_to_worker_graph.items():
            node = batch.node_objects[rid]
            node.reset_outputs()
            req_output = output.get(rid)
            uuids: set[str] = set()
            if req_output:
                infos = self.tensor_manager.store_and_populate_graph_edges(
                    request_id=rid,
                    tensors=req_output,
                    graph_edges=node.outputs,
                    node_name=node.name,
                    graph_walk=batch.graph_walk,
                    skip_cuda_sync=True,
                    skip_ref_count=True,
                )
                uuids = {info.uuid for entries in infos.values() for info in entries}
            completion = self.graphs_manager.mark_node_complete(rid, wg_id, batch.node_name)
            routing = self.graphs_manager.process_node_outputs(
                rid,
                node_name=batch.node_name,
                outputs=[edge.clone() for edge in completion.output_edges],
                graph_walk=batch.graph_walk,
            )
            if uuids:
                for edge in routing.persist:
                    for info in edge.tensor_info:
                        self.tensor_manager.set_persist(rid, info.uuid, True)
                routed = (
                    routing.routed_to_this_worker_graph
                    + routing.emit_to_client
                    + routing.streaming_local
                    + sum(routing.to_workers.values(), start=[])
                    + sum(routing.streaming_to_workers.values(), start=[])
                )
                self.tensor_manager.set_output_ref_counts(rid, uuids, routed)
            assert not routing.to_workers, f"single-worker harness routed edges off-worker: {routing.to_workers}"
            self._send_outputs(rid, routing, batch.graph_walk)

    def _send_outputs(self, rid: str, routing, graph_walk: str) -> None:
        if routing.persist:
            self.graphs_manager.buffer_persist_signals(rid, routing.persist)
        if routing.new_token_outputs:
            counts = {}
            for edge in routing.new_token_outputs:
                if edge.name in counts:
                    continue
                counts[edge.name] = sum(
                    self.tensor_manager.get_tensor(rid, info.uuid).numel() for info in edge.tensor_info
                )
            self.graphs_manager.buffer_new_token_counts(rid, counts)
        for edge in routing.emit_to_client:
            self.graphs_manager.buffer_output_signals(rid, [edge])
            if edge.name == "new_token":
                for info in edge.tensor_info:
                    token = self.tensor_manager.get_tensor(rid, info.uuid)
                    self.records[rid].tokens.extend(int(t) for t in token.reshape(-1).tolist())
        if routing.completed_worker_graph_ids:
            self._conductor_worker_graphs_done(rid)

    def _conductor_worker_graphs_done(self, rid: str) -> None:
        """Inline stand-in for the conductor's WORKER_GRAPHS_DONE handling."""
        persist = self.graphs_manager.flush_persist_signals(rid)
        for name, infos in persist.items():
            self._conductor_persist[rid].setdefault(name, []).extend(infos)
        for count in self.graphs_manager.flush_new_token_counts(rid).values():
            self._conductor_num_output_tokens[rid] += count
        self.graphs_manager.flush_output_signals(rid)
        metadata = self._conductor_metadata[rid]
        fwd_args = self.model.get_partition_forward_pass_args(PARTITION, metadata, self._conductor_persist[rid])
        self._conductor_metadata[rid] = fwd_args.full_metadata
        fwd_args.full_metadata.kwargs.update(fwd_args.step_metadata)
        if self._conductor_num_output_tokens[rid] >= self.max_output_tokens:
            fwd_args.request_done = True
        if fwd_args.request_done:
            self._finish_request(rid)
            return
        previous = self.graphs_manager.get_fwd_info(rid, PARTITION)
        next_seed = previous.random_seed + 1
        next_info = CurrentForwardPassInfo(
            request_id=rid,
            graph_walk=fwd_args.full_metadata.graph_walk,
            fwd_index=previous.fwd_index + 1,
            random_seed=next_seed,
            max_tokens=self.max_output_tokens,
            resource_configs=previous.resource_configs,
            resource_publish_info=previous.resource_publish_info,
            partition_name=PARTITION,
        )
        self.graphs_manager.update_request_info(rid, PARTITION, current_fwd_info=next_info)
        leftover = self.graphs_manager.process_new_inputs(rid, fwd_args.inputs)
        assert not leftover, f"decode inputs for {rid} were not claimed: {leftover}"
        for info in fwd_args.unpersist_tensors:
            self.tensor_manager.set_persist(rid, info.uuid, False)

    def _finish_request(self, rid: str) -> None:
        self.engine_manager.remove_request(rid)
        self.graphs_manager.remove_request(rid)
        self.tensor_manager.cleanup_request(rid)
        self.scheduler.clear_rid(rid)
        self.records[rid].done = True


# ---------------------------------------------------------------------------
# Helpers for tests
# ---------------------------------------------------------------------------


def llm_batches(observed: list[BatchObservation], graph_walk: str | None = None) -> list[BatchObservation]:
    return [obs for obs in observed if obs.node == H.LLM_NODE and (graph_walk is None or obs.graph_walk == graph_walk)]


def isolated_run(
    target: H.Target,
    rid: str,
    prompt: H.TextPrompt | H.VisionPrompt,
    *,
    seed: int,
    max_output_tokens: int,
    pixel_seed: int | None = None,
    **worker_kwargs,
) -> RequestRecord:
    """Run one request alone on a fresh worker with identical weights."""
    worker = QwenVLWorker(target, seed=seed, max_output_tokens=max_output_tokens, **worker_kwargs)
    worker.submit(rid, prompt, pixel_seed=pixel_seed)
    worker.run_until_done([rid])
    assert worker.free_pages == worker.total_pages
    return worker.records[rid]


def edge_names(edges: list[GraphEdge]) -> list[str]:
    return [edge.name for edge in edges]
