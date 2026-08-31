# QwenVL design record

This directory is the design record for the QwenVL integration stack for
MStar issue [#127](https://github.com/mstar-project/mstar/issues/127).

The stack begins with a single-GPU, non-tensor-parallel correctness baseline
for `Qwen/Qwen3-VL-30B-A3B-Instruct`. Each later slice is design-only until its
own runtime acceptance gates pass.

## Included design

| Document | Scope | Merge claim allowed |
| --- | --- | --- |
| [Single-GPU correctness](PR_0_SINGLE_GPU_CORRECTNESS.md) | Official config/checkpoint mapping, image/text graph, MRoPE, bounded KV residency, and greedy decode | Local non-TP baseline only |
| [Continuous batching](PR_1_CONTINUOUS_BATCHING.md) | Same-walk batching guards, paged-attention ownership, isolation/lifecycle gates, engine + worker integration harness, eager-only CUDA-graph scope | Continuous batching + paged attention on one GPU, once `benchmark/qwenvl_acceptance.py batch` reports every P1 gate `pass` on a CUDA + FlashInfer host |
| [Tensor parallelism](PR_2_TENSOR_PARALLELISM.md) | TP topology, exact shard contracts, and TP2-vs-TP1 acceptance | Design only; distributed parity evidence required |

## Evidence labels

- **Unit**: isolated tensor/helper behavior.
- **Component**: real model component with fake engine boundaries.
- **Integration**: real MStar scheduler/engine path.
- **System**: launched server with the published checkpoint.

PR 0 has unit/component evidence only until its real-checkpoint CUDA protocol
passes. “Tests pass” without its environment and evidence label is not an
acceptance claim.

## Where the tests live

| Location | Label | Notes |
| --- | --- | --- |
| `test/modular/qwenvl/` | Component | fake `Cache`, direct `preprocess`/`forward_batched`; pins the P1-G6 eager-only decision |
| `test/integration/test_qwenvl_attention_parity.py` | Integration (`cuda-flashinfer-bf16` rows) | P1-G5: resource-pool FlashInfer versus dense SDPA reference over the same production KV pages |
| `test/integration/test_qwenvl_batched_engine.py` | Integration | P1-G2: packed resource-engine forward versus its per-request fallback, B∈{2,4,8} |
| `test/integration/test_qwenvl_scheduler_batching.py` | Integration | P1-G1: batch shapes observed at `MicroScheduler` and engine entry |
| `test/integration/test_qwenvl_lifecycle.py` | Integration | P1-G3/G4: perturbation isolation, completion, cancel, page reuse, OOM hold |

Each integration test is also parametrised with a `cpu-dense-fp32` row. That
row runs on a laptop and validates the harness and engine plumbing; it carries
the Component label and is never acceptance evidence.

```bash
uv run --extra qwenvl --extra dev pytest -q test/modular/qwenvl test/integration/test_qwenvl_*.py   # CPU dry run
uv run --extra qwenvl --extra dev pytest -q test/integration/test_qwenvl_*.py -m cuda                  # PR1 acceptance
uv run --extra qwenvl --extra dev python benchmark/qwenvl_acceptance.py batch --skip-cpu-rows --output qwenvl-batch-evidence.json
```
