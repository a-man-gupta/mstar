# Qwen3-VL-30B-A3B platform scenarios

These tests are organized around the capabilities an inference platform must
provide, rather than the private helper that happens to implement them.

| Scenario | What a failure means |
| --- | --- |
| `test_checkpoint_onboarding.py` | The published Qwen3-VL MoE config cannot be safely admitted, or interleaved MRoPE semantics drift. |
| `test_checkpoint_loading.py` | A 30B-A3B checkpoint can be partially or incorrectly mapped into the serving graph. |
| `test_image_chat_serving.py` | An API image-plus-text chat request cannot be processed, routed through graph walks, or deployed with the declared TP topology. |
| `test_continuous_batch_serving.py` | Packed LLM preprocessing, same-walk homogeneity (`can_batch`/`preprocess`), cache positions, last-token sampling, decode-never-sees-vision, and the eager-only CUDA-graph decision (P1-G6) preserve per-request boundaries. **Component** label with a fake `Cache`: it is not proof that the scheduler co-batches, nor that FlashInfer agrees with the reference. |
| `test_reference_compatibility.py` | Tiny MoE text and DeepStack image execution no longer match Hugging Face's Qwen3-VL reference. |

Run the platform contract locally:

```bash
uv run --extra qwenvl --extra dev --with pytest-cov \
  pytest -q test/modular/qwenvl \
  --cov=mstar.model.qwenvl --cov-report=term-missing
```

The suite is intentionally CPU-sized. It proves graph, loading, packed-LLM,
and reference contracts. It does not prove end-to-end scheduler co-batching
between `prefill` and `prefill_vision`, actual 30B checkpoint admission,
single-GPU CUDA serving, TP2 numerical parity, or throughput.

The PR1 continuous-batching gates (P1-G1 .. P1-G5) are proven by the
**Integration** suites in `test/integration/test_qwenvl_*.py`, which enter
through the real `KVCacheEngine`, `MicroScheduler` and `WorkerGraphsManager`
(see `docs/qwenvl/PR_1_CONTINUOUS_BATCHING.md` §Evidence). Their
`cpu-dense-fp32` parametrisations run here on CPU as a harness dry run; only
the `cuda-flashinfer-bf16` rows (`-m cuda`) are acceptance evidence.
