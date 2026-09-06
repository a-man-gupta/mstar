"""Collect evidence for the QwenVL acceptance gates.

PR-0 (single-GPU correctness): run ``checkpoint`` on the same GPU that will
serve the model, then start ``mstar serve qwenvl`` and run ``server`` against
it.

PR-1 (continuous batching + paged attention): run ``batch`` on a CUDA GPU with
FlashInfer installed. It executes the PR1 integration suites
(``test/integration/test_qwenvl_*.py``) plus the component test pinning the
P1-G6 eager-only decision, and folds the per-test outcomes into a per-gate
verdict (P1-G1 .. P1-G6).

Results are emitted as JSON so they can be attached to the PR without treating
CPU tests as acceptance.
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
REPO_ROOT = Path(__file__).resolve().parents[1]

PR1_GATES = {
    "P1-G1": "Each claimed batch shape is observed through the real scheduler",
    "P1-G2": "Batched outputs match isolated outputs within declared tolerance",
    "P1-G3": "No cache, visual, sampler, or output cross-request contamination",
    "P1-G4": "Page lifecycle releases and safely reuses storage",
    "P1-G5": "GQA, causal, variable-length, and page-boundary cases pass",
    "P1-G6": "CUDA-graph parity passes or eager-only scope is documented",
}
PR1_TEST_PATHS = [
    "test/integration/test_qwenvl_attention_parity.py",
    "test/integration/test_qwenvl_batched_engine.py",
    "test/integration/test_qwenvl_scheduler_batching.py",
    "test/integration/test_qwenvl_lifecycle.py",
    "test/modular/qwenvl/test_continuous_batch_serving.py",
]
# Parametrised test ids carry the harness target; only the CUDA FlashInfer
# rows count as Integration evidence. CPU rows are a harness dry run.
CUDA_TARGET_ID = "cuda-flashinfer-bf16"
CPU_TARGET_ID = "cpu-dense-fp32"


def _write_evidence(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if output is not None:
        output.write_text(rendered + "\n")


def checkpoint_evidence(args: argparse.Namespace) -> None:
    import torch
    from huggingface_hub import snapshot_download

    from mstar.model.qwenvl.qwenvl_model import QwenVLModel

    if not torch.cuda.is_available():
        raise RuntimeError("PR-0 checkpoint acceptance requires a CUDA GPU.")
    local_dir = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)
    model = QwenVLModel(local_dir)
    config = model.config
    model.get_submodule("LLM", device=args.device, autocast_dtype=torch.bfloat16)
    model.get_submodule("vision_encoder", device=args.device, autocast_dtype=torch.bfloat16)
    torch.cuda.synchronize(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    allocated = torch.cuda.max_memory_allocated(args.device)
    reserved = torch.cuda.max_memory_reserved(args.device)
    total = properties.total_memory
    _write_evidence(
        {
            "gates": ["P0-G1", "P0-G2", "P0-G7"],
            "model": args.model,
            "revision": args.revision,
            "snapshot": local_dir,
            "device": properties.name,
            "config": {
                "model_type": config.model_type,
                "hidden_size": config.text_config.hidden_size,
                "num_hidden_layers": config.text_config.num_hidden_layers,
                "num_experts": config.text_config.num_experts,
                "spatial_merge_size": config.vision_config.spatial_merge_size,
            },
            "load": {
                "llm": "complete",
                "vision_encoder": "complete",
            },
            "memory_gib": {
                "peak_allocated": allocated / 2**30,
                "peak_reserved": reserved / 2**30,
                "device_total": total / 2**30,
                "reserved_headroom": (total - reserved) / 2**30,
            },
        },
        args.output,
    )


def _checkerboard_png() -> bytes:
    from PIL import Image

    image = Image.new("RGB", (96, 64))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = (255, 255, 255) if (x // 16 + y // 16) % 2 else (0, 0, 0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def position_evidence(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeModel,
        Qwen3VLMoeTextRotaryEmbedding,
    )

    from mstar.model.qwenvl.components import compute_mrope_cos_sin
    from mstar.model.qwenvl.qwenvl_model import QwenVLModel

    if not torch.cuda.is_available():
        raise RuntimeError("PR-0 position acceptance requires a CUDA GPU.")
    local_dir = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken"],
    )
    model = QwenVLModel(local_dir)
    image = Image.open(io.BytesIO(_checkerboard_png())).convert("RGB")
    image_tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    processed = model.process_prompt(
        "Describe the dominant colors and pattern in this image in one sentence.",
        ["image", "text"],
        ["text"],
        {"image_inputs": [image_tensor]},
    )
    input_ids = processed["text_inputs"][0]
    grid = processed["image_grid_thw"][0]
    actual_positions = processed["position_ids"][0]
    expected_positions, _ = Qwen3VLMoeModel.get_rope_index(
        SimpleNamespace(config=model.config),
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=grid,
    )
    expected_positions = expected_positions[:, 0]
    torch.testing.assert_close(actual_positions, expected_positions, atol=0, rtol=0)

    device = torch.device(args.device)
    actual_cos, actual_sin = compute_mrope_cos_sin(
        actual_positions.to(device),
        head_dim=model.config.text_config.head_dim,
        rope_theta=model.config.text_config.rope_theta,
        mrope_section=tuple(model.config.text_config.rope_scaling["mrope_section"]),
        dtype=torch.bfloat16,
    )
    reference_rope = Qwen3VLMoeTextRotaryEmbedding(model.config.text_config, device=device)
    expected_cos, expected_sin = reference_rope(
        torch.empty(1, device=device, dtype=torch.bfloat16),
        expected_positions[:, None, :].to(device),
    )
    expected_cos, expected_sin = expected_cos[0], expected_sin[0]
    torch.testing.assert_close(actual_cos, expected_cos, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(actual_sin, expected_sin, atol=args.atol, rtol=args.rtol)
    _write_evidence(
        {
            "gate": "P0-G3",
            "model": args.model,
            "revision": args.revision,
            "device": torch.cuda.get_device_name(device),
            "image_grid_thw": grid.tolist(),
            "position_ids_exact": True,
            "cos_max_abs_error": (actual_cos - expected_cos).abs().max().item(),
            "sin_max_abs_error": (actual_sin - expected_sin).abs().max().item(),
            "atol": args.atol,
            "rtol": args.rtol,
        },
        args.output,
    )


def server_evidence(args: argparse.Namespace) -> None:
    from mstar import MStarClient
    from mstar.client import TextChunk

    client = MStarClient(args.url, timeout=args.timeout)
    if not client.health():
        raise RuntimeError(f"QwenVL server is not healthy at {args.url}.")
    chunks: list[str] = []
    events = client.chat(
        "Describe the dominant colors and pattern in this image in one sentence.",
        images=[("qwenvl-pr0-checkerboard.png", _checkerboard_png())],
        stream=True,
        temperature=0.0,
        max_output_tokens=args.max_output_tokens,
    )
    for event in events:
        if isinstance(event, TextChunk):
            chunks.append(event.text)
    text = "".join(chunks)
    if not text.strip():
        raise RuntimeError("QwenVL server returned no decoded text.")
    if "<|" in text:
        raise RuntimeError(f"QwenVL server leaked a special token: {text!r}")
    _write_evidence(
        {
            "gate": "P0-G6",
            "url": args.url,
            "prompt": "Describe the dominant colors and pattern in this image in one sentence.",
            "temperature": 0.0,
            "max_output_tokens": args.max_output_tokens,
            "stream_chunk_count": len(chunks),
            "decoded_text": text,
            "human_coherence_review_required": True,
        },
        args.output,
    )


class _GateRecorder:
    """pytest plugin: map every executed test to its PR1 gates and outcome."""

    def __init__(self) -> None:
        self.gates_by_nodeid: dict[str, list[str]] = {}
        self.target_by_nodeid: dict[str, tuple[str, str]] = {}
        self.results: dict[str, dict[str, Any]] = {}

    def pytest_collection_modifyitems(self, session, config, items) -> None:
        for item in items:
            gates = sorted({mark.args[0] for mark in item.iter_markers("p1_gate") if mark.args})
            if not gates:
                continue
            self.gates_by_nodeid[item.nodeid] = gates
            integration = "test/integration/" in item.nodeid.replace("\\", "/")
            if CPU_TARGET_ID in item.nodeid:
                self.target_by_nodeid[item.nodeid] = (
                    "cpu",
                    "Component (harness dry run on CPU dense reference; not acceptance)",
                )
            elif integration and (CUDA_TARGET_ID in item.nodeid or item.get_closest_marker("cuda") is not None):
                self.target_by_nodeid[item.nodeid] = ("cuda", "Integration")
            else:
                self.target_by_nodeid[item.nodeid] = ("component", "Component")

    def pytest_runtest_logreport(self, report) -> None:
        if report.nodeid not in self.gates_by_nodeid:
            return
        # A failed/skipped setup is the test's outcome; otherwise use the call phase.
        if report.when == "setup" and report.outcome == "passed":
            return
        if report.when == "teardown":
            return
        if report.nodeid in self.results and report.when == "call" and self.results[report.nodeid]["phase"] == "setup":
            return
        entry: dict[str, Any] = {
            "outcome": report.outcome,
            "phase": report.when,
            "duration_s": round(report.duration, 3),
        }
        if report.outcome == "skipped" and report.longrepr is not None:
            entry["reason"] = str(report.longrepr[-1] if isinstance(report.longrepr, tuple) else report.longrepr)
        if report.outcome == "failed":
            entry["longrepr"] = str(report.longrepr)[-4000:]
        self.results[report.nodeid] = entry


def batch_evidence(args: argparse.Namespace) -> None:
    import pytest
    import torch

    cuda = torch.cuda.is_available()
    if not cuda and not args.allow_cpu_only:
        raise RuntimeError(
            "PR-1 batch acceptance requires a CUDA GPU with FlashInfer. "
            "Pass --allow-cpu-only to record a CPU dry run (not acceptance evidence)."
        )
    try:
        import flashinfer  # type: ignore

        flashinfer_version = getattr(flashinfer, "__version__", "unknown")
    except ImportError:
        flashinfer_version = None

    recorder = _GateRecorder()
    pytest_args = [str(REPO_ROOT / path) for path in PR1_TEST_PATHS]
    pytest_args += ["-q", "-p", "no:cacheprovider", "-W", "ignore", "--rootdir", str(REPO_ROOT)]
    if args.skip_cpu_rows:
        pytest_args += ["-k", f"not {CPU_TARGET_ID}"]
    if args.pytest_args:
        pytest_args += args.pytest_args
    exit_code = int(pytest.main(pytest_args, plugins=[recorder]))

    tests: list[dict[str, Any]] = []
    for nodeid, gates in sorted(recorder.gates_by_nodeid.items()):
        result = recorder.results.get(nodeid, {"outcome": "not run", "phase": None})
        target, label = recorder.target_by_nodeid[nodeid]
        tests.append({"nodeid": nodeid, "gates": gates, "target": target, "evidence_label": label, **result})

    gate_report: dict[str, dict[str, Any]] = {}
    for gate, description in PR1_GATES.items():
        rows = [t for t in tests if gate in t["gates"]]
        if gate == "P1-G6":
            # Eager-only decision: proven by the component test that pins
            # ``get_cuda_graph_configs() == []`` plus the design doc.
            label = "Component + design doc (eager-only scope, option A)"
            acceptance_rows = [t for t in rows if t["target"] == "component"]
        else:
            label = "Integration"
            acceptance_rows = [t for t in rows if t["target"] == "cuda"]
        passed = [t for t in acceptance_rows if t["outcome"] == "passed"]
        failed = [t for t in rows if t["outcome"] == "failed"]
        skipped = [t for t in acceptance_rows if t["outcome"] != "passed"]
        if failed:
            verdict = "fail"
        elif not acceptance_rows:
            verdict = "no acceptance tests"
        elif passed and not skipped:
            verdict = "pass"
        elif passed:
            verdict = "partial (some acceptance rows skipped)"
        else:
            verdict = "not collected (CUDA/FlashInfer rows skipped)"
        gate_report[gate] = {
            "description": description,
            "evidence_label": label,
            "verdict": verdict,
            "acceptance_tests": len(acceptance_rows),
            "passed": len(passed),
            "failed": len(failed),
            "skipped": len(skipped),
            "dry_run_rows_passed": sum(1 for t in rows if t["target"] == "cpu" and t["outcome"] == "passed"),
        }

    payload = {
        "gates": gate_report,
        "all_gates_pass": all(g["verdict"] == "pass" for g in gate_report.values()),
        "pytest_exit_code": exit_code,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": cuda,
            "device": torch.cuda.get_device_name(0) if cuda else None,
            "flashinfer": flashinfer_version,
            "platform": platform.platform(),
        },
        "tolerance": {
            "bf16_flashinfer": {
                "rtol": 1e-2,
                "atol": 2e-2,
                "note": "last-token logits; greedy streams may only diverge on close-call top-2 margins",
            },
            "fp32_dense_reference": {"rtol": 1e-5, "atol": 1e-5},
        },
        "cuda_graphs": "eager-only (P1-G6 option A); QwenVLLLMSubmodule.get_cuda_graph_configs() == []",
        "tests": tests,
    }
    _write_evidence(payload, args.output)
    if not payload["all_gates_pass"] and not args.allow_cpu_only:
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    batch = subparsers.add_parser("batch", help="Run the PR1 integration suites and emit per-gate P1-G* evidence")
    batch.add_argument("--output", type=Path)
    batch.add_argument(
        "--allow-cpu-only",
        action="store_true",
        help="Do not require CUDA; records a CPU dense-reference dry run that is NOT acceptance evidence",
    )
    batch.add_argument("--skip-cpu-rows", action="store_true", help="Deselect the CPU dry-run parametrisations")
    batch.add_argument("pytest_args", nargs="*", help="Extra arguments forwarded to pytest (after --)")
    batch.set_defaults(run=batch_evidence)

    checkpoint = subparsers.add_parser("checkpoint", help="Validate real config/load and record peak memory")
    checkpoint.add_argument("--model", default=DEFAULT_MODEL)
    checkpoint.add_argument("--revision", required=True, help="Immutable Hub commit SHA")
    checkpoint.add_argument("--cache-dir")
    checkpoint.add_argument("--device", default="cuda:0")
    checkpoint.add_argument("--output", type=Path)
    checkpoint.set_defaults(run=checkpoint_evidence)

    positions = subparsers.add_parser("positions", help="Compare real processor positions and rotary tensors")
    positions.add_argument("--model", default=DEFAULT_MODEL)
    positions.add_argument("--revision", required=True, help="Immutable Hub commit SHA")
    positions.add_argument("--cache-dir")
    positions.add_argument("--device", default="cuda:0")
    positions.add_argument("--atol", type=float, default=2e-3)
    positions.add_argument("--rtol", type=float, default=2e-3)
    positions.add_argument("--output", type=Path)
    positions.set_defaults(run=position_evidence)

    server = subparsers.add_parser("server", help="Run a deterministic image-chat streaming smoke")
    server.add_argument("--url", default="http://localhost:8000")
    server.add_argument("--timeout", type=float, default=600.0)
    server.add_argument("--max-output-tokens", type=int, default=64)
    server.add_argument("--output", type=Path)
    server.set_defaults(run=server_evidence)

    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.run(parsed)
