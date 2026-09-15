"""Stub out broken lerobot subpackages before any test imports them.

Some lerobot versions have a malformed ``@dataclass`` in
``lerobot.policies.groot.groot_n1`` that crashes at import time. We don't
need groot for Pi0.5 tests, so we pre-register fake stub modules in
``sys.modules`` to satisfy the eager import chain in
``lerobot.policies.__init__``.
"""

import sys
import types


def _make_stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__path__ = []
    return m


if "lerobot.policies.groot.groot_n1" not in sys.modules:
    pkg = _make_stub("lerobot.policies.groot")
    g_n1 = _make_stub("lerobot.policies.groot.groot_n1")
    g_n1.GR00TN15 = type("GR00TN15", (), {})
    cfg = _make_stub("lerobot.policies.groot.configuration_groot")
    cfg.GrootConfig = type("GrootConfig", (), {})
    modg = _make_stub("lerobot.policies.groot.modeling_groot")
    modg.GrootPolicy = type("GrootPolicy", (), {})

    sys.modules["lerobot.policies.groot"] = pkg
    sys.modules["lerobot.policies.groot.groot_n1"] = g_n1
    sys.modules["lerobot.policies.groot.configuration_groot"] = cfg
    sys.modules["lerobot.policies.groot.modeling_groot"] = modg


# --- CPU-only import stubs (mirrors test/modular/conftest.py) ---------------
#
# The QwenVL PR1 integration tests run their dense-reference variant on CPU.
# ``mstar.utils.sampling`` imports Triton at module load and
# ``mstar.engine.__init__`` writes ``torch._dynamo.config`` flags that older
# CUDA-less torch builds lack; neither is needed for the CPU path.

import torch  # noqa: E402


class _DynamoConfigSink:
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        return None


try:
    torch._dynamo.config.recompile_limit = 64
except (AttributeError, RuntimeError):
    torch._dynamo.config = _DynamoConfigSink()

if "triton" not in sys.modules:
    try:
        import triton  # noqa: F401
    except ImportError:
        triton = types.ModuleType("triton")
        triton.language = types.ModuleType("triton.language")
        triton.language.constexpr = int
        triton.jit = lambda *a, **k: (lambda f: f)
        triton.cdiv = lambda a, b: -(-a // b)
        triton.Config = lambda *a, **k: a
        triton.autotune = lambda *a, **k: (lambda f: f)
        triton.heuristics = lambda *a, **k: (lambda f: f)
        sys.modules["triton"] = triton
        sys.modules["triton.language"] = triton.language


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: requires a CUDA device (and FlashInfer) — PR1 acceptance evidence")
    config.addinivalue_line(
        "markers",
        "p1_gate(name): QwenVL PR1 acceptance gate this test provides evidence for (P1-G1 .. P1-G6)",
    )
