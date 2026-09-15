"""QwenVL PR1 / WS-C — scheduler co-batching through the real worker stack (P1-G1).

``qwenvl_worker_harness.QwenVLWorker`` wires the production
``WorkerGraphsManager`` + ``MicroScheduler`` + ``EngineManager`` (with a
``KVCacheEngine`` for the LLM node and a ``StatelessEngine`` for the vision
encoder) around ``QwenVLModel``'s real graph walks, and replays the
conductor's NEW_REQUEST / WORKER_GRAPHS_DONE / decode-loop transitions
in-process. Every batch is observed twice:

* at the scheduler (``ScheduledBatch.graph_walk`` + request ids), and
* at engine entry (``NodeBatch.request_ids`` + the number/shape of ``lm_head``
  launches, which distinguishes one packed ``forward_batched`` from N
  per-request forwards).

The batch shapes PR1 claims are asserted here, and every co-batched request's
greedy token stream is compared with the same request run alone on a fresh
worker with identical weights (P1-G2 through the scheduler, P1-G3 for
mixed-origin decode).
"""

from __future__ import annotations

import pytest
import qwenvl_harness as H
import qwenvl_worker_harness as W

pytestmark = pytest.mark.p1_gate("P1-G1")

MAX_OUTPUT_TOKENS = 5
WORKER_KW = dict(page_size=16, max_num_pages=96)


@pytest.fixture(params=H.TARGET_PARAMS)
def target(request, monkeypatch) -> H.Target:
    return H.prepare_target(request.param, monkeypatch)


def _worker(target: H.Target, **overrides) -> W.QwenVLWorker:
    kwargs = {**WORKER_KW, **overrides}
    return W.QwenVLWorker(target, seed=0, max_output_tokens=MAX_OUTPUT_TOKENS, **kwargs)


def _assert_matches_isolated(worker: W.QwenVLWorker, rid: str, target: H.Target, **isolated_kw) -> None:
    record = worker.records[rid]
    alone = W.isolated_run(
        target, rid, record.prompt, seed=0, max_output_tokens=MAX_OUTPUT_TOKENS, **{**WORKER_KW, **isolated_kw}
    )
    assert len(record.tokens) == len(alone.tokens) == MAX_OUTPUT_TOKENS + 1, (rid, record.tokens, alone.tokens)
    try:
        H.assert_greedy_streams_match(record.tokens, alone.tokens, alone.margins, target, f"{rid} co-batched vs alone")
    except AssertionError as error:
        raise AssertionError(f"{error}\nco-batched top2={record.top2}\nisolated top2={alone.top2}") from error


def _walk_batches(observed, walk):
    return [obs for obs in W.llm_batches(observed, walk)]


# ---------------------------------------------------------------------------
# prefill ↔ prefill and shared decode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_scheduler_co_batches_text_prefill_and_decode(target: H.Target, batch_size: int) -> None:
    worker = _worker(target)
    rids = [f"t{i}" for i in range(batch_size)]
    for i, rid in enumerate(rids):
        worker.submit(rid, H.text_prompt(8 + 5 * i, seed=i))

    observed = worker.run_until_done()

    prefill = _walk_batches(observed, "prefill")
    assert len(prefill) == 1, [o.request_ids for o in prefill]
    assert sorted(prefill[0].request_ids) == rids
    assert prefill[0].engine_request_ids == prefill[0].request_ids
    assert prefill[0].packed, prefill

    decode = _walk_batches(observed, "decode")
    assert len(decode) == MAX_OUTPUT_TOKENS
    for obs in decode:
        assert sorted(obs.request_ids) == rids
        assert obs.packed, obs
    assert not _walk_batches(observed, "prefill_vision")
    assert all(obs.node == H.LLM_NODE for obs in observed)

    assert worker.free_pages == worker.total_pages
    for rid in rids:
        assert worker.records[rid].llm_walks() == ["prefill"] + ["decode"] * MAX_OUTPUT_TOKENS
        _assert_matches_isolated(worker, rid, target)


# ---------------------------------------------------------------------------
# prefill_vision ↔ prefill_vision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [2, 4])
def test_scheduler_co_batches_vision_prefill(target: H.Target, batch_size: int) -> None:
    worker = _worker(target)
    grids = [(1, 4, 4), (1, 6, 4), (1, 2, 8), (2, 2, 2)][:batch_size]
    rids = [f"v{i}" for i in range(batch_size)]
    for i, (rid, grid) in enumerate(zip(rids, grids, strict=True)):
        worker.submit(rid, H.vision_prompt(grid, seed=40 + i, prefix=2 + i))

    observed = worker.run_until_done()

    encoder = [obs for obs in observed if obs.node == W.VISION_NODE]
    assert len(encoder) == 1 and sorted(encoder[0].request_ids) == rids
    assert encoder[0].graph_walk == "prefill_vision"

    vision_prefill = _walk_batches(observed, "prefill_vision")
    assert len(vision_prefill) == 1 and sorted(vision_prefill[0].request_ids) == rids
    assert vision_prefill[0].packed, vision_prefill
    assert not _walk_batches(observed, "prefill")
    # The encoder must run before the LLM consumes its embeddings.
    assert encoder[0].step < vision_prefill[0].step

    decode = _walk_batches(observed, "decode")
    assert len(decode) == MAX_OUTPUT_TOKENS and all(sorted(o.request_ids) == rids and o.packed for o in decode)

    assert worker.free_pages == worker.total_pages
    for rid in rids:
        assert worker.records[rid].llm_walks() == ["prefill_vision"] + ["decode"] * MAX_OUTPUT_TOKENS
        _assert_matches_isolated(worker, rid, target)


# ---------------------------------------------------------------------------
# mixed-origin: separate prefill walks, shared decode (PR1 required test #4)
# ---------------------------------------------------------------------------


def test_text_and_image_requests_prefill_separately_then_share_decode(target: H.Target) -> None:
    worker = _worker(target)
    text_rids = ["t0", "t1"]
    image_rids = ["v0", "v1"]
    worker.submit("t0", H.text_prompt(12, seed=1))
    worker.submit("v0", H.vision_prompt((1, 4, 4), seed=2))
    worker.submit("t1", H.text_prompt(20, seed=3))
    worker.submit("v1", H.vision_prompt((1, 2, 8), seed=4, prefix=4))

    observed = worker.run_until_done()
    all_rids = sorted(text_rids + image_rids)

    # No LLM batch ever mixes walks, and each prefill walk holds only its kind.
    for obs in W.llm_batches(observed):
        kinds = {("image" if rid.startswith("v") else "text") for rid in obs.request_ids}
        if obs.graph_walk == "prefill":
            assert kinds == {"text"}, obs
        elif obs.graph_walk == "prefill_vision":
            assert kinds == {"image"}, obs
        else:
            assert obs.graph_walk == "decode"
    prefill = _walk_batches(observed, "prefill")
    vision_prefill = _walk_batches(observed, "prefill_vision")
    assert len(prefill) == 1 and sorted(prefill[0].request_ids) == text_rids and prefill[0].packed
    assert len(vision_prefill) == 1 and sorted(vision_prefill[0].request_ids) == image_rids and vision_prefill[0].packed

    # Once both prefill kinds are done, decode is one shared packed batch.
    decode = _walk_batches(observed, "decode")
    shared = [obs for obs in decode if sorted(obs.request_ids) == all_rids]
    assert shared, [o.request_ids for o in decode]
    assert all(obs.packed for obs in shared)
    # Requests that prefilled first may decode alone for a step or two while
    # the other kind is still prefilling, but never with a foreign walk.
    for obs in decode:
        assert set(obs.request_ids) <= set(all_rids)

    assert worker.free_pages == worker.total_pages
    for rid in all_rids:
        _assert_matches_isolated(worker, rid, target)


def test_late_arrival_prefills_alone_then_joins_the_running_decode_batch(target: H.Target) -> None:
    """Continuous batching proper: a request admitted mid-flight is prefilled
    on its own walk and merged into the existing decode batch without
    disturbing the in-flight request's token stream."""
    worker = _worker(target)
    worker.submit("early", H.text_prompt(10, seed=5))
    first = [worker.step() for _ in range(3)]  # prefill + 2 decode steps
    assert [b.graph_walk for b in first] == ["prefill", "decode", "decode"]

    worker.submit("late", H.vision_prompt((1, 4, 4), seed=6))
    observed = worker.run_until_done()

    late_prefill = _walk_batches(observed, "prefill_vision")
    assert len(late_prefill) == 1 and late_prefill[0].request_ids == ("late",)
    joined = [obs for obs in _walk_batches(observed, "decode") if set(obs.request_ids) == {"early", "late"}]
    assert joined, "late request never shared a decode batch with the in-flight one"
    assert all(obs.packed for obs in joined)

    assert worker.records["early"].llm_walks() == ["prefill"] + ["decode"] * MAX_OUTPUT_TOKENS
    assert worker.records["late"].llm_walks() == ["prefill_vision"] + ["decode"] * MAX_OUTPUT_TOKENS
    for rid in ("early", "late"):
        _assert_matches_isolated(worker, rid, target)


def test_engine_observes_exactly_the_scheduled_request_set(target: H.Target) -> None:
    """The ids the scheduler picked are the ids the engine executed, in order,
    for every batch of every walk (no silent splitting or reordering)."""
    worker = _worker(target)
    for i in range(3):
        worker.submit(f"t{i}", H.text_prompt(6 + i, seed=i))
        worker.submit(f"v{i}", H.vision_prompt((1, 2 + 2 * (i % 2), 4), seed=10 + i))
    observed = worker.run_until_done()
    assert observed
    for obs in observed:
        assert obs.engine_request_ids == obs.request_ids, obs
        if obs.node == H.LLM_NODE:
            assert obs.packed, obs
