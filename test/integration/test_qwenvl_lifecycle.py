"""QwenVL PR1 / WS-E — request lifecycle and cross-request isolation (P1-G3, P1-G4).

Engine-level tests drive ``KVCacheEngine`` directly (``qwenvl_harness``);
worker-level tests go through the scheduler stack (``qwenvl_worker_harness``)
so completion, cancellation and OOM back-off take the production code paths
(``pending_removes``, ``hold_requests`` / ``push_back_node``, page release
in ``remove_request``).

P1-G3 (no bleed): perturbing one request's image, prompt length or sampler
must leave every co-batched request's logits and tokens unchanged.
P1-G4 (lifecycle): completion by EOS and by the token limit frees pages;
cancelling one request mid-decode leaves the others' streams intact; pages
recycled from finished requests carry no stale KV into new admissions,
including when the pool is oversubscribed and the scheduler has to hold
requests.
"""

from __future__ import annotations

import pytest
import qwenvl_harness as H
import qwenvl_worker_harness as W
import torch

MAX_OUTPUT_TOKENS = 5
DECODE_STEPS = 4


@pytest.fixture(params=H.TARGET_PARAMS)
def target(request, monkeypatch) -> H.Target:
    return H.prepare_target(request.param, monkeypatch)


@pytest.fixture
def paired(target: H.Target) -> tuple[H.LLMEngine, H.LLMEngine]:
    config = H.make_tiny_config()
    weights = H.build_language_model(config, target, seed=0)

    def build() -> H.LLMEngine:
        return H.build_llm_engine(target, config=config, page_size=16, max_num_pages=128, language_model=weights)

    return build(), build()


def _run_stream(handle: H.LLMEngine, prompts: dict, steps: int) -> dict[str, list[H.StepResult]]:
    """Prefill (one launch per walk) then ``steps`` batched decode steps."""
    text = {rid: p for rid, p in prompts.items() if isinstance(p, H.TextPrompt)}
    image = {rid: p for rid, p in prompts.items() if isinstance(p, H.VisionPrompt)}
    tokens: dict[str, int] = {}
    per_rid: dict[str, list[H.StepResult]] = {rid: [] for rid in prompts}
    for group in (text, image):
        if group:
            result = H.run_prefill(handle, group)
            tokens.update(result.tokens)
            for rid in group:
                per_rid[rid].append(result)
    for _ in range(steps):
        result = H.run_decode(handle, tokens)
        tokens = result.tokens
        for rid in prompts:
            per_rid[rid].append(result)
    return per_rid


def _assert_unchanged(
    baseline: dict[str, list[H.StepResult]], perturbed: dict[str, list[H.StepResult]], rids, target: H.Target, what: str
) -> None:
    for rid in rids:
        margins = [step.margin(rid) for step in baseline[rid]]
        for i, (b, p) in enumerate(zip(baseline[rid], perturbed[rid], strict=True)):
            H.assert_logits_close(p.logits[rid], b.logits[rid], target, f"{what}: {rid} step {i}")
        H.assert_greedy_streams_match(
            [s.tokens[rid] for s in perturbed[rid]],
            [s.tokens[rid] for s in baseline[rid]],
            margins,
            target,
            f"{what}: {rid}",
        )


# ---------------------------------------------------------------------------
# P1-G3 — isolation under perturbation
# ---------------------------------------------------------------------------


@pytest.mark.p1_gate("P1-G3")
@pytest.mark.parametrize("perturbation", ["image_grid", "image_features", "prompt_length", "sampler"])
def test_perturbing_one_request_leaves_co_batched_requests_unchanged(paired, target, perturbation: str) -> None:
    base_engine, perturbed_engine = paired
    steady = {
        "t_a": H.text_prompt(14, seed=1),
        "v_a": H.vision_prompt((1, 4, 4), seed=2),
        "t_b": H.text_prompt(31, seed=3),
    }
    victim_base = H.vision_prompt((1, 2, 8), seed=4, prefix=3)
    if perturbation == "image_grid":
        victim_new = H.vision_prompt((1, 6, 4), seed=4, prefix=3)
    elif perturbation == "image_features":
        victim_new = H.vision_prompt((1, 2, 8), seed=99, prefix=3, feature_scale=3.0)
    elif perturbation == "prompt_length":
        victim_new = H.vision_prompt((1, 2, 8), seed=4, prefix=40, suffix=9)
    else:
        victim_new = victim_base

    for handle in (base_engine, perturbed_engine):
        for rid in steady:
            handle.add_request(rid)
    base_engine.add_request("victim")
    if perturbation == "sampler":
        hot = H.stochastic_sampling(base_engine.config, seed=1234, temperature=1.5)
        perturbed_engine.add_request("victim", sampling=hot)
    else:
        perturbed_engine.add_request("victim")

    baseline = _run_stream(base_engine, {**steady, "victim": victim_base}, DECODE_STEPS)
    perturbed = _run_stream(perturbed_engine, {**steady, "victim": victim_new}, DECODE_STEPS)

    _assert_unchanged(baseline, perturbed, steady, target, perturbation)
    # The perturbation itself must be observable on the victim, or the test
    # would pass vacuously.
    victim_tokens_base = [s.tokens["victim"] for s in baseline["victim"]]
    victim_tokens_new = [s.tokens["victim"] for s in perturbed["victim"]]
    victim_logits_differ = any(
        not torch.allclose(b.logits["victim"], p.logits["victim"], atol=1e-4)
        for b, p in zip(baseline["victim"], perturbed["victim"], strict=True)
    )
    assert victim_logits_differ or victim_tokens_base != victim_tokens_new, "perturbation had no effect on the victim"


@pytest.mark.p1_gate("P1-G3")
def test_sampler_state_is_per_request(paired, target) -> None:
    """Two requests with identical prompts but different sampling configs
    in one decode batch: the greedy one must reproduce its solo stream, and
    per-request sampler state must vanish on removal."""
    handle, _ = paired
    prompt = H.text_prompt(20, seed=5)
    handle.add_request("greedy")
    handle.add_request("hot", sampling=H.stochastic_sampling(handle.config, seed=7, temperature=2.0))
    stream = _run_stream(handle, {"greedy": prompt, "hot": prompt}, DECODE_STEPS)
    assert handle.sampler._sampling_config["greedy"].temperature == 0.0
    assert handle.sampler._sampling_config["hot"].temperature == 2.0

    solo_engine = H.build_llm_engine(
        target, config=handle.config, page_size=16, max_num_pages=128, language_model=handle.submodule.language_model
    )
    solo_engine.add_request("greedy")
    solo = _run_stream(solo_engine, {"greedy": prompt}, DECODE_STEPS)
    H.assert_greedy_streams_match(
        [s.tokens["greedy"] for s in stream["greedy"]],
        [s.tokens["greedy"] for s in solo["greedy"]],
        [s.margin("greedy") for s in solo["greedy"]],
        target,
        "greedy request next to a hot sampler",
    )
    # Same prompt → same prefill logits: only the sampler differs between the
    # two rows (after that the hot request feeds itself different tokens).
    prefill = stream["greedy"][0]
    H.assert_logits_close(prefill.logits["hot"], prefill.logits["greedy"], target, "identical prompts share logits")
    hot_tokens = [s.tokens["hot"] for s in stream["hot"]]
    greedy_tokens = [s.tokens["greedy"] for s in stream["greedy"]]
    assert hot_tokens != greedy_tokens, "temperature-2 sampling reproduced the greedy stream; sampler not applied"

    handle.remove_request("hot")
    assert "hot" not in handle.sampler._sampling_config
    assert "hot" not in handle.sampler._seen_token_mask
    assert not handle.has_request("hot")
    assert "greedy" in handle.sampler._sampling_config


# ---------------------------------------------------------------------------
# P1-G4 — completion, cancellation, page reuse (engine level)
# ---------------------------------------------------------------------------


@pytest.mark.p1_gate("P1-G4")
def test_remove_request_returns_pages_and_clears_all_per_request_state(paired) -> None:
    handle, _ = paired
    prompts = {f"r{i}": H.text_prompt(20 + 16 * i, seed=i) for i in range(4)}
    for rid in prompts:
        handle.add_request(rid)
    tokens = H.run_prefill(handle, prompts).tokens
    H.run_decode(handle, tokens)
    used = {rid: handle.page_indices(rid) for rid in prompts}
    assert all(used.values())
    assert handle.free_pages == handle.total_pages - sum(len(p) for p in used.values())

    handle.remove_request("r1")
    assert not handle.has_request("r1")
    assert "r1" not in handle.sampler._sampling_config
    assert handle.free_pages == handle.total_pages - sum(len(p) for rid, p in used.items() if rid != "r1")
    # Others keep their pages and their KV length.
    for rid in ("r0", "r2", "r3"):
        assert handle.page_indices(rid) == used[rid]
        assert handle.seq_len(rid) == prompts[rid].length + 1

    for rid in ("r0", "r2", "r3"):
        handle.remove_request(rid)
    assert handle.free_pages == handle.total_pages
    assert not handle.alloc_manager.request_states


@pytest.mark.p1_gate("P1-G4")
def test_cancel_mid_decode_does_not_disturb_remaining_requests(paired, target) -> None:
    full, reference = paired
    prompts = {f"r{i}": H.text_prompt(10 + 7 * i, seed=20 + i) for i in range(4)}
    for handle in (full, reference):
        for rid in prompts:
            handle.add_request(rid)

    tokens_f = H.run_prefill(full, prompts).tokens
    tokens_r = H.run_prefill(reference, prompts).tokens
    survivors = ["r0", "r2", "r3"]
    streams_f = {rid: [] for rid in survivors}
    streams_r = {rid: [] for rid in survivors}
    margins_r = {rid: [] for rid in survivors}
    for step in range(DECODE_STEPS):
        if step == 1:
            full.remove_request("r1")
            tokens_f.pop("r1")
            reference.remove_request("r1")
            tokens_r.pop("r1")
            assert full.free_pages == reference.free_pages
        result_f = H.run_decode(full, tokens_f)
        result_r = H.run_decode(reference, tokens_r, sequential=True)
        for rid in survivors:
            H.assert_logits_close(result_f.logits[rid], result_r.logits[rid], target, f"survivor {rid} step {step}")
            streams_f[rid].append(result_f.tokens[rid])
            streams_r[rid].append(result_r.tokens[rid])
            margins_r[rid].append(result_r.margin(rid))
        tokens_f, tokens_r = result_f.tokens, result_r.tokens
    for rid in survivors:
        H.assert_greedy_streams_match(streams_f[rid], streams_r[rid], margins_r[rid], target, f"survivor {rid}")
        assert full.seq_len(rid) == prompts[rid].length + DECODE_STEPS
    assert not full.has_request("r1")


@pytest.mark.p1_gate("P1-G4")
def test_recycled_pages_carry_no_stale_kv(target) -> None:
    """Fill the pool, finish everything, admit new requests onto the recycled
    pages and check they reproduce a fresh-engine run exactly."""
    config = H.make_tiny_config()
    weights = H.build_language_model(config, target, seed=0)
    page_size, pool = 16, 12
    kwargs = dict(config=config, page_size=page_size, max_num_pages=pool, language_model=weights)
    recycled = H.build_llm_engine(target, **kwargs)
    fresh = H.build_llm_engine(target, **kwargs)

    # Round 1: three requests × 3 pages (+1 decode step inside the last page) = 9 of 12 pages.
    first = {f"old{i}": H.text_prompt(40 + i, seed=50 + i) for i in range(3)}
    for rid in first:
        recycled.add_request(rid)
    tokens = H.run_prefill(recycled, first).tokens
    for _ in range(3):
        tokens = H.run_decode(recycled, tokens).tokens
    old_pages = {p for rid in first for p in recycled.page_indices(rid)}
    assert len(old_pages) == 9
    for rid in first:
        recycled.remove_request(rid)
    assert recycled.free_pages == pool

    # Round 2: different prompts that need 8 pages, so FIFO recycling must
    # hand out pages that held round-1 KV.
    second = {"new_a": H.text_prompt(50, seed=60), "new_b": H.vision_prompt((1, 6, 4), seed=61, prefix=30, suffix=10)}
    for handle in (recycled, fresh):
        for rid in second:
            handle.add_request(rid)
    stream_recycled = _run_stream(recycled, second, DECODE_STEPS)
    stream_fresh = _run_stream(fresh, second, DECODE_STEPS)
    new_pages = {p for rid in second for p in recycled.page_indices(rid)}
    assert new_pages & old_pages, "round 2 did not land on any recycled page; test is vacuous"
    _assert_unchanged(stream_fresh, stream_recycled, second, target, "recycled pages")


# ---------------------------------------------------------------------------
# P1-G4 — through the scheduler / worker stack
# ---------------------------------------------------------------------------


@pytest.mark.p1_gate("P1-G4")
def test_worker_completion_by_token_limit_frees_pages_and_state(target) -> None:
    worker = W.QwenVLWorker(target, seed=0, max_output_tokens=MAX_OUTPUT_TOKENS, page_size=16, max_num_pages=64)
    for i in range(4):
        worker.submit(f"r{i}", H.text_prompt(9 + 4 * i, seed=i))
    worker.run_until_done()
    assert worker.free_pages == worker.total_pages
    assert not worker.active_requests()
    assert not worker.alloc_manager.request_states
    assert not worker.sampler._sampling_config
    for rid, record in worker.records.items():
        assert record.done and len(record.tokens) == MAX_OUTPUT_TOKENS + 1, (rid, record.tokens)


@pytest.mark.p1_gate("P1-G4")
def test_worker_completion_by_eos_stops_the_decode_loop_early(target) -> None:
    worker = W.QwenVLWorker(target, seed=0, max_output_tokens=MAX_OUTPUT_TOKENS, page_size=16, max_num_pages=64)
    eos = worker.config.text_config.eos_token_id
    honor_eos = H.greedy_sampling(worker.config, ignore_eos=False)
    worker.submit("stops", H.text_prompt(12, seed=1), sampling=honor_eos)
    worker.submit("runs", H.text_prompt(12, seed=1))  # same prompt, ignore_eos=True
    # Bias the head so both requests emit EOS from the first token on.
    with torch.no_grad():
        worker.llm_submodule.lm_head.weight[eos] += 50.0
    worker.run_until_done()
    stops, runs = worker.records["stops"], worker.records["runs"]
    assert stops.tokens[:2] == [eos, eos] and len(stops.tokens) == 2, {
        "tokens": stops.tokens,
        "top2": stops.top2,
    }
    assert runs.tokens == [eos] * (MAX_OUTPUT_TOKENS + 1), {
        "tokens": runs.tokens,
        "top2": runs.top2,
    }
    assert worker.free_pages == worker.total_pages


@pytest.mark.p1_gate("P1-G4")
def test_worker_cancel_mid_decode_frees_pages_and_others_match_isolated(target) -> None:
    worker = W.QwenVLWorker(target, seed=0, max_output_tokens=MAX_OUTPUT_TOKENS, page_size=16, max_num_pages=64)
    rids = [f"r{i}" for i in range(4)]
    for i, rid in enumerate(rids):
        worker.submit(rid, H.text_prompt(11 + 6 * i, seed=30 + i))
    assert worker.step().graph_walk == "prefill"
    assert worker.step().graph_walk == "decode"
    before = worker.free_pages
    victim_pages = len(worker.page_indices("r2"))
    worker.cancel("r2")
    assert worker.free_pages == before + victim_pages
    assert "r2" not in worker.active_requests()

    observed = worker.run_until_done([r for r in rids if r != "r2"])
    for obs in W.llm_batches(observed, "decode"):
        assert sorted(obs.request_ids) == ["r0", "r1", "r3"] and obs.packed, obs
    assert worker.free_pages == worker.total_pages
    for rid in ("r0", "r1", "r3"):
        record = worker.records[rid]
        alone = W.isolated_run(
            target, rid, record.prompt, seed=0, max_output_tokens=MAX_OUTPUT_TOKENS, page_size=16, max_num_pages=64
        )
        H.assert_greedy_streams_match(record.tokens, alone.tokens, alone.margins, target, f"{rid} after cancel")


@pytest.mark.p1_gate("P1-G4")
def test_worker_late_arrivals_that_do_not_fit_are_held_then_admitted_on_recycled_pages(target) -> None:
    """Pool near capacity with requests mid-decode; late arrivals that cannot
    get pages must be pushed back and held (allocation failure path), then
    admitted onto the pages the finished requests release, with token streams
    identical to a fresh run (no stale KV).

    Note: the platform holds *the whole failing batch*; a single prefill batch
    whose combined footprint exceeds the pool would be retried as-is, so this
    test admits the oversubscribing requests only after the first wave is
    already decoding — the shape the conductor's admission produces.
    """
    page_size, pool = 16, 10
    worker = W.QwenVLWorker(
        target, seed=0, max_output_tokens=MAX_OUTPUT_TOKENS, page_size=page_size, max_num_pages=pool
    )
    first_wave = {f"r{i}": H.text_prompt(40 + 3 * i, seed=70 + i) for i in range(3)}  # 3 pages each → 9 of 10
    for rid, prompt in first_wave.items():
        worker.submit(rid, prompt)
    assert worker.step().graph_walk == "prefill"
    assert worker.step().graph_walk == "decode"
    old_pages = {p for rid in first_wave for p in worker.page_indices(rid)}
    assert worker.free_pages == 1

    late = {"late_a": H.text_prompt(44, seed=80), "late_b": H.vision_prompt((1, 6, 4), seed=81, prefix=30, suffix=8)}
    for rid, prompt in late.items():
        worker.submit(rid, prompt)

    late_pages: dict[str, set[int]] = {}

    def record_late_pages(w: W.QwenVLWorker) -> None:
        for rid in late:
            if rid not in late_pages and rid in w.alloc_manager.request_states and w.page_indices(rid):
                late_pages[rid] = set(w.page_indices(rid))

    worker.run_until_done(after_step=record_late_pages)

    assert worker.oom_events, "late arrivals were never refused pages; test is vacuous"
    assert all(set(rids) <= set(late) for _, _, rids in worker.oom_events), worker.oom_events
    late_prefills = [
        obs for obs in W.llm_batches(worker.observed) if obs.graph_walk != "decode" and set(obs.request_ids) & set(late)
    ]
    refused = [obs for obs in late_prefills if obs.allocation_failed]
    admitted = [obs for obs in late_prefills if not obs.allocation_failed]
    assert refused and admitted
    assert all(obs.packed for obs in admitted), admitted
    assert late_pages.keys() == late.keys()
    assert all(pages & old_pages for pages in late_pages.values()), (late_pages, old_pages)
    assert worker.free_pages == worker.total_pages

    for rid, prompt in {**first_wave, **late}.items():
        record = worker.records[rid]
        assert record.done and len(record.tokens) == MAX_OUTPUT_TOKENS + 1, (rid, record.tokens)
        alone = W.isolated_run(
            target, rid, prompt, seed=0, max_output_tokens=MAX_OUTPUT_TOKENS, page_size=page_size, max_num_pages=pool
        )
        H.assert_greedy_streams_match(record.tokens, alone.tokens, alone.margins, target, f"{rid} after OOM back-off")
