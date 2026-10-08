import asyncio
import time

import numpy as np
import pytest

from auxein.aggregators import Aggregator, mean
from auxein.backend import Backend
from auxein.core import Status
from auxein.environments import ScenarioSet
from auxein.evaluators import EpisodeEvaluator, EvaluationError
from auxein.execution import make_executor
from tests.support import episodes as ep
from tests.support import pointmass as pm
from tests.support.helpers import array_batch, eval_context, problem

pytestmark = pytest.mark.filterwarnings(
    "ignore::auxein.execution.AbandonedEvaluationWarning"
)  # the thread timeouts abandon threads on purpose

SCENARIOS = ScenarioSet.from_params([{}, {}, {}, {}], seed=1)  # ids s0000 .. s0003


def value_aggregator() -> Aggregator:
    return Aggregator({"value": mean("gene")})


def make(environment=None, decoder=None, scenarios=SCENARIOS, aggregator=None, **kwargs) -> EpisodeEvaluator:
    return EpisodeEvaluator(decoder or ep.ListDecoder(), environment or ep.Probe(), scenarios, aggregator or value_aggregator(), **kwargs)


def evaluate(evaluator, batch, *, backend=None, concurrency=1, kind="inline", policy="infeasible", timeout=None, seed=0, spec=None):
    backend = backend or Backend()
    executor = make_executor(kind, concurrency)
    try:
        ctx = eval_context(
            spec or problem(), backend, seed, concurrency=concurrency, executor=executor, failure_policy=policy, timeout=timeout
        )
        return asyncio.run(evaluator.evaluate(batch, ctx))
    finally:
        executor.shutdown()


def batch_of(n=3, backend=None):
    return array_batch(backend or Backend(), n=n)


# --- the per-episode path ---


@pytest.mark.parametrize(("concurrency", "kind"), [(1, "inline"), (1, "thread"), (4, "inline"), (4, "thread"), (4, "process")])
def test_the_per_episode_path_gives_the_same_evaluations_with_every_executor(concurrency: int, kind: str):
    results = evaluate(make(), batch_of(5), concurrency=concurrency, kind=kind)
    assert [e.candidate.id for e in results] == list(range(5))
    assert [e.status for e in results] == [Status.OK] * 5
    assert [e.objectives["value"] for e in results] == [1.0 + 3 * i for i in range(5)]  # the mean of one gene over the scenarios
    assert all(e.raw is not None and e.raw.key == f"episodes/{e.candidate.id}" for e in results)


def test_the_episodes_of_every_candidate_and_scenario_travel_with_the_evaluations():
    results = evaluate(make(), batch_of(3))
    episodes = results.episodes
    assert episodes is not None
    assert (
        episodes.candidate_ids == (0, 1, 2)
        and episodes.scenario_ids == SCENARIOS.ids
        and episodes.names == ("agent", "gene", "scenario", "world")
    )
    assert episodes.values.shape == (3, 4, 4) and not episodes.failures
    np.testing.assert_array_equal(episodes.values[:, :, episodes.names.index("scenario")], [[0, 1, 2, 3]] * 3)


def test_async_environments_run_natively_with_any_concurrency():
    for concurrency in (1, 4):
        results = evaluate(make(ep.AsyncProbe()), batch_of(3), concurrency=concurrency)
        assert [e.objectives["value"] for e in results] == [1.0, 4.0, 7.0]


def test_an_async_environment_with_a_process_executor_is_an_error():
    with pytest.raises(ValueError, match="async environment.*cannot be sent to worker processes"):
        evaluate(make(ep.AsyncProbe()), batch_of(2), concurrency=2, kind="process")


@pytest.mark.parametrize("kind", ["thread", "inline"])
def test_episodes_in_progress_never_exceed_the_concurrency_across_candidates_and_scenarios(kind: str):
    gauge = ep.Gauge()
    environment = ep.SlowProbe(gauge) if kind == "thread" else ep.AsyncProbe(gauge)
    evaluate(make(environment), batch_of(3), concurrency=5, kind=kind)  # 3 candidates x 4 scenarios = 12 episodes
    assert gauge.calls == 12 and gauge.peak == 5  # the limit is on the whole batch, and it is used


def test_the_limit_is_not_per_candidate():
    gauge = ep.Gauge()
    evaluate(make(ep.AsyncProbe(gauge, 0.02)), batch_of(6), concurrency=8)  # one candidate has only 4 episodes
    assert gauge.peak == 8


def test_results_are_in_ask_order_even_when_later_candidates_finish_first():
    finished: list[float] = []

    class Reversed(ep.Probe):
        def run_episode(self, agents, scenario, rng):
            time.sleep(0.04 - 0.012 * agents["agent"][0] / 3)  # the first candidate is the slowest
            finished.append(agents["agent"][0])
            return super().run_episode(agents, scenario, rng)

    spec_scenarios = ScenarioSet.from_params([{}], seed=1)
    results = evaluate(make(Reversed(), scenarios=spec_scenarios), batch_of(3), concurrency=3, kind="thread")
    assert finished != sorted(finished)
    assert [e.candidate.id for e in results] == [0, 1, 2] and results.episodes is not None and results.episodes.candidate_ids == (0, 1, 2)


def test_each_candidates_wall_time_is_the_time_of_its_episodes():
    results = evaluate(make(ep.SlowProbe(pause=0.02)), batch_of(2))
    assert all(0.07 < e.cost.wall_time < 0.5 for e in results)  # four episodes of 20 ms each


# --- common random numbers ---


def test_every_candidate_faces_the_same_world_but_has_its_own_agent_stream():
    results = evaluate(make(), batch_of(3))
    episodes = results.episodes
    assert episodes is not None
    world = episodes.values[:, :, episodes.names.index("world")]
    agent = episodes.values[:, :, episodes.names.index("agent")]
    np.testing.assert_array_equal(world[0], world[1])
    np.testing.assert_array_equal(world[0], world[2])  # the identical realisation for all candidates
    assert len(set(world[0])) == 4  # and different scenarios have different worlds
    assert len({tuple(row) for row in agent}) == 3  # while the agents' own randomness differs per candidate
    assert len(set(agent[0])) == 4  # and per scenario


def test_the_world_depends_on_the_scenario_seed_and_the_agent_on_the_run_seed():
    first = evaluate(make(), batch_of(2), seed=1).episodes
    second = evaluate(make(), batch_of(2), seed=2).episodes
    assert first is not None and second is not None
    world, agent = first.names.index("world"), first.names.index("agent")
    np.testing.assert_array_equal(first.values[:, :, world], second.values[:, :, world])  # the run seed does not touch the world
    assert not np.array_equal(first.values[:, :, agent], second.values[:, :, agent])
    again = evaluate(make(), batch_of(2), seed=1).episodes
    assert again is not None
    np.testing.assert_array_equal(first.values, again.values)


@pytest.mark.parametrize("kind", ["inline", "thread", "process"])
def test_the_streams_are_the_same_whatever_the_executor(kind: str):
    reference = evaluate(make(), batch_of(3)).episodes
    other = evaluate(make(), batch_of(3), concurrency=3, kind=kind).episodes
    assert reference is not None and other is not None
    np.testing.assert_array_equal(reference.values, other.values)


# --- failures ---


@pytest.mark.parametrize(
    "environment", [ep.Troubled(raises=("s0002",)), ep.Troubled(returns=("s0002",)), ep.AsyncTroubled(raises=("s0002",))]
)
def test_one_failing_scenario_fails_the_candidate_and_the_error_lists_it(environment):
    results = evaluate(make(environment), batch_of(2))
    for evaluation in results:
        assert evaluation.status is Status.FAILED and not evaluation.objectives
        assert evaluation.error is not None and "1 of 4 episodes did not succeed" in evaluation.error and "'s0002'" in evaluation.error
    assert "the simulator diverged in s0002" in results[0].error or "the agent crashed in s0002" in results[0].error  # type: ignore[operator]


def test_the_error_lists_every_failing_scenario_and_counts_the_rest():
    results = evaluate(make(ep.Troubled(raises=("s0000", "s0001", "s0002", "s0003"))), batch_of(1))
    error = results[0].error or ""
    assert (
        "4 of 4 episodes" in error
        and all(f"'s000{i}'" in error for i in range(4))
        and error.count("ValueError: the simulator diverged") == 3
    )
    assert "and in scenario(s) 's0003'" in error  # the first three are described, the rest named


def test_the_measurements_of_the_successful_episodes_are_still_recorded():
    results = evaluate(make(ep.Troubled(raises=("s0001",))), batch_of(2))
    episodes = results.episodes
    assert episodes is not None and set(episodes.failures) == {(0, 1), (1, 1)}
    assert episodes.failures[(0, 1)][0] is Status.FAILED and "diverged" in episodes.failures[(0, 1)][1]
    scenario = episodes.names.index("scenario")
    assert episodes.values[0, 0, scenario] == 0.0 and episodes.values[0, 2, scenario] == 2.0  # the others measured
    assert np.isnan(episodes.values[0, 1, scenario])  # the failed one has nothing to record


def test_other_candidates_are_unaffected():
    class OnlyFirst(ep.Probe):
        def run_episode(self, agents, scenario, rng):
            if agents["agent"][0] == 1.0 and scenario.index == 3:
                raise RuntimeError("boom")
            return super().run_episode(agents, scenario, rng)

    results = evaluate(make(OnlyFirst()), batch_of(3))
    assert [e.status for e in results] == [Status.FAILED, Status.OK, Status.OK]


def test_fail_fast_raises_at_the_first_exception_naming_the_candidate_and_chaining_it():
    with pytest.raises(EvaluationError, match="candidate 0") as raised:
        evaluate(make(ep.Troubled(raises=("s0002",))), batch_of(2), policy="fail_fast")
    assert isinstance(raised.value.__cause__, ValueError)


def test_a_returned_failure_is_left_to_the_driver_under_fail_fast():
    """The environment reported it: the evaluator returns the FAILED evaluation and the driver's backstop applies the policy."""
    results = evaluate(make(ep.Troubled(returns=("s0001",))), batch_of(2), policy="fail_fast")
    assert [e.status for e in results] == [Status.FAILED, Status.FAILED]


def test_a_non_finite_aggregated_objective_fails_the_candidate_naming_the_objective():
    results = evaluate(make(ep.Diverging(), aggregator=Aggregator({"value": mean("world")})), batch_of(2))
    assert all(e.status is Status.FAILED and "non-finite objective value: 'value' is nan" in (e.error or "") for e in results)


def test_a_genome_that_cannot_be_decoded_fails_only_that_candidate():
    results = evaluate(make(decoder=ep.BrokenDecoder()), array_batch(Backend(), n=2) if False else _with_first_genes([1.0, 500.0, 2.0]))
    assert [e.status for e in results] == [Status.OK, Status.FAILED, Status.OK]
    assert "decoding the genome failed" in (results[1].error or "") and "cannot decode this genome" in (results[1].error or "")
    assert results.episodes is not None and results.episodes.candidate_ids == (0, 2)  # the undecodable one ran nothing
    with pytest.raises(EvaluationError):
        evaluate(make(decoder=ep.BrokenDecoder()), _with_first_genes([1.0, 500.0]), policy="fail_fast")


def _with_first_genes(genes):
    from auxein.core import ArrayBatch, CandidateId

    backend = Backend()
    return ArrayBatch(backend.asarray(np.array([[g, 0.0, 0.0] for g in genes])), [CandidateId(i) for i in range(len(genes))], 0, "random")


# --- misconfiguration is not a failed episode ---


def test_an_environment_that_returns_something_else_is_misconfiguration_under_every_policy():
    class WrongDecoder(ep.ListDecoder):
        pass

    for policy in ("infeasible", "fail_fast"):
        with pytest.raises(TypeError, match="must return an EpisodeResult, got dict"):
            evaluate(make(ep.Wrong()), batch_of(1), policy=policy)


def test_episodes_that_report_different_measurements_are_rejected():
    with pytest.raises(ValueError, match="must report the same measurements"):
        evaluate(make(ep.Shifty(), aggregator=Aggregator({"value": mean("a")})), batch_of(1))


def test_an_aggregator_that_does_not_match_the_problem_is_rejected_like_a_typo_in_a_result():
    with pytest.raises(ValueError, match=r"objectives the problem declares but the aggregator lacks: \['value'\]"):
        evaluate(make(aggregator=Aggregator({"vlaue": mean("gene")})), batch_of(1))


def test_an_aggregator_reading_an_unknown_measurement_says_what_exists():
    with pytest.raises(ValueError, match=r"reads the measurement 'nope'.*\['agent', 'gene', 'scenario', 'world'\]"):
        evaluate(make(aggregator=Aggregator({"value": mean("nope")})), batch_of(1))


def test_a_negative_constraint_from_the_aggregator_is_rejected():
    from auxein.core import ProblemSpec

    spec = ProblemSpec(problem().space, problem().objectives, ("c",))
    aggregator = Aggregator({"value": mean("gene")}, {"c": mean(lambda m: m["world"] - 100.0)})  # always below zero
    with pytest.raises(ValueError, match="constraint"):
        evaluate(make(aggregator=aggregator), batch_of(1), spec=spec)


# --- roles ---


def test_the_evolved_role_must_be_named_when_the_environment_has_several():
    with pytest.raises(ValueError, match="say which one"):
        make(ep.TwoRoles())
    with pytest.raises(ValueError, match="not 'nobody'"):
        make(ep.TwoRoles(), role="nobody")

    seen: list[tuple[str, ...]] = []

    class Spy(ep.TwoRoles):
        def run_episode(self, agents, scenario, rng):
            seen.append(tuple(agents))
            return ep.Probe.run_episode(self, {"agent": agents["evolved"]}, scenario, rng)

    evaluate(make(Spy(), role="evolved"), batch_of(1))
    assert set(seen) == {("evolved",)}  # agents are passed by role; the other participants belong to the scenario


# --- the description ---


def test_the_description_names_the_components_and_the_scenario_fingerprint():
    text = repr(make())
    assert text.startswith("EpisodeEvaluator(environment=Probe(), decoder=ListDecoder(), aggregator=Aggregator(")
    assert f"scenarios=4:{SCENARIOS.fingerprint[:16]}" in text and "role='agent'" in text
    assert text == repr(make())  # the same in every process: no memory addresses
    other = ScenarioSet.from_params([{}, {}, {}, {"x": 1}], seed=1)
    assert repr(make(scenarios=other)) != text
    assert "at 0x" not in repr(make(ep.AsyncProbe())) and "AsyncProbe" in repr(make(ep.AsyncProbe()))


def test_a_context_without_episode_streams_is_refused():
    from dataclasses import replace

    ctx = replace(eval_context(problem(), Backend()), episode_rng_for=None)
    with pytest.raises(ValueError, match="no episode streams"):
        asyncio.run(make().evaluate(batch_of(1), ctx))


# --- the batched path ---


def pointmass_evaluator(decoder=None) -> EpisodeEvaluator:
    return EpisodeEvaluator(decoder or pm.GainsDecoder(), pm.PointMassEnvironment(), pm.scenario_set(6), pm.aggregator())


def test_the_batched_path_is_taken_when_environment_and_decoder_offer_it_and_the_batch_is_an_array(backend: Backend):
    calls: list[str] = []

    class Spy(pm.PointMassEnvironment):
        def run_episode(self, *args, **kwargs):
            calls.append("episode")
            return super().run_episode(*args, **kwargs)

        def run_batch(self, *args, **kwargs):
            calls.append("batch")
            return super().run_batch(*args, **kwargs)

    evaluator = EpisodeEvaluator(pm.GainsDecoder(), Spy(), pm.scenario_set(6), pm.aggregator())
    assert evaluator.batched and evaluator.batch_sensitive
    batch = _gains_batch(backend, 4)
    results = evaluate(evaluator, batch, backend=backend, spec=pm.problem())
    assert calls == ["batch"] and all(e.status is Status.OK for e in results)  # once for the whole batch, never per episode
    assert results.episodes is not None and results.episodes.values.shape == (4, 6, 5)


def test_a_decoder_without_decode_batch_takes_the_per_episode_path_even_if_the_environment_could_batch(backend: Backend):
    evaluator = pointmass_evaluator(pm.PerEpisodeDecoder())
    assert not evaluator.batched
    batched = evaluate(pointmass_evaluator(), _gains_batch(backend, 3), backend=backend, spec=pm.problem())
    single = evaluate(evaluator, _gains_batch(backend, 3), backend=backend, spec=pm.problem())
    atol = 1e-12 if backend.precision == "float64" and backend.name == "numpy" else 1e-4
    for a, b in zip(batched, single, strict=True):
        assert a.objectives["error"] == pytest.approx(b.objectives["error"], abs=atol)


def test_the_batched_path_rejects_a_timeout():
    with pytest.raises(ValueError, match="batched EpisodeEvaluator does not support a timeout"):
        evaluate(pointmass_evaluator(), _gains_batch(Backend(), 2), timeout=5.0, spec=pm.problem())


def test_the_batched_path_reports_a_failed_episode_per_candidate_and_scenario():
    class Failing(pm.PointMassEnvironment):
        def run_batch(self, agents, scenarios, rng):
            from auxein.environments import EpisodeBatchResult, EpisodeFailure

            out = super().run_batch(agents, scenarios, rng)
            return EpisodeBatchResult(dict(out.measurements), {(1, 2): EpisodeFailure(Status.TIMEOUT, "the simulator timed out")})

    evaluator = EpisodeEvaluator(pm.GainsDecoder(), Failing(), pm.scenario_set(6), pm.aggregator())
    results = evaluate(evaluator, _gains_batch(Backend(), 3), spec=pm.problem())
    assert [e.status for e in results] == [Status.OK, Status.TIMEOUT, Status.OK]
    assert (
        "the simulator timed out" in (results[1].error or "")
        and results.episodes is not None
        and set(results.episodes.failures) == {(1, 2)}
    )


def test_an_exception_in_run_batch_fails_every_candidate_or_stops_the_run():
    class Exploding(pm.PointMassEnvironment):
        def run_batch(self, agents, scenarios, rng):
            raise RuntimeError("the GPU simulator crashed")

    evaluator = EpisodeEvaluator(pm.GainsDecoder(), Exploding(), pm.scenario_set(6), pm.aggregator())
    results = evaluate(evaluator, _gains_batch(Backend(), 3), spec=pm.problem())
    assert all(e.status is Status.FAILED and "the GPU simulator crashed" in (e.error or "") for e in results)
    with pytest.raises(EvaluationError):
        evaluate(evaluator, _gains_batch(Backend(), 3), spec=pm.problem(), policy="fail_fast")


def test_run_batch_must_return_the_right_shape():
    class Short(pm.PointMassEnvironment):
        def run_batch(self, agents, scenarios, rng):
            out = super().run_batch(agents, scenarios, rng)
            from auxein.environments import EpisodeBatchResult

            return EpisodeBatchResult({k: v[:, :2] for k, v in out.measurements.items()})

    with pytest.raises(ValueError, match="shape"):
        evaluate(
            EpisodeEvaluator(pm.GainsDecoder(), Short(), pm.scenario_set(6), pm.aggregator()), _gains_batch(Backend(), 3), spec=pm.problem()
        )


def test_the_batched_world_noise_is_the_same_for_every_candidate_and_the_stream_is_per_batch():
    seen: list[object] = []

    class Spy(pm.PointMassEnvironment):
        def run_batch(self, agents, scenarios, rng):
            seen.append(rng.uniform((1,)).tolist())
            return super().run_batch(agents, scenarios, rng)

    evaluator = EpisodeEvaluator(pm.GainsDecoder(), Spy(), pm.scenario_set(6), pm.aggregator())
    one = _gains_batch(Backend(), 3)
    evaluate(evaluator, one, spec=pm.problem(), seed=2)
    evaluate(evaluator, one, spec=pm.problem(), seed=2)
    assert seen[0] == seen[1]  # derived from the first candidate's id and the run seed: reproducible
    evaluate(evaluator, one, spec=pm.problem(), seed=3)
    assert seen[2] != seen[0]


def _gains_batch(backend: Backend, n: int):
    from auxein.core import ArrayBatch, CandidateId

    gains = np.array([[2.0 + i, 3.0, 0.1 * i] for i in range(n)])
    return ArrayBatch(backend.asarray(gains), [CandidateId(i) for i in range(n)], 0, "random")


# --- timeouts, per episode ---


def test_a_timed_out_episode_makes_the_candidate_time_out_with_a_thread_executor():
    started = time.perf_counter()
    results = evaluate(make(ep.Troubled(slow=("s0001",), seconds=5.0)), batch_of(2), concurrency=4, kind="thread", timeout=0.3)
    assert time.perf_counter() - started < 3.0
    for evaluation in results:
        assert evaluation.status is Status.TIMEOUT and "'s0001'" in (evaluation.error or "") and "timed out" in (evaluation.error or "")
        assert evaluation.cost.wall_time > 0.2
    assert results.episodes is not None and results.episodes.failures[(0, 1)][0] is Status.TIMEOUT


def test_a_timed_out_episode_is_cancelled_when_the_environment_is_async():
    started = time.perf_counter()
    results = evaluate(make(ep.AsyncTroubled(slow=("s0003",), seconds=5.0)), batch_of(2), concurrency=4, timeout=0.2)
    assert time.perf_counter() - started < 3.0
    assert [e.status for e in results] == [Status.TIMEOUT, Status.TIMEOUT]


def test_a_timed_out_episode_kills_its_worker_with_a_process_executor():
    started = time.perf_counter()
    results = evaluate(make(ep.Troubled(slow=("s0002",), seconds=30.0)), batch_of(3), concurrency=3, kind="process", timeout=1.0)
    assert time.perf_counter() - started < 20.0
    assert [e.status for e in results] == [Status.TIMEOUT] * 3 and all("'s0002'" in (e.error or "") for e in results)
    others = results.episodes
    assert others is not None and set(others.failures) == {(0, 2), (1, 2), (2, 2)}  # only that scenario: the others ran


def test_a_worker_that_dies_fails_the_candidate_with_the_cause_in_its_error():
    results = evaluate(make(ep.Troubled(kills=("s0001",))), batch_of(2), concurrency=2, kind="process")
    assert all(e.status is Status.FAILED and "died" in (e.error or "") for e in results)


def test_a_candidate_is_a_timeout_only_if_every_failing_episode_timed_out():
    results = evaluate(
        make(ep.Troubled(slow=("s0001",), raises=("s0002",), seconds=5.0)), batch_of(1), concurrency=4, kind="thread", timeout=0.3
    )
    assert results[0].status is Status.FAILED  # a mix of a timeout and an exception is a failure
