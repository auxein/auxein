"""The agent layer through the driver: evolving agents, recording their episodes, judging them held out, and resuming (design doc §6)."""

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import auxein
from auxein.aggregators import Aggregator, cvar_upper, maximum, mean
from auxein.core import Objective, Status
from auxein.driver import (
    ConfigurationMismatchError,
    EvaluationFailureWarning,
    SteadyStateVectorisationWarning,
    aevaluate_held_out,
    evaluate_held_out,
)
from auxein.environments import ScenarioSet
from auxein.evaluators import EvaluationError
from auxein.recording import open_run
from tests.driver.resume_test import comparable, same_result
from tests.support import episodes as ep
from tests.support import pointmass as pm
from tests.support.eventlog import event_log
from tests.support.fixtures import integration_backend
from tests.support.reading import peek
from tests.support.resumable import backend_config, settings

ROOT = Path(__file__).resolve().parents[2]
SELECTION, HELD_OUT = ScenarioSet.generate_split(pm.scenario_params, 8, 8, seed=3)

pytestmark = [pytest.mark.filterwarnings("ignore::auxein.driver.errors.EvaluationFailureWarning"), pytest.mark.usefixtures("use_corner_backend")]


def evaluator(
    scenarios: ScenarioSet = SELECTION, *, batched: bool = True, aggregator: Aggregator | None = None
) -> auxein.EpisodeEvaluator[Any]:
    decoder = pm.GainsDecoder() if batched else pm.PerEpisodeDecoder()
    return auxein.EpisodeEvaluator(decoder, pm.PointMassEnvironment(), scenarios, aggregator or pm.aggregator())


def arguments(strategy: Any, run_dir: Path | None = None, **over: Any) -> dict[str, Any]:
    settings_: dict[str, Any] = {
        "strategy": strategy,
        "evaluator": evaluator(),
        "space": pm.SPACE,
        "objectives": [Objective("error")],
        "constraints": ["overshoot"],
        "descriptors": ["success_rate"],
        "seed": 4,
        "batch_size": 20,
        "run_dir": run_dir,
        "backend": integration_backend(),
    }
    settings_.update(over)
    if run_dir is None:
        warnings.simplefilter("ignore", auxein.RecordingDisabledWarning)
    return settings_


def same_values(a: Any, b: Any) -> bool:
    """Whether two dicts of measurements agree: exactly in float64; to float32 rounding where the two sides did the arithmetic
    on different paths (batched in float32 against per-episode in Python floats, or re-reduced on the host)."""
    if integration_backend().precision == "float64":
        return bool(a == b)
    return a.keys() == b.keys() and all(a[k] == pytest.approx(b[k], rel=1e-4, abs=1e-6) for k in a)


def genetic() -> auxein.GeneticAlgorithm:
    return auxein.GeneticAlgorithm(population_size=20, offspring_size=20)


# --- evolving agents ---


@pytest.mark.parametrize("batched", [True, False])
def test_a_genetic_algorithm_evolves_a_controller_better_than_random_search_and_satisfies_the_constraint(batched: bool):
    ga, rs = [], []
    for seed in range(4):
        found = auxein.run(budget=auxein.Budget(evaluations=300), **arguments(genetic(), seed=seed, evaluator=evaluator(batched=batched)))
        random = auxein.run(
            budget=auxein.Budget(evaluations=300), **arguments(auxein.RandomSearch(), seed=seed, evaluator=evaluator(batched=batched))
        )
        assert found.best is not None and found.best.constraints["overshoot"] == 0.0  # feasible on the selection set
        ga.append(found.best.objectives["error"])
        rs.append(random.best.objectives["error"])  # type: ignore[union-attr]
    assert np.mean(ga) < np.mean(rs) and sum(g <= r for g, r in zip(ga, rs, strict=True)) >= 3


def test_the_step_level_environment_runs_through_the_same_evaluator():
    evolved = auxein.EpisodeEvaluator(pm.StepGainsDecoder(), pm.step_environment(), SELECTION, pm.aggregator())
    result = auxein.run(budget=auxein.Budget(evaluations=100), **arguments(genetic(), evaluator=evolved))
    reference = auxein.run(budget=auxein.Budget(evaluations=100), **arguments(genetic()))  # batched numpy
    assert result.best is not None and reference.best is not None
    assert same_values(result.best.objectives, reference.best.objectives)  # the same maths, so the same run


@pytest.mark.parametrize(("executor", "concurrency"), [("inline", 1), ("thread", 4), ("process", 3)])
def test_the_run_is_the_same_with_every_executor(executor: str, concurrency: int):
    kwargs = {"evaluator": evaluator(batched=False), "delivery": "steady_state", "concurrency": concurrency, "executor": executor}
    one = auxein.run(budget=auxein.Budget(evaluations=60), **arguments(genetic(), **kwargs))
    base = auxein.run(budget=auxein.Budget(evaluations=60), **arguments(genetic(), **{**kwargs, "concurrency": 1, "executor": "inline"}))
    same_result(one, base)


def test_a_timeout_with_a_batched_evaluator_is_an_error_at_start_up():
    with pytest.raises(ValueError, match="timeout cannot be combined with a batched EpisodeEvaluator"):
        auxein.run(budget=auxein.Budget(evaluations=20), timeout=5.0, **arguments(genetic()))
    auxein.run(
        budget=auxein.Budget(evaluations=20), timeout=5.0, **arguments(genetic(), evaluator=evaluator(batched=False))
    )  # per episode: fine


def test_steady_state_delivery_warns_about_a_batched_evaluator_once():
    with pytest.warns(SteadyStateVectorisationWarning, match="one candidate at a time") as caught:
        auxein.run(budget=auxein.Budget(evaluations=30), delivery="steady_state", **arguments(genetic()))
    assert len([w for w in caught if issubclass(w.category, SteadyStateVectorisationWarning)]) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", SteadyStateVectorisationWarning)
        auxein.run(
            budget=auxein.Budget(evaluations=30), delivery="steady_state", **arguments(genetic(), evaluator=evaluator(batched=False))
        )


# --- failures through the driver ---


def troubled(environment: Any) -> auxein.EpisodeEvaluator[Any]:
    scenarios = ScenarioSet.from_params([{}, {}, {}], seed=1)
    return auxein.EpisodeEvaluator(ep.ListDecoder(), environment, scenarios, Aggregator({"error": mean("gene")}))


def trouble_arguments(environment: Any, run_dir: Path | None = None, **over: Any) -> dict[str, Any]:
    return arguments(
        auxein.RandomSearch(), run_dir, evaluator=troubled(environment), constraints=[], descriptors=[], space=pm.SPACE, **over
    )


def test_a_failing_scenario_fails_the_candidate_and_is_counted_under_infeasible():
    class Sometimes(ep.Probe):
        def run_episode(self, agents, scenario, rng):
            if scenario.index == 1 and float(agents["agent"][0]) < 5.0:  # the low first genes fail in one scenario
                raise RuntimeError("the simulator diverged")
            return super().run_episode(agents, scenario, rng)

    with pytest.warns(EvaluationFailureWarning, match="1 of 3 episodes did not succeed"):
        result = auxein.run(budget=auxein.Budget(evaluations=40), **trouble_arguments(Sometimes()))
    assert 0 < result.status_counts["failed"] < 40 and result.status_counts["ok"] + result.status_counts["failed"] == 40
    assert result.best is not None and result.best.status is Status.OK


def test_the_driver_applies_fail_fast_to_a_failure_the_environment_returned():
    with pytest.raises(EvaluationError, match="failed"):
        auxein.run(budget=auxein.Budget(evaluations=40), failure_policy="fail_fast", **trouble_arguments(ep.Troubled(returns=("s0001",))))


def test_a_run_where_every_episode_fails_trips_the_guard():
    from auxein.driver import AllEvaluationsFailedError

    with pytest.raises(AllEvaluationsFailedError, match="agent crashed"):
        auxein.run(budget=auxein.Budget(evaluations=40), **trouble_arguments(ep.Troubled(returns=("s0000", "s0001", "s0002"))))


# --- recording the episodes ---


def test_every_candidates_episodes_are_recorded_with_its_evaluation(tmp_path: Path):
    result = auxein.run(budget=auxein.Budget(evaluations=60), **arguments(genetic(), tmp_path / "r"))
    db = sqlite3.connect(tmp_path / "r" / "events.sqlite")
    assert db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 60 * 8
    assert db.execute("SELECT COUNT(DISTINCT candidate_id) FROM episodes").fetchone()[0] == 60
    assert db.execute("SELECT COUNT(*) FROM episodes WHERE status != 'ok'").fetchone()[0] == 0
    db.close()
    run = open_run(tmp_path / "r")
    try:
        episodes = run.episodes(7)
        assert [e.scenario_index for e in episodes] == list(range(8)) and [e.scenario_id for e in episodes] == list(SELECTION.ids)
        assert set(episodes[0].measurements) == {"final_distance", "steps", "effort", "success", "max_overshoot"}
        assert all(e.status is Status.OK and e.error is None for e in episodes)
        assert run.episodes(10_000) == []
        best = result.best
        assert best is not None and best.raw is not None and best.raw.key == f"episodes/{best.candidate.id}"
    finally:
        run.close()


def test_failed_episodes_are_recorded_too_and_the_others_keep_their_measurements(tmp_path: Path):
    class Sometimes(ep.Probe):
        def run_episode(self, agents, scenario, rng):
            if scenario.index == 1:
                raise RuntimeError("diverged")
            return super().run_episode(agents, scenario, rng)

    auxein.run(budget=auxein.Budget(evaluations=12), initial_failure_guard=None, **trouble_arguments(Sometimes(), tmp_path / "r"))
    run = open_run(tmp_path / "r")
    try:
        episodes = run.episodes(0)
        assert [e.status for e in episodes] == [Status.OK, Status.FAILED, Status.OK]
        assert episodes[1].measurements == {} and "diverged" in (episodes[1].error or "") and episodes[0].measurements["gene"] > 0
    finally:
        run.close()


def test_re_aggregating_a_recorded_run_matches_a_fresh_evaluation_with_that_aggregator(tmp_path: Path):
    """Random search ignores results, so the same seed proposes the same candidates whatever the aggregator."""
    other = Aggregator(
        objectives={"error": cvar_upper(lambda m: m["final_distance"] + 0.05 * m["steps"] / pm.STEPS, 0.25)},
        constraints={"overshoot": maximum(lambda m: pm.relu(m["max_overshoot"] - 0.2))},
        descriptors={"success_rate": mean("success")},
    )
    auxein.run(budget=auxein.Budget(evaluations=80), **arguments(auxein.RandomSearch(), tmp_path / "a"))
    auxein.run(
        budget=auxein.Budget(evaluations=80), **arguments(auxein.RandomSearch(), tmp_path / "b", evaluator=evaluator(aggregator=other))
    )
    with open_run(tmp_path / "a") as first, open_run(tmp_path / "b") as second:
        rejudged = first.reaggregate(other)
        fresh = list(second.evaluations())
    assert len(rejudged) == len(fresh) == 80
    for judged, evaluation in zip(rejudged, fresh, strict=True):
        assert judged.candidate_id == evaluation.candidate_id and judged.status is Status.OK
        assert same_values(judged.objectives, evaluation.objectives) and same_values(judged.constraints, evaluation.constraints)
        assert same_values(judged.descriptors, evaluation.descriptors)
    with open_run(tmp_path / "a") as original:  # and the original aggregator gives the original evaluations back
        for judged, evaluation in zip(original.reaggregate(pm.aggregator()), original.evaluations(), strict=True):
            assert same_values(judged.objectives, evaluation.objectives)


def test_re_aggregating_reports_failed_candidates_and_checks_the_names(tmp_path: Path):
    class Sometimes(ep.Probe):
        def run_episode(self, agents, scenario, rng):
            if scenario.index == 1 and float(agents["agent"][0]) < 5.0:
                raise RuntimeError("diverged")
            return super().run_episode(agents, scenario, rng)

    auxein.run(budget=auxein.Budget(evaluations=30), initial_failure_guard=None, **trouble_arguments(Sometimes(), tmp_path / "r"))
    with open_run(tmp_path / "r") as run:
        judged = run.reaggregate(Aggregator({"error": mean("gene")}))
        statuses = [j.status for j in judged]
        assert Status.FAILED in statuses and Status.OK in statuses and all(not j.objectives for j in judged if j.status is Status.FAILED)
        with pytest.raises(ValueError, match="reads the measurement 'nope'"):
            run.reaggregate(Aggregator({"error": mean("nope")}))


def test_recording_thousands_of_candidates_by_tens_of_scenarios_is_batched(tmp_path: Path):
    scenarios = ScenarioSet.generate(pm.scenario_params, 30, seed=1)
    started = time.perf_counter()
    auxein.run(
        budget=auxein.Budget(evaluations=2000),
        **arguments(auxein.RandomSearch(), tmp_path / "r", evaluator=evaluator(scenarios), batch_size=200),
    )
    elapsed = time.perf_counter() - started
    db = sqlite3.connect(tmp_path / "r" / "events.sqlite")
    assert db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 60_000
    db.close()
    assert elapsed < 30.0


# --- held-out evaluation ---


def test_held_out_evaluation_judges_the_best_candidate_on_scenarios_the_run_never_saw(tmp_path: Path):
    result = auxein.run(budget=auxein.Budget(evaluations=300), **arguments(genetic(), tmp_path / "r"))
    before = comparable(tmp_path / "r")
    report = evaluate_held_out(tmp_path / "r", evaluator(HELD_OUT), HELD_OUT)
    assert comparable(tmp_path / "r") == before  # nothing was added to the run's event log
    (candidate,) = report.candidates
    assert result.best is not None and candidate.candidate_id == result.best.candidate.id and candidate.status is Status.OK
    assert [s.scenario_id for s in candidate.scenarios] == list(HELD_OUT.ids) and not set(HELD_OUT.ids) & set(SELECTION.ids)
    assert (
        set(candidate.objectives) == {"error"}
        and set(candidate.constraints) == {"overshoot"}
        and set(candidate.descriptors) == {"success_rate"}
    )
    assert all(s.status is Status.OK and "final_distance" in s.measurements for s in candidate.scenarios)
    assert 0.0 < candidate.objectives["error"] < 1.0  # a sensible number, near what the selection set gave
    assert abs(candidate.objectives["error"] - result.best.objectives["error"]) < 0.2
    assert report.scenarios_fingerprint == HELD_OUT.fingerprint and "EpisodeEvaluator(" in report.evaluator


def test_the_report_is_written_into_the_run_directory_and_can_be_skipped(tmp_path: Path):
    auxein.run(budget=auxein.Budget(evaluations=100), **arguments(genetic(), tmp_path / "r"))
    report = evaluate_held_out(tmp_path / "r", evaluator(HELD_OUT), HELD_OUT)
    assert report.path == tmp_path / "r" / "held_out.json"
    written = json.loads(report.path.read_text())  # type: ignore[union-attr]
    assert written["scenarios_fingerprint"] == HELD_OUT.fingerprint and len(written["candidates"][0]["scenarios"]) == 8
    assert written["candidates"][0]["scenarios"][0]["measurements"]["final_distance"] >= 0.0
    assert evaluate_held_out(tmp_path / "r", evaluator(HELD_OUT), HELD_OUT, write=False).path is None


def test_held_out_candidates_can_be_the_front_a_result_or_chosen_ids(tmp_path: Path):
    result = auxein.run(budget=auxein.Budget(evaluations=100), **arguments(genetic(), tmp_path / "r"))
    by_id = evaluate_held_out(tmp_path / "r", evaluator(HELD_OUT), HELD_OUT, candidates=[3, 11, 50], write=False)
    assert [c.candidate_id for c in by_id.candidates] == [3, 11, 50]
    front = evaluate_held_out(tmp_path / "r", evaluator(HELD_OUT), HELD_OUT, candidates="pareto", write=False)
    assert [c.candidate_id for c in front.candidates] == [e.candidate.id for e in result.pareto_front]
    from_result = evaluate_held_out(result, evaluator(HELD_OUT), HELD_OUT, write=False)
    assert (
        from_result.candidates[0].objectives
        == evaluate_held_out(tmp_path / "r", evaluator(HELD_OUT), HELD_OUT, write=False).candidates[0].objectives
    )
    with pytest.raises(ValueError, match="did not record candidates \\[9999\\]"):
        evaluate_held_out(tmp_path / "r", evaluator(HELD_OUT), HELD_OUT, candidates=[9999])


def test_a_result_without_a_run_directory_needs_the_problem(tmp_path: Path):
    result = auxein.run(budget=auxein.Budget(evaluations=60), **arguments(genetic()))
    with pytest.raises(ValueError, match="pass problem="):
        evaluate_held_out(result, evaluator(HELD_OUT), HELD_OUT)
    report = evaluate_held_out(result, evaluator(HELD_OUT), HELD_OUT, problem=pm.problem())
    assert report.path is None and report.candidates[0].status is Status.OK


def test_held_out_evaluation_is_the_same_with_the_per_episode_path_and_works_inside_a_loop(tmp_path: Path):
    import asyncio

    auxein.run(budget=auxein.Budget(evaluations=100), **arguments(genetic(), tmp_path / "r"))
    batched = evaluate_held_out(tmp_path / "r", evaluator(HELD_OUT), HELD_OUT, write=False)
    single = evaluate_held_out(tmp_path / "r", evaluator(HELD_OUT, batched=False), HELD_OUT, concurrency=3, executor="thread", write=False)
    assert same_values(batched.candidates[0].objectives, single.candidates[0].objectives)

    async def inside() -> Any:
        return await aevaluate_held_out(tmp_path / "r", evaluator(HELD_OUT), HELD_OUT, write=False)

    assert asyncio.run(inside()).candidates[0].objectives == batched.candidates[0].objectives
    assert (
        evaluate_held_out(tmp_path / "r", evaluator(HELD_OUT), HELD_OUT, write=False).candidates[0].objectives
        == batched.candidates[0].objectives
    )


# --- resuming ---


def resumed_equals_uninterrupted(tmp_path: Path, delivery: str, batched: bool, short: int, long: int) -> None:
    kwargs: dict[str, Any] = {"evaluator": evaluator(batched=batched), "delivery": delivery}
    auxein.run(budget=auxein.Budget(evaluations=short), **arguments(genetic(), tmp_path / "a", **kwargs))
    extended = auxein.resume(budget=auxein.Budget(evaluations=long), **arguments(genetic(), tmp_path / "a", **kwargs))
    reference = auxein.run(budget=auxein.Budget(evaluations=long), **arguments(genetic(), tmp_path / "b", **kwargs))
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")  # events, candidates, evaluations and the episodes table
    assert (
        event_log(tmp_path / "a")["episodes"] == event_log(tmp_path / "b")["episodes"]
        and len(event_log(tmp_path / "a")["episodes"]) == long * 8
    )
    same_result(extended, reference)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_extending_a_run_keeps_its_recorded_episodes_without_duplicates(tmp_path: Path, delivery: str, batched: bool):
    if batched and delivery == "steady_state":
        pytest.skip("a batched evaluator in steady-state delivery is called one candidate at a time: the case is the per-batch one")
    resumed_equals_uninterrupted(tmp_path, delivery, batched, 130, 300)  # 130 is not a multiple of the batch: a final batch was cut


def test_a_changed_scenario_set_is_refused_on_resume(tmp_path: Path):
    auxein.run(budget=auxein.Budget(evaluations=60), **arguments(genetic(), tmp_path / "r"))
    changed = ScenarioSet.generate(pm.scenario_params, 8, seed=99)
    before = comparable(tmp_path / "r")
    with pytest.raises(ConfigurationMismatchError, match="evaluator: recorded") as raised:
        auxein.resume(budget=auxein.Budget(evaluations=120), **arguments(genetic(), tmp_path / "r", evaluator=evaluator(changed)))
    assert "scenarios=8:" in str(raised.value) and comparable(tmp_path / "r") == before
    other_aggregator = Aggregator(
        {"error": mean("final_distance")}, {"overshoot": maximum("max_overshoot")}, {"success_rate": mean("success")}
    )
    with pytest.raises(ConfigurationMismatchError, match="evaluator: recorded"):
        auxein.resume(
            budget=auxein.Budget(evaluations=120), **arguments(genetic(), tmp_path / "r", evaluator=evaluator(aggregator=other_aggregator))
        )


# --- killing an episode run and resuming it ---

KILLS = [
    ({"delivery": "generation", "executor": "inline", "concurrency": 1, "batched": False}, 60),
    ({"delivery": "generation", "executor": "inline", "concurrency": 1, "batched": True}, 90),
    ({"delivery": "steady_state", "executor": "thread", "concurrency": 4, "batched": False}, 70),
]


def recorded(run_dir: Path) -> int:
    try:
        db = sqlite3.connect(f"file:{run_dir / 'events.sqlite'}?mode=ro", uri=True)
        try:
            return int(db.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0])
        finally:
            db.close()
    except sqlite3.Error:
        return -1


@pytest.mark.parametrize(("config", "kill_after"), KILLS)
def test_a_killed_episode_run_resumes_to_the_same_log_and_episodes_without_duplicates(
    tmp_path: Path, config: dict[str, Any], kill_after: int
):
    total = 200
    run_dir = tmp_path / "run"
    full = {
        "strategy": "ga",
        "evaluator": "episode",
        "run_dir": str(run_dir),
        "evaluations": total,
        "scenarios": 4,
        "backend": backend_config(integration_backend()),
        **config,
    }
    victim = subprocess.Popen(
        [sys.executable, "-m", "tests.support.resume_cli", json.dumps(full)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 120
    try:
        while victim.poll() is None and recorded(run_dir) < kill_after and time.monotonic() < deadline:
            time.sleep(0.003)
        victim.send_signal(signal.SIGKILL)
        victim.wait(timeout=30)
    finally:
        if victim.poll() is None:
            victim.kill()
            victim.wait()
    done = recorded(run_dir)
    assert 0 <= done < total or victim.returncode is not None

    calls = tmp_path / "calls"
    env = {**os.environ, "AUXEIN_CALL_LOG": str(calls)}
    survivor = subprocess.run(
        [sys.executable, "-m", "tests.support.resume_cli", json.dumps({**full, "resume": True})],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert survivor.returncode == 0, survivor.stderr[-3000:]
    logged = len(calls.read_text().split()) if calls.exists() else 0
    if not config["batched"]:
        assert logged == (total - done) * 4  # the episodes of the candidates that were not recorded, and no others

    reference = tmp_path / "reference"
    auxein.run(budget=auxein.Budget(evaluations=total), **settings({**full, "run_dir": str(reference)}))
    assert comparable(run_dir) == comparable(reference)
    assert len(event_log(run_dir)["episodes"]) == total * 4  # one row per candidate and scenario: none lost, none doubled
    assert peek(run_dir).metadata["status"] == "completed"
