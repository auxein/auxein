"""Failure policies, the first-failure warning and the all-failures guard (design doc §6.6)."""

import json
import multiprocessing
import warnings
from pathlib import Path

import numpy as np
import pytest

from auxein.core import Evaluation, EvaluationBatch, Result, Status
from auxein.driver import (
    AllEvaluationsFailedError,
    Budget,
    EvaluationFailureWarning,
    RecordingDisabledWarning,
    run,
)
from auxein.evaluators import EvaluationError, FunctionEvaluator, VectorisedEvaluator
from auxein.execution import ExecutorError
from auxein.recording import open_run
from auxein.spaces import Box
from auxein.strategies import GeneticAlgorithm, RandomSearch
from tests.support import workers
from tests.support.eventlog import event_log
from tests.support.fakes import ScriptedStrategy

SPACE = Box(-5.0, 5.0, dim=3)


def go(strategy=None, evaluator=None, budget=None, batch_size=8, seed=1, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RecordingDisabledWarning)
        return run(
            strategy=strategy or RandomSearch(),
            evaluator=evaluator or FunctionEvaluator(workers.sphere),
            space=kwargs.pop("space", SPACE),
            budget=budget or Budget(evaluations=60),
            seed=seed,
            batch_size=batch_size,
            **kwargs,
        )


FLAKY = FunctionEvaluator(workers.flaky_sphere, uses_rng=True)


# --- the infeasible policy (the default) ---


def test_an_exception_becomes_a_failed_evaluation_with_its_type_message_and_traceback(tmp_path: Path):
    with pytest.warns(EvaluationFailureWarning):
        result = go(evaluator=FLAKY, run_dir=tmp_path / "r")
    assert result.evaluations_used == 60 and result.stop_reason == "budget:evaluations"  # the run completes
    counts = result.status_counts
    assert counts["failed"] > 5 and counts["ok"] > 5 and counts["ok"] + counts["failed"] == 60 and counts["timeout"] == 0
    with open_run(tmp_path / "r") as recorded:
        rows = list(recorded.evaluations())
    failed = [r for r in rows if r.status is Status.FAILED]
    assert len(failed) == counts["failed"] and len(rows) == 60
    error = failed[0].error
    assert error is not None and "ValueError: flaky evaluation" in error and "Traceback (most recent call last)" in error
    assert "workers.py" in error  # the traceback points at the user's code
    summary = json.loads((tmp_path / "r" / "metadata.json").read_text())["summary"]
    assert summary["status_counts"] == counts


def test_a_failed_candidate_never_becomes_the_best():
    with pytest.warns(EvaluationFailureWarning):
        result = go(evaluator=FLAKY)
    assert result.best is not None and result.best.status is Status.OK
    assert all(e.status is Status.OK for e in result.pareto_front)


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_fail_fast_stops_at_the_first_failure_and_chains_the_original_exception(delivery: str):
    with pytest.raises(EvaluationError, match="flaky evaluation") as raised:
        go(evaluator=FLAKY, failure_policy="fail_fast", delivery=delivery)
    assert isinstance(raised.value.__cause__, ValueError) and len(raised.value.candidate_ids) == 1


def test_on_the_vectorised_evaluator_an_exception_fails_every_candidate_of_the_batch(tmp_path: Path):
    calls = {"n": 0}

    def fn(X):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ArithmeticError("batch exploded")
        return (X * X).sum(axis=1)

    with pytest.warns(EvaluationFailureWarning, match="batch exploded"):
        result = go(evaluator=VectorisedEvaluator(fn), budget=Budget(evaluations=32), batch_size=8, run_dir=tmp_path / "r")
    assert result.status_counts == {"ok": 24, "failed": 8, "timeout": 0}
    with open_run(tmp_path / "r") as recorded:
        errors = {r.error for r in recorded.evaluations() if r.status is Status.FAILED}
    assert len(errors) == 1  # the same error for each of the eight
    calls["n"] = 1  # a new run: its first call is the second call of `fn`, which raises
    with pytest.raises(EvaluationError, match=r"candidates 0, 1, 2, 3, 4, \.\.\. \(8 candidates\)"):
        go(evaluator=VectorisedEvaluator(fn), budget=Budget(evaluations=32), batch_size=8, failure_policy="fail_fast")


class FailingEvaluator:
    """A custom evaluator that reports failures itself: the driver is the backstop for them."""

    def __init__(self, status: Status = Status.FAILED, every: int = 3) -> None:
        self.status, self.every = status, every

    async def evaluate(self, batch, ctx):
        out = []
        for c in batch.candidates:
            if c.id % self.every == 0:
                out.append(Evaluation.failed(c, self.status, f"custom failure of {c.id}", 0.5))
            else:
                out.append(Evaluation(c, Status.OK, {"value": float(c.id)}))
        return EvaluationBatch(out)


def test_the_driver_is_the_backstop_for_custom_evaluators():
    with pytest.warns(EvaluationFailureWarning, match="custom failure of 0"):
        result = go(evaluator=FailingEvaluator(), budget=Budget(evaluations=30))
    assert result.status_counts == {"ok": 20, "failed": 10, "timeout": 0}
    with pytest.warns(EvaluationFailureWarning):
        result = go(evaluator=FailingEvaluator(Status.TIMEOUT), budget=Budget(evaluations=30))
    assert result.status_counts == {"ok": 20, "failed": 0, "timeout": 10}
    with pytest.raises(EvaluationError, match="candidate 0 failed: failed: custom failure of 0"):
        go(evaluator=FailingEvaluator(), failure_policy="fail_fast")
    with pytest.raises(EvaluationError, match="timeout: custom failure of 0"):
        go(evaluator=FailingEvaluator(Status.TIMEOUT), failure_policy="fail_fast", delivery="steady_state")


def test_a_non_finite_objective_is_a_failure_with_a_message():
    def diverges(genome, rng):
        return float("nan") if _first(rng) < 0.3 else 1.0

    with pytest.warns(EvaluationFailureWarning, match="non-finite objective value: 'value' is nan"):
        result = go(evaluator=FunctionEvaluator(diverges, uses_rng=True))
    assert result.status_counts["failed"] > 5
    with pytest.raises(EvaluationError, match="non-finite objective value"):
        go(evaluator=FunctionEvaluator(diverges, uses_rng=True), failure_policy="fail_fast")


def _first(rng):
    return float(rng.uniform(1)[0])


# --- misconfiguration always fails the run ---


def test_a_plain_dict_fails_the_run_under_the_infeasible_policy():
    with pytest.raises(TypeError, match="returned a dict"):
        go(evaluator=FunctionEvaluator(lambda g: {"value": 1.0}))


def test_a_result_with_a_missing_or_unknown_name_fails_the_run():
    with pytest.raises(ValueError, match="missing objectives 'value'"):
        go(evaluator=FunctionEvaluator(lambda g: Result({"loss": 1.0})))
    with pytest.raises(ValueError, match="unknown objectives 'extra'"):
        go(evaluator=FunctionEvaluator(lambda g: Result({"value": 1.0, "extra": 2.0})))
    with pytest.raises(TypeError, match="must return a number or a Result"):
        go(evaluator=FunctionEvaluator(lambda g: "three"))


def test_a_wrong_vectorised_return_fails_the_run():
    with pytest.raises(ValueError, match="values for a batch"):
        go(evaluator=VectorisedEvaluator(lambda X: np.zeros(3)))
    with pytest.raises(TypeError, match="returned a dict"):
        go(evaluator=VectorisedEvaluator(lambda X: {"value": np.zeros(len(X))}))


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_an_unpicklable_function_with_a_process_executor_fails_the_run(delivery: str):
    with pytest.raises(EvaluationError, match="executor='thread'") as raised:
        go(evaluator=FunctionEvaluator(lambda g: 1.0), executor="process", concurrency=2, delivery=delivery)
    assert isinstance(raised.value.__cause__, ExecutorError)
    assert multiprocessing.active_children() == []


# --- the safety net ---


def test_the_first_failure_is_warned_about_once_with_the_traceback():
    with pytest.warns(EvaluationFailureWarning) as caught:
        result = go(evaluator=FLAKY)
    ours = [w for w in caught if issubclass(w.category, EvaluationFailureWarning)]
    assert result.status_counts["failed"] > 1 and len(ours) == 1
    text = str(ours[0].message)
    assert "first failure of this run" in text and "Traceback (most recent call last)" in text and "ValueError: flaky evaluation" in text
    assert "candidate " in text and "'failed'" in text


def test_no_warning_when_nothing_fails():
    with warnings.catch_warnings():
        warnings.simplefilter("error", EvaluationFailureWarning)
        go()


def test_the_guard_stops_a_run_whose_first_ten_evaluations_all_fail(tmp_path: Path):
    with pytest.warns(EvaluationFailureWarning), pytest.raises(AllEvaluationsFailedError) as raised:
        go(evaluator=FunctionEvaluator(workers.always_fails), run_dir=tmp_path / "r", budget=Budget(evaluations=500))
    message = str(raised.value)
    assert "first 10 evaluation(s) all failed" in message and "this evaluation function is broken" in message
    assert "bug" in message and "initial_failure_guard=None" in message and "Traceback" in message
    meta = json.loads((tmp_path / "r" / "metadata.json").read_text())
    assert meta["status"] == "failed" and meta["summary"]["evaluations_used"] == 16  # the batch that was running is kept


def test_the_guard_does_not_stop_a_run_where_one_of_the_first_ten_succeeds():
    calls = {"n": 0}

    def fn(genome):
        calls["n"] += 1
        if calls["n"] != 7:
            raise RuntimeError("failing")
        return 1.0

    with pytest.warns(EvaluationFailureWarning):
        result = go(evaluator=FunctionEvaluator(fn), budget=Budget(evaluations=100))
    assert result.status_counts == {"ok": 1, "failed": 99, "timeout": 0}  # and later failures never trigger it


def test_the_guard_can_be_disabled_or_changed():
    with pytest.warns(EvaluationFailureWarning):
        result = go(evaluator=FunctionEvaluator(workers.always_fails), initial_failure_guard=None, budget=Budget(evaluations=40))
    assert result.status_counts["failed"] == 40 and result.best is None
    with pytest.warns(EvaluationFailureWarning), pytest.raises(AllEvaluationsFailedError, match="first 3 evaluation"):
        go(evaluator=FunctionEvaluator(workers.always_fails), initial_failure_guard=3, batch_size=4)


def test_a_run_too_short_for_the_guard_in_which_everything_failed_is_stopped_at_the_end():
    with pytest.warns(EvaluationFailureWarning), pytest.raises(AllEvaluationsFailedError, match="first 4 evaluation"):
        go(evaluator=FunctionEvaluator(workers.always_fails), budget=Budget(evaluations=4), batch_size=4)


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_the_guard_outcome_does_not_depend_on_concurrency_or_the_executor(delivery: str):
    def fails_at_first(genome, rng):
        raise RuntimeError("nope")

    outcomes = set()
    for concurrency, executor in [(1, "inline"), (4, "thread"), (3, "auto")]:
        with warnings.catch_warnings(), pytest.raises(AllEvaluationsFailedError) as raised:
            warnings.simplefilter("ignore", EvaluationFailureWarning)
            go(evaluator=FunctionEvaluator(fails_at_first, uses_rng=True), delivery=delivery, concurrency=concurrency, executor=executor)
        outcomes.add(str(raised.value).split("\n")[0])
    assert len(outcomes) == 1


def test_the_guard_counts_in_ask_order_when_the_first_candidates_fail_and_later_ones_succeed():
    # candidates 0..9 fail, 10 succeeds: the guard stops the run even though a later evaluation would have succeeded
    def fn(genome):
        if genome < 10:
            raise RuntimeError("early failure")
        return 1.0

    with pytest.warns(EvaluationFailureWarning), pytest.raises(AllEvaluationsFailedError):
        go(strategy=ScriptedStrategy(tell_mode="both"), evaluator=FunctionEvaluator(fn), space=SPACE, batch_size=16)


# --- determinism with failures ---


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_the_event_log_with_failures_is_identical_across_concurrency_and_executors(tmp_path: Path, delivery: str):
    sync, async_ = FLAKY, FunctionEvaluator(workers.async_flaky_sphere, uses_rng=True)
    configurations = [(sync, 1, "inline"), (sync, 4, "thread"), (async_, 1, "auto"), (async_, 8, "auto"), (sync, 2, "process")]
    logs = []
    for i, (evaluator, concurrency, executor) in enumerate(configurations):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EvaluationFailureWarning)
            go(
                GeneticAlgorithm(population_size=6, offspring_size=4), evaluator, Budget(evaluations=50), seed=4, delivery=delivery,
                concurrency=concurrency, executor=executor, run_dir=tmp_path / str(i),
            )  # fmt: skip
        logs.append(event_log(tmp_path / str(i)))
    assert all(log == logs[0] for log in logs)
    statuses = [row[1] for row in logs[0]["evaluations"]]
    assert "failed" in statuses and "ok" in statuses  # and the failures really happened
    assert multiprocessing.active_children() == []
