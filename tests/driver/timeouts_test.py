"""Timeouts, process isolation and crashes (design doc §5.3 and §6.6). Durations are milliseconds; assertions are on outcomes."""

import asyncio
import json
import multiprocessing
import subprocess
import sys
import textwrap
import threading
import time
import warnings
from functools import partial
from pathlib import Path

import pytest

from auxein.core import Status
from auxein.driver import Budget, EvaluationFailureWarning, RecordingDisabledWarning, run
from auxein.evaluators import EvaluationError, FunctionEvaluator, VectorisedEvaluator
from auxein.execution import AbandonedEvaluationWarning, EvaluationTimeout
from auxein.recording import open_run
from auxein.spaces import Box
from auxein.strategies import RandomSearch
from tests.support import workers
from tests.support.fixtures import integration_backend

pytestmark = pytest.mark.usefixtures("use_corner_backend")


SPACE = Box(-5.0, 5.0, dim=3)


def go(evaluator, budget=None, batch_size=8, seed=1, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RecordingDisabledWarning)
        warnings.simplefilter("ignore", EvaluationFailureWarning)
        kwargs.setdefault("backend", integration_backend())
        return run(
            strategy=kwargs.pop("strategy", RandomSearch()), evaluator=evaluator, space=SPACE,
            budget=budget or Budget(evaluations=40), seed=seed, batch_size=batch_size, **kwargs,
        )  # fmt: skip


def no_stray_workers() -> bool:
    return multiprocessing.active_children() == []


# --- an async def is cancelled ---


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_an_async_function_that_times_out_is_cancelled_and_recorded_as_a_timeout(tmp_path: Path, delivery: str):
    cancelled: list[int] = []

    async def fn(genome, rng):
        if float(rng.uniform(1)[0]) < 0.25:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(1)
                raise
        return float((genome * genome).sum())

    t0 = time.perf_counter()
    result = go(FunctionEvaluator(fn, uses_rng=True), timeout=0.05, concurrency=4, delivery=delivery, run_dir=tmp_path / "r")
    assert time.perf_counter() - t0 < 10
    counts = result.status_counts
    assert counts["timeout"] > 3 and counts["ok"] > 3 and sum(counts.values()) == 40 == result.evaluations_used  # they count
    assert len(cancelled) == counts["timeout"]  # really cancelled, not abandoned
    with open_run(tmp_path / "r") as recorded:
        timed_out = [r for r in recorded.evaluations() if r.status is Status.TIMEOUT]
    assert len(timed_out) == counts["timeout"]
    assert all(r.wall_time == pytest.approx(0.05) and "timed out" in (r.error or "") for r in timed_out)  # its wall time is the limit


def test_a_timeout_error_of_the_function_itself_is_an_ordinary_failure():
    async def fn(genome):
        raise TimeoutError("the server timed out")

    result = go(FunctionEvaluator(fn), timeout=5.0, initial_failure_guard=None, budget=Budget(evaluations=8))
    assert result.status_counts == {"ok": 0, "failed": 8, "timeout": 0}


def test_fail_fast_turns_a_timeout_into_an_error_naming_the_candidate():
    async def fn(genome):
        await asyncio.sleep(30)

    with pytest.raises(EvaluationError, match="candidate 0 failed: EvaluationTimeout") as raised:
        go(FunctionEvaluator(fn), timeout=0.02, failure_policy="fail_fast", concurrency=1)
    assert isinstance(raised.value.__cause__, EvaluationTimeout)


# --- a worker process is killed and replaced ---


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_a_process_that_times_out_is_killed_and_replaced_and_the_others_are_unaffected(tmp_path: Path, delivery: str):
    evaluator = FunctionEvaluator(partial(workers.slow_if_unlucky, 60.0), uses_rng=True)
    if integration_backend().name != "numpy":
        # killing and replacing a worker does not depend on the backend, and on torch every spawned worker first imports it to
        # unpickle a tensor genome (about a second, counted towards the timeout), which would need a timeout that makes this
        # test last half a minute; the torch corner of the process executor is covered by the kill-and-resume and episode tests
        pytest.skip("worker kill and replacement is independent of the backend")
    timeout = 0.5
    t0 = time.perf_counter()
    result = go(
        evaluator,
        timeout=timeout,
        concurrency=3,
        executor="process",
        delivery=delivery,
        run_dir=tmp_path / "r",
        budget=Budget(evaluations=30),
    )
    assert time.perf_counter() - t0 < 40  # nothing waited for the 60 s
    counts = result.status_counts
    assert counts["timeout"] > 2 and counts["ok"] > 10 and sum(counts.values()) == 30
    with open_run(tmp_path / "r") as recorded:
        rows = list(recorded.evaluations())
    for row in rows:
        if row.status is Status.OK:  # every evaluation that did finish has the right value, whatever happened to its neighbours
            assert row.objectives["value"] == pytest.approx(float((row.genome * row.genome).sum()))
    assert no_stray_workers()


def test_fail_fast_with_a_process_timeout_stops_the_run_and_leaves_no_workers():
    evaluator = FunctionEvaluator(partial(workers.slow_if_unlucky, 60.0), uses_rng=True)
    with pytest.raises(EvaluationError, match="EvaluationTimeout"):
        go(evaluator, timeout=0.5, concurrency=2, executor="process", failure_policy="fail_fast")
    assert no_stray_workers()


# --- a thread is abandoned ---


def test_a_thread_that_times_out_is_abandoned_its_late_result_is_discarded_and_the_slot_is_freed(tmp_path: Path):
    release = threading.Event()
    started: list[int] = []

    def fn(genome, rng):
        started.append(1)
        if float(rng.uniform(1)[0]) < 0.25:
            release.wait(30)  # stuck until the test lets it go
            return -1.0  # a late result that must not be used
        return float((genome * genome).sum())

    try:
        with pytest.warns(AbandonedEvaluationWarning) as caught:
            result = go(FunctionEvaluator(fn, uses_rng=True), timeout=0.05, concurrency=1, executor="thread", run_dir=tmp_path / "r")
        warned = [w for w in caught if issubclass(w.category, AbandonedEvaluationWarning)]
        assert len(warned) == 2  # the explanation, once, and the count of the threads still running at the end
        assert "cannot be stopped" in str(warned[0].message) and "still running" in str(warned[1].message)
        counts = result.status_counts
        assert counts["timeout"] > 3 and sum(counts.values()) == 40 and len(started) == 40  # concurrency 1 did not block on them
        summary = json.loads((tmp_path / "r" / "metadata.json").read_text())["summary"]
        assert summary["abandoned_evaluations"] == counts["timeout"] and summary["status_counts"] == counts
    finally:
        release.set()  # the abandoned threads now finish: nothing happens to the (finished) run
    time.sleep(0.1)
    assert result.best is not None and result.best.objectives["value"] >= 0


def test_abandoned_threads_do_not_stop_the_interpreter_from_exiting():
    script = textwrap.dedent(
        """
        import time
        import warnings

        import auxein


        def stuck(genome):
            time.sleep(600)  # an evaluation that never returns
            return 0.0


        if __name__ == "__main__":
            warnings.simplefilter("ignore")
            result = auxein.run(
                strategy=auxein.RandomSearch(), evaluator=auxein.FunctionEvaluator(stuck), space=auxein.Box(-1, 1, dim=2),
                budget=auxein.Budget(evaluations=12), seed=1, batch_size=4, timeout=0.05, concurrency=2, initial_failure_guard=None,
            )
            print("done", result.status_counts["timeout"])
        """
    )
    start = time.perf_counter()
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "done 12" in done.stdout and time.perf_counter() - start < 100  # it exited long before the 600 s


# --- how executors resolve with a timeout ---


def test_auto_with_a_timeout_resolves_to_threads_even_for_one_worker(tmp_path: Path):
    go(FunctionEvaluator(workers.sphere), timeout=5.0, concurrency=1, run_dir=tmp_path / "a", budget=Budget(evaluations=8))
    meta = json.loads((tmp_path / "a" / "metadata.json").read_text())
    assert meta["executor"] == "thread" and meta["timeout"] == 5.0
    go(FunctionEvaluator(workers.sphere), concurrency=1, run_dir=tmp_path / "b", budget=Budget(evaluations=8))
    assert json.loads((tmp_path / "b" / "metadata.json").read_text())["executor"] == "inline"


def test_an_inline_executor_with_a_timeout_is_refused_at_start_up():
    with pytest.raises(ValueError, match="executor='inline' cannot be combined with a timeout.*'thread'.*'process'.*async def"):
        go(FunctionEvaluator(workers.sphere), timeout=1.0, executor="inline")


def test_a_vectorised_evaluator_with_a_timeout_is_refused_at_start_up():
    with pytest.raises(ValueError, match="cannot be combined with a VectorisedEvaluator"):
        go(VectorisedEvaluator(lambda X: (X * X).sum(axis=1)), timeout=1.0)


@pytest.mark.parametrize("bad", [0, -1.0, float("nan")])
def test_a_timeout_must_be_positive(bad: float):
    with pytest.raises(ValueError, match="timeout must be a positive number"):
        go(FunctionEvaluator(workers.sphere), timeout=bad)


# --- a worker that dies ---


@pytest.mark.parametrize(("function", "expected"), [(workers.exit_if_unlucky, "exited with code 7"), (workers.kill_if_unlucky, "SIGKILL")])
@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_a_worker_that_dies_fails_its_candidate_and_is_replaced(tmp_path: Path, function, expected: str, delivery: str):
    result = go(
        FunctionEvaluator(function, uses_rng=True), concurrency=3, executor="process", delivery=delivery, run_dir=tmp_path / "r",
        budget=Budget(evaluations=30),
    )  # fmt: skip
    counts = result.status_counts
    assert counts["failed"] > 2 and counts["ok"] > 10 and sum(counts.values()) == 30  # the run went on after the crashes
    with open_run(tmp_path / "r") as recorded:
        rows = list(recorded.evaluations())
    failed = [r for r in rows if r.status is Status.FAILED]
    assert all("worker process died" in (r.error or "") and expected in (r.error or "") for r in failed)
    assert len([r for r in rows if r.status is Status.OK]) == counts["ok"]
    assert no_stray_workers()


def test_fail_fast_with_a_crash_stops_the_run_naming_the_candidate():
    with pytest.raises(EvaluationError, match="worker process died"):
        go(FunctionEvaluator(workers.exit_if_unlucky, uses_rng=True), executor="process", concurrency=2, failure_policy="fail_fast")
    assert no_stray_workers()


def test_a_script_that_cannot_start_workers_is_a_misconfiguration(monkeypatch):
    from auxein.execution import processes

    monkeypatch.setattr(processes, "worker_main", _failing_worker_main)
    with pytest.raises(EvaluationError, match="failed to start") as raised:
        go(FunctionEvaluator(workers.sphere), executor="process", concurrency=1)
    assert "if __name__" in str(raised.value.__cause__)  # whatever the policy
    assert no_stray_workers()


def _failing_worker_main(connection):
    raise SystemExit(3)


# --- nothing is left behind ---


def _non_daemon_threads() -> set[threading.Thread]:
    return {t for t in threading.enumerate() if not t.daemon and t is not threading.main_thread()}


@pytest.mark.parametrize("how", ["normally", "error", "interrupt", "guard"])
def test_no_worker_processes_and_no_non_daemon_threads_remain(how: str):
    before = _non_daemon_threads()
    evaluator = {
        "normally": FunctionEvaluator(workers.sphere),
        "error": FunctionEvaluator(workers.flaky_sphere, uses_rng=True),
        "interrupt": FunctionEvaluator(workers.raise_keyboard_interrupt),
        "guard": FunctionEvaluator(workers.always_fails),
    }[how]
    kwargs = {"error": {"failure_policy": "fail_fast"}}.get(how, {})
    if how == "normally":
        go(evaluator, executor="process", concurrency=2, budget=Budget(evaluations=12), **kwargs)
    elif how == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            go(evaluator, executor="process", concurrency=2, **kwargs)
    else:
        with pytest.raises(Exception, match="flaky|first 10"):
            go(evaluator, executor="process", concurrency=2, **kwargs)
    assert no_stray_workers() and _non_daemon_threads() == before


def test_cancelling_a_run_while_workers_are_busy_kills_them():
    from auxein.driver import arun

    async def main() -> None:
        task = asyncio.ensure_future(
            arun(
                strategy=RandomSearch(),
                evaluator=FunctionEvaluator(workers.pause),
                space=SPACE,
                budget=Budget(evaluations=50),
                seed=1,
                batch_size=4,
                executor="process",
                concurrency=3,
            )  # fmt: skip
        )
        await asyncio.sleep(1.5)  # the workers are up and sleeping
        assert multiprocessing.active_children()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        asyncio.run(main())
    assert no_stray_workers()
