"""Resuming in throughput mode, after failures, across budgets, and with two processes (design doc §10.4)."""

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
from auxein.core import Batch, EvalContext, Evaluation, EvaluationBatch, Result, Status
from auxein.driver import AllEvaluationsFailedError, EvaluationFailureWarning, ResumeError, ResumeWarning
from auxein.evaluators import EvaluationError
from tests.driver.resume_test import arguments, comparable, pretend_killed, same_result, sphere
from tests.support import workers
from tests.support.eventlog import event_log
from tests.support.reading import peek

ROOT = Path(__file__).resolve().parents[2]

pytestmark = [pytest.mark.filterwarnings("ignore::auxein.driver.errors.EvaluationFailureWarning"), pytest.mark.usefixtures("use_corner_backend")]


def forget_checkpoints_after(run_dir: Path, keep: int) -> int:
    """Make the `keep`-th checkpoint the latest, as if the process had been killed before it wrote the others. Returns its sequence."""
    db = sqlite3.connect(run_dir / "events.sqlite")
    rows = db.execute("SELECT id, event_seq FROM checkpoints ORDER BY id").fetchall()
    for identifier, _ in rows[keep:]:
        db.execute("DELETE FROM checkpoints WHERE id = ?", (identifier,))
    db.commit()
    db.close()
    return int(rows[keep - 1][1])


def rows_after(run_dir: Path, seq: int) -> dict[int, bytes]:
    db = sqlite3.connect(run_dir / "events.sqlite")
    try:
        return {int(i): bytes(g) for i, g in db.execute("SELECT id, genome FROM candidates WHERE event_seq > ?", (seq,))}
    finally:
        db.close()


def consistent(run_dir: Path, evaluations: int) -> None:
    """No duplicate or orphan rows, lineage that points at recorded candidates, and the budget respected."""
    db = sqlite3.connect(run_dir / "events.sqlite")
    try:
        candidates = {i for (i,) in db.execute("SELECT id FROM candidates")}
        assert len(candidates) == db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == evaluations
        assert {i for (i,) in db.execute("SELECT candidate_id FROM evaluations")} == candidates
        assert db.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == evaluations
        for parent, child in db.execute("SELECT parent_id, child_id FROM lineage"):
            assert child in candidates and parent in candidates
        kinds = [k for (k,) in db.execute("SELECT kind FROM events ORDER BY seq")]
        assert kinds.count("ask") == kinds.count("tell") == (evaluations if "steady" in str(run_dir) else kinds.count("ask"))
        assert kinds.count("stop") == 1 and kinds[-1] == "stop"
        sequences = [s for (s,) in db.execute("SELECT event_seq FROM candidates")]
        asks = {s for (s,) in db.execute("SELECT seq FROM events WHERE kind = 'ask'")}
        assert set(sequences) <= asks  # every candidate belongs to an ask event that still exists
    finally:
        db.close()


# --- throughput mode: truncate and redo ---


@pytest.mark.parametrize("delivery", ["steady_state", "generation"])
def test_throughput_resume_deletes_what_was_recorded_after_the_checkpoint_and_redoes_it(tmp_path: Path, delivery: str):
    run_dir = tmp_path / f"{delivery}-run"
    extra = {"deterministic": False, "concurrency": 4, "executor": "thread"}
    evaluator = auxein.FunctionEvaluator(workers.jittery_sphere, uses_rng=True)
    first = auxein.run(
        budget=auxein.Budget(evaluations=300),
        checkpoint_every_evaluations=40,
        keep_checkpoints=20,
        **arguments("random", delivery, run_dir, evaluator, **extra),
    )
    assert first.evaluations_used == 300
    before_genomes = rows_after(run_dir, 0)
    seq = forget_checkpoints_after(run_dir, 3)  # killed after its third checkpoint
    pretend_killed(run_dir)
    db = sqlite3.connect(run_dir / "events.sqlite")
    db.execute("DELETE FROM events WHERE kind = 'stop'")
    db.commit()
    db.close()
    doomed = rows_after(run_dir, seq)
    assert 0 < len(doomed) < 300

    resumed = auxein.resume(budget=auxein.Budget(evaluations=300), **arguments("random", delivery, run_dir, evaluator, **extra))

    assert resumed.evaluations_used == 300 and resumed.stop_reason == "budget:evaluations"
    consistent(run_dir, 300)
    db = sqlite3.connect(run_dir / "events.sqlite")
    (payload,) = [json.loads(p) for (p,) in db.execute("SELECT payload FROM events WHERE kind = 'resume'")]
    db.close()
    assert payload["mode"] == "truncate" and payload["checkpoint"] == seq and payload["truncated"] == len(doomed)
    assert resumed.evaluations_used == 300
    # a random search proposes the same candidate for the same id whenever it is asked, so the redone work has the same genomes
    after = rows_after(run_dir, 0)
    assert set(after) == set(before_genomes) and all(after[i] == before_genomes[i] for i in after)
    if delivery == "generation":  # nothing to reorder: the redone run is the run
        reference = auxein.run(budget=auxein.Budget(evaluations=300), **arguments("random", delivery, tmp_path / "ref", evaluator, **extra))
        assert comparable(run_dir) == comparable(tmp_path / "ref")
        same_result(resumed, reference)


def test_throughput_resume_without_a_checkpoint_starts_again_after_deleting_everything(tmp_path: Path):
    run_dir = tmp_path / "steady-run"
    extra = {"deterministic": False, "concurrency": 3, "executor": "thread"}
    evaluator = auxein.FunctionEvaluator(workers.jittery_sphere, uses_rng=True)
    auxein.run(budget=auxein.Budget(evaluations=120), keep_checkpoints=0, **arguments("ga", "steady_state", run_dir, evaluator, **extra))
    pretend_killed(run_dir)
    resumed = auxein.resume(
        budget=auxein.Budget(evaluations=150), keep_checkpoints=0, **arguments("ga", "steady_state", run_dir, evaluator, **extra)
    )
    assert resumed.evaluations_used == 150
    consistent(run_dir, 150)
    db = sqlite3.connect(run_dir / "events.sqlite")
    (payload,) = [json.loads(p) for (p,) in db.execute("SELECT payload FROM events WHERE kind = 'resume'")]
    db.close()
    assert payload["mode"] == "truncate" and payload["checkpoint"] is None and payload["truncated"] == 120


def test_throughput_resume_reissues_the_candidates_that_were_in_flight_with_their_ids_and_genomes(tmp_path: Path):
    """An interrupt checkpoint holds the window: those candidates were asked but not told, and come back as they were."""
    run_dir = tmp_path / "steady-run"
    extra = {"deterministic": False, "concurrency": 2, "executor": "thread"}

    class Interrupt:
        def __init__(self, at: int | None) -> None:
            self.at, self.seen = at, 0

        def __repr__(self) -> str:
            return "Interrupt()"

        async def evaluate(self, batch: Batch[Any], ctx: EvalContext[Any]) -> EvaluationBatch[Any]:
            self.seen += 1
            if self.at is not None and self.seen == self.at:
                raise KeyboardInterrupt
            return EvaluationBatch(
                [Evaluation(c, Status.OK, {"value": float(np.sum(np.asarray(c.genome) ** 2))}) for c in batch.candidates]
            )

    with pytest.raises(KeyboardInterrupt):
        auxein.run(
            budget=auxein.Budget(evaluations=100),
            checkpoint_every=10_000,
            **arguments("random", "steady_state", run_dir, Interrupt(40), **extra),
        )
    (checkpoint,) = peek(run_dir).checkpoints()
    state_file = json.loads((run_dir / checkpoint.path / "state.json").read_text())
    window = state_file["state"]["driver"]["window"]
    assert len(window["ids"]) == 16  # batch_size candidates were asked and not told
    resumed = auxein.resume(budget=auxein.Budget(evaluations=100), **arguments("random", "steady_state", run_dir, Interrupt(None), **extra))
    assert resumed.evaluations_used == 100
    consistent(run_dir, 100)
    db = sqlite3.connect(run_dir / "events.sqlite")
    told = {i for (i,) in db.execute("SELECT id FROM candidates")}
    db.close()
    assert set(window["ids"]) <= told  # every candidate that was in flight at the checkpoint was evaluated after it


# --- failures ---


class Scripted:
    """An evaluator whose failures and interrupt are decided by candidate id, so that they repeat in a resumed run."""

    def __init__(self, fail_below: int = 0, interrupt_at: int | None = None, fail_at: int | None = None) -> None:
        self.fail_below, self.interrupt_at, self.fail_at = fail_below, interrupt_at, fail_at

    def __repr__(self) -> str:
        return "Scripted()"

    async def evaluate(self, batch: Batch[Any], ctx: EvalContext[Any]) -> EvaluationBatch[Any]:
        results: list[Evaluation[Any]] = []
        for candidate in batch.candidates:
            if candidate.id == self.interrupt_at:
                raise KeyboardInterrupt
            if candidate.id < self.fail_below or candidate.id == self.fail_at:
                results.append(Evaluation.failed(candidate, Status.FAILED, f"scripted failure of candidate {candidate.id}"))
            else:
                results.append(Evaluation(candidate, Status.OK, {"value": float(np.sum(np.asarray(candidate.genome) ** 2))}))
        return EvaluationBatch(results)


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_a_fail_fast_run_resumed_after_the_evaluator_is_fixed_evaluates_the_failed_candidate_again(tmp_path: Path, delivery: str):
    run_dir = tmp_path / "r"
    with pytest.raises(EvaluationError):
        auxein.run(
            budget=auxein.Budget(evaluations=120), failure_policy="fail_fast", **arguments("ga", delivery, run_dir, Scripted(fail_at=50))
        )
    recorded = {e.candidate_id for e in peek(run_dir).evaluations()}
    assert 50 not in recorded and len(recorded) < 50 + 16  # the failed evaluation was never recorded
    assert peek(run_dir).metadata["status"] == "failed"

    fixed = auxein.resume(
        budget=auxein.Budget(evaluations=120), failure_policy="fail_fast", **arguments("ga", delivery, run_dir, Scripted())
    )
    assert fixed.evaluations_used == 120 and dict(fixed.status_counts) == {"ok": 120, "failed": 0, "timeout": 0}
    reference = auxein.run(
        budget=auxein.Budget(evaluations=120), failure_policy="fail_fast", **arguments("ga", delivery, tmp_path / "ref", Scripted())
    )
    assert comparable(run_dir) == comparable(tmp_path / "ref")
    same_result(fixed, reference)
    assert [s["status"] for s in peek(run_dir).sessions] == ["failed", "completed"]


def test_the_failure_guard_and_its_count_survive_a_resume(tmp_path: Path):
    """Seven failures were seen when the run was interrupted; the three more that follow must still trip a guard of ten."""
    run_dir = tmp_path / "r"
    with pytest.raises(KeyboardInterrupt):
        auxein.run(
            budget=auxein.Budget(evaluations=100),
            initial_failure_guard=10,
            checkpoint_every=10_000,
            **arguments("random", "steady_state", run_dir, Scripted(fail_below=12, interrupt_at=7)),
        )
    (checkpoint,) = peek(run_dir).checkpoints()
    driver = json.loads((run_dir / checkpoint.path / "state.json").read_text())["state"]["driver"]
    assert driver["guard_told"] == 7 and driver["guard_open"] is True and driver["failures"]["failed"] == 7
    assert driver["first_failure"]["candidate_id"] == 0 and "scripted failure" in driver["first_failure"]["error"]
    with pytest.raises(AllEvaluationsFailedError, match="first 10 evaluation"):
        auxein.resume(
            budget=auxein.Budget(evaluations=100),
            initial_failure_guard=10,
            **arguments("random", "steady_state", run_dir, Scripted(fail_below=12)),
        )
    assert peek(run_dir).metadata["status"] == "failed"


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_failure_counts_and_the_failed_evaluations_are_the_same_after_a_resume(tmp_path: Path, delivery: str):
    evaluator = auxein.FunctionEvaluator(workers.flaky_sphere, uses_rng=True)
    with pytest.warns(EvaluationFailureWarning):
        auxein.run(budget=auxein.Budget(evaluations=80), **arguments("ga", delivery, tmp_path / "a", evaluator))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the first failure was reported by the first session: a replayed one is not new
        resumed = auxein.resume(budget=auxein.Budget(evaluations=250), **arguments("ga", delivery, tmp_path / "a", evaluator))
    with pytest.warns(EvaluationFailureWarning):
        reference = auxein.run(budget=auxein.Budget(evaluations=250), **arguments("ga", delivery, tmp_path / "b", evaluator))
    assert resumed.status_counts["failed"] > 20
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")
    same_result(resumed, reference)


# --- budgets across sessions ---


def test_wall_time_is_the_cumulative_active_time_of_all_sessions(tmp_path: Path):
    now = [0.0]

    def clock() -> float:
        return now[0]

    def tick(genome: np.ndarray) -> float:
        now[0] += 1.0  # every evaluation takes a second
        return sphere(genome)

    evaluator = auxein.FunctionEvaluator(tick)
    first = auxein.run(
        budget=auxein.Budget(wall_time=50.0), clock=clock, **arguments("random", "generation", tmp_path / "r", evaluator, batch_size=10)
    )
    assert first.stop_reason == "budget:wall_time" and first.evaluations_used == 50 and first.wall_time == 50.0
    now[0] += 10_000.0  # a day passes between the sessions: it does not count
    second = auxein.resume(
        budget=auxein.Budget(wall_time=80.0), clock=clock, **arguments("random", "generation", tmp_path / "r", evaluator, batch_size=10)
    )
    assert second.stop_reason == "budget:wall_time" and second.evaluations_used == 80 and second.wall_time == 80.0
    metadata = peek(tmp_path / "r").metadata
    assert metadata["summary"]["wall_time"] == 80.0  # type: ignore[index]
    assert [s["wall_time"] for s in peek(tmp_path / "r").sessions] == [50.0, 80.0]
    with pytest.warns(ResumeWarning):
        exhausted = auxein.resume(
            budget=auxein.Budget(wall_time=80.0), clock=clock, **arguments("random", "generation", tmp_path / "r", evaluator, batch_size=10)
        )
    assert exhausted.evaluations_used == 80  # nothing more to do, and nothing evaluated


def test_a_wall_time_budget_that_is_already_used_up_evaluates_nothing_on_resume(tmp_path: Path):
    now = [0.0]
    calls: list[int] = []

    def tick(genome: np.ndarray) -> float:
        now[0] += 1.0
        calls.append(1)
        return sphere(genome)

    evaluator = auxein.FunctionEvaluator(tick)
    auxein.run(
        budget=auxein.Budget(wall_time=30.0, evaluations=1000),
        clock=lambda: now[0],
        **arguments("random", "generation", tmp_path / "r", evaluator, batch_size=10),
    )
    pretend_killed(tmp_path / "r")  # killed before it could say it was finished, with the budget used up
    calls.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = auxein.resume(
            budget=auxein.Budget(wall_time=30.0, evaluations=1000),
            clock=lambda: now[0],
            **arguments("random", "generation", tmp_path / "r", evaluator, batch_size=10),
        )
    assert calls == [] and again.stop_reason == "budget:wall_time" and again.evaluations_used == 30


def test_cost_totals_are_cumulative_across_sessions_and_a_new_unit_is_counted_from_the_recording(tmp_path: Path):
    def spend(genome: np.ndarray) -> Result:
        return Result(objectives={"value": sphere(genome)}, cost={"tokens": 10.0, "euros": 0.5})

    evaluator = auxein.FunctionEvaluator(spend)
    first = auxein.run(budget=auxein.Budget(cost={"tokens": 200.0}), **arguments("ga", "generation", tmp_path / "a", evaluator))
    assert first.stop_reason == "budget:cost:tokens" and first.evaluations_used == 32
    second = auxein.resume(budget=auxein.Budget(cost={"tokens": 500.0}), **arguments("ga", "generation", tmp_path / "a", evaluator))
    reference = auxein.run(budget=auxein.Budget(cost={"tokens": 500.0}), **arguments("ga", "generation", tmp_path / "b", evaluator))
    assert second.evaluations_used == reference.evaluations_used == 64  # 640 tokens: the batch that crosses 500 completes
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")
    third = auxein.resume(budget=auxein.Budget(cost={"euros": 40.0}), **arguments("ga", "generation", tmp_path / "a", evaluator))
    assert third.evaluations_used == 80  # 32 euros were spent already (64 evaluations): the new unit starts from what was recorded
    assert third.stop_reason == "budget:cost:euros"


# --- two processes ---


def start_long_run(run_dir: Path, evaluations: int = 20_000) -> subprocess.Popen[str]:
    config = {"strategy": "random", "run_dir": str(run_dir), "evaluations": evaluations, "executor": "inline", "concurrency": 1}
    return subprocess.Popen(
        [sys.executable, "-m", "tests.support.resume_cli", json.dumps(config)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def wait_for_progress(run_dir: Path, process: subprocess.Popen[str], at_least: int = 5) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        assert process.poll() is None, "the run ended before the test could look at it"
        try:
            db = sqlite3.connect(f"file:{run_dir / 'events.sqlite'}?mode=ro", uri=True)
            count = db.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
            db.close()
            if count >= at_least:
                return
        except sqlite3.Error:
            pass
        time.sleep(0.02)
    raise AssertionError("the run made no progress")


def test_a_second_process_cannot_resume_a_run_that_is_being_written_and_takes_over_the_lock_of_a_killed_one(tmp_path: Path):
    run_dir = tmp_path / "r"
    writer = start_long_run(run_dir)
    try:
        wait_for_progress(run_dir, writer)
        before = comparable(run_dir)["candidates"][:3]
        with pytest.raises(ResumeError, match=rf"being written by process {writer.pid}"):
            auxein.resume(
                budget=auxein.Budget(evaluations=30_000), **arguments("random", "generation", run_dir, batch_size=10, executor="inline")
            )
        assert (run_dir / "writer.lock").read_text() == str(writer.pid)
        assert comparable(run_dir)["candidates"][:3] == before
        writer.send_signal(signal.SIGKILL)
        writer.wait()
    finally:
        if writer.poll() is None:
            writer.kill()
            writer.wait()
    assert (run_dir / "writer.lock").exists()  # the dead writer left its lock behind
    killed_at = len(list(peek(run_dir).evaluations()))
    from tests.support.resumable import settings

    result = auxein.resume(
        budget=auxein.Budget(evaluations=killed_at + 20),
        **{**settings({"strategy": "random", "run_dir": str(run_dir), "executor": "inline"})},
    )
    assert result.evaluations_used == killed_at + 20
    assert not (run_dir / "writer.lock").exists()


def test_the_lock_is_released_when_a_run_ends_fails_or_is_interrupted(tmp_path: Path):
    auxein.run(budget=auxein.Budget(evaluations=20), **arguments("random", "generation", tmp_path / "a"))
    assert not (tmp_path / "a" / "writer.lock").exists()
    with pytest.raises(EvaluationError):
        auxein.run(
            budget=auxein.Budget(evaluations=20),
            failure_policy="fail_fast",
            **arguments("random", "generation", tmp_path / "b", Scripted(fail_at=3)),
        )
    assert not (tmp_path / "b" / "writer.lock").exists()
    with pytest.raises(KeyboardInterrupt):
        auxein.run(budget=auxein.Budget(evaluations=20), **arguments("random", "steady_state", tmp_path / "c", Scripted(interrupt_at=3)))
    assert not (tmp_path / "c" / "writer.lock").exists()


def test_a_second_writer_in_the_same_process_is_refused_too(tmp_path: Path):
    from auxein.recording import SQLiteRecorder

    auxein.run(budget=auxein.Budget(evaluations=20), **arguments("random", "generation", tmp_path / "r"))
    first = SQLiteRecorder(tmp_path / "r", resume=True)
    first.open_existing()
    try:
        with pytest.raises(ResumeError, match="being written by this process"):
            SQLiteRecorder(tmp_path / "r", resume=True).open_existing()
    finally:
        first.abandon()
    SQLiteRecorder(tmp_path / "r", resume=True).open_existing()  # and the lock is free again
    assert os.path.exists(tmp_path / "r" / "writer.lock")
    # clean up the lock this test just took
    (tmp_path / "r" / "writer.lock").unlink()


def test_the_event_log_helper_is_unchanged_by_a_resume_of_nothing(tmp_path: Path):
    auxein.run(budget=auxein.Budget(evaluations=40), **arguments("random", "generation", tmp_path / "r"))
    assert event_log(tmp_path / "r")["events"][-1][0] == "stop"
