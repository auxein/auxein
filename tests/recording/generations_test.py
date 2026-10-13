"""The generations of a recorded run (design doc §10.2): the reader's answers after the fact are the run's answers at the time.

Every test that runs something compares the reader with the `Progress` the callbacks of the same run saw, generation by
generation, and with the `RunResult`: same candidates, same history rows. Generation k is the k-th tell across sessions.
"""

import asyncio
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import auxein
from auxein.callbacks import ProgressHistory, stop_when
from auxein.core import Candidate, CandidateId, Evaluation, EvaluationBatch, ListBatch, Objective, Result, Status
from auxein.driver import Progress
from auxein.recording import RecordedEvaluation, RunReader, SQLiteRecorder, open_run
from auxein.recording import reader as reader_module
from tests.driver.resume_kill_test import launch, recorded
from tests.driver.resume_test import pretend_killed
from tests.support import sequences
from tests.support.fixtures import integration_backend
from tests.support.resumable import backend_config
from tests.support.resumable import settings as resumable_settings

pytestmark = [
    pytest.mark.filterwarnings("ignore::auxein.driver.errors.EvaluationFailureWarning"),
    pytest.mark.filterwarnings("ignore::auxein.RecordingDisabledWarning"),
    pytest.mark.usefixtures("use_corner_backend"),
]


# --- problems: a constrained single objective with failures, and two objectives in opposite directions ---


def constrained(genome: Any) -> Result:
    """The sphere with a constraint, failing for a few genomes: what the champion ranking must get right."""
    if float(genome[0]) > 4.0:
        raise RuntimeError("diverged")
    return Result({"value": float((genome * genome).sum())}, {"c": max(0.0, float(genome[1]) - 2.0)})


async def constrained_async(genome: Any) -> Result:
    """`constrained`, finishing after a delay that depends on the genome, so that throughput mode tells out of ask order."""
    await asyncio.sleep(0.0005 * (int(abs(float(genome[2])) * 1000) % 7))
    return constrained(genome)


def opposed(genome: Any) -> Result:
    """Two objectives in conflict, one minimised and one maximised (`directed`)."""
    return Result({"near": float((genome * genome).sum()), "far": -float(((genome - 1.0) ** 2).sum())})


DIRECTED = [Objective("near"), Objective("far", "maximise")]


def arguments(kind: str, run_dir: Path | None, evaluations: int, **over: Any) -> dict[str, Any]:
    if kind == "nsga2":
        base: dict[str, Any] = {
            "strategy": auxein.NSGA2(population_size=10, offspring_size=10),
            "evaluator": auxein.FunctionEvaluator(opposed),
            "objectives": DIRECTED,
        }
    else:
        base = {
            "strategy": auxein.GeneticAlgorithm(population_size=10, offspring_size=10),
            "evaluator": auxein.FunctionEvaluator(constrained),
            "constraints": ["c"],
        }
    base.update(
        {
            "space": auxein.Box(-5.0, 5.0, dim=3),
            "budget": auxein.Budget(evaluations=evaluations),
            "seed": 3,
            "batch_size": 10,
            "run_dir": run_dir,
            "backend": integration_backend(),
        }
    )
    base.update(over)
    return base


class Spy:
    """A callback that keeps every `Progress`, and a `ProgressHistory` of the same calls."""

    def __init__(self) -> None:
        self.calls: list[Progress[Any]] = []
        self.history = ProgressHistory()

    def __call__(self, progress: Progress[Any]) -> None:
        self.calls.append(progress)
        self.history(progress)

    @property
    def told(self) -> list[Progress[Any]]:
        return [p for p in self.calls if not p.final]


def ident(evaluation: Any) -> int | None:
    if evaluation is None:
        return None
    return int(evaluation.candidate_id if isinstance(evaluation, RecordedEvaluation) else evaluation.candidate.id)


def same_value(a: object, b: object) -> bool:
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def assert_agrees_with_callbacks(run: RunReader, spy: Spy) -> None:
    """Every tell the callbacks saw has the same champion, front and history row in the reader."""
    rows = {row["generation"]: row for row in run.generation_history()}
    told = spy.told
    assert told, "the run told nothing"
    for progress, live in zip(told, [r for r in spy.history.rows if not r["final"]], strict=True):
        k = progress.generation
        assert ident(run.best_up_to(k)) == ident(progress.best), k
        assert [ident(e) for e in run.pareto_front_up_to(k)] == [ident(e) for e in progress.pareto_front], k
        row = rows[k]
        assert row["step"] == progress.step
        for key, value in row.items():
            if key in live:
                assert same_value(value, live[key]), (k, key, value, live[key])
        assert {"elapsed", "replaying", "final"}.isdisjoint(row)
    if spy.calls[-1].final:
        assert run.generations() == told[-1].generation
        assert ident(run.best_up_to()) == ident(spy.calls[-1].best)
        assert [ident(e) for e in run.pareto_front_up_to()] == [ident(e) for e in spy.calls[-1].pareto_front]


def same_generations(a: Path, b: Path) -> None:
    """Two recordings have the same generations: numbering, champions, fronts and history."""
    with open_run(a) as one, open_run(b) as two:
        assert one.generations() == two.generations()
        ids = [e.candidate_id for e in one.evaluations()]
        assert ids == [e.candidate_id for e in two.evaluations()]
        assert [one.generation_of(i) for i in ids] == [two.generation_of(i) for i in ids]
        for k in range(1, one.generations() + 1):
            assert ident(one.best_up_to(k)) == ident(two.best_up_to(k))
            assert [ident(e) for e in one.pareto_front_up_to(k)] == [ident(e) for e in two.pareto_front_up_to(k)]
        for x, y in zip(one.generation_history(), two.generation_history(), strict=True):
            assert x.keys() == y.keys() and all(same_value(x[key], y[key]) for key in x)


# --- the reader agrees with the live view ---


@pytest.mark.parametrize(
    ("kind", "delivery", "deterministic"),
    [
        ("ga", "generation", True),
        ("ga", "steady_state", True),
        ("ga", "steady_state", False),
        ("nsga2", "generation", True),
        ("nsga2", "steady_state", False),
    ],
)
def test_the_reader_agrees_with_the_callbacks_at_every_generation(tmp_path: Path, kind: str, delivery: str, deterministic: bool):
    spy = Spy()
    over: dict[str, Any] = {"delivery": delivery, "deterministic": deterministic, "callbacks": [spy]}
    if not deterministic:  # throughput mode tells as evaluations finish: make them finish out of order
        over.update(concurrency=4, evaluator=auxein.FunctionEvaluator(constrained_async if kind == "ga" else opposed))
    result = auxein.run(**arguments(kind, tmp_path / "r", 150, **over))
    with open_run(tmp_path / "r") as run:
        assert_agrees_with_callbacks(run, spy)
        assert ident(run.best_up_to()) == ident(result.best)
        if kind == "ga":
            assert result.status_counts["failed"] > 0 and run.best_up_to() is not None  # failures happened and ranked last
        if delivery == "steady_state":
            assert run.generations() == 150 and all(len(run.evaluations_in(k)) == 1 for k in (1, 75, 150))


def test_throughput_mode_numbers_tells_in_completion_order(tmp_path: Path):
    over = {"delivery": "steady_state", "deterministic": False, "concurrency": 4, "evaluator": auxein.FunctionEvaluator(constrained_async)}
    auxein.run(**arguments("ga", tmp_path / "r", 80, **over))
    with open_run(tmp_path / "r") as run:
        told = [run.evaluations_in(k)[0].candidate_id for k in range(1, run.generations() + 1)]
    assert sorted(told) == list(range(80)) and told != sorted(told)  # every candidate once, and not in ask order


# --- the overall answers, and candidates that were never told ---


@pytest.mark.parametrize("kind", ["ga", "nsga2"])
def test_the_overall_answers_are_the_run_result_including_a_truncated_final_batch(tmp_path: Path, kind: str):
    result = auxein.run(**arguments(kind, tmp_path / "r", 155))  # batches of 10: the last one is cut to 5 and never told
    with open_run(tmp_path / "r") as run:
        assert run.generations() == 15
        assert [run.generation_of(i) for i in range(150, 155)] == [None] * 5
        assert run.generation_of(0) == 1 and run.generation_of(149) == 15
        assert ident(run.best_up_to()) == ident(result.best)
        assert [ident(e) for e in run.pareto_front_up_to()] == [ident(e) for e in result.pareto_front]
        assert run.generation_history()[-1]["evaluations"] == 150
        with pytest.raises(ValueError, match="candidate 155 is not in the recording"):
            run.generation_of(155)


def test_a_batch_never_told_that_holds_the_best_is_the_overall_best_and_in_no_generation(tmp_path: Path):
    """Built through the recorder, so that the batch that was never told certainly holds the best."""
    recorder = SQLiteRecorder(tmp_path / "r")
    recorder.on_start({"name": "t", "problem": {"objectives": [{"name": "value", "direction": "minimise"}], "constraints": []}})
    told = ListBatch([Candidate(CandidateId(i), f"g{i}", (), "init", 0) for i in range(3)])
    recorder.on_batch(0, told, EvaluationBatch([Evaluation(c, Status.OK, {"value": 5.0 - c.id}) for c in told.candidates]))
    recorder.on_tell(0, 3)
    cut = ListBatch([Candidate(CandidateId(i), f"g{i}", (), "init", 1) for i in range(3, 5)])
    recorder.on_batch(1, cut, EvaluationBatch([Evaluation(c, Status.OK, {"value": -float(c.id)}) for c in cut.candidates]))
    recorder.on_end("completed", "budget:evaluations", {})
    with open_run(tmp_path / "r") as run:
        assert run.generations() == 1 and [run.generation_of(i) for i in range(5)] == [1, 1, 1, None, None]
        assert ident(run.best_up_to(1)) == 2 and ident(run.best_in(1)) == 2
        best = run.best_up_to()
        assert best is not None and best.candidate_id == 4 and best.genome == "g4"
        assert [ident(e) for e in run.pareto_front_up_to()] == [4]
        assert run.generation_history()[0]["evaluations"] == 3


def test_a_run_with_no_tell_has_no_generations_but_an_overall_best(tmp_path: Path):
    result = auxein.run(**arguments("ga", tmp_path / "r", 7))  # one ask, cut to 7: never told
    with open_run(tmp_path / "r") as run:
        assert run.generations() == 0 and run.generation_history() == [] and run.generation_history_csv() == ""
        assert ident(run.best_up_to()) == ident(result.best) and list(run.champions(every=1)) == []
        with pytest.raises(ValueError, match="no generation 1: the run has no generations"):
            run.best_up_to(1)


# --- resumed runs ---


@pytest.mark.parametrize("vectorised", [False, True])
def test_an_extended_run_has_the_generations_of_the_longer_run(tmp_path: Path, vectorised: bool):
    """Deterministic replay, with the cut final batch grown (a function evaluator) or recorded again whole (a vectorised one)."""

    def sphere_batch(X: Any) -> Any:
        return (X * X).sum(axis=1)

    over: dict[str, Any] = {"evaluator": auxein.VectorisedEvaluator(sphere_batch), "constraints": []} if vectorised else {}
    auxein.run(**arguments("ga", tmp_path / "ref", 180, **over))
    auxein.run(**arguments("ga", tmp_path / "ext", 155, **over))
    spy = Spy()
    auxein.resume(**arguments("ga", tmp_path / "ext", 180, callbacks=[spy], **over))
    same_generations(tmp_path / "ext", tmp_path / "ref")
    with open_run(tmp_path / "ext") as run:
        assert run.generation_of(150) == 16 and run.generations() == 18  # the batch cut at 155 was told in the longer run
        assert_agrees_with_callbacks(run, spy)
        assert spy.told[-1].generation == 18


def test_a_resume_that_replays_from_the_start_numbers_generations_across_sessions(tmp_path: Path):
    auxein.run(**arguments("ga", tmp_path / "r", 100, keep_checkpoints=0))
    spy = Spy()
    auxein.resume(**arguments("ga", tmp_path / "r", 160, keep_checkpoints=0, callbacks=[spy]))
    assert [p.generation for p in spy.told] == list(range(1, 17)) and spy.told[0].replaying and not spy.told[-1].replaying
    with open_run(tmp_path / "r") as run:
        assert_agrees_with_callbacks(run, spy)


def test_a_throughput_run_truncated_and_redone_numbers_generations_across_sessions(tmp_path: Path):
    def crash_at_60(progress: Progress[Any]) -> None:
        if progress.generation == 60:
            raise RuntimeError("the renderer crashed")

    over: dict[str, Any] = {
        "delivery": "steady_state",
        "deterministic": False,
        "concurrency": 4,
        "evaluator": auxein.FunctionEvaluator(constrained_async),
        "checkpoint_every_evaluations": 25,
    }
    with pytest.raises(RuntimeError, match="renderer crashed"):
        auxein.run(**arguments("ga", tmp_path / "r", 150, callbacks=[crash_at_60], **over))
    spy = Spy()
    auxein.resume(**arguments("ga", tmp_path / "r", 150, callbacks=[spy], **over))
    first = spy.told[0].generation
    assert 1 < first <= 60  # it continues from the checkpoint the work after it was truncated to
    assert [p.generation for p in spy.told] == list(range(first, first + len(spy.told)))
    with open_run(tmp_path / "r") as run:
        assert run.sessions[-1]["mode"] == "truncate"
        assert_agrees_with_callbacks(run, spy)
        assert run.generations() == 150 and sorted(e.candidate_id for k in range(1, 151) for e in run.evaluations_in(k)) == sorted(
            e.candidate_id for e in run.evaluations()
        )


def test_a_run_stopped_by_a_callback_then_extended_has_the_generations_of_the_run_never_stopped(tmp_path: Path):
    auxein.run(**arguments("ga", tmp_path / "ref", 200))
    first = Spy()
    stop = stop_when(lambda p: p.generation == 7, "enough")
    stopped = auxein.run(**arguments("ga", tmp_path / "r", 200, callbacks=[first, stop]))
    assert stopped.stop_reason == "callback:enough"
    with open_run(tmp_path / "r") as run:
        assert run.generations() == 7
        assert_agrees_with_callbacks(run, first)
    second = Spy()
    auxein.resume(**arguments("ga", tmp_path / "r", 200, callbacks=[second]))
    assert second.told[0].generation == 8 or second.told[0].replaying
    with open_run(tmp_path / "r") as run:
        assert_agrees_with_callbacks(run, second)
    same_generations(tmp_path / "r", tmp_path / "ref")


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_a_batch_recorded_but_never_told_has_no_generation_until_a_resume_tells_it(tmp_path: Path, delivery: str):
    """A crash between recording a batch and telling it (simulated as `tests/driver/resume_test.py` does)."""
    over: dict[str, Any] = {"delivery": delivery, "checkpoint_every_evaluations": 48, "keep_checkpoints": 10}
    auxein.run(**arguments("ga", tmp_path / "ref", 200, **over))
    auxein.run(**arguments("ga", tmp_path / "a", 200, **over))
    db = sqlite3.connect(tmp_path / "a" / "events.sqlite")
    (last_tell,) = db.execute("SELECT MAX(seq) FROM events WHERE kind = 'tell'").fetchone()
    for table, column in (("lineage", "child_id"), ("evaluations", "candidate_id")):
        db.execute(f"DELETE FROM {table} WHERE {column} IN (SELECT id FROM candidates WHERE event_seq > ?)", (last_tell,))
    db.execute("DELETE FROM candidates WHERE event_seq > ?", (last_tell,))
    db.execute("DELETE FROM events WHERE seq >= ?", (last_tell,))
    db.execute("DELETE FROM events WHERE kind = 'stop'")
    db.execute("DELETE FROM checkpoints WHERE event_seq >= ?", (last_tell,))
    db.commit()
    db.close()
    pretend_killed(tmp_path / "a")
    with open_run(tmp_path / "a") as run:
        untold = [e.candidate_id for e in run.evaluations() if run.generation_of(e.candidate_id) is None]
        assert untold and run.generations() == (19 if delivery == "generation" else 199)
        assert run.best_up_to() is not None
    spy = Spy()
    auxein.resume(**arguments("ga", tmp_path / "a", 200, callbacks=[spy], **over))
    with open_run(tmp_path / "a") as run:
        assert all(run.generation_of(i) is not None for i in untold)
        assert_agrees_with_callbacks(run, spy)
    same_generations(tmp_path / "a", tmp_path / "ref")


@pytest.mark.parametrize("deterministic", [True, False])
def test_a_killed_run_resumes_with_its_generations_numbered_across_sessions(tmp_path: Path, deterministic: bool):
    """SIGKILL in the middle (the kill tests' runner), then resume: replay in deterministic mode, truncate-and-redo otherwise."""
    backend = backend_config(integration_backend())
    config: dict[str, Any] = {"strategy": "ga", "delivery": "steady_state", "deterministic": deterministic, "backend": backend}
    reference = tmp_path / "ref"
    auxein.run(budget=auxein.Budget(evaluations=150), **resumable_settings({**config, "run_dir": str(reference)}))
    run_dir = tmp_path / "killed"
    victim = launch({**config, "run_dir": str(run_dir), "evaluations": 150}, None)
    deadline = time.monotonic() + 60
    while victim.poll() is None and recorded(run_dir) < 70 and time.monotonic() < deadline:
        time.sleep(0.005)
    victim.kill()
    victim.wait()
    spy = Spy()
    auxein.resume(budget=auxein.Budget(evaluations=150), callbacks=[spy], **resumable_settings({**config, "run_dir": str(run_dir)}))
    with open_run(run_dir) as run:
        assert_agrees_with_callbacks(run, spy)
        assert run.generations() == 150
        assert sorted(run.evaluations_in(k)[0].candidate_id for k in range(1, 151)) == list(range(150))
    if deterministic:
        same_generations(run_dir, reference)


# --- the queries themselves ---


def test_best_in_a_generation_can_be_worse_than_the_champion_so_far(tmp_path: Path):
    auxein.run(**arguments("ga", tmp_path / "r", 200))
    with open_run(tmp_path / "r") as run:
        worse = 0
        for k in range(1, run.generations() + 1):
            own, champion = run.best_in(k), run.best_up_to(k)
            told = run.evaluations_in(k)
            assert own is not None and champion is not None and own.candidate_id in {e.candidate_id for e in told}
            feasible = [e for e in told if e.status is Status.OK and not any(e.constraints.values())]
            assert own.objectives["value"] == min(e.objectives["value"] for e in feasible)
            assert champion.objectives["value"] <= own.objectives["value"]
            worse += champion.candidate_id != own.candidate_id
        assert worse > 0
        assert ident(run.best_in(1)) == ident(run.best_up_to(1))  # nothing came before the first generation


def test_evaluations_in_a_generation_come_in_tell_order_with_genomes(tmp_path: Path):
    spy = Spy()
    auxein.run(**arguments("ga", tmp_path / "r", 60, callbacks=[spy]))
    with open_run(tmp_path / "r") as run:
        third = run.evaluations_in(3)
        assert [e.candidate_id for e in third] == list(range(20, 30))
        assert all(isinstance(e.genome, np.ndarray) and e.genome.shape == (3,) for e in third)
        assert all(e.parents for e in third)  # bred: the lineage comes with them
        best = spy.told[2].best
        assert best is not None
        champion = run.best_up_to(3)
        assert champion is not None
        np.testing.assert_array_equal(champion.genome, integration_backend().to_numpy(best.candidate.genome))


def test_generations_out_of_range_name_the_valid_range(tmp_path: Path):
    auxein.run(**arguments("ga", tmp_path / "r", 50))
    with open_run(tmp_path / "r") as run:
        for query in (run.best_up_to, run.best_in, run.pareto_front_up_to, run.evaluations_in):
            for k in (0, 6, -1):
                with pytest.raises(ValueError, match=f"no generation {k}: generations are numbered 1 to 5"):
                    query(k)
            with pytest.raises(TypeError):
                query(2.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="every must be a positive int"):
            list(run.champions(every=0))


def test_a_recording_of_schema_version_4_is_refused(tmp_path: Path):
    auxein.run(**arguments("ga", tmp_path / "r", 20))
    db = sqlite3.connect(tmp_path / "r" / "events.sqlite")
    db.execute("UPDATE schema_version SET version = 4")
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="unsupported run schema version 4: this reader understands version 5"):
        open_run(tmp_path / "r")


def test_champions_are_every_n_generations_and_the_last(tmp_path: Path):
    auxein.run(**arguments("ga", tmp_path / "r", 150))  # 15 generations
    with open_run(tmp_path / "r") as run:
        assert [k for k, _ in run.champions(every=4)] == [4, 8, 12, 15]
        assert [k for k, _ in run.champions(every=5)] == [5, 10, 15]
        assert [k for k, _ in run.champions(every=20)] == [15]
        for k, champion in run.champions(every=3):
            assert ident(champion) == ident(run.best_up_to(k))
    auxein.run(**arguments("nsga2", tmp_path / "mo", 30))
    with open_run(tmp_path / "mo") as run, pytest.raises(ValueError, match="several objectives has no single champion"):
        list(run.champions(every=1))


def test_a_run_with_several_objectives_has_fronts_and_no_single_best(tmp_path: Path):
    spy = Spy()
    auxein.run(**arguments("nsga2", tmp_path / "r", 100, callbacks=[spy]))
    with open_run(tmp_path / "r") as run:
        assert run.best_up_to() is None and run.best_up_to(3) is None and run.best_in(3) is None
        front = run.pareto_front_up_to(10)
        assert front and all(e.status is Status.OK for e in front)
        row = run.generation_history()[-1]
        assert row["best"] is None and row["front_size"] == len(front) and "near_mean" in row and "far_worst" in row
        assert row["far_best"] >= row["far_worst"]  # a maximised objective: its best is the largest


def test_the_history_as_csv(tmp_path: Path):
    auxein.run(**arguments("ga", tmp_path / "r", 50))
    with open_run(tmp_path / "r") as run:
        text = run.generation_history_csv(tmp_path / "history.csv")
        assert (tmp_path / "history.csv").read_text() == text
        lines = text.splitlines()
        assert len(lines) == 6 and lines[0].startswith("generation,step,evaluations,best,front_size,failed,timeout,batch_ok")


# --- decoding: only what is returned, through the codec ---


class DecodeCounter:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls = 0
        original = reader_module.decode_genome

        def counting(*args: Any) -> Any:
            self.calls += 1
            return original(*args)

        monkeypatch.setattr(reader_module, "decode_genome", counting)

    def since(self) -> int:
        calls, self.calls = self.calls, 0
        return calls


def test_queries_decode_only_the_genomes_they_return(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    auxein.run(**arguments("ga", tmp_path / "r", 300))
    counter = DecodeCounter(monkeypatch)
    with open_run(tmp_path / "r") as run:
        assert run.generations() == 30 and counter.since() == 0  # indexing decodes nothing
        run.generation_history()
        run.generation_history_csv()
        run.generation_of(5)
        assert counter.since() == 0
        run.best_up_to(10)
        assert counter.since() == 1
        run.best_in(10)
        assert counter.since() == 1
        assert counter.since() == 0
        assert len(run.evaluations_in(10)) == 10 and counter.since() == 10
        front = run.pareto_front_up_to(10)
        assert counter.since() == len(front) == 1
        champions = list(run.champions(every=5))
        assert counter.since() == len({ident(c) for _, c in champions})


def test_structured_genomes_come_back_through_the_codec(tmp_path: Path):
    spy = Spy()
    auxein.run(
        strategy=auxein.StructuredGeneticAlgorithm(population_size=10, offspring_size=10),
        evaluator=auxein.FunctionEvaluator(sequences.distance),
        space=sequences.SPACE,
        constraints=["too_long"],
        objectives=[auxein.Objective("distance")],
        budget=auxein.Budget(evaluations=100),
        seed=2,
        batch_size=10,
        run_dir=tmp_path / "r",
        backend=integration_backend(),
        callbacks=[spy],
    )
    with open_run(tmp_path / "r", codec=sequences.SPACE.codec) as run:
        assert_agrees_with_callbacks(run, spy)
        champion, live = run.best_up_to(6), spy.told[5].best
        assert champion is not None and live is not None
        assert isinstance(champion.genome, tuple) and champion.genome == live.candidate.genome
        assert all(sequences.SPACE.contains(e.genome) for e in run.evaluations_in(4))
