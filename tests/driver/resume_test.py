"""Resuming recorded runs: replay, extension, interruption, divergence and configuration checks (design doc §10.4)."""

import json
import sqlite3
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import auxein
from auxein.core import Objective
from auxein.driver import ConfigurationMismatchError, ReplayMismatchError, ResumeError, ResumeWarning
from auxein.strategies.ga import GeneticAlgorithm
from tests.support.eventlog import event_log
from tests.support.reading import peek

pytestmark = pytest.mark.filterwarnings("ignore::auxein.driver.errors.EvaluationFailureWarning")


def sphere(genome: np.ndarray) -> float:
    return float((genome * genome).sum())


def make_strategy(name: str) -> Any:
    return auxein.RandomSearch() if name == "random" else GeneticAlgorithm(population_size=16, offspring_size=16)


def arguments(strategy: str, delivery: str, run_dir: Path, evaluator: Any = None, **over: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "strategy": make_strategy(strategy),
        "evaluator": auxein.FunctionEvaluator(sphere) if evaluator is None else evaluator,
        "space": auxein.Box(-5.0, 5.0, dim=3),
        "seed": 11,
        "batch_size": 16,
        "delivery": delivery,
        "run_dir": run_dir,
    }
    settings.update(over)
    return settings


def comparable(run_dir: Path) -> dict[str, list[object]]:
    """The deterministic part of a recording, without the `resume` events that only a resumed run has."""
    log = event_log(run_dir)
    log["events"] = [event for event in log["events"] if event[0] != "resume"]  # type: ignore[index]
    return log


def same_result(a: auxein.RunResult[Any], b: auxein.RunResult[Any]) -> None:
    assert a.stop_reason == b.stop_reason
    assert a.evaluations_used == b.evaluations_used
    assert a.trace == b.trace
    assert dict(a.status_counts) == dict(b.status_counts)
    assert (a.best is None) == (b.best is None)
    if a.best is not None and b.best is not None:
        assert a.best.candidate.id == b.best.candidate.id
        assert dict(a.best.objectives) == dict(b.best.objectives)
        assert np.array_equal(np.asarray(a.best.candidate.genome), np.asarray(b.best.candidate.genome))
    assert [e.candidate.id for e in a.pareto_front] == [e.candidate.id for e in b.pareto_front]


STRATEGIES = ["random", "ga"]
DELIVERIES = ["generation", "steady_state"]


# --- extension: a finished run with a larger budget is the run with that budget ---


@pytest.mark.parametrize("delivery", DELIVERIES)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_extending_a_finished_run_gives_the_run_that_a_larger_budget_would_have_made(tmp_path: Path, strategy: str, delivery: str):
    short = auxein.run(budget=auxein.Budget(evaluations=2_000), **arguments(strategy, delivery, tmp_path / "a"))
    extended = auxein.resume(budget=auxein.Budget(evaluations=5_000), **arguments(strategy, delivery, tmp_path / "a"))
    uninterrupted = auxein.run(budget=auxein.Budget(evaluations=5_000), **arguments(strategy, delivery, tmp_path / "b"))
    assert short.evaluations_used == 2_000 and extended.evaluations_used == 5_000
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")
    same_result(extended, uninterrupted)
    assert extended.best is not None and short.best is not None
    assert extended.best.objectives["value"] <= short.best.objectives["value"]


@pytest.mark.parametrize("delivery", DELIVERIES)
def test_extending_works_whatever_checkpoints_exist(tmp_path: Path, delivery: str):
    """Replaying from a checkpoint, from scratch (none kept) and from scratch (deleted) all give the same run."""
    reference = auxein.run(budget=auxein.Budget(evaluations=700), **arguments("ga", delivery, tmp_path / "ref"))
    for name, keep in (("kept", None), ("none", 0), ("deleted", None)):
        run_dir = tmp_path / name
        auxein.run(budget=auxein.Budget(evaluations=300), keep_checkpoints=keep, **arguments("ga", delivery, run_dir))
        if name == "deleted":
            db = sqlite3.connect(run_dir / "events.sqlite")
            db.execute("DELETE FROM checkpoints")
            db.commit()
            db.close()
        elif name == "none":
            assert peek(run_dir).checkpoints() == []
        resumed = auxein.resume(budget=auxein.Budget(evaluations=700), keep_checkpoints=keep, **arguments("ga", delivery, run_dir))
        assert comparable(run_dir) == comparable(tmp_path / "ref"), name
        same_result(resumed, reference)


def test_a_finished_run_with_an_unchanged_budget_returns_its_result_without_evaluating(tmp_path: Path):
    calls = []

    def counted(genome: np.ndarray) -> float:
        calls.append(1)
        return sphere(genome)

    evaluator = auxein.FunctionEvaluator(counted)
    first = auxein.run(budget=auxein.Budget(evaluations=100), **arguments("ga", "generation", tmp_path / "r", evaluator))
    before = comparable(tmp_path / "r"), (tmp_path / "r" / "metadata.json").read_text()
    calls.clear()
    with pytest.warns(ResumeWarning, match="already complete"):
        again = auxein.resume(budget=auxein.Budget(evaluations=100), **arguments("ga", "generation", tmp_path / "r", evaluator))
    assert calls == []
    same_result(again, first)
    assert (comparable(tmp_path / "r"), (tmp_path / "r" / "metadata.json").read_text()) == before  # not even a session was added
    assert not (tmp_path / "r" / "writer.lock").exists()


# --- no recorded evaluation is repeated ---


def interrupting(after: int, calls: list[int]) -> Callable[[np.ndarray], float]:
    def evaluate(genome: np.ndarray) -> float:
        calls.append(1)
        if len(calls) == after:
            raise KeyboardInterrupt
        return sphere(genome)

    return evaluate


@pytest.mark.parametrize("delivery", DELIVERIES)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_an_interrupted_run_resumes_to_the_uninterrupted_run_and_repeats_no_evaluation(tmp_path: Path, strategy: str, delivery: str):
    calls: list[int] = []
    evaluator = auxein.FunctionEvaluator(interrupting(150, calls))
    with pytest.raises(KeyboardInterrupt):
        auxein.run(
            budget=auxein.Budget(evaluations=400),
            checkpoint_every_evaluations=50,
            **arguments(strategy, delivery, tmp_path / "a", evaluator),
        )
    recorded = peek(tmp_path / "a").metadata["status"], len(list(peek(tmp_path / "a").evaluations()))
    assert recorded[0] == "interrupted" and 0 < recorded[1] < 150
    calls.clear()
    resumed = auxein.resume(
        budget=auxein.Budget(evaluations=400),
        **arguments(strategy, delivery, tmp_path / "a", auxein.FunctionEvaluator(interrupting(10**9, calls))),
    )
    assert len(calls) == 400 - recorded[1]  # exactly the evaluations that were not recorded
    reference = auxein.run(budget=auxein.Budget(evaluations=400), **arguments(strategy, delivery, tmp_path / "b"))
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")
    same_result(resumed, reference)


def test_interrupting_a_steady_state_run_writes_a_checkpoint_and_a_generation_run_cannot(tmp_path: Path):
    """A steady-state window is part of the checkpoint, so it is consistent whenever the driver waits; a batch is not."""
    for delivery, expect in (("steady_state", True), ("generation", False)):
        calls: list[int] = []
        run_dir = tmp_path / delivery
        with pytest.raises(KeyboardInterrupt):
            auxein.run(
                budget=auxein.Budget(evaluations=300),
                checkpoint_every=10_000,  # no periodic checkpoint: any that exists is the interrupt's
                **arguments("ga", delivery, run_dir, auxein.FunctionEvaluator(interrupting(100, calls))),
            )
        assert bool(peek(run_dir).checkpoints()) is expect


# --- a crash between recording a batch and telling it ---


def pretend_killed(run_dir: Path) -> None:
    """Make a finished run's metadata read as that of a process that was killed: still 'running', no end."""
    path = run_dir / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["status"] = "running"
    for key in ("ended_at", "stop_reason", "summary"):
        metadata.pop(key, None)
    for key in ("ended_at", "stop_reason", "evaluations_used", "wall_time"):
        metadata["sessions"][-1].pop(key, None)
    metadata["sessions"][-1]["status"] = "running"
    path.write_text(json.dumps(metadata))


@pytest.mark.parametrize("delivery", DELIVERIES)
def test_a_run_killed_between_recording_a_candidate_and_telling_it_resumes_to_the_same_log(tmp_path: Path, delivery: str):
    auxein.run(
        budget=auxein.Budget(evaluations=200),
        checkpoint_every_evaluations=48,
        keep_checkpoints=10,
        **arguments("ga", delivery, tmp_path / "a"),
    )
    reference = comparable(tmp_path / "a")
    db = sqlite3.connect(tmp_path / "a" / "events.sqlite")
    (last_tell,) = db.execute("SELECT MAX(seq) FROM events WHERE kind = 'tell'").fetchone()
    for table, column in (("lineage", "child_id"), ("evaluations", "candidate_id")):  # ...nor did anything after it...
        db.execute(f"DELETE FROM {table} WHERE {column} IN (SELECT id FROM candidates WHERE event_seq > ?)", (last_tell,))
    db.execute("DELETE FROM candidates WHERE event_seq > ?", (last_tell,))
    db.execute("DELETE FROM events WHERE seq >= ?", (last_tell,))  # ...so the batch is recorded and was never told
    db.execute("DELETE FROM events WHERE kind = 'stop'")
    db.execute("DELETE FROM checkpoints WHERE event_seq >= ?", (last_tell,))  # ...so no checkpoint was taken after it
    db.commit()
    db.close()
    pretend_killed(tmp_path / "a")
    auxein.resume(budget=auxein.Budget(evaluations=200), **arguments("ga", delivery, tmp_path / "a"))
    assert comparable(tmp_path / "a") == reference


# --- divergence ---


def recorded_state(run_dir: Path) -> tuple[object, str]:
    return comparable(run_dir), (run_dir / "metadata.json").read_text()


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("seed", 12),
        ("batch_size", 8),
        ("delivery", "generation"),  # recorded: steady_state
        ("deterministic", False),
        ("concurrency", 2),
        ("executor", "thread"),
        ("timeout", 5.0),
        ("failure_policy", "fail_fast"),
        ("initial_failure_guard", None),
        ("strategy", GeneticAlgorithm(population_size=13, offspring_size=12)),
        ("evaluator", auxein.FunctionEvaluator(lambda g: float(g.sum()))),
        ("space", auxein.Box(-4.0, 5.0, dim=3)),
        ("objectives", [Objective("value", "maximise")]),
        ("constraints", ["c"]),
        ("descriptors", ["d"]),
        ("backend", auxein.Backend("numpy", "cpu", "float32")),
    ],
)
def test_each_setting_that_is_not_the_budget_is_refused_by_name_and_nothing_is_recorded(tmp_path: Path, setting: str, value: object):
    run_dir = tmp_path / "r"
    auxein.run(budget=auxein.Budget(evaluations=60), **arguments("ga", "steady_state", run_dir))
    before = recorded_state(run_dir)
    changed = arguments("ga", "steady_state", run_dir)
    changed[setting] = value
    expected = (
        "precision" if setting == "backend" else setting.replace("space", "problem.space").replace("objectives", "problem.objectives")
    )
    expected = expected.replace("constraints", "problem.constraints") if setting == "constraints" else expected
    expected = "problem.descriptors" if setting == "descriptors" else expected
    with pytest.raises(ConfigurationMismatchError, match=f"{expected}: recorded") as raised:
        auxein.resume(budget=auxein.Budget(evaluations=120), **changed)
    assert "only the budget may change" in str(raised.value)
    assert recorded_state(run_dir) == before
    assert not (run_dir / "writer.lock").exists()


def test_every_mismatching_setting_is_listed_with_its_recorded_and_given_value(tmp_path: Path):
    run_dir = tmp_path / "r"
    auxein.run(budget=auxein.Budget(evaluations=60), **arguments("random", "generation", run_dir))
    with pytest.raises(ConfigurationMismatchError) as raised:
        auxein.resume(budget=auxein.Budget(evaluations=120), **arguments("random", "generation", run_dir, seed=12, batch_size=8))
    message = str(raised.value)
    assert "seed: recorded 11, given 12" in message and "batch_size: recorded 16, given 8" in message


def test_only_the_budget_may_change_and_a_larger_one_is_accepted(tmp_path: Path):
    auxein.run(budget=auxein.Budget(evaluations=60), **arguments("random", "generation", tmp_path / "r"))
    result = auxein.resume(
        budget=auxein.Budget(evaluations=100, wall_time=1_000.0), checkpoint_every=5.0, **arguments("random", "generation", tmp_path / "r")
    )
    assert result.evaluations_used == 100


def test_a_budget_below_what_is_recorded_is_refused(tmp_path: Path):
    auxein.run(budget=auxein.Budget(evaluations=60), **arguments("random", "generation", tmp_path / "r"))
    with pytest.raises(ResumeError, match="smaller than the 60 evaluations"):
        auxein.resume(budget=auxein.Budget(evaluations=30), **arguments("random", "generation", tmp_path / "r"))


class Drifting(auxein.RandomSearch):  # type: ignore[type-arg]
    """A strategy whose code changed: same name and repr, other candidates."""

    def ask(self, n: int) -> Any:
        batch = super().ask(n)
        return auxein.core.ArrayBatch(batch.as_array() + 1e-9, batch.ids, batch.step, "random")


Drifting.__name__ = Drifting.__qualname__ = "RandomSearch"
Drifting.__module__ = "auxein.strategies.random_search"


@pytest.mark.parametrize("delivery", DELIVERIES)
def test_changed_strategy_code_is_found_by_replay_and_nothing_is_recorded(tmp_path: Path, delivery: str):
    run_dir = tmp_path / "r"
    auxein.run(budget=auxein.Budget(evaluations=100), keep_checkpoints=0, **arguments("random", delivery, run_dir))
    before = recorded_state(run_dir)
    changed = arguments("random", delivery, run_dir)
    changed["strategy"] = Drifting()
    with pytest.raises(ReplayMismatchError, match=r"candidate 0 differently.*its genome.*configuration or code changed") as raised:
        auxein.resume(budget=auxein.Budget(evaluations=200), keep_checkpoints=0, **changed)
    assert "seed" in str(raised.value)
    assert recorded_state(run_dir) == before
    assert not (run_dir / "writer.lock").exists()


@pytest.mark.parametrize("delivery", DELIVERIES)
def test_a_tampered_recorded_genome_names_the_candidate(tmp_path: Path, delivery: str):
    run_dir = tmp_path / "r"
    auxein.run(budget=auxein.Budget(evaluations=100), keep_checkpoints=0, **arguments("ga", delivery, run_dir))
    db = sqlite3.connect(run_dir / "events.sqlite")
    genome = bytearray(db.execute("SELECT genome FROM candidates WHERE id = 37").fetchone()[0])
    genome[0] ^= 1
    db.execute("UPDATE candidates SET genome = ? WHERE id = 37", (bytes(genome),))
    db.commit()
    db.close()
    before = recorded_state(run_dir)
    with pytest.raises(ReplayMismatchError, match=r"candidate 37 differently.*its genome"):
        auxein.resume(budget=auxein.Budget(evaluations=200), keep_checkpoints=0, **arguments("ga", delivery, run_dir))
    assert recorded_state(run_dir) == before


def test_a_run_directory_that_is_not_a_run_or_has_an_old_schema_cannot_be_resumed(tmp_path: Path):
    with pytest.raises(ResumeError, match="not a recorded run"):
        auxein.resume(budget=auxein.Budget(evaluations=10), **arguments("random", "generation", tmp_path / "missing"))
    run_dir = tmp_path / "old"
    auxein.run(budget=auxein.Budget(evaluations=20), **arguments("random", "generation", run_dir))
    db = sqlite3.connect(run_dir / "events.sqlite")
    db.execute("UPDATE schema_version SET version = 1")
    db.commit()
    db.close()
    with pytest.raises(ResumeError, match="schema version 1.*cannot be migrated"):
        auxein.resume(budget=auxein.Budget(evaluations=40), **arguments("random", "generation", run_dir))
    assert not (run_dir / "writer.lock").exists()


@pytest.mark.filterwarnings("ignore::auxein.RecordingDisabledWarning")
def test_checkpoint_options_need_a_run_directory():
    for option in ({"checkpoint_every": 10.0}, {"checkpoint_every_evaluations": 10}, {"keep_checkpoints": 3}):
        with pytest.raises(ValueError, match="need run_dir"):
            auxein.run(
                strategy=auxein.RandomSearch(),
                evaluator=auxein.FunctionEvaluator(sphere),
                space=auxein.Box(-1.0, 1.0, dim=2),
                budget=auxein.Budget(evaluations=10),
                seed=1,
                **option,
            )


def test_resume_needs_a_run_dir():
    with pytest.raises(TypeError):
        auxein.resume(  # type: ignore[call-arg]
            strategy=auxein.RandomSearch(),
            evaluator=auxein.FunctionEvaluator(sphere),
            space=auxein.Box(-1.0, 1.0, dim=2),
            budget=auxein.Budget(evaluations=10),
            seed=1,
        )


# --- sessions, resume events, results ---


def test_sessions_and_resume_events_are_recorded_and_the_original_configuration_is_kept(tmp_path: Path):
    run_dir = tmp_path / "r"
    auxein.run(budget=auxein.Budget(evaluations=100), **arguments("ga", "generation", run_dir))
    original = json.loads((run_dir / "metadata.json").read_text())
    auxein.resume(budget=auxein.Budget(evaluations=250), **arguments("ga", "generation", run_dir))
    run = peek(run_dir)
    metadata = run.metadata
    for key in ("seed", "budget", "batch_size", "strategy", "evaluator", "problem", "started_at", "versions", "git"):
        assert metadata[key] == original[key]
    assert metadata["budget"] == {"evaluations": 100, "wall_time": None, "cost": {}}
    first, second = run.sessions
    assert first["mode"] == "start" and first["status"] == "completed" and first["budget"]["evaluations"] == 100
    assert second["mode"] == "replay" and second["status"] == "completed" and second["budget"]["evaluations"] == 250
    assert second["evaluations_used"] == 250 and "versions" in second and "git" in second
    assert metadata["status"] == "completed" and metadata["summary"]["evaluations_used"] == 250  # type: ignore[index]
    db = sqlite3.connect(run_dir / "events.sqlite")
    resume_events = [json.loads(p) for (p,) in db.execute("SELECT payload FROM events WHERE kind = 'resume'")]
    stops = db.execute("SELECT COUNT(*) FROM events WHERE kind = 'stop'").fetchone()[0]
    db.close()
    assert stops == 1  # the log of a run has one stop, at its end; the earlier one lives on in the sessions
    assert len(resume_events) == 1
    event = resume_events[0]
    assert event["mode"] == "replay" and event["checkpoint"] is not None and event["replayed"] >= 0
    assert event["budget_old"]["evaluations"] == 100 and event["budget_new"]["evaluations"] == 250


def test_the_result_of_a_resumed_run_covers_the_whole_run(tmp_path: Path):
    def constrained(genome: np.ndarray) -> auxein.Result:
        return auxein.Result(objectives={"value": float((genome * genome).sum())}, constraints={"c": float(max(0.0, 1.0 - genome[0]))})

    def args(run_dir: Path) -> dict[str, Any]:
        return arguments("ga", "generation", run_dir, auxein.FunctionEvaluator(constrained), constraints=["c"])

    auxein.run(budget=auxein.Budget(evaluations=120), **args(tmp_path / "a"))
    resumed = auxein.resume(budget=auxein.Budget(evaluations=300), **args(tmp_path / "a"))
    reference = auxein.run(budget=auxein.Budget(evaluations=300), **args(tmp_path / "b"))
    same_result(resumed, reference)
    assert resumed.pareto_front and resumed.trace and resumed.evaluations_used == 300
    assert resumed.wall_time >= 0


def test_a_multi_objective_run_resumes_with_its_front(tmp_path: Path):
    def two(genome: np.ndarray) -> auxein.Result:
        return auxein.Result(objectives={"a": float((genome**2).sum()), "b": float(((genome - 1) ** 2).sum())})

    def args(run_dir: Path) -> dict[str, Any]:
        return arguments("random", "steady_state", run_dir, auxein.FunctionEvaluator(two), objectives=[Objective("a"), Objective("b")])

    auxein.run(budget=auxein.Budget(evaluations=150), **args(tmp_path / "a"))
    resumed = auxein.resume(budget=auxein.Budget(evaluations=400), **args(tmp_path / "a"))
    reference = auxein.run(budget=auxein.Budget(evaluations=400), **args(tmp_path / "b"))
    same_result(resumed, reference)
    assert resumed.best is None and len(resumed.pareto_front) > 1


def test_resume_works_inside_a_running_event_loop_and_asynchronously(tmp_path: Path):
    import asyncio

    auxein.run(budget=auxein.Budget(evaluations=50), **arguments("random", "generation", tmp_path / "r"))

    async def main() -> auxein.RunResult[Any]:
        return await auxein.aresume(budget=auxein.Budget(evaluations=90), **arguments("random", "generation", tmp_path / "r"))

    assert asyncio.run(main()).evaluations_used == 90

    async def inside() -> auxein.RunResult[Any]:
        return auxein.resume(budget=auxein.Budget(evaluations=130), **arguments("random", "generation", tmp_path / "r"))

    assert asyncio.run(inside()).evaluations_used == 130


# --- checkpoints during a run ---


def test_checkpoints_are_written_on_the_evaluation_count_at_the_end_and_only_the_last_two_are_kept(tmp_path: Path):
    auxein.run(budget=auxein.Budget(evaluations=320), checkpoint_every_evaluations=64, **arguments("ga", "generation", tmp_path / "r"))
    checkpoints = peek(tmp_path / "r").checkpoints()
    assert len(checkpoints) == 2
    assert [c.evaluations_used for c in checkpoints][-1] == 320  # the end of the run, after the one taken when the budget came close
    for info in checkpoints:
        assert (tmp_path / "r" / info.path / "state.json").exists() and (tmp_path / "r" / info.path / "arrays.npz").exists()
    assert sorted(p.name for p in (tmp_path / "r" / "checkpoints").iterdir()) == sorted(Path(c.path).name for c in checkpoints)


def test_keep_checkpoints_sets_how_many_stay_and_zero_writes_none(tmp_path: Path):
    auxein.run(
        budget=auxein.Budget(evaluations=320),
        checkpoint_every_evaluations=32,
        keep_checkpoints=4,
        **arguments("ga", "generation", tmp_path / "a"),
    )
    assert len(peek(tmp_path / "a").checkpoints()) == 4
    auxein.run(budget=auxein.Budget(evaluations=100), keep_checkpoints=0, **arguments("ga", "generation", tmp_path / "b"))
    assert peek(tmp_path / "b").checkpoints() == [] and not (tmp_path / "b" / "checkpoints").exists()


def test_time_based_checkpoints_follow_the_injected_clock(tmp_path: Path):
    now = [0.0]

    def clock() -> float:
        now[0] += 1.0  # every look at the clock is a second
        return now[0]

    auxein.run(
        budget=auxein.Budget(evaluations=160),
        checkpoint_every=3.0,
        clock=clock,
        keep_checkpoints=50,
        **arguments("ga", "generation", tmp_path / "r"),
    )
    assert len(peek(tmp_path / "r").checkpoints()) >= 2


def test_a_run_that_ends_exactly_on_its_budget_checkpoints_at_the_end_and_can_be_extended_from_it(tmp_path: Path):
    auxein.run(
        budget=auxein.Budget(evaluations=160), checkpoint_every=10_000, **arguments("ga", "generation", tmp_path / "a")
    )  # 10 full batches
    ends = peek(tmp_path / "a").checkpoints()
    assert [c.evaluations_used for c in ends][-1] == 160
    resumed = auxein.resume(budget=auxein.Budget(evaluations=330), checkpoint_every=10_000, **arguments("ga", "generation", tmp_path / "a"))
    reference = auxein.run(budget=auxein.Budget(evaluations=330), checkpoint_every=10_000, **arguments("ga", "generation", tmp_path / "b"))
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")
    same_result(resumed, reference)


def test_a_damaged_newest_checkpoint_falls_back_to_the_one_before(tmp_path: Path):
    run_dir = tmp_path / "r"
    auxein.run(
        budget=auxein.Budget(evaluations=200),
        checkpoint_every_evaluations=64,
        keep_checkpoints=3,
        **arguments("ga", "steady_state", run_dir),
    )
    newest = max(peek(run_dir).checkpoints(), key=lambda c: c.id)
    (run_dir / newest.path / "state.json").write_text("not json")
    with pytest.warns(RuntimeWarning, match="unreadable checkpoint"):
        resumed = auxein.resume(budget=auxein.Budget(evaluations=300), **arguments("ga", "steady_state", run_dir))
    reference = auxein.run(budget=auxein.Budget(evaluations=300), **arguments("ga", "steady_state", tmp_path / "ref"))
    assert comparable(run_dir) == comparable(tmp_path / "ref")
    same_result(resumed, reference)


def test_warnings_stay_quiet_for_a_plain_resume(tmp_path: Path):
    auxein.run(budget=auxein.Budget(evaluations=50), **arguments("random", "generation", tmp_path / "r"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        auxein.resume(budget=auxein.Budget(evaluations=80), **arguments("random", "generation", tmp_path / "r"))


# --- backends ---

BACKENDS = [("numpy", "float32"), ("torch", "float64"), ("torch", "float32")]


@pytest.mark.parametrize("delivery", DELIVERIES)
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(("name", "precision"), BACKENDS)
def test_extension_gives_the_same_run_on_every_backend_and_precision(
    tmp_path: Path, name: str, precision: str, strategy: str, delivery: str
):
    if name == "torch":
        pytest.importorskip("torch")
    backend = auxein.Backend(name, "cpu", precision)  # type: ignore[arg-type]
    kwargs: dict[str, Any] = {"backend": backend}
    evaluator = auxein.FunctionEvaluator(sphere)
    auxein.run(budget=auxein.Budget(evaluations=130), **arguments(strategy, delivery, tmp_path / "a", evaluator, **kwargs))
    extended = auxein.resume(budget=auxein.Budget(evaluations=300), **arguments(strategy, delivery, tmp_path / "a", evaluator, **kwargs))
    reference = auxein.run(budget=auxein.Budget(evaluations=300), **arguments(strategy, delivery, tmp_path / "b", evaluator, **kwargs))
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")
    same_result(extended, reference)


def test_a_strategy_whose_ask_depends_on_the_size_asked_cannot_be_extended_in_generation_delivery_but_can_in_steady_state(tmp_path: Path):
    """`offspring_size=None` breeds exactly n children, so the cut final batch of the short run is not a prefix of the longer run's."""

    def args(delivery: str, run_dir: Path) -> dict[str, Any]:
        settings = arguments("ga", delivery, run_dir)
        settings["strategy"] = GeneticAlgorithm(population_size=16, offspring_size=None)
        return settings

    auxein.run(budget=auxein.Budget(evaluations=100), **args("generation", tmp_path / "g"))
    with pytest.raises(ReplayMismatchError, match="differently"):
        auxein.resume(budget=auxein.Budget(evaluations=300), **args("generation", tmp_path / "g"))
    auxein.run(budget=auxein.Budget(evaluations=96), **args("generation", tmp_path / "m"))  # a multiple of batch_size: nothing was cut
    auxein.resume(budget=auxein.Budget(evaluations=300), **args("generation", tmp_path / "m"))
    auxein.run(budget=auxein.Budget(evaluations=300), **args("generation", tmp_path / "ref"))
    assert comparable(tmp_path / "m") == comparable(tmp_path / "ref")
    auxein.run(budget=auxein.Budget(evaluations=100), **args("steady_state", tmp_path / "s"))
    auxein.resume(budget=auxein.Budget(evaluations=300), **args("steady_state", tmp_path / "s"))
    auxein.run(budget=auxein.Budget(evaluations=300), **args("steady_state", tmp_path / "sref"))
    assert comparable(tmp_path / "s") == comparable(tmp_path / "sref")
