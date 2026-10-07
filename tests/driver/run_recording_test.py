import json
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import Objective, Result, Status
from auxein.driver import Budget, RecordingDisabledWarning, run
from auxein.evaluators import EvaluationError, FunctionEvaluator, VectorisedEvaluator
from auxein.recording import RunDirectoryError, open_run
from auxein.spaces import Box
from auxein.strategies import RandomSearch
from tests.support.fakes import ScriptedStrategy


def sphere(g):
    return float((g * g).sum())


def record(path: Path, *, strategy=None, evaluator=None, seed=3, backend=None, budget=None, batch_size=16, **kwargs):
    return run(
        strategy=strategy or RandomSearch(),
        evaluator=evaluator or FunctionEvaluator(sphere),
        space=kwargs.pop("space", Box(-5.0, 5.0, dim=3)),
        budget=budget or Budget(evaluations=50),
        seed=seed,
        backend=backend,
        batch_size=batch_size,
        run_dir=path,
        **kwargs,
    )


def test_a_recorded_run_has_every_candidate_and_evaluation(tmp_path: Path, backend: Backend):
    result = record(tmp_path / "r", backend=backend)
    assert result.run_dir == tmp_path / "r"
    with open_run(tmp_path / "r") as recorded:
        evaluations = list(recorded.evaluations())
    assert [e.candidate_id for e in evaluations] == list(range(50))
    assert all(e.status is Status.OK and e.origin == "random" and e.parents == () for e in evaluations)
    assert {e.step for e in evaluations} == {0, 1, 2, 3}  # 16 + 16 + 16 + 2
    for e in evaluations:
        genome = np.asarray(e.genome)
        assert genome.dtype == np.dtype(backend.precision)
        assert e.objectives["value"] == pytest.approx(float((genome.astype(np.float64) ** 2).sum()), rel=1e-5)
    assert result.best is not None and result.best.objectives["value"] == min(e.objectives["value"] for e in evaluations)


def test_the_metadata_describes_the_run(tmp_path: Path):
    record(
        tmp_path / "my-run",
        seed=17,
        budget=Budget(evaluations=30, wall_time=60.0, cost={"tokens": 5.0}),
        batch_size=8,
        backend=Backend("numpy", "cpu", "float32"),
    )
    with open_run(tmp_path / "my-run") as recorded:
        meta = recorded.metadata
    assert meta["name"] == "my-run"  # the name defaults to the directory name
    assert meta["seed"] == 17 and meta["batch_size"] == 8 and meta["precision"] == "float32"
    assert meta["backend"] == {"name": "numpy", "device": "cpu"}
    assert meta["budget"] == {"evaluations": 30, "wall_time": 60.0, "cost": {"tokens": 5.0}}
    assert meta["problem"] == {
        "space": {"type": "Box", "dim": 3, "lower": [-5.0] * 3, "upper": [5.0] * 3, "log_scale": [False] * 3},
        "objectives": [{"name": "value", "direction": "minimise"}],
        "constraints": [],
        "descriptors": [],
    }
    assert meta["strategy"] == {"class": "RandomSearch", "module": "auxein.strategies.random_search", "repr": "RandomSearch()"}
    assert meta["evaluator"]["class"] == "FunctionEvaluator" and "sphere" in meta["evaluator"]["repr"]
    assert meta["status"] == "completed" and meta["stop_reason"] == "budget:evaluations" and meta["started_at"] <= meta["ended_at"]
    assert meta["summary"]["evaluations_used"] == 30 and meta["summary"]["best_candidate_id"] is not None
    assert meta["summary"]["pareto_size"] == 1 and "value" in meta["summary"]["best_objectives"]
    assert {"auxein", "python", "numpy"} <= set(meta["versions"]) and "git" in meta


def test_an_explicit_name_wins_and_declared_names_are_in_the_metadata(tmp_path: Path):
    record(
        tmp_path / "dir", name="friendly", objectives=[Objective("loss"), Objective("score", "maximise")], constraints=["cpa"], descriptors=["speed"],
        evaluator=FunctionEvaluator(lambda g: Result({"loss": sphere(g), "score": -sphere(g)}, {"cpa": 0.0}, {"speed": 1.0})),
    )  # fmt: skip
    with open_run(tmp_path / "dir") as recorded:
        meta = recorded.metadata
        first = next(iter(recorded.evaluations()))
    assert meta["name"] == "friendly"
    assert meta["problem"]["objectives"] == [{"name": "loss", "direction": "minimise"}, {"name": "score", "direction": "maximise"}]  # type: ignore[index]
    assert meta["problem"]["constraints"] == ["cpa"] and meta["problem"]["descriptors"] == ["speed"]  # type: ignore[index]
    assert set(first.objectives) == {"loss", "score"} and first.constraints == {"cpa": 0.0} and first.descriptors == {"speed": 1.0}


def test_a_failing_run_is_recorded_as_failed_and_the_error_propagates(tmp_path: Path):
    def fn(genome):
        if fn.calls >= 20:
            raise RuntimeError("simulator crashed")
        fn.calls += 1
        return sphere(genome)

    fn.calls = 0  # type: ignore[attr-defined]
    with pytest.raises(EvaluationError, match="simulator crashed"):
        record(tmp_path / "r", evaluator=FunctionEvaluator(fn), budget=Budget(evaluations=100), batch_size=8)
    meta = json.loads((tmp_path / "r" / "metadata.json").read_text())
    assert meta["status"] == "failed" and meta["stop_reason"] is None and meta["summary"]["evaluations_used"] == 16
    with open_run(tmp_path / "r") as recorded:
        assert len(list(recorded.evaluations())) == 16  # the batches that completed are kept


def test_an_interrupted_run_is_recorded_as_interrupted(tmp_path: Path):
    def fn(genome):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        record(tmp_path / "r", evaluator=FunctionEvaluator(fn))
    assert json.loads((tmp_path / "r" / "metadata.json").read_text())["status"] == "interrupted"


def test_a_strategy_that_rejects_the_problem_is_recorded_as_failed(tmp_path: Path):
    with pytest.raises(ValueError, match="no thanks"):
        record(tmp_path / "r", strategy=ScriptedStrategy(reject="no thanks"))
    assert json.loads((tmp_path / "r" / "metadata.json").read_text())["status"] == "failed"


def test_a_non_empty_run_dir_is_refused_before_anything_runs(tmp_path: Path):
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "file").write_text("x")
    strategy = ScriptedStrategy()
    with pytest.raises(RunDirectoryError):
        record(tmp_path / "taken", strategy=strategy)
    assert strategy.bound is None and strategy.asks == 0


def test_two_runs_cannot_share_a_directory(tmp_path: Path):
    record(tmp_path / "r")
    with pytest.raises(RunDirectoryError):
        record(tmp_path / "r")


def test_a_run_without_run_dir_writes_nothing_to_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RecordingDisabledWarning)
        run(
            strategy=RandomSearch(), evaluator=FunctionEvaluator(sphere), space=Box(-5.0, 5.0, dim=3), budget=Budget(evaluations=20), seed=1
        )
    assert list(tmp_path.iterdir()) == []


# --- determinism ---

TIMING_COLUMNS = {"created_at", "finished_at", "wall_time"}


def events_content(path: Path) -> dict[str, list[tuple]]:
    """Everything in events.sqlite except the timestamps and the measured wall times."""
    db = sqlite3.connect(path / "events.sqlite")
    content = {}
    for table in ("candidates", "lineage", "evaluations", "events"):
        columns = [c[1] for c in db.execute(f"PRAGMA table_info({table})") if c[1] not in TIMING_COLUMNS]
        content[table] = db.execute(f"SELECT {', '.join(columns)} FROM {table} ORDER BY 1, 2").fetchall()
    return content


def test_the_same_seed_gives_identical_evaluations_and_event_logs(tmp_path: Path, backend: Backend):
    first = record(tmp_path / "a", seed=5, backend=backend, budget=Budget(evaluations=100))
    second = record(tmp_path / "b", seed=5, backend=backend, budget=Budget(evaluations=100))
    assert events_content(tmp_path / "a") == events_content(tmp_path / "b")
    assert first.best is not None and second.best is not None
    assert first.best.candidate.id == second.best.candidate.id and first.best.objectives == second.best.objectives
    assert first.trace == second.trace and first.evaluations_used == second.evaluations_used


def test_different_seeds_differ(tmp_path: Path, backend: Backend):
    record(tmp_path / "a", seed=5, backend=backend)
    record(tmp_path / "b", seed=6, backend=backend)
    assert events_content(tmp_path / "a")["candidates"] != events_content(tmp_path / "b")["candidates"]
    assert events_content(tmp_path / "a")["evaluations"] != events_content(tmp_path / "b")["evaluations"]


def test_the_same_seed_with_a_random_evaluator_is_reproducible(tmp_path: Path, backend: Backend):
    def noisy(genome, rng):
        return sphere(genome) + float(backend.to_numpy(rng.normal(1))[0])

    def noisy_run(name: str, seed: int):
        record(tmp_path / name, seed=seed, backend=backend, evaluator=FunctionEvaluator(noisy, uses_rng=True))
        return events_content(tmp_path / name)

    assert noisy_run("a", 4) == noisy_run("b", 4)
    assert noisy_run("c", 5)["evaluations"] != noisy_run("d", 4)["evaluations"]


def test_vectorised_runs_are_reproducible_and_their_batch_streams_follow_the_first_id(tmp_path: Path, backend: Backend):
    def fn(X, rng):
        return (X * X).sum(axis=1) + rng.normal(X.shape[0])

    def vector_run(name: str):
        record(tmp_path / name, backend=backend, evaluator=VectorisedEvaluator(fn, uses_rng=True), batch_size=10)
        return events_content(tmp_path / name)

    assert vector_run("a") == vector_run("b")


def test_the_event_log_does_not_depend_on_whether_the_evaluator_is_a_function_or_vectorised(tmp_path: Path):
    record(tmp_path / "f", evaluator=FunctionEvaluator(sphere))
    record(tmp_path / "v", evaluator=VectorisedEvaluator(lambda X: (X * X).sum(axis=1)))
    a, b = events_content(tmp_path / "f"), events_content(tmp_path / "v")
    assert a["candidates"] == b["candidates"] and a["events"] == b["events"]
    values = [json.loads(r[2])["value"] for r in a["evaluations"]]
    assert values == pytest.approx([json.loads(r[2])["value"] for r in b["evaluations"]])
