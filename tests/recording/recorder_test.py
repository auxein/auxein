import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import ArrayBatch, Candidate, CandidateId, Cost, Evaluation, EvaluationBatch, ListBatch, Status
from auxein.recording import GenomeEncodingError, NoopRecorder, RunDirectoryError, SQLiteRecorder, open_run

METADATA = {"name": "test-run", "seed": 7, "batch_size": 4}


def started(path: Path, metadata: dict | None = None) -> SQLiteRecorder:
    recorder = SQLiteRecorder(path)
    recorder.on_start(metadata or METADATA)
    return recorder


def evaluations_of(batch, status: Status = Status.OK) -> EvaluationBatch:
    return EvaluationBatch(
        [Evaluation(c, status, {"value": float(c.id)}, {"cpa": 0.5}, {"speed": 2.0}, Cost(0.01, {"tokens": 3.0})) for c in batch.candidates]
    )


def array_batch(backend: Backend, n: int = 3, d: int = 2, first_id: int = 0, step: int = 0, parents=None) -> ArrayBatch:
    genomes = backend.asarray(np.arange(n * d, dtype=np.float64).reshape(n, d) / 7.0)
    return ArrayBatch(genomes, [CandidateId(first_id + i) for i in range(n)], step, "random", parents)


def test_noop_recorder_does_nothing():
    noop = NoopRecorder()
    batch = ListBatch([Candidate(CandidateId(0), "g", (), "init", 0)])
    noop.on_start({"a": 1})
    noop.on_batch(0, batch, evaluations_of(batch))
    noop.on_tell(0, 1)
    noop.on_end("completed", "budget:evaluations", {})


def test_creates_the_directory_with_metadata_and_a_versioned_schema(tmp_path: Path):
    run_dir = tmp_path / "runs" / "nested" / "run-1"
    recorder = started(run_dir)
    assert (run_dir / "metadata.json").exists() and (run_dir / "events.sqlite").exists()
    recorder.on_end("completed", "budget:evaluations", {})
    with open_run(run_dir) as run:
        assert run.schema_version == 1
    db = sqlite3.connect(run_dir / "events.sqlite")
    assert db.execute("SELECT version FROM schema_version").fetchall() == [(1,)]
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"schema_version", "candidates", "lineage", "evaluations", "events"} <= tables
    indexes = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert {"lineage_parent", "lineage_child"} <= indexes


def test_metadata_at_the_start_and_at_the_end(tmp_path: Path):
    recorder = started(tmp_path / "r")
    start = json.loads((tmp_path / "r" / "metadata.json").read_text())
    assert start["name"] == "test-run" and start["seed"] == 7 and start["batch_size"] == 4
    assert start["status"] == "running" and start["started_at"].endswith("+00:00")
    assert {"auxein", "python", "numpy"} <= set(start["versions"]) and start["versions"]["python"]
    assert "git" in start

    recorder.on_end("completed", "budget:evaluations", {"evaluations_used": 12})
    end = json.loads((tmp_path / "r" / "metadata.json").read_text())
    assert end["status"] == "completed" and end["stop_reason"] == "budget:evaluations" and end["summary"] == {"evaluations_used": 12}
    assert end["ended_at"] >= end["started_at"] and end["name"] == "test-run"
    assert not list((tmp_path / "r").glob("*.tmp"))  # written atomically: no temporary file is left


def test_torch_is_listed_in_the_versions_when_installed(tmp_path: Path):
    pytest.importorskip("torch")
    started(tmp_path / "r")
    assert json.loads((tmp_path / "r" / "metadata.json").read_text())["versions"]["torch"]


def test_a_failed_or_interrupted_status_is_recorded(tmp_path: Path):
    for status in ("failed", "interrupted"):
        recorder = started(tmp_path / status)
        recorder.on_end(status, None, {"error": "boom"})
        meta = json.loads((tmp_path / status / "metadata.json").read_text())
        assert meta["status"] == status and meta["stop_reason"] is None and meta["summary"] == {"error": "boom"}
        stop = sqlite3.connect(tmp_path / status / "events.sqlite").execute("SELECT payload FROM events WHERE kind='stop'").fetchone()
        assert json.loads(stop[0]) == {"status": status, "stop_reason": None}


def test_it_refuses_an_existing_non_empty_directory(tmp_path: Path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "something.txt").write_text("hi")
    with pytest.raises(RunDirectoryError, match="already exists and is not empty"):
        SQLiteRecorder(occupied)
    assert isinstance(RunDirectoryError("x"), FileExistsError)
    (tmp_path / "a-file").write_text("x")
    with pytest.raises(RunDirectoryError):
        SQLiteRecorder(tmp_path / "a-file")


def test_two_runs_never_mix(tmp_path: Path):
    started(tmp_path / "r").on_end("completed", "x", {})
    with pytest.raises(RunDirectoryError):
        SQLiteRecorder(tmp_path / "r")


def test_an_empty_or_missing_directory_is_fine(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    started(tmp_path / "empty").on_end("completed", "x", {})
    started(tmp_path / "missing").on_end("completed", "x", {})
    SQLiteRecorder(tmp_path / "never-started")  # creating a recorder writes nothing
    assert not (tmp_path / "never-started").exists()


def test_hooks_before_start_are_an_error(tmp_path: Path):
    recorder = SQLiteRecorder(tmp_path / "r")
    batch = array_batch(Backend())
    with pytest.raises(RuntimeError, match="not been started"):
        recorder.on_batch(0, batch, evaluations_of(batch))
    recorder.on_end("failed", None, {})  # ending a run that never started is harmless


def test_array_genomes_round_trip_in_every_backend_and_precision(tmp_path: Path, backend: Backend):
    batch = array_batch(backend, 5, 3, parents=[(), (CandidateId(0),), (), (CandidateId(1), CandidateId(2)), ()])
    recorder = started(tmp_path / "r")
    recorder.on_batch(0, batch, evaluations_of(batch))
    recorder.on_end("completed", "x", {})
    expected = backend.to_numpy(batch.as_array())
    with open_run(tmp_path / "r") as run:
        recorded = list(run.evaluations())
    assert [r.candidate_id for r in recorded] == [0, 1, 2, 3, 4]
    for i, r in enumerate(recorded):
        assert isinstance(r.genome, np.ndarray) and r.genome.dtype == np.dtype(backend.precision) and r.genome.shape == (3,)
        np.testing.assert_array_equal(r.genome, expected[i])  # exactly: raw bytes, no pickle
        assert r.step == 0 and r.origin == "random"
    assert recorded[3].parents == (1, 2) and recorded[1].parents == (0,) and recorded[0].parents == ()


def test_genomes_are_stored_as_raw_bytes(tmp_path: Path):
    batch = array_batch(Backend("numpy", "cpu", "float32"), 2, 2)
    recorder = started(tmp_path / "r")
    recorder.on_batch(0, batch, evaluations_of(batch))
    recorder.on_end("completed", "x", {})
    row = (
        sqlite3.connect(tmp_path / "r" / "events.sqlite")
        .execute("SELECT genome_kind, genome, genome_dtype, genome_shape FROM candidates WHERE id=1")
        .fetchone()
    )
    assert row[0] == "array" and row[2] == "float32" and json.loads(row[3]) == [2]
    assert row[1] == batch.as_array()[1].tobytes()


def test_json_genomes_round_trip(tmp_path: Path):
    genomes = [{"instructions": ["be brief", "cite sources"], "temperature": 0.5, "tools": {"search": True}}, [1, 2, 3], "text", 3.5, None]
    batch = ListBatch([Candidate(CandidateId(i), g, (), "init", 0) for i, g in enumerate(genomes)])
    recorder = started(tmp_path / "r")
    recorder.on_batch(0, batch, evaluations_of(batch))
    recorder.on_end("completed", "x", {})
    with open_run(tmp_path / "r") as run:
        assert [r.genome for r in run.evaluations()] == genomes


def test_a_genome_that_cannot_be_stored_is_a_clear_error_and_writes_nothing(tmp_path: Path):
    batch = ListBatch([Candidate(CandidateId(0), {1, 2}, (), "init", 0), Candidate(CandidateId(1), "ok", (), "init", 0)])
    recorder = started(tmp_path / "r")
    with pytest.raises(GenomeEncodingError, match=r"set cannot be recorded.*Structured genomes get proper storage in a later step"):
        recorder.on_batch(0, batch, evaluations_of(batch))
    assert sqlite3.connect(tmp_path / "r" / "events.sqlite").execute("SELECT COUNT(*) FROM candidates").fetchone() == (0,)
    assert issubclass(GenomeEncodingError, TypeError)


def test_evaluations_are_recorded_with_all_their_fields(tmp_path: Path):
    ok = Candidate(CandidateId(0), [1.0], (), "random", 0)
    failed = Candidate(CandidateId(1), [2.0], (), "random", 0)
    batch = ListBatch([ok, failed])
    results = EvaluationBatch(
        [
            Evaluation(ok, Status.OK, {"a": 1.5, "b": -2.0}, {"cpa": 0.0}, {"speed": 3.0}, Cost(0.25, {"tokens": 10.0, "money": 0.5})),
            Evaluation(failed, Status.FAILED, {"a": float("nan"), "b": float("inf")}, cost=Cost(0.1), error="simulator crashed"),
        ]
    )
    recorder = started(tmp_path / "r")
    recorder.on_batch(2, batch, results)
    recorder.on_end("completed", "x", {})
    with open_run(tmp_path / "r") as run:
        first, second = list(run.evaluations())
    assert first.status is Status.OK and first.objectives == {"a": 1.5, "b": -2.0} and first.constraints == {"cpa": 0.0}
    assert (
        first.descriptors == {"speed": 3.0}
        and first.cost_units == {"tokens": 10.0, "money": 0.5}
        and first.wall_time == 0.25
        and first.error is None
    )
    assert second.status is Status.FAILED and np.isnan(second.objectives["a"]) and second.objectives["b"] == float("inf")
    assert second.error == "simulator crashed" and second.wall_time == 0.1


def test_events_log_asks_tells_and_the_stop(tmp_path: Path):
    batch = array_batch(Backend(), 3)
    recorder = started(tmp_path / "r")
    recorder.on_batch(0, batch, evaluations_of(batch))
    recorder.on_tell(0, 3)
    recorder.on_end("completed", "strategy", {})
    rows = sqlite3.connect(tmp_path / "r" / "events.sqlite").execute("SELECT seq, kind, step, payload FROM events ORDER BY seq").fetchall()
    assert [(r[1], r[2]) for r in rows] == [("ask", 0), ("tell", 0), ("stop", None)]
    assert [r[0] for r in rows] == [1, 2, 3]
    assert json.loads(rows[0][3]) == {"count": 3, "first_id": 0, "last_id": 2} and json.loads(rows[1][3]) == {"count": 3}
    assert json.loads(rows[2][3]) == {"status": "completed", "stop_reason": "strategy"}


def hand_built_lineage(tmp_path: Path) -> Path:
    """0, 1 are roots; 2 = (0, 1); 3 = (2); 4 = (2, 1); 5 = (4)."""
    parents = {0: (), 1: (), 2: (0, 1), 3: (2,), 4: (2, 1), 5: (4,)}
    candidates = [
        Candidate(CandidateId(i), [float(i)], tuple(CandidateId(p) for p in ps), "crossover" if ps else "init", 0)
        for i, ps in parents.items()
    ]
    batch = ListBatch(candidates)
    recorder = started(tmp_path / "lineage")
    recorder.on_batch(0, batch, evaluations_of(batch))
    recorder.on_end("completed", "x", {})
    return tmp_path / "lineage"


def test_ancestry_and_descendants_follow_the_lineage(tmp_path: Path):
    with open_run(hand_built_lineage(tmp_path)) as run:
        assert run.ancestry(5) == [4, 1, 2, 0]  # nearest first, then by id
        assert run.ancestry(3) == [2, 0, 1]
        assert run.ancestry(2) == [0, 1]
        assert run.ancestry(0) == [] and run.ancestry(99) == []
        assert run.descendants(1) == [2, 4, 3, 5]
        assert run.descendants(0) == [2, 3, 4, 5]
        assert run.descendants(2) == [3, 4, 5]
        assert run.descendants(5) == [] and run.descendants(99) == []
        assert [r.parents for r in run.evaluations()][2:] == [(0, 1), (2,), (2, 1), (4,)]


def test_the_reader_rejects_other_directories_and_schema_versions(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="not a recorded run"):
        open_run(tmp_path)
    run_dir = hand_built_lineage(tmp_path)
    db = sqlite3.connect(run_dir / "events.sqlite")
    db.execute("UPDATE schema_version SET version = 2")
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="unsupported run schema version 2"):
        open_run(run_dir)


def test_the_reader_exposes_the_metadata(tmp_path: Path):
    with open_run(hand_built_lineage(tmp_path)) as run:
        assert run.metadata["name"] == "test-run" and run.metadata["status"] == "completed"
