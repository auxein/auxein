"""The content-addressed genome store, in `events.sqlite` (design doc §10.3)."""

import hashlib
import json
import os
import sqlite3
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import auxein
from auxein.recording import open_run
from auxein.recording.genomes import genome_hash
from auxein.spaces import SequenceSpace
from tests.driver.resume_modes_test import forget_checkpoints_after, rows_after
from tests.driver.resume_test import comparable, pretend_killed
from tests.support.fakes import ScriptedStrategy
from tests.support.reading import peek

LONG = tuple(f"instruction-number-{i:03d}" for i in range(250))  # about 6 KB encoded
SPACE = SequenceSpace(LONG, 1, 300)
_ = (forget_checkpoints_after, rows_after, pretend_killed, os)


def run_scripted(run_dir: Path, genomes, space, *, threshold: Any = 4096, evaluations: int = 40, **kw: Any):
    warnings.simplefilter("ignore", auxein.RecordingDisabledWarning)
    return auxein.run(
        strategy=ScriptedStrategy(genomes),
        evaluator=auxein.FunctionEvaluator(lambda g: float(len(g))),
        space=space,
        budget=auxein.Budget(evaluations=evaluations),
        seed=1,
        batch_size=10,
        run_dir=run_dir,
        genome_store_threshold=threshold,
        **{"keep_checkpoints": 0, **kw},
    )


def db_of(run_dir: Path) -> sqlite3.Connection:
    return sqlite3.connect(run_dir / "events.sqlite")


def test_large_structured_genomes_are_stored_once_by_hash_however_many_candidates_share_them(tmp_path: Path):
    shared = lambda i: LONG if i % 4 else LONG[:200]  # noqa: E731  - two distinct large genomes, repeated
    run_scripted(tmp_path / "r", shared, SPACE)
    db = db_of(tmp_path / "r")
    assert db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 40
    assert db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 2  # two genomes, however many candidates
    assert db.execute("SELECT COUNT(*) FROM candidates WHERE genome_hash IS NOT NULL").fetchone()[0] == 40
    assert db.execute("SELECT COUNT(*) FROM candidates WHERE LENGTH(genome) > 0").fetchone()[0] == 0  # nothing inline
    (digest, data) = db.execute("SELECT hash, data FROM blobs ORDER BY LENGTH(data) DESC").fetchone()
    db.close()
    assert digest == hashlib.sha256(data).hexdigest() == genome_hash(data)
    assert json.loads(data) == list(LONG)  # the canonical encoding of the genome


def test_large_array_genomes_go_into_the_store_too(tmp_path: Path):
    genome = np.arange(1000, dtype=np.float64)  # 8 KB
    run_scripted(tmp_path / "r", lambda i: genome if i % 2 else genome + 1, auxein.Box(0.0, 1e6, dim=1000))
    db = db_of(tmp_path / "r")
    assert db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 2
    assert db.execute("SELECT genome_dtype FROM candidates WHERE id = 0").fetchone() == ("float64",)  # dtype and shape stay per candidate
    db.close()
    with open_run(tmp_path / "r") as run:
        first, second = list(run.evaluations())[:2]
        np.testing.assert_array_equal(first.genome, genome + 1)  # type: ignore[arg-type]
        np.testing.assert_array_equal(second.genome, genome)  # type: ignore[arg-type]


def test_small_genomes_stay_inline_and_the_threshold_is_configurable(tmp_path: Path):
    small = ("instruction-number-000", "instruction-number-001")
    run_scripted(tmp_path / "small", lambda i: small, SPACE)
    db = db_of(tmp_path / "small")
    assert (
        db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 0
        and db.execute("SELECT COUNT(*) FROM candidates WHERE genome_hash IS NULL").fetchone()[0] == 40
    )
    db.close()
    run_scripted(tmp_path / "tiny-threshold", lambda i: small, SPACE, threshold=10)
    db = db_of(tmp_path / "tiny-threshold")
    assert db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 1
    db.close()
    run_scripted(tmp_path / "off", lambda i: LONG, SPACE, threshold=None)  # the store can be switched off
    db = db_of(tmp_path / "off")
    assert (
        db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 0
        and db.execute("SELECT LENGTH(genome) FROM candidates WHERE id = 0").fetchone()[0] > 4096
    )
    db.close()
    with pytest.raises(ValueError, match="threshold"):
        run_scripted(tmp_path / "bad", lambda i: small, SPACE, threshold=-1)


def test_the_reader_resolves_references_transparently_with_and_without_the_store(tmp_path: Path):
    run_scripted(tmp_path / "with", lambda i: LONG[: 200 + i], SPACE)
    run_scripted(tmp_path / "without", lambda i: LONG[: 200 + i], SPACE, threshold=None)
    with open_run(tmp_path / "with", SPACE.codec) as a, open_run(tmp_path / "without", SPACE.codec) as b:
        with_store = [(e.candidate_id, e.genome) for e in a.evaluations()]
        without = [(e.candidate_id, e.genome) for e in b.evaluations()]
        stats = a.genome_store()
    assert with_store == without and with_store[7][1] == LONG[:207]
    assert stats.blobs == 40 and stats.references == 40 and stats.bytes > 40 * 4096


def test_replay_still_checks_genomes_byte_for_byte_through_the_store(tmp_path: Path):
    from auxein.driver import ReplayMismatchError

    auxein.run(
        strategy=auxein.StructuredGeneticAlgorithm(population_size=12, offspring_size=12),
        evaluator=auxein.FunctionEvaluator(lambda g: float(len(g))),
        space=SPACE,
        budget=auxein.Budget(evaluations=60),
        seed=2,
        batch_size=12,
        run_dir=tmp_path / "r",
        genome_store_threshold=64,  # almost every genome goes to the store
        keep_checkpoints=0,
    )
    assert db_of(tmp_path / "r").execute("SELECT COUNT(*) FROM blobs").fetchone()[0] > 10
    extended = auxein.resume(
        strategy=auxein.StructuredGeneticAlgorithm(population_size=12, offspring_size=12),
        evaluator=auxein.FunctionEvaluator(lambda g: float(len(g))),
        space=SPACE,
        budget=auxein.Budget(evaluations=120),
        seed=2,
        batch_size=12,
        run_dir=tmp_path / "r",
        genome_store_threshold=64,
        keep_checkpoints=0,
    )
    assert extended.evaluations_used == 120
    db = db_of(tmp_path / "r")
    (blob_hash,) = db.execute("SELECT genome_hash FROM candidates WHERE id = 3").fetchone()
    db.execute("UPDATE blobs SET data = ? WHERE hash = ?", (b'["instruction-number-001"]', blob_hash))  # tamper with the stored genome
    db.commit()
    db.close()
    with pytest.raises(ReplayMismatchError, match="its genome"):
        auxein.resume(
            strategy=auxein.StructuredGeneticAlgorithm(population_size=12, offspring_size=12),
            evaluator=auxein.FunctionEvaluator(lambda g: float(len(g))),
            space=SPACE,
            budget=auxein.Budget(evaluations=180),
            seed=2,
            batch_size=12,
            run_dir=tmp_path / "r",
            genome_store_threshold=64,
            keep_checkpoints=0,
        )


def test_throughput_truncation_removes_the_blobs_nobody_references_any_more(tmp_path: Path):
    kw: dict[str, Any] = {}
    unique = lambda i: tuple(LONG[(i + j) % 250] for j in range(200))  # noqa: E731  - every candidate has its own large genome
    run_scripted(tmp_path / "r", unique, SPACE, evaluations=40, checkpoint_every_evaluations=10, keep_checkpoints=20, **kw)
    before = db_of(tmp_path / "r").execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
    assert before == 40
    seq = forget_checkpoints_after(tmp_path / "r", 2)
    pretend_killed(tmp_path / "r")
    doomed = len(rows_after(tmp_path / "r", seq))
    assert 0 < doomed < 40
    from auxein.recording import SQLiteRecorder

    recorder = SQLiteRecorder(tmp_path / "r", resume=True)
    recorder.open_existing()
    assert recorder.truncate_after(seq) == doomed
    recorder.abandon()
    db = db_of(tmp_path / "r")
    kept = db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    assert kept == 40 - doomed and db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == kept  # the orphaned genomes went with them
    assert db.execute("SELECT COUNT(*) FROM blobs WHERE hash NOT IN (SELECT genome_hash FROM candidates)").fetchone()[0] == 0
    db.close()


def test_a_blob_shared_with_a_surviving_candidate_is_kept_on_truncation(tmp_path: Path):
    shared = lambda i: LONG  # noqa: E731
    run_scripted(tmp_path / "r", shared, SPACE, evaluations=40, checkpoint_every_evaluations=10, keep_checkpoints=20)
    seq = forget_checkpoints_after(tmp_path / "r", 2)
    from auxein.recording import SQLiteRecorder

    recorder = SQLiteRecorder(tmp_path / "r", resume=True)
    recorder.open_existing()
    recorder.truncate_after(seq)
    recorder.abandon()
    assert db_of(tmp_path / "r").execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 1


def test_the_store_makes_a_run_with_large_repetitive_genomes_much_smaller(tmp_path: Path):
    repetitive = lambda i: tuple(LONG[(i % 5 + j) % 250] for j in range(200))  # noqa: E731  - five distinct genomes, 200 candidates
    run_scripted(tmp_path / "with", repetitive, SPACE, evaluations=200)
    run_scripted(tmp_path / "without", repetitive, SPACE, evaluations=200, threshold=None)
    with_store = os.path.getsize(tmp_path / "with" / "events.sqlite")
    without = os.path.getsize(tmp_path / "without" / "events.sqlite")
    assert with_store * 5 < without
    assert comparable(tmp_path / "with")["evaluations"] == comparable(tmp_path / "without")["evaluations"]
    _ = peek
