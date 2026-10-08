"""`SQLiteRecorder`: the run directory (design doc §10).

runs/<name>/
  metadata.json   configuration, environment, and on finish the status and summary (written atomically)
  events.sqlite   candidates, lineage, evaluations and events (the driver is the single writer)
"""

import json
import os
import sqlite3
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from auxein.backend import Backend
from auxein.core import Batch, EvaluationBatch
from auxein.recording import environment
from auxein.recording.genomes import EncodedGenome, encode_array, encode_genome

G = TypeVar("G")

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
CREATE TABLE candidates (
    id INTEGER PRIMARY KEY,
    step INTEGER NOT NULL,
    origin TEXT NOT NULL,
    genome_kind TEXT NOT NULL,      -- 'array' (raw bytes) or 'json'
    genome BLOB NOT NULL,
    genome_dtype TEXT,              -- arrays only
    genome_shape TEXT,              -- arrays only: a JSON list
    created_at REAL NOT NULL
);
CREATE TABLE lineage (
    parent_id INTEGER NOT NULL,
    child_id INTEGER NOT NULL
);
CREATE INDEX lineage_parent ON lineage (parent_id);
CREATE INDEX lineage_child ON lineage (child_id);
CREATE TABLE evaluations (
    candidate_id INTEGER PRIMARY KEY REFERENCES candidates (id),
    status TEXT NOT NULL,
    objectives TEXT NOT NULL,       -- JSON objects: name -> value
    constraints TEXT NOT NULL,
    descriptors TEXT NOT NULL,
    cost_units TEXT NOT NULL,
    wall_time REAL NOT NULL,
    error TEXT,
    finished_at REAL NOT NULL
);
CREATE TABLE events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,             -- 'ask', 'tell' or 'stop'
    step INTEGER,
    payload TEXT NOT NULL           -- JSON
);
"""


class RunDirectoryError(FileExistsError):
    """The run directory already holds something: runs never mix."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class SQLiteRecorder:
    """Writes a run to a directory: `metadata.json` and `events.sqlite`.

    The directory must not exist or must be empty, so that runs never mix; this is checked when the recorder is
    created, before anything runs. Each batch is written in one transaction. Genomes are stored without pickle: array
    genomes as raw bytes with their dtype and shape, other genomes as JSON (a genome that is neither is an error).
    Read a recorded run back with `auxein.recording.open_run`.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        if self.run_dir.exists() and (not self.run_dir.is_dir() or any(self.run_dir.iterdir())):
            raise RunDirectoryError(
                f"the run directory {self.run_dir} already exists and is not empty: runs never mix, so choose a new directory"
            )
        self._db: sqlite3.Connection | None = None
        self._metadata: dict[str, object] = {}

    def _connection(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError("the recorder has not been started (or the run has already ended)")
        return self._db

    def _write_metadata(self) -> None:
        path = self.run_dir / "metadata.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._metadata, indent=2, default=str) + "\n")
        os.replace(temporary, path)

    def on_start(self, metadata: Mapping[str, object]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metadata = {
            **metadata,
            "versions": environment.versions(),
            "git": environment.git_state(),
            "started_at": _now(),
            "status": "running",
        }
        self._write_metadata()
        db = sqlite3.connect(self.run_dir / "events.sqlite")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        with db:
            db.executescript(SCHEMA)
            db.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        self._db = db

    def on_batch(self, step: int, batch: Batch[G], results: EvaluationBatch[G]) -> None:
        db = self._connection()
        candidates = batch.candidates
        genomes = batch.as_array()
        if genomes is not None:  # one device-to-host copy for the whole batch
            host = Backend().to_numpy(genomes)
            encoded: list[EncodedGenome] = [encode_array(host[i]) for i in range(len(candidates))]
        else:
            encoded = [encode_genome(c.genome) for c in candidates]

        now = time.time()
        candidate_rows = [(c.id, c.step, c.origin, e.kind, e.data, e.dtype, e.shape, now) for c, e in zip(candidates, encoded, strict=True)]
        lineage_rows = [(parent, c.id) for c in candidates for parent in c.parents]
        evaluation_rows = [
            (
                e.candidate.id,
                e.status.value,
                json.dumps(dict(e.objectives)),
                json.dumps(dict(e.constraints)),
                json.dumps(dict(e.descriptors)),
                json.dumps(dict(e.cost.units)),
                e.cost.wall_time,
                e.error,
                now,
            )
            for e in results
        ]
        payload = json.dumps(
            {"count": len(candidates), "first_id": candidates[0].id, "last_id": candidates[-1].id} if candidates else {"count": 0}
        )
        with db:
            db.executemany("INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?)", candidate_rows)
            db.executemany("INSERT INTO lineage VALUES (?, ?)", lineage_rows)
            db.executemany("INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", evaluation_rows)
            db.execute("INSERT INTO events (kind, step, payload) VALUES ('ask', ?, ?)", (step, payload))

    def on_tell(self, step: int, count: int) -> None:
        db = self._connection()
        with db:
            db.execute("INSERT INTO events (kind, step, payload) VALUES ('tell', ?, ?)", (step, json.dumps({"count": count})))

    def on_end(self, status: str, stop_reason: str | None, summary: Mapping[str, object]) -> None:
        db = self._db
        if db is None:  # never started: nothing to finalise
            return
        with db:
            db.execute(
                "INSERT INTO events (kind, step, payload) VALUES ('stop', NULL, ?)",
                (json.dumps({"status": status, "stop_reason": stop_reason}),),
            )
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.close()
        self._db = None
        self._metadata.update({"ended_at": _now(), "status": status, "stop_reason": stop_reason, "summary": dict(summary)})
        self._write_metadata()
