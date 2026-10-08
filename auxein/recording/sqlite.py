"""`SQLiteRecorder`: the run directory (design doc §10).

runs/<name>/
  metadata.json   configuration, environment, the sessions of the run, and on finish the status and summary (atomic)
  events.sqlite   candidates, lineage, evaluations, events and checkpoints (the driver is the single writer)
  checkpoints/    ckpt-<event seq>/state.json and arrays.npz, to resume from
  writer.lock     the process id of the writer, while there is one
"""

import json
import os
import sqlite3
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar, cast

from auxein.backend import Backend
from auxein.core import Batch, EpisodeRecords, EvaluationBatch, StateDict, Status
from auxein.recording import checkpoints, environment
from auxein.recording.genomes import EncodedGenome, encode_batch
from auxein.recording.lock import RunLock, RunLockedError

G = TypeVar("G")

SCHEMA_VERSION = 3

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
    created_at REAL NOT NULL,
    event_seq INTEGER NOT NULL      -- the 'ask' event whose transaction recorded the candidate and its evaluation
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
    kind TEXT NOT NULL,             -- 'ask', 'tell', 'stop' or 'resume'
    step INTEGER,
    payload TEXT NOT NULL           -- JSON
);
CREATE TABLE episodes (
    candidate_id INTEGER NOT NULL REFERENCES candidates (id),
    scenario_index INTEGER NOT NULL,
    scenario_id TEXT NOT NULL,
    status TEXT NOT NULL,
    measurements TEXT NOT NULL,     -- JSON object: measurement name -> value (empty for an episode that did not succeed)
    error TEXT,
    PRIMARY KEY (candidate_id, scenario_index)
) WITHOUT ROWID;
CREATE TABLE checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_seq INTEGER NOT NULL,     -- the checkpoint is consistent with the events up to this one
    evaluations_used INTEGER NOT NULL,
    path TEXT NOT NULL,             -- relative to the run directory
    created_at REAL NOT NULL
);
"""


class RunDirectoryError(FileExistsError):
    """The run directory already holds something: runs never mix."""


class ResumeError(RuntimeError):
    """A run cannot be resumed: it is not a run, its schema is not the current one, or it is being written by another process."""


@dataclass(frozen=True)
class CheckpointInfo:
    """A row of the `checkpoints` table."""

    id: int
    event_seq: int
    evaluations_used: int
    path: str
    created_at: float


@dataclass(frozen=True)
class ExistingRun:
    """What a recorder found when it opened a recorded run to resume it."""

    metadata: dict[str, object]
    checkpoints: tuple[CheckpointInfo, ...]
    """Newest first."""
    last_seq: int
    evaluations_recorded: int


def _episode_rows(episodes: EpisodeRecords) -> list[tuple[int, int, str, str, str, str | None]]:
    """The rows of the `episodes` table for the episodes of a batch: one per candidate and scenario, in that order."""
    rows: list[tuple[int, int, str, str, str, str | None]] = []
    names = episodes.names
    for row, candidate_id in enumerate(episodes.candidate_ids):
        values = episodes.values[row].tolist()
        for scenario, scenario_id in enumerate(episodes.scenario_ids):
            failure = episodes.failures.get((row, scenario))
            if failure is None:
                rows.append(
                    (
                        candidate_id,
                        scenario,
                        scenario_id,
                        Status.OK.value,
                        json.dumps(dict(zip(names, values[scenario], strict=True))),
                        None,
                    )
                )
            else:
                rows.append((candidate_id, scenario, scenario_id, failure[0].value, "{}", failure[1]))
    return rows


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class SQLiteRecorder:
    """Writes a run to a directory: `metadata.json`, `events.sqlite` and its checkpoints.

    A new run needs a directory that does not exist or is empty, so that runs never mix; this is checked when the recorder
    is created, before anything runs. With `resume=True` the directory must hold a recorded run, which `open_existing`
    opens. Each batch is written in one transaction. Genomes are stored without pickle: array genomes as raw bytes with
    their dtype and shape, other genomes as JSON (a genome that is neither is an error). While it writes, the recorder
    holds the run's lock file, so that no second process can. Read a recorded run back with `auxein.recording.open_run`.
    """

    def __init__(self, run_dir: str | Path, *, resume: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.resume = resume
        if resume:
            if not (self.run_dir / "events.sqlite").exists() or not (self.run_dir / "metadata.json").exists():
                raise ResumeError(f"{self.run_dir} is not a recorded run: it has no events.sqlite and metadata.json to resume")
        elif self.run_dir.exists() and (not self.run_dir.is_dir() or any(self.run_dir.iterdir())):
            raise RunDirectoryError(
                f"the run directory {self.run_dir} already exists and is not empty: runs never mix, so choose a new directory "
                "(to continue an earlier run, use auxein.resume)"
            )
        self._db: sqlite3.Connection | None = None
        self._metadata: dict[str, object] = {}
        self._lock = RunLock(self.run_dir)
        self._last_seq = 0
        self._replayed_ask = 0
        self._pending_session: tuple[str, object, dict[str, object]] | None = None

    # --- plumbing ---

    def _connection(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError("the recorder has not been started (or the run has already ended)")
        return self._db

    def _write_metadata(self) -> None:
        path = self.run_dir / "metadata.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._metadata, indent=2, default=str) + "\n")
        os.replace(temporary, path)

    @property
    def last_seq(self) -> int:
        """The sequence number of the last event written (or, while replaying, matched): what a checkpoint is consistent with."""
        return self._last_seq

    def _sessions(self) -> list[dict[str, object]]:
        return cast("list[dict[str, object]]", self._metadata.setdefault("sessions", []))

    def _new_session(self, mode: str, budget: object) -> dict[str, object]:
        return {
            "started_at": _now(),
            "mode": mode,
            "budget": budget,
            "versions": environment.versions(),
            "git": environment.git_state(),
            "status": "running",
        }

    # --- a new run ---

    def on_start(self, metadata: Mapping[str, object]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock.acquire()
        try:
            session = self._new_session("start", metadata.get("budget"))
            self._metadata = {
                **metadata,
                "versions": session["versions"],
                "git": session["git"],
                "started_at": session["started_at"],
                "status": "running",
                "sessions": [session],
            }
            self._write_metadata()
            db = sqlite3.connect(self.run_dir / "events.sqlite")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            with db:
                db.executescript(SCHEMA)
                db.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            self._db = db
        except BaseException:
            self._lock.release()
            raise

    # --- an existing run ---

    def open_existing(self) -> ExistingRun:
        """Take the lock, open a recorded run and describe it. Raises `ResumeError` if it cannot be resumed."""
        try:
            self._lock.acquire()
        except RunLockedError as error:
            raise ResumeError(str(error)) from error
        try:
            db = sqlite3.connect(self.run_dir / "events.sqlite")
            db.execute("PRAGMA synchronous=NORMAL")
            try:
                row = db.execute("SELECT version FROM schema_version").fetchone()
            except sqlite3.DatabaseError as error:
                db.close()
                raise ResumeError(f"{self.run_dir} is not a recorded run: {error}") from error
            version = None if row is None else row[0]
            if version != SCHEMA_VERSION:
                db.close()
                raise ResumeError(
                    f"{self.run_dir} was recorded with schema version {version}, and only runs with version {SCHEMA_VERSION} can be "
                    "resumed. Runs recorded by earlier versions cannot be migrated: start a new run"
                )
            self._db = db
            self._metadata = cast("dict[str, object]", json.loads((self.run_dir / "metadata.json").read_text()))
            last = db.execute("SELECT COALESCE(MAX(seq), 0) FROM events WHERE kind IN ('ask', 'tell')").fetchone()[0]
            self._last_seq = int(last)
            rows = db.execute("SELECT id, event_seq, evaluations_used, path, created_at FROM checkpoints ORDER BY id DESC").fetchall()
            recorded = int(db.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0])
            checkpoints.clear_temporary(self.run_dir / "checkpoints")
            return ExistingRun(self._metadata, tuple(CheckpointInfo(*row) for row in rows), self._last_seq, recorded)
        except BaseException:
            self._db = None
            self._lock.release()
            raise

    def load_checkpoint(self, infos: Sequence[CheckpointInfo], backend: Backend) -> tuple[CheckpointInfo, StateDict] | None:
        """The newest checkpoint that can be read, or None. A damaged one is skipped with a warning: an older checkpoint
        gives the same run, only with more to replay."""
        for info in infos:
            try:
                _, state = checkpoints.read(self.run_dir / info.path, backend)
            except checkpoints.CheckpointError as error:
                warnings.warn(f"skipping an unreadable checkpoint: {error}", RuntimeWarning, stacklevel=2)
                continue
            return info, state
        return None

    def truncate_after(self, event_seq: int) -> int:
        """Delete everything recorded after `event_seq` (candidates, lineage, evaluations, events) in one transaction, and
        return how many evaluations that removed. With 0 it deletes everything recorded."""
        db = self._connection()
        with db:
            removed = int(db.execute("SELECT COUNT(*) FROM candidates WHERE event_seq > ?", (event_seq,)).fetchone()[0])
            doomed = "SELECT id FROM candidates WHERE event_seq > ?"
            db.execute(f"DELETE FROM lineage WHERE child_id IN ({doomed})", (event_seq,))
            db.execute(f"DELETE FROM evaluations WHERE candidate_id IN ({doomed})", (event_seq,))
            db.execute(f"DELETE FROM episodes WHERE candidate_id IN ({doomed})", (event_seq,))
            db.execute("DELETE FROM candidates WHERE event_seq > ?", (event_seq,))
            db.execute("DELETE FROM events WHERE seq > ? AND kind != 'resume'", (event_seq,))
            db.execute("DELETE FROM checkpoints WHERE event_seq > ?", (event_seq,))
        self._last_seq = event_seq
        return removed

    def prepare_session(self, mode: str, budget: object, event: Mapping[str, object], carried_seq: int) -> None:
        """A resumed run is about to start. Nothing is written yet: the session is recorded with the first thing the run
        writes, so that an attempt that is refused (a configuration or a replay that does not match) leaves the run exactly
        as it was.

        `carried_seq` is the event sequence number the run continues from."""
        self._pending_session = (mode, budget, dict(event))
        self._last_seq = carried_seq

    def _flush_session(self) -> None:
        """Record the resume: delete the `stop` events of earlier sessions (the sessions in `metadata.json` keep that
        history, and a run has one stop in its log, at its end), write the `resume` event and open the new session."""
        if self._pending_session is None:
            return
        mode, budget, event = self._pending_session
        self._pending_session = None
        db = self._connection()
        with db:
            db.execute("DELETE FROM events WHERE kind = 'stop'")
            db.execute("INSERT INTO events (kind, step, payload) VALUES ('resume', NULL, ?)", (json.dumps(event),))
        self._sessions().append(self._new_session(mode, budget))
        self._metadata.update({"status": "running"})
        for key in ("ended_at", "stop_reason", "summary"):
            self._metadata.pop(key, None)
        self._write_metadata()

    # --- events ---

    def _first_event_after(self, seq: int, kinds: str) -> tuple[int, str] | None:
        row = (
            self._connection()
            .execute(f"SELECT seq, kind FROM events WHERE seq > ? AND kind IN ({kinds}) ORDER BY seq LIMIT 1", (seq,))
            .fetchone()
        )
        return None if row is None else (int(row[0]), str(row[1]))

    def _delete_group(self, db: sqlite3.Connection, ask_seq: int) -> None:
        """Remove a recorded batch (and its tell, if any): replay re-records it whole, because a larger budget makes it larger."""
        doomed = "SELECT id FROM candidates WHERE event_seq = ?"
        db.execute(f"DELETE FROM lineage WHERE child_id IN ({doomed})", (ask_seq,))
        db.execute(f"DELETE FROM evaluations WHERE candidate_id IN ({doomed})", (ask_seq,))
        db.execute(f"DELETE FROM episodes WHERE candidate_id IN ({doomed})", (ask_seq,))
        db.execute("DELETE FROM candidates WHERE event_seq = ?", (ask_seq,))
        following = self._first_event_after(ask_seq, "'ask', 'tell'")
        db.execute("DELETE FROM events WHERE seq = ?", (ask_seq,))
        if following is not None and following[1] == "tell":
            db.execute("DELETE FROM events WHERE seq = ?", (following[0],))

    def on_batch(self, step: int, batch: Batch[G], results: EvaluationBatch[G], recorded: int = 0, rerecord: bool = False) -> None:
        """Record a batch, its evaluations and the episodes behind them, in one transaction.

        `recorded` is how many of its first candidates replay took from the recording: all of them (the batch is already
        there, so nothing is written), none (the usual case), or some (a final batch that a larger budget has made longer).
        In the last case the recorded candidates stay as they are and the batch grows by the new ones, unless `rerecord`:
        the evaluator draws randomness per batch, so the whole batch was evaluated again and replaces the recorded one."""
        db = self._connection()
        candidates = list(batch.candidates)  # an array batch builds its candidates on access: do it once
        first_id, last_id = candidates[0].id, candidates[-1].id
        if recorded >= len(candidates):
            row = db.execute("SELECT event_seq FROM candidates WHERE id = ?", (first_id,)).fetchone()
            if row is None:
                raise RuntimeError(f"replay matched candidate {first_id} but the recording does not hold it")
            self._replayed_ask = self._last_seq = int(row[0])
            return
        self._flush_session()
        grow = 0 < recorded and not rerecord
        skip = recorded if grow else 0  # candidates that are in the recording already and stay there
        encoded: list[EncodedGenome] = encode_batch(batch)
        now = time.time()
        lineage_rows = [(parent, c.id) for c in candidates[skip:] for parent in c.parents]
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
            for e in list(results)[skip:]
        ]
        payload = json.dumps({"count": len(candidates), "first_id": first_id, "last_id": last_id})
        seq = 0
        with db:
            if recorded > 0:
                row = db.execute("SELECT event_seq FROM candidates WHERE id = ?", (first_id,)).fetchone()
                if grow:
                    seq = int(row[0])
                    following = self._first_event_after(seq, "'ask', 'tell'")
                    if following is not None and following[1] == "tell":
                        db.execute("DELETE FROM events WHERE seq = ?", (following[0],))
                    db.execute("UPDATE events SET payload = ? WHERE seq = ?", (payload, seq))
                else:
                    self._delete_group(db, int(row[0]))
            if not grow:
                seq = int(db.execute("INSERT INTO events (kind, step, payload) VALUES ('ask', ?, ?)", (step, payload)).lastrowid or 0)
            candidate_rows = [
                (c.id, c.step, c.origin, e.kind, e.data, e.dtype, e.shape, now, seq)
                for c, e in zip(candidates[skip:], encoded[skip:], strict=True)
            ]
            db.executemany("INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", candidate_rows)
            db.executemany("INSERT INTO lineage VALUES (?, ?)", lineage_rows)
            db.executemany("INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", evaluation_rows)
            if results.episodes is not None:
                db.executemany("INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?)", _episode_rows(results.episodes))
        self._last_seq = seq

    def on_tell(self, step: int, count: int, recorded: bool = False) -> None:
        """The strategy was told a batch. `recorded` says that the batch came whole from the recording, whose tell event is
        then there already, unless the run was killed between recording the batch and telling it."""
        db = self._connection()
        if recorded:
            following = self._first_event_after(self._replayed_ask, "'ask', 'tell'")
            if following is not None and following[1] == "tell":
                self._last_seq = following[0]
                return
        self._flush_session()
        with db:
            cursor = db.execute("INSERT INTO events (kind, step, payload) VALUES ('tell', ?, ?)", (step, json.dumps({"count": count})))
        self._last_seq = int(cursor.lastrowid or 0)

    # --- checkpoints ---

    def checkpoint(self, state: StateDict, evaluations_used: int, keep: int) -> None:
        """Write a checkpoint consistent with the events so far, register it, and keep only the newest `keep`."""
        db = self._connection()
        self._flush_session()
        seq = self._last_seq
        known = db.execute("SELECT 1 FROM checkpoints WHERE event_seq = ?", (seq,)).fetchone()
        if known is None:
            directory = checkpoints.write(self.run_dir / "checkpoints", seq, state)
            with db:
                db.execute(
                    "INSERT INTO checkpoints (event_seq, evaluations_used, path, created_at) VALUES (?, ?, ?, ?)",
                    (seq, evaluations_used, str(directory.relative_to(self.run_dir)), time.time()),
                )
        stale = db.execute("SELECT id, path FROM checkpoints ORDER BY id DESC LIMIT -1 OFFSET ?", (keep,)).fetchall()
        if stale:
            with db:
                db.executemany("DELETE FROM checkpoints WHERE id = ?", [(row[0],) for row in stale])
            for _, path in stale:
                checkpoints.remove(self.run_dir / path)

    # --- the end ---

    def on_end(self, status: str, stop_reason: str | None, summary: Mapping[str, object]) -> None:
        db = self._db
        if db is None:  # never started: nothing to finalise
            self._lock.release()
            return
        try:
            self._flush_session()
            with db:
                db.execute(
                    "INSERT INTO events (kind, step, payload) VALUES ('stop', NULL, ?)",
                    (json.dumps({"status": status, "stop_reason": stop_reason}),),
                )
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            db.close()
            self._db = None
            ended = _now()
            session = self._sessions()[-1]
            session.update(
                {
                    "ended_at": ended,
                    "status": status,
                    "stop_reason": stop_reason,
                    "evaluations_used": summary.get("evaluations_used"),
                    "wall_time": summary.get("wall_time"),
                }
            )
            self._metadata.update({"ended_at": ended, "status": status, "stop_reason": stop_reason, "summary": dict(summary)})
            self._write_metadata()
        finally:
            self._lock.release()

    def abandon(self) -> None:
        """Release the lock and close the database without finalising anything (a resume that was refused before it began)."""
        if self._db is not None:
            self._db.close()
            self._db = None
        self._lock.release()


__all__ = ["SCHEMA_VERSION", "CheckpointInfo", "ExistingRun", "ResumeError", "RunDirectoryError", "SQLiteRecorder"]
