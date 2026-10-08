"""Reading a recording back in the order it was told, to resume a run (design doc §10.4).

Two uses: **replay** feeds the recorded evaluations that came after a checkpoint to a restored strategy instead of
evaluating again (`ReplayStream`), and **rebuilding** the result tracker folds every recorded evaluation back in
(`iter_records`). Both read in told order, which is the order of the events: the tables are keyed by candidate id, but
only the `events` table, through each candidate's `event_seq`, says when it was told.
"""

import json
import sqlite3
from collections.abc import Generator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

from auxein.backend import Backend
from auxein.core import Candidate, CandidateId, Cost, Evaluation, RawRef, Status
from auxein.recording.genomes import EncodedGenome, decode_genome

_CHUNK = 2048

G = TypeVar("G")

_SELECT = """
SELECT c.id, c.step, c.origin, c.genome_kind, c.genome, c.genome_dtype, c.genome_shape,
       e.status, e.objectives, e.constraints, e.descriptors, e.cost_units, e.wall_time, e.error,
       EXISTS (SELECT 1 FROM episodes p WHERE p.candidate_id = c.id)
FROM candidates c JOIN evaluations e ON e.candidate_id = c.id
"""


@dataclass(frozen=True)
class ReplayRecord:
    """One recorded candidate and evaluation, with the genome still encoded so that it can be compared byte for byte."""

    candidate_id: CandidateId
    step: int
    origin: str
    parents: tuple[CandidateId, ...]
    genome: EncodedGenome
    status: Status
    objectives: Mapping[str, float]
    constraints: Mapping[str, float]
    descriptors: Mapping[str, float]
    cost_units: Mapping[str, float]
    wall_time: float
    error: str | None
    has_episodes: bool = False
    """Whether the recording holds per-scenario episodes for the candidate (the evaluation then references them)."""

    def evaluation(self, candidate: Candidate[G]) -> Evaluation[G]:
        """The recorded evaluation, attached to `candidate` (the regenerated one, which replay has checked to be the same)."""
        cost = Cost(self.wall_time, self.cost_units)
        raw = RawRef(f"episodes/{self.candidate_id}") if self.has_episodes else None
        return Evaluation(candidate, self.status, self.objectives, self.constraints, self.descriptors, cost, raw, self.error)

    def decoded_candidate(self, backend: Backend) -> Candidate[object]:
        """The candidate with its genome decoded: array genomes on the run's backend, JSON genomes as plain values."""
        genome = decode_genome(self.genome.kind, self.genome.data, self.genome.dtype, self.genome.shape)
        if self.genome.kind == "array":
            genome = backend.asarray(genome)  # type: ignore[arg-type]
        return Candidate(self.candidate_id, genome, self.parents, self.origin, self.step)


def _records(db: sqlite3.Connection, rows: list[tuple[object, ...]]) -> list[ReplayRecord]:
    ids = [cast("int", row[0]) for row in rows]
    parents: dict[int, list[CandidateId]] = {}
    if ids:
        lineage = db.execute("SELECT child_id, parent_id FROM lineage WHERE child_id BETWEEN ? AND ? ORDER BY rowid", (min(ids), max(ids)))
        for child, parent in lineage:
            parents.setdefault(child, []).append(CandidateId(parent))
    records: list[ReplayRecord] = []
    for row in rows:
        candidate_id = cast("int", row[0])
        genome = EncodedGenome(cast("str", row[3]), bytes(cast("bytes", row[4])), cast("str | None", row[5]), cast("str | None", row[6]))
        records.append(
            ReplayRecord(
                candidate_id=CandidateId(candidate_id),
                step=cast("int", row[1]),
                origin=cast("str", row[2]),
                parents=tuple(parents.get(candidate_id, ())),
                genome=genome,
                status=Status(cast("str", row[7])),
                objectives=json.loads(cast("str", row[8])),
                constraints=json.loads(cast("str", row[9])),
                descriptors=json.loads(cast("str", row[10])),
                cost_units=json.loads(cast("str", row[11])),
                wall_time=cast("float", row[12]),
                error=cast("str | None", row[13]),
                has_episodes=bool(row[14]),
            )
        )
    return records


def iter_records(database: Path, *, after: int | None = None, upto: int | None = None) -> Generator[ReplayRecord]:
    """Recorded evaluations in told order, those recorded after event `after` and up to event `upto` (each optional).

    Reads through its own read-only connection, so it can run while the recorder writes.
    """
    db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        clauses: list[str] = []
        parameters: list[int] = []
        if after is not None:
            clauses.append("c.event_seq > ?")
            parameters.append(after)
        if upto is not None:
            clauses.append("c.event_seq <= ?")
            parameters.append(upto)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = db.execute(f"{_SELECT} {where} ORDER BY c.event_seq, c.id", parameters)
        while True:
            rows = cursor.fetchmany(_CHUNK)
            if not rows:
                return
            yield from _records(db, rows)
    finally:
        db.close()


def count_records(database: Path, *, after: int | None = None) -> int:
    db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        if after is None:
            return int(db.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0])
        return int(db.execute("SELECT COUNT(*) FROM candidates WHERE event_seq > ?", (after,)).fetchone()[0])
    finally:
        db.close()


class ReplayStream:
    """The recorded evaluations after a checkpoint, handed out in told order as the restored run asks for them.

    In deterministic mode a run's candidates are told in ask order, so the recorded evaluations are a prefix of the
    candidates the restored strategy will ask for: the k-th candidate asked after the checkpoint has the k-th record.
    """

    def __init__(self, database: Path, after: int) -> None:
        self.total = count_records(database, after=after)
        self._iterator = iter_records(database, after=after)
        self._taken = 0

    @property
    def remaining(self) -> int:
        """How many recorded evaluations have not been handed out yet."""
        return self.total - self._taken

    def take(self, n: int) -> list[ReplayRecord]:
        """The next `n` records, or fewer when the recording runs out."""
        records: list[ReplayRecord] = []
        for record in self._iterator:
            records.append(record)
            if len(records) == n:
                break
        self._taken += len(records)
        return records

    def close(self) -> None:
        self._iterator.close()
