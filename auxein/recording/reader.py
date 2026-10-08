"""A minimal reader of a recorded run: `open_run(path)`: its evaluations, lineage, sessions and checkpoints."""

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import cast

from auxein.core import CandidateId, Status
from auxein.recording.genomes import decode_genome
from auxein.recording.sqlite import SCHEMA_VERSION, CheckpointInfo


@dataclass(frozen=True)
class RecordedEvaluation:
    """One candidate and its evaluation, as recorded. Array genomes come back as read-only numpy arrays."""

    candidate_id: CandidateId
    step: int
    origin: str
    parents: tuple[CandidateId, ...]
    genome: object
    status: Status
    objectives: dict[str, float]
    constraints: dict[str, float]
    descriptors: dict[str, float]
    cost_units: dict[str, float]
    wall_time: float
    error: str | None


_EVALUATIONS = """
SELECT c.id, c.step, c.origin, c.genome_kind, c.genome, c.genome_dtype, c.genome_shape,
       e.status, e.objectives, e.constraints, e.descriptors, e.cost_units, e.wall_time, e.error
FROM evaluations e JOIN candidates c ON c.id = e.candidate_id
ORDER BY c.id
"""

_ANCESTRY = """
WITH RECURSIVE walk (id, depth) AS (
    SELECT parent_id, 1 FROM lineage WHERE child_id = ?
    UNION
    SELECT l.parent_id, walk.depth + 1 FROM lineage l JOIN walk ON l.child_id = walk.id
)
SELECT id FROM walk GROUP BY id ORDER BY MIN(depth), id
"""

_DESCENDANTS = """
WITH RECURSIVE walk (id, depth) AS (
    SELECT child_id, 1 FROM lineage WHERE parent_id = ?
    UNION
    SELECT l.child_id, walk.depth + 1 FROM lineage l JOIN walk ON l.parent_id = walk.id
)
SELECT id FROM walk GROUP BY id ORDER BY MIN(depth), id
"""


class RunReader:
    """A recorded run opened for reading (read-only)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        database = self.path / "events.sqlite"
        if not database.exists():
            raise FileNotFoundError(f"{self.path} is not a recorded run: there is no events.sqlite in it")
        self._db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        version = self._db.execute("SELECT version FROM schema_version").fetchone()
        if version is None or version[0] != SCHEMA_VERSION:
            raise ValueError(f"unsupported run schema version {version and version[0]}: this reader understands version {SCHEMA_VERSION}")

    @property
    def schema_version(self) -> int:
        return int(self._db.execute("SELECT version FROM schema_version").fetchone()[0])

    @property
    def metadata(self) -> dict[str, object]:
        """The contents of `metadata.json`."""
        return cast("dict[str, object]", json.loads((self.path / "metadata.json").read_text()))

    @property
    def sessions(self) -> list[dict[str, object]]:
        """The sessions of the run, oldest first: the first start and each resume, with its mode, budget, versions, and (once it
        ended) its end time, status, stop reason, evaluations used and cumulative wall time. A session that was killed has
        no end."""
        return cast("list[dict[str, object]]", self.metadata.get("sessions", []))

    def checkpoints(self) -> list[CheckpointInfo]:
        """The checkpoints that are kept, oldest first."""
        rows = self._db.execute("SELECT id, event_seq, evaluations_used, path, created_at FROM checkpoints ORDER BY id").fetchall()
        return [CheckpointInfo(*row) for row in rows]

    def evaluations(self) -> Iterator[RecordedEvaluation]:
        """Every evaluated candidate, in id order, with its genome decoded."""
        for row in self._db.execute(_EVALUATIONS).fetchall():
            candidate_id = CandidateId(row[0])
            parents = tuple(
                CandidateId(r[0])
                for r in self._db.execute("SELECT parent_id FROM lineage WHERE child_id = ? ORDER BY rowid", (candidate_id,))
            )
            yield RecordedEvaluation(
                candidate_id=candidate_id,
                step=row[1],
                origin=row[2],
                parents=parents,
                genome=decode_genome(row[3], row[4], row[5], row[6]),
                status=Status(row[7]),
                objectives=json.loads(row[8]),
                constraints=json.loads(row[9]),
                descriptors=json.loads(row[10]),
                cost_units=json.loads(row[11]),
                wall_time=row[12],
                error=row[13],
            )

    def ancestry(self, candidate_id: int) -> list[int]:
        """The ancestors of a candidate (parents, grandparents, ...), nearest first, then by id. One recursive query."""
        return [r[0] for r in self._db.execute(_ANCESTRY, (candidate_id,))]

    def descendants(self, candidate_id: int) -> list[int]:
        """The descendants of a candidate (children, grandchildren, ...), nearest first, then by id. One recursive query."""
        return [r[0] for r in self._db.execute(_DESCENDANTS, (candidate_id,))]

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "RunReader":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None) -> None:
        self.close()


def open_run(path: str | Path) -> RunReader:
    """Open a recorded run directory for reading."""
    return RunReader(path)
