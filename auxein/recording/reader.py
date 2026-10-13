"""A reader of a recorded run: `open_run(path)`: its evaluations, lineage, sessions and checkpoints, and its generations."""

import csv
import io
import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from auxein.aggregators import Aggregator
from auxein.backend import Backend
from auxein.core import CandidateId, Status
from auxein.recording.generations import Generations, Ranked, check_generation, problem_of
from auxein.recording.genomes import decode_genome
from auxein.recording.sqlite import SCHEMA_VERSION, CheckpointInfo
from auxein.spaces.codec import GenomeCodec


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


@dataclass(frozen=True)
class RecordedEpisode:
    """One recorded episode: a candidate in a scenario, with the raw measurements it produced."""

    candidate_id: CandidateId
    scenario_index: int
    scenario_id: str
    status: Status
    measurements: dict[str, float]
    error: str | None


@dataclass(frozen=True)
class Reaggregated:
    """A candidate re-judged by another aggregator from its recorded measurements.

    A candidate with an episode that did not succeed has the status of that failure and no values, as in the run itself.
    """

    candidate_id: CandidateId
    status: Status
    objectives: dict[str, float]
    constraints: dict[str, float]
    descriptors: dict[str, float]
    cost: dict[str, float]
    error: str | None = None


@dataclass(frozen=True)
class RecordedOperatorCall:
    """One recorded call of an external operator: what it proposed, what it cost and how long it took."""

    key: str
    operator: str
    output: object
    cost: dict[str, float]
    wall_time: float
    metadata: dict[str, object]
    event_seq: int


@dataclass(frozen=True)
class GenomeStoreStats:
    """What the genome store holds (design doc §10.3)."""

    blobs: int
    bytes: int
    references: int
    """How many candidates refer to a blob; more than `blobs` when genomes are shared."""


_SELECT_EVALUATIONS = """
SELECT c.id, c.step, c.origin, c.genome_kind, CASE WHEN c.genome_hash IS NULL THEN c.genome ELSE b.data END, c.genome_dtype, c.genome_shape,
       e.status, e.objectives, e.constraints, e.descriptors, e.cost_units, e.wall_time, e.error
FROM evaluations e JOIN candidates c ON c.id = e.candidate_id LEFT JOIN blobs b ON b.hash = c.genome_hash
"""

_EVALUATIONS = f"{_SELECT_EVALUATIONS} ORDER BY c.id"

_IDS_PER_QUERY = 500  # well below SQLite's limit on the parameters of one statement

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


def _row(columns: Mapping[str, npt.NDArray[np.float64]], row: int) -> dict[str, float]:
    return {name: float(column[row]) for name, column in columns.items()}


class RunReader:
    """A recorded run opened for reading (read-only)."""

    def __init__(self, path: str | Path, codec: GenomeCodec[Any] | None = None) -> None:
        self.path = Path(path)
        self.codec = codec
        """The codec of the run's search space, if known: structured genomes then come back decoded, and as plain JSON otherwise."""
        database = self.path / "events.sqlite"
        if not database.exists():
            raise FileNotFoundError(f"{self.path} is not a recorded run: there is no events.sqlite in it")
        self._db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        self._generations: Generations | None = None
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

    def _recorded(self, row: Sequence[Any], parents: tuple[CandidateId, ...]) -> RecordedEvaluation:
        """A row of `_SELECT_EVALUATIONS` as a `RecordedEvaluation`, its genome decoded (through the codec, if there is one)."""
        return RecordedEvaluation(
            candidate_id=CandidateId(row[0]),
            step=row[1],
            origin=row[2],
            parents=parents,
            genome=decode_genome(row[3], row[4], row[5], row[6], self.codec),
            status=Status(row[7]),
            objectives=json.loads(row[8]),
            constraints=json.loads(row[9]),
            descriptors=json.loads(row[10]),
            cost_units=json.loads(row[11]),
            wall_time=row[12],
            error=row[13],
        )

    def evaluations(self) -> Iterator[RecordedEvaluation]:
        """Every evaluated candidate, in id order, with its genome decoded."""
        for row in self._db.execute(_EVALUATIONS).fetchall():
            parents = tuple(
                CandidateId(r[0]) for r in self._db.execute("SELECT parent_id FROM lineage WHERE child_id = ? ORDER BY rowid", (row[0],))
            )
            yield self._recorded(row, parents)

    def _fetch(self, ids: Sequence[int]) -> list[RecordedEvaluation]:
        """The recorded evaluations of these candidates, in this order, with only their genomes decoded."""
        rows: dict[int, Sequence[Any]] = {}
        parents: dict[int, list[CandidateId]] = {}
        unique = list(dict.fromkeys(ids))
        for start in range(0, len(unique), _IDS_PER_QUERY):
            chunk = unique[start : start + _IDS_PER_QUERY]
            marks = ", ".join("?" * len(chunk))
            for row in self._db.execute(f"{_SELECT_EVALUATIONS} WHERE c.id IN ({marks})", chunk):
                rows[row[0]] = row
            for child, parent in self._db.execute(
                f"SELECT child_id, parent_id FROM lineage WHERE child_id IN ({marks}) ORDER BY rowid", chunk
            ):
                parents.setdefault(child, []).append(CandidateId(parent))
        decoded = {i: self._recorded(rows[i], tuple(parents.get(i, ()))) for i in unique}
        return [decoded[i] for i in ids]

    def _decoded(self, ranked: Sequence[Ranked]) -> list[RecordedEvaluation]:
        return self._fetch([int(e.candidate.id) for e in ranked])

    # --- generations (design doc §10.2) ---

    def _index(self) -> Generations:
        """The generation index, built on first use and kept: one query over objectives and statuses, no genome decoded."""
        if self._generations is None:
            objectives, constrained = problem_of(self.metadata)
            self._generations = Generations(self._db, objectives, constrained)
        return self._generations

    def generations(self) -> int:
        """How many generations the run has: its tells, across the sessions of a resumed run.

        A generation is what `Progress.generation` counts: one generation of a generational strategy in generation delivery,
        one told candidate in steady-state delivery."""
        return len(self._index())

    def generation_of(self, candidate_id: int) -> int | None:
        """The generation (the tell) that delivered a candidate, from 1; None for a candidate that was recorded but never told
        (a final batch cut by the budget, a batch recorded just before a crash). A `ValueError` if the candidate is not recorded."""
        return self._index().generation_of(candidate_id)

    def best_up_to(self, generation: int | None = None) -> RecordedEvaluation | None:
        """The champion so far: the run's best evaluation after generation `generation`, exactly what `Progress.best` was at
        that generation (single objective only; None for several objectives or before anything succeeded).

        Without a generation it is the run's overall best, `RunResult.best`, which includes a final batch that was never told.
        It is ranked by the run's own ranking (feasible first, then a lower total violation, then the better objective in its
        declared direction, then the earliest id), and only its genome is decoded."""
        index = self._index()
        k = None if generation is None else check_generation(generation, len(index))
        best = index.best_up_to(k)
        return None if best is None else self._decoded([best])[0]

    def best_in(self, generation: int) -> RecordedEvaluation | None:
        """The best of the candidates generation `generation` told, by the same ranking, ignoring every other generation: it can
        be worse than `best_up_to(generation)`, the champion so far. None if none of them succeeded, or for several objectives."""
        index = self._index()
        best = index.best_in(check_generation(generation, len(index)))
        return None if best is None else self._decoded([best])[0]

    def pareto_front_up_to(self, generation: int | None = None) -> tuple[RecordedEvaluation, ...]:
        """The Pareto front after generation `generation`, exactly `Progress.pareto_front` at that generation (sorted by candidate
        id; for a single objective, the best feasible evaluation). Without a generation it is `RunResult.pareto_front`."""
        index = self._index()
        k = None if generation is None else check_generation(generation, len(index))
        return tuple(self._decoded(index.pareto_front_up_to(k)))

    def evaluations_in(self, generation: int) -> list[RecordedEvaluation]:
        """The evaluations generation `generation` told, in the order they were told, genomes decoded."""
        index = self._index()
        return self._decoded(index.told_in(check_generation(generation, len(index))))

    def champions(self, every: int) -> Iterator[tuple[int, RecordedEvaluation | None]]:
        """`(k, best_up_to(k))` for k = `every`, 2·`every`, ... and the last generation, for rendering the champion at milestones:

            with open_run("runs/team") as run:
                for k, champion in run.champions(every=25):
                    render(decoder.decode(champion.genome), f"generation-{k}.mp4")

        A champion is decoded once however many milestones it holds. The best is None at a milestone before anything
        succeeded. A run with several objectives has no single champion: use `pareto_front_up_to`."""
        if isinstance(every, bool) or not isinstance(every, int) or every < 1:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"every must be a positive int, got {every!r}")
        index = self._index()
        if len(index.objectives) > 1:
            raise ValueError(
                "a run with several objectives has no single champion: use pareto_front_up_to(k) for its front at generation k"
            )
        count = len(index)
        milestones = list(range(every, count + 1, every))
        if count and (not milestones or milestones[-1] != count):
            milestones.append(count)
        decoded: dict[int, RecordedEvaluation] = {}
        for k in milestones:
            best = index.best_up_to(k)
            if best is None:
                yield k, None
                continue
            key = int(best.candidate.id)
            if key not in decoded:
                decoded[key] = self._decoded([best])[0]
            yield k, decoded[key]

    def generation_history(self) -> list[dict[str, object]]:
        """One row per generation, for fitness-over-generations charts, computed from the recording without decoding any genome.

        The rows have the fields of `ProgressHistory` rows that the recording holds faithfully, with the same names, values and
        natural units: `generation`, `evaluations` (used so far), `best` (the run's best objective value, single objective
        only, else None) and `front_size`, the cumulative `failed` and `timeout`, and for the generation's own told
        candidates `batch_ok`, `batch_failed`, `batch_timeout`, `batch_feasible` and per objective `<name>_best`,
        `<name>_mean`, `<name>_median` and `<name>_worst` (NaN when none was OK). It adds `step`, the generation's ask round.

        `elapsed` is left out: the recording has the clock time of each batch, which counts the time between the sessions of
        a resumed run and the time a killed session lost after its last checkpoint, so it is not the active time that
        `Progress.elapsed` reports. `replaying` and `final` describe a callback's call, not the run."""
        return self._index().history()

    def generation_history_csv(self, path: str | Path | None = None) -> str:
        """`generation_history()` as CSV text, also written to `path` if one is given (like `ProgressHistory.to_csv`)."""
        rows = self.generation_history()
        buffer = io.StringIO()
        if rows:
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        text = buffer.getvalue()
        if path is not None:
            Path(path).write_text(text)
        return text

    def episodes(self, candidate_id: int) -> list[RecordedEpisode]:
        """The recorded episodes of a candidate, one per scenario in scenario order (empty if it ran none)."""
        rows = self._db.execute(
            "SELECT scenario_index, scenario_id, status, measurements, error FROM episodes WHERE candidate_id = ? ORDER BY scenario_index",
            (candidate_id,),
        ).fetchall()
        return [RecordedEpisode(CandidateId(candidate_id), r[0], r[1], Status(r[2]), json.loads(r[3]), r[4]) for r in rows]

    def reaggregate(self, aggregator: Aggregator, backend: Backend | None = None) -> list[Reaggregated]:
        """Re-judge every recorded candidate with `aggregator`, from the stored measurements and without re-simulating.

        This is what recording the raw per-scenario measurements is for (design doc §6.4): the same run, seen through another
        aggregator (a worst case instead of a mean, a different constraint). The result is what an evaluation with that
        aggregator would have had. The aggregator is not checked against the run's problem, since re-judging may use new
        names. Candidates are in id order; those that ran no episodes are left out, and those with a failed episode are
        reported with that status and no values.
        """
        backend = Backend() if backend is None else backend
        grouped: dict[int, list[tuple[int, Status, dict[str, float], str | None]]] = {}
        for candidate, index, status, measurements, error in self._db.execute(
            "SELECT candidate_id, scenario_index, status, measurements, error FROM episodes ORDER BY candidate_id, scenario_index"
        ):
            grouped.setdefault(candidate, []).append((index, Status(status), json.loads(measurements), error))
        results: dict[int, Reaggregated] = {}
        ok: list[int] = []
        for candidate, episodes in grouped.items():
            failed = [e for e in episodes if e[1] is not Status.OK]
            if failed:
                status = Status.TIMEOUT if all(e[1] is Status.TIMEOUT for e in failed) else Status.FAILED
                results[candidate] = Reaggregated(
                    CandidateId(candidate), status, {}, {}, {}, {}, f"{len(failed)} of {len(episodes)} episodes did not succeed"
                )
            else:
                ok.append(candidate)
        if ok:
            names = sorted(grouped[ok[0]][0][2])
            arrays = {name: backend.asarray(np.array([[e[2][name] for e in grouped[c]] for c in ok], dtype=np.float64)) for name in names}
            try:
                aggregated = aggregator.aggregate(arrays, backend)
            except KeyError as error:
                raise ValueError(str(error.args[0])) from None
            for row, candidate in enumerate(ok):
                if row in aggregated.invalid:
                    results[candidate] = Reaggregated(
                        CandidateId(candidate), Status.FAILED, {}, {}, {}, {}, f"non-finite value: {aggregated.invalid[row]}"
                    )
                    continue
                results[candidate] = Reaggregated(
                    CandidateId(candidate),
                    Status.OK,
                    _row(aggregated.objectives, row),
                    _row(aggregated.constraints, row),
                    _row(aggregated.descriptors, row),
                    _row(aggregated.cost, row),
                )
        return [results[c] for c in sorted(results)]

    def operator_calls(self) -> list[RecordedOperatorCall]:
        """The recorded calls of external proposal operators, oldest first (design doc §3.5)."""
        rows = self._db.execute(
            "SELECT key, operator, output, cost_units, wall_time, metadata, event_seq FROM operator_calls ORDER BY rowid"
        ).fetchall()
        return [RecordedOperatorCall(r[0], r[1], json.loads(r[2]), json.loads(r[3]), r[4], json.loads(r[5]), r[6]) for r in rows]

    def genome_store(self) -> GenomeStoreStats:
        """How much the genome store holds: blobs, their bytes, and how many candidates refer to them."""
        blobs, size = self._db.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(data)), 0) FROM blobs").fetchone()
        references = self._db.execute("SELECT COUNT(*) FROM candidates WHERE genome_hash IS NOT NULL").fetchone()[0]
        return GenomeStoreStats(int(blobs), int(size), int(references))

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


def open_run(path: str | Path, codec: GenomeCodec[Any] | None = None) -> RunReader:
    """Open a recorded run directory for reading. Pass the codec of its search space to get structured genomes decoded."""
    return RunReader(path, codec)
