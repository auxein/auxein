"""The generations of a recorded run: which tell delivered each candidate, and the run's results as they stood after each one.

A run's **generation k** is its k-th tell, counted across the sessions of a resumed run from 1: exactly what
`Progress.generation` counts (design doc §9.6). In generation delivery that is one generation of a generational strategy; in
steady-state delivery, one told candidate. A candidate's generation is the tell that delivered it, and a candidate that was
recorded but never told (a final batch cut by the budget, a batch recorded just before a crash or the all-failures guard) has
none, though it still counts towards the run's results, as it does in `RunResult`.

Since schema version 5 every `tell` event names the batch it delivered by the id of the batch's first candidate, so the mapping
is read, not inferred (§10.2). Everything here works on the objectives, constraints and statuses only: genomes are never
decoded, so that a run of hundreds of generations of large genomes is indexed in one query and the caller decodes only what it
returns.

The answers must be the ones the run gave. So the results are not ranked here: the recorded evaluations are folded, in the
order the run told them, through the run's own `ResultTracker`, and `best_up_to(k)` is the tracker's `best` after the
evaluations the run had accounted for when it called the callbacks of generation k.
"""

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from auxein.core import Candidate, CandidateId, Cost, Evaluation, Objective, Status
from auxein.core.evaluation import unchecked_evaluation
from auxein.core.tracking import ResultTracker, batch_stats

Ranked = Evaluation[None]
"""A recorded evaluation without its genome: all the ranking reads. Its candidate's genome is None."""

_ROWS = """
SELECT c.id, c.event_seq, c.step, c.origin, e.status, e.objectives, e.constraints
FROM candidates c JOIN evaluations e ON e.candidate_id = c.id
ORDER BY c.event_seq, c.id
"""
"""Every recorded evaluation, without genomes, in the order the run told them: the batches in the order they were recorded, and
in ask order within a batch. It is the order in which a resumed run rebuilds its tracker, and so the order that decides ties in
the Pareto archive (the earlier point keeps its place)."""

_TELLS = "SELECT step, json_extract(payload, '$.first_id') FROM events WHERE kind = 'tell' ORDER BY seq"


@dataclass(frozen=True)
class _Snapshot:
    """The run's results as the callbacks of one generation saw them."""

    best: Ranked | None
    front_size: int
    evaluations: int
    """How many evaluations the run had accounted for (`Progress.evaluations_used`): those told so far, and any recorded but
    never told before them."""
    failed: int
    timeout: int


class Generations:
    """The generation index of a recorded run, built by one query. See the module docstring."""

    def __init__(self, db: sqlite3.Connection, objectives: Sequence[Objective], constrained: bool) -> None:
        self.objectives = tuple(objectives)
        self.constrained = constrained
        self._order: list[Ranked] = []
        """Every recorded evaluation in told order."""
        position_of_seq: dict[int, tuple[int, int]] = {}  # a batch's event sequence -> its slice of `_order`
        seq_of_id: dict[int, int] = {}
        for candidate_id, event_seq, step, origin, status, objectives_json, constraints_json in db.execute(_ROWS):
            index = len(self._order)
            start, _ = position_of_seq.get(event_seq, (index, index))
            position_of_seq[event_seq] = (start, index + 1)
            seq_of_id[candidate_id] = event_seq
            candidate = Candidate(CandidateId(candidate_id), None, (), origin, step)
            # recorded values were validated when the run made them: no need to validate (and copy) them again
            self._order.append(
                unchecked_evaluation(
                    candidate,
                    Status(status),
                    MappingProxyType(json.loads(objectives_json)),
                    MappingProxyType(json.loads(constraints_json)),
                    MappingProxyType({}),
                    Cost(),
                )
            )
        self._slices: list[tuple[int, int]] = []
        """Generation k's evaluations are `_order[start:end]` with `(start, end) = _slices[k - 1]`."""
        self._steps: list[int] = []
        self._generation_of: dict[int, int | None] = dict.fromkeys(seq_of_id)
        for step, first_id in db.execute(_TELLS):
            if first_id is None or first_id not in seq_of_id:
                raise ValueError(
                    f"the recording is inconsistent: tell event {len(self._slices) + 1} names candidate {first_id}, which is not recorded"
                )
            start, end = position_of_seq[seq_of_id[first_id]]
            if self._slices and start < self._slices[-1][1]:
                raise ValueError(
                    f"the recording is inconsistent: tell event {len(self._slices) + 1} delivers a batch recorded before the previous one"
                )
            self._slices.append((start, end))
            self._steps.append(int(step))
            for evaluation in self._order[start:end]:
                self._generation_of[int(evaluation.candidate.id)] = len(self._slices)
        self._snapshots, self._final = self._fold()

    def _tracker(self) -> ResultTracker[None]:
        return ResultTracker(self.objectives, constrained=self.constrained)

    def _fold(self) -> tuple[list[_Snapshot], ResultTracker[None]]:
        """Fold every evaluation through the run's ranking once, noting the results at the end of each generation."""
        tracker = self._tracker()
        snapshots: list[_Snapshot] = []
        done = failed = timeout = 0
        for _, end in self._slices:
            chunk = self._order[done:end]
            tracker.add(chunk, done)
            failed += sum(1 for e in chunk if e.status is Status.FAILED)
            timeout += sum(1 for e in chunk if e.status is Status.TIMEOUT)
            done = end
            snapshots.append(_Snapshot(tracker.best, tracker.front_size, done, failed, timeout))
        tracker.add(self._order[done:], done)  # never told: still part of the run's results, as in `RunResult`
        return snapshots, tracker

    # --- the queries; `k` has been checked by the caller ---

    def __len__(self) -> int:
        return len(self._slices)

    def generation_of(self, candidate_id: int) -> int | None:
        if candidate_id not in self._generation_of:
            raise ValueError(f"candidate {candidate_id} is not in the recording")
        return self._generation_of[candidate_id]

    def best_up_to(self, k: int | None) -> Ranked | None:
        return self._final.best if k is None else self._snapshots[k - 1].best

    def pareto_front_up_to(self, k: int | None) -> tuple[Ranked, ...]:
        """The archive is not kept per generation (a many-objective run's would be large), so it is folded again up to `k`."""
        if k is None:
            return self._final.pareto_front
        tracker = self._tracker()
        tracker.add(self._order[: self._slices[k - 1][1]], 0)
        return tracker.pareto_front

    def told_in(self, k: int) -> list[Ranked]:
        start, end = self._slices[k - 1]
        return self._order[start:end]

    def best_in(self, k: int) -> Ranked | None:
        tracker = self._tracker()
        tracker.add(self.told_in(k), 0)
        return tracker.best

    def history(self) -> list[dict[str, object]]:
        """One row per generation, with the fields of `ProgressHistory` that the recording holds faithfully (see
        `RunReader.generation_history`)."""
        rows: list[dict[str, object]] = []
        first = self.objectives[0].name
        for k, snapshot in enumerate(self._snapshots, start=1):
            batch = batch_stats(self.told_in(k), self.objectives)
            row: dict[str, object] = {
                "generation": k,
                "step": self._steps[k - 1],
                "evaluations": snapshot.evaluations,
                "best": snapshot.best.objectives[first] if snapshot.best is not None else None,
                "front_size": snapshot.front_size,
                "failed": snapshot.failed,
                "timeout": snapshot.timeout,
                "batch_ok": batch.ok,
                "batch_failed": batch.failed,
                "batch_timeout": batch.timeout,
                "batch_feasible": batch.feasible_fraction,
            }
            for objective in self.objectives:
                stats = batch.objectives[objective.name]
                for field in ("best", "mean", "median", "worst"):
                    row[f"{objective.name}_{field}"] = getattr(stats, field)
            rows.append(row)
        return rows


def check_generation(k: object, count: int) -> int:
    """`k` as a generation of a run with `count` of them, or a `ValueError` that names the valid range."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"a generation is an int, got {k!r}")
    if not 1 <= k <= count:
        valid = "the run has no generations (nothing was told)" if count == 0 else f"generations are numbered 1 to {count}"
        raise ValueError(f"no generation {k}: {valid}")
    return k


def problem_of(metadata: Mapping[str, Any]) -> tuple[tuple[Objective, ...], bool]:
    """The objectives (with their directions) and whether there are constraints, as the run's metadata records them."""
    problem = cast("dict[str, Any]", metadata.get("problem") or {})
    objectives = tuple(Objective(o["name"], o["direction"]) for o in problem.get("objectives", []))
    if not objectives:
        raise ValueError("the run's metadata does not describe its objectives, so its results cannot be ranked")
    return objectives, bool(problem.get("constraints"))
