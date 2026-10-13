"""The ranking of a run's results and the statistics of a told batch, shared by the driver and the recording's reader.

`ResultTracker` is **the** ranking of `RunResult.best` and `RunResult.pareto_front` (design doc §9.5). It lives in the core,
not in the driver, so that the reader of a recorded run (`RunReader.best_up_to` and the other generation queries, §10.2) uses
the very same code and gives the same answers as the run did, without the recording package importing the driver (which
imports the recording). `batch_stats` is what a progress callback reports about the last told batch (§9.6), and what the
reader's generation history reports per generation, for the same reason.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Generic

import numpy as np
import numpy.typing as npt

from auxein.core._typing import G
from auxein.core.evaluation import Evaluation, Objective, Status


class ResultTracker(Generic[G]):
    """Keeps the best evaluation, the Pareto archive and the trace incrementally.

    For one objective its memory is constant: it holds at most one evaluation, so the footprint of a run does not grow
    with its length. For several objectives it holds the non-dominated archive, which is what the user asked for.
    """

    def __init__(self, objectives: Sequence[Objective], *, constrained: bool = True) -> None:
        self._objectives = tuple(objectives)
        # one objective and no constraints: nothing can be infeasible, and a much cheaper loop gives the same results
        self._unconstrained_single = len(objectives) == 1 and not constrained
        self._names = tuple(o.name for o in objectives)
        self._signs = tuple(o.sign for o in objectives)
        self._best: Evaluation[G] | None = None
        self._best_key: tuple[int, float, float, int] | None = None
        self._front: list[tuple[tuple[float, ...], Evaluation[G]]] = []
        self._points: npt.NDArray[np.float64] | None = None  # the objective values of the archive, one row per member
        self._trace: list[tuple[int, float]] = []

    @property
    def best(self) -> Evaluation[G] | None:
        return self._best

    @property
    def trace(self) -> tuple[tuple[int, float], ...]:
        return tuple(self._trace)

    @property
    def pareto_front(self) -> tuple[Evaluation[G], ...]:
        return tuple(sorted((e for _, e in self._front), key=lambda e: e.candidate.id))

    @property
    def front_size(self) -> int:
        """`len(pareto_front)`, without sorting the archive: a reader asks it once per generation of a long run."""
        return len(self._front)

    def add(self, evaluations: Sequence[Evaluation[G]], used_before: int) -> None:
        """Take in a batch of evaluations, `used_before` being the number of evaluations made before it."""
        if self._unconstrained_single:
            self._add_unconstrained_single(evaluations, used_before)
            return
        single = len(self._names) == 1
        for position, evaluation in enumerate(evaluations):
            if evaluation.status is not Status.OK:
                continue
            values = tuple(sign * evaluation.objectives[name] for sign, name in zip(self._signs, self._names, strict=True))
            violation = sum(evaluation.constraints.values())
            if single:
                key = (0 if violation == 0 else 1, violation, values[0], evaluation.candidate.id)
                if self._best_key is None or key < self._best_key:
                    self._best, self._best_key = evaluation, key
                    self._trace.append((used_before + position + 1, evaluation.objectives[self._names[0]]))
            if violation == 0:
                self._archive(values, evaluation)

    def _add_unconstrained_single(self, evaluations: Sequence[Evaluation[G]], used_before: int) -> None:
        """`add` for one objective and no constraints: the same results as the general loop, with less work per evaluation."""
        name, sign = self._names[0], self._signs[0]
        best_value = None if self._best_key is None else self._best_key[2]
        best_id = -1 if self._best_key is None else self._best_key[3]
        for position, evaluation in enumerate(evaluations):
            if evaluation.status is not Status.OK:
                continue
            value = sign * evaluation.objectives[name]
            candidate_id = evaluation.candidate.id
            if best_value is None or value < best_value or (value == best_value and candidate_id < best_id):
                best_value, best_id = value, candidate_id
                self._best, self._best_key = evaluation, (0, 0.0, value, candidate_id)
                self._trace.append((used_before + position + 1, evaluation.objectives[name]))
            if not self._front or value < self._front[0][0][0]:  # strictly better: an equal value keeps the earlier candidate
                self._front = [((value,), evaluation)]

    def _archive(self, values: tuple[float, ...], evaluation: Evaluation[G]) -> None:
        """Add a point to the archive of non-dominated points, unless something covers it (dominates it or equals it: the
        earliest candidate keeps the place), and drop what it dominates.

        The comparison against the whole archive is one numpy expression, since an archive of a many-objective or a
        well-converged run holds thousands of points and a Python loop over it per evaluation made the driver, not the
        strategy, the cost of a multi-objective run."""
        point = np.asarray(values, dtype=np.float64)
        members = self._points
        if members is not None and members.shape[0]:
            if bool((members <= point).all(axis=1).any()):  # a member dominates the point or equals it
                return
            dominated = (point <= members).all(axis=1) & (point < members).any(axis=1)
            if bool(dominated.any()):
                keep = ~dominated
                self._front = [item for item, kept in zip(self._front, keep.tolist(), strict=True) if kept]
                members = members[keep]
        self._front.append((values, evaluation))
        row: npt.NDArray[np.float64] = point[None, :]
        self._points = row if members is None or not members.shape[0] else np.concatenate((members, row), axis=0)  # pyright: ignore[reportUnknownMemberType]


@dataclass(frozen=True)
class ObjectiveStats:
    """One objective over the `OK` evaluations of a batch, in natural units: `best` and `worst` follow the declared direction
    (for a maximised objective the best is the largest). All four are NaN when the batch has no `OK` evaluation."""

    best: float
    mean: float
    median: float
    worst: float


@dataclass(frozen=True)
class BatchStats:
    """The last batch that was told to the strategy (a generation, in generation delivery; a single candidate in steady state).

    `ok`, `failed` and `timeout` count its evaluations by status, and `feasible_fraction` is the share of **all** its
    evaluations that are `OK` with no constraint violated (a failed evaluation is infeasible). `objectives` are the
    statistics per objective, over the `OK` evaluations only.
    """

    size: int
    ok: int
    failed: int
    timeout: int
    feasible_fraction: float
    objectives: Mapping[str, ObjectiveStats]


def batch_stats(evaluations: Sequence[Evaluation[Any]], objectives: Sequence[Objective]) -> BatchStats:
    """The statistics of a told batch (see `BatchStats`): plain numpy on the host, since it is a few numbers per objective."""
    ok = [e for e in evaluations if e.status is Status.OK]
    failed = sum(1 for e in evaluations if e.status is Status.FAILED)
    timeout = sum(1 for e in evaluations if e.status is Status.TIMEOUT)
    feasible = sum(1 for e in ok if all(v == 0 for v in e.constraints.values()))
    per_objective: dict[str, ObjectiveStats] = {}
    for objective in objectives:
        if not ok:
            per_objective[objective.name] = ObjectiveStats(math.nan, math.nan, math.nan, math.nan)
            continue
        values = np.asarray([e.objectives[objective.name] for e in ok], dtype=np.float64)
        low, high = float(values.min()), float(values.max())
        best, worst = (low, high) if objective.direction == "minimise" else (high, low)
        per_objective[objective.name] = ObjectiveStats(best, float(values.mean()), float(np.median(values)), worst)
    return BatchStats(
        size=len(evaluations),
        ok=len(ok),
        failed=failed,
        timeout=timeout,
        feasible_fraction=feasible / len(evaluations) if evaluations else math.nan,
        objectives=MappingProxyType(per_objective),
    )


__all__ = ["BatchStats", "ObjectiveStats", "ResultTracker", "batch_stats"]
