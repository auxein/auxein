"""Tracking results as they arrive, and the final `RunResult` (design doc §9.5)."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic

from auxein.core import Evaluation, Objective, Status
from auxein.core._typing import G


@dataclass(frozen=True)
class RunResult(Generic[G]):
    """What a run found.

    `best` is, for a single-objective problem, the best `OK` evaluation: feasible beats infeasible, then a lower total
    violation, then a lower objective in minimisation form (the declared direction is respected), and ties go to the
    earliest id. It is None for several objectives, or when no `OK` evaluation exists.

    `pareto_front` holds the non-dominated feasible `OK` evaluations, in minimisation form, sorted by candidate id; for a
    single objective it is just the best feasible one.

    `trace` (single objective only) lists `(evaluations_used, best_value)` each time `best` changed, with the value in
    natural units, for quick plots. It follows `best`, so with constraints it can get worse when the first feasible
    candidate replaces an infeasible one with a better value.
    """

    stop_reason: str
    """`budget:evaluations`, `budget:wall_time`, `budget:cost:<unit>` or `strategy`."""
    evaluations_used: int
    wall_time: float
    run_dir: Path | None
    best: Evaluation[G] | None
    pareto_front: tuple[Evaluation[G], ...]
    trace: tuple[tuple[int, float], ...]


def _dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


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

    def add(self, evaluations: Sequence[Evaluation[G]], used_before: int) -> None:
        """Take in a batch of evaluations, `used_before` being the number of evaluations made before it."""
        if self._unconstrained_single:
            self._add_unconstrained_single(evaluations, used_before)
            return
        single = len(self._names) == 1
        for position, evaluation in enumerate(evaluations):
            if evaluation.status is not Status.OK:
                continue
            values = tuple(sign * evaluation.objectives[name] for sign, name in zip(self._signs, self._names))
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
        for member, _ in self._front:
            if member == values or _dominates(member, values):
                return  # dominated, or equal to an earlier one: the earliest candidate keeps the place
        self._front = [(m, e) for m, e in self._front if not _dominates(values, m)]
        self._front.append((values, evaluation))
