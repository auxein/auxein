"""Evaluation results of a batch, as records or as columns (design doc §4.3)."""

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, overload

import numpy as np

from auxein.backend import Array, Backend
from auxein.core._typing import G
from auxein.core.candidate import Candidate
from auxein.core.episodes import EpisodeRecords
from auxein.core.evaluation import Evaluation, Objective, Status

NO_VALUE = math.nan
INFEASIBLE = math.inf


def to_minimisation(values: Array, objectives: Sequence[Objective], backend: Backend) -> Array:
    """Convert an `(n, k)` array of objective values in natural units to minimisation form.

    Maximised columns are negated, using each objective's declared direction, so that strategies never handle signs
    themselves: lower is always better in the result.
    """
    signs = backend.asarray([o.sign for o in objectives])
    return values * signs


@dataclass(frozen=True)
class EvaluationBatch(Generic[G]):
    """The evaluations of a batch of candidates, in the order the candidates were asked for.

    It reads as a sequence of `Evaluation` records, or as columnar arrays on a backend, for strategies that work on
    arrays. Failed and timed-out evaluations have no usable values: they appear as NaN in the objective and descriptor
    columns, and as infinite violation in the constraint columns, so that they are infeasible and rank last (the
    default failure policy, design doc §6.6). A missing value in an evaluation with status OK is an error.
    """

    evaluations: Sequence[Evaluation[G]]
    episodes: EpisodeRecords | None = None
    """The per-scenario measurements behind the evaluations, for the recorder (an episode evaluator sets it; design doc §6.4)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluations", tuple(self.evaluations))

    def __len__(self) -> int:
        return len(self.evaluations)

    def __iter__(self) -> Iterator[Evaluation[G]]:
        return iter(self.evaluations)

    @overload
    def __getitem__(self, index: int) -> Evaluation[G]: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Evaluation[G]]: ...
    def __getitem__(self, index: int | slice) -> Evaluation[G] | Sequence[Evaluation[G]]:
        return self.evaluations[index]

    @property
    def candidates(self) -> tuple[Candidate[G], ...]:
        return tuple(e.candidate for e in self.evaluations)

    def _column(
        self, names: Sequence[str], pick: Callable[[Evaluation[G]], Mapping[str, float]], fill: float, kind: str
    ) -> list[list[float]]:
        rows: list[list[float]] = []
        for e in self.evaluations:
            if e.status is not Status.OK:
                rows.append([fill] * len(names))
                continue
            values = pick(e)
            row: list[float] = []
            for name in names:
                try:
                    row.append(float(values[name]))
                except KeyError:
                    raise KeyError(f"candidate {e.candidate.id} has status OK but reported no {kind} {name!r}") from None
            rows.append(row)
        return rows

    def _matrix(self, rows: list[list[float]], width: int, backend: Backend) -> Array:
        return backend.xp.reshape(backend.asarray(rows), (len(rows), width))

    def objectives_matrix(self, objectives: Sequence[Objective], backend: Backend) -> Array:
        """An `(n, k)` array of objective values in natural units, columns in the order of `objectives`."""
        names = [o.name for o in objectives]
        return self._matrix(self._column(names, lambda e: e.objectives, NO_VALUE, "objective"), len(names), backend)

    def minimisation_matrix(self, objectives: Sequence[Objective], backend: Backend) -> Array:
        """Like `objectives_matrix`, converted to minimisation form (see `to_minimisation`)."""
        return to_minimisation(self.objectives_matrix(objectives, backend), objectives, backend)

    def constraints_matrix(self, names: Sequence[str], backend: Backend) -> Array:
        """An `(n, m)` array of constraint violations (0 = satisfied), columns in the order of `names`."""
        return self._matrix(self._column(names, lambda e: e.constraints, INFEASIBLE, "constraint"), len(names), backend)

    def descriptors_matrix(self, names: Sequence[str], backend: Backend) -> Array:
        """An `(n, m)` array of descriptor values, columns in the order of `names`."""
        return self._matrix(self._column(names, lambda e: e.descriptors, NO_VALUE, "descriptor"), len(names), backend)

    def total_violation(self, backend: Backend, names: Sequence[str] | None = None) -> Array:
        """An `(n,)` array with the summed violation of the given constraints (all reported ones by default).

        Failed and timed-out evaluations have infinite violation.
        """
        return backend.asarray(self.violation_list(names))

    def violation_list(self, names: Sequence[str] | None = None) -> list[float]:
        """`total_violation` as a plain list on the host. A strategy that must know whether anything failed can test
        `INFEASIBLE in violation_list()` at the speed of a list scan, without a pass over an array."""
        return [
            INFEASIBLE
            if e.status is not Status.OK
            else float(sum(e.constraints[n] for n in names) if names is not None else sum(e.constraints.values()))
            for e in self.evaluations
        ]

    def feasible_mask(self, backend: Backend) -> Array:
        """An `(n,)` boolean array: status OK and no constraint violated."""
        feasible = [e.status is Status.OK and all(v == 0 for v in e.constraints.values()) for e in self.evaluations]
        return backend.asarray(feasible, dtype=backend.bool_dtype)

    def status_mask(self, status: Status, backend: Backend | None = None) -> Array:
        """An `(n,)` boolean array: which evaluations have this status.

        Status is host metadata (design doc §7.2), so without a backend the mask is a numpy array on the host.
        """
        mask = [e.status is status for e in self.evaluations]
        if backend is None:
            return np.asarray(mask, dtype=bool)
        return backend.asarray(mask, dtype=backend.bool_dtype)
