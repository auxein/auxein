"""The aggregator (design doc §6.5): per-scenario measurements in, objectives, constraints and descriptors out."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from auxein.aggregators.reductions import Reduction
from auxein.backend import Array, Backend
from auxein.core import ProblemSpec

Column = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Aggregated:
    """What an aggregator made of a batch of candidates: host float64 columns of shape `(n,)`, by name."""

    objectives: Mapping[str, Column]
    constraints: Mapping[str, Column]
    descriptors: Mapping[str, Column]
    cost: Mapping[str, Column]
    invalid: Mapping[int, str] = field(default_factory=dict[int, str])
    """Rows whose constraint, descriptor or cost values are not finite, with why: a diverged simulation. Those candidates fail."""


class Aggregator:
    """Maps the measurements of every candidate on every scenario to its objectives, constraints and descriptors.

    It is declarative and swappable: each of `objectives`, `constraints`, `descriptors` (and `cost`, user-defined cost units
    such as tokens) maps a name to a `Reduction`, a source and a reducer applied across scenarios (see
    `auxein.aggregators`). For example::

        Aggregator(
            objectives={"fuel": mean("fuel_used"), "time": mean("time_to_waypoint")},
            constraints={"cpa": maximum(lambda m: relu(0.5 - m["closest_approach_nm"]))},
            descriptors={"mean_speed": mean("mean_speed")},
        )

    **It always works on arrays of shape `(n, s)`** (`n` candidates, `s` scenarios): the per-episode path of the episode
    evaluator stacks its results into the same shape, so one vectorised, backend-generic code path serves both. Its names must
    match the `ProblemSpec` exactly (`validate`), so a typo is caught like a typo in a `Result`.
    """

    def __init__(
        self,
        objectives: Mapping[str, Reduction],
        constraints: Mapping[str, Reduction] | None = None,
        descriptors: Mapping[str, Reduction] | None = None,
        cost: Mapping[str, Reduction] | None = None,
    ) -> None:
        if not objectives:
            raise ValueError("an aggregator needs at least one objective")
        self.objectives = dict(objectives)
        self.constraints = dict(constraints or {})
        self.descriptors = dict(descriptors or {})
        self.cost = dict(cost or {})
        for kind, group in (
            ("objective", self.objectives),
            ("constraint", self.constraints),
            ("descriptor", self.descriptors),
            ("cost unit", self.cost),
        ):
            for name, reduction in group.items():
                if not name:
                    raise ValueError(f"{kind} names must be non-empty")
                if not isinstance(reduction, Reduction):  # pyright: ignore[reportUnnecessaryIsInstance]
                    raise TypeError(f"the {kind} {name!r} must be a Reduction such as mean('fuel'), got {type(reduction).__name__}")

    def validate(self, problem: ProblemSpec[object]) -> None:
        """Raise `ValueError` unless the aggregator's objectives, constraints and descriptors are exactly the problem's."""
        details: list[str] = []
        for kind, declared, given in (
            ("objectives", problem.objective_names, self.objectives),
            ("constraints", problem.constraints, self.constraints),
            ("descriptors", problem.descriptors, self.descriptors),
        ):
            missing = [n for n in declared if n not in given]
            unknown = [n for n in given if n not in declared]
            if missing:
                details.append(f"{kind} the problem declares but the aggregator lacks: {missing}")
            if unknown:
                details.append(f"{kind} the aggregator computes but the problem does not declare: {unknown}")
        if details:
            raise ValueError(f"the aggregator does not match the problem: {'; '.join(details)}")

    def aggregate(self, measurements: Mapping[str, Array], backend: Backend) -> Aggregated:
        """Reduce `(n, s)` measurement arrays to one value per candidate, as host float64 columns."""
        xp = backend.xp
        groups = [
            {name: _to_host(reduction.reduce(measurements, xp), backend) for name, reduction in group.items()}
            for group in (self.objectives, self.constraints, self.descriptors, self.cost)
        ]
        objectives, constraints, descriptors, cost = groups
        invalid: dict[int, str] = {}
        # the columns are on the host by now (see `_to_host`), so scanning for non-finite values is a numpy job
        for kind, columns in (("constraint", constraints), ("descriptor", descriptors), ("cost unit", cost)):
            for name, column in columns.items():
                for row in np.nonzero(~np.isfinite(column))[0].tolist():
                    text = f"{kind} {name!r} is {float(column[row])}"
                    invalid[row] = f"{invalid[row]}, {text}" if row in invalid else text
        return Aggregated(objectives, constraints, descriptors, cost, invalid)

    def describe(self) -> str:
        parts = [f"objectives={self.objectives!r}"]
        for name, group in (("constraints", self.constraints), ("descriptors", self.descriptors), ("cost", self.cost)):
            if group:
                parts.append(f"{name}={group!r}")
        return f"Aggregator({', '.join(parts)})"

    def __repr__(self) -> str:
        return self.describe()


def _to_host(column: Array, backend: Backend) -> Column:
    """One reduced column to the host, as float64. Host-side by design (design doc §7.2): the reduction itself ran on the
    backend's device over `(n, s)` arrays, and what leaves it is one number per candidate, the numbers an `Evaluation` holds
    as Python floats and the recorder writes. It is one small transfer per name per batch, never one per scenario or episode."""
    host = np.asarray(backend.to_numpy(column), dtype=np.float64)
    if host.ndim != 1:
        raise ValueError(f"a reduction must produce one value per candidate (shape (n,)), got shape {host.shape}")
    return host
