"""Budgets (design doc §9.3)."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class Budget:
    """Limits on a run; it stops when any one of them is exhausted. At least one limit is required.

    - `evaluations` is a **hard** limit: if fewer evaluations remain than a strategy asks for, only the remaining
      candidates are evaluated (in ask order) and recorded, and the run ends without telling the strategy about the
      incomplete batch. Every evaluation counts, including initialisation.
    - `wall_time` (seconds) and `cost` (limits on the summed user-defined cost units, e.g. tokens or money) are checked
      **between batches**: the batch in progress always completes, so they can be exceeded by up to one batch.
    """

    evaluations: int | None = None
    wall_time: float | None = None
    cost: Mapping[str, float] = field(default_factory=dict[str, float])

    def __post_init__(self) -> None:
        if self.evaluations is None and self.wall_time is None and not self.cost:
            raise ValueError("a budget needs at least one limit: evaluations, wall_time or cost")
        if self.evaluations is not None and self.evaluations < 1:
            raise ValueError(f"the evaluation budget must be at least 1, got {self.evaluations}")
        if self.wall_time is not None and not (math.isfinite(self.wall_time) and self.wall_time > 0):
            raise ValueError(f"the wall-time budget must be a positive number of seconds, got {self.wall_time}")
        for unit, limit in self.cost.items():
            if not (math.isfinite(limit) and limit > 0):
                raise ValueError(f"the budget for cost unit {unit!r} must be positive and finite, got {limit}")
        object.__setattr__(self, "cost", MappingProxyType(dict(self.cost)))

    def describe(self) -> dict[str, object]:
        """A JSON-serialisable description, for run metadata."""
        return {"evaluations": self.evaluations, "wall_time": self.wall_time, "cost": dict(self.cost)}
