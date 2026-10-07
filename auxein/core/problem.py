"""The problem specification handed to strategies (design doc §3.2)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

from auxein.core._typing import G
from auxein.core.evaluation import Objective
from auxein.spaces import Space


def _check_names(kind: str, names: Sequence[str]) -> None:
    for name in names:
        if not name:
            raise ValueError(f"{kind} names must be non-empty")
    if len(set(names)) != len(names):
        duplicated = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"{kind} names must be unique, but {duplicated} appear more than once")


@dataclass(frozen=True)
class ProblemSpec(Generic[G]):
    """What is being searched: the space, what to optimise, and what to record.

    Names must be non-empty and unique within objectives, constraints and descriptors. At least one objective is
    required. Sequences are stored as tuples.
    """

    space: Space[G]
    objectives: tuple[Objective, ...]
    constraints: tuple[str, ...] = ()
    descriptors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "objectives", tuple(self.objectives))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "descriptors", tuple(self.descriptors))
        if not self.objectives:
            raise ValueError("a problem needs at least one objective")
        _check_names("objective", [o.name for o in self.objectives])
        _check_names("constraint", self.constraints)
        _check_names("descriptor", self.descriptors)

    @property
    def objective_names(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.objectives)
