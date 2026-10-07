"""Evaluation records (design doc §5.1 and §5.2)."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Generic, Literal

from auxein.core._typing import G
from auxein.core.candidate import Candidate


class Status(Enum):
    """How an evaluation ended. A failure is a result, not a crash (§6.6)."""

    OK = "ok"
    FAILED = "failed"
    TIMEOUT = "timeout"


Direction = Literal["minimise", "maximise"]


@dataclass(frozen=True)
class Objective:
    """A quantity to optimise. Directions are explicit: users never negate values, strategies convert (§5.2)."""

    name: str
    direction: Direction = "minimise"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("an objective needs a non-empty name")
        if self.direction not in ("minimise", "maximise"):
            raise ValueError(f"direction must be 'minimise' or 'maximise', got {self.direction!r}")

    @property
    def sign(self) -> float:
        """1 for a minimised objective and -1 for a maximised one: multiply natural values by it to minimise."""
        return 1.0 if self.direction == "minimise" else -1.0


def _frozen(values: Mapping[str, float], what: str) -> Mapping[str, float]:
    for key in values:
        if not key:
            raise ValueError(f"{what} names must be non-empty")
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class Cost:
    """What an evaluation cost: wall time in seconds, plus user-defined units (tokens, money, sim-seconds)."""

    wall_time: float = 0.0
    units: Mapping[str, float] = field(default_factory=dict[str, float])

    def __post_init__(self) -> None:
        if not (math.isfinite(self.wall_time) and self.wall_time >= 0):
            raise ValueError(f"wall_time must be finite and non-negative, got {self.wall_time}")
        for name, amount in self.units.items():
            if not (math.isfinite(amount) and amount >= 0):
                raise ValueError(f"cost unit {name!r} must be finite and non-negative, got {amount}")
        object.__setattr__(self, "units", _frozen(self.units, "cost unit"))


@dataclass(frozen=True)
class RawRef:
    """A reference to the per-scenario measurements of an evaluation (§6.4). Storage arrives with the recorder."""

    key: str

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("a reference needs a non-empty key")


@dataclass(frozen=True)
class ArtifactRef:
    """A reference to a heavy output of an episode: a trajectory, a transcript, a log (§6.2)."""

    key: str

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("a reference needs a non-empty key")


@dataclass(frozen=True)
class Evaluation(Generic[G]):
    """The result of evaluating one candidate.

    Objective values are in natural units (the direction is declared by the problem). They must be finite when the
    status is OK; a failed or timed-out evaluation may carry NaN or infinite values, which strategies must never read
    (the batch views mark such rows through the status mask, and rank them as infeasible). Constraint values are
    violation amounts: 0 means satisfied, above 0 violated, and they are always finite. Descriptors say how the agent
    behaved and are never optimised. The mappings are copied and made read-only.
    """

    candidate: Candidate[G]
    status: Status
    objectives: Mapping[str, float]
    constraints: Mapping[str, float] = field(default_factory=dict[str, float])
    descriptors: Mapping[str, float] = field(default_factory=dict[str, float])
    cost: Cost = field(default_factory=Cost)
    raw: RawRef | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status is Status.OK:
            for name, value in self.objectives.items():
                if not math.isfinite(value):
                    raise ValueError(
                        f"objective {name!r} is {value} but the evaluation is OK: only FAILED or TIMEOUT evaluations may be non-finite"
                    )
        for name, violation in self.constraints.items():
            if not (math.isfinite(violation) and violation >= 0):
                raise ValueError(f"constraint {name!r} must be a finite violation amount of at least 0, got {violation}")
        object.__setattr__(self, "objectives", _frozen(self.objectives, "objective"))
        object.__setattr__(self, "constraints", _frozen(self.constraints, "constraint"))
        object.__setattr__(self, "descriptors", _frozen(self.descriptors, "descriptor"))
