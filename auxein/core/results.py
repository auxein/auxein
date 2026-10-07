"""Explicit results returned by user code (design doc §5.3).

A fitness function that returns anything richer than one number must say what each value is, with a `Result`
(per candidate) or a `BatchResult` (per batch, from a vectorised function). There is no flat-dict format: a dict can't
say which keys are objectives, constraints or descriptors.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import numpy as np
import numpy.typing as npt

from auxein.backend import Array, Backend


def _number(value: object, kind: str, name: str) -> float:
    try:
        return float(value)  # type: ignore[arg-type]  # numbers of any supported backend: int, float, numpy scalar, 0-d array
    except (TypeError, ValueError):
        raise TypeError(f"{kind} {name!r} must be a number, got {type(value).__name__}") from None


def _check_names(names: Mapping[str, object], kind: str) -> None:
    for name in names:
        if not isinstance(name, str) or not name:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"{kind} names must be non-empty strings, got {name!r}")


@dataclass(frozen=True)
class Result:
    """What a fitness function returns for one candidate, when a bare number is not enough.

    `objectives`, `constraints` and `descriptors` map names to numbers, and must match the problem exactly: every
    declared name present, no unknown name (this catches typos). Objective values are in natural units. Constraint values
    are violation amounts (0 = satisfied, above 0 = violated). `cost` holds user-defined cost units (tokens, money,
    simulated seconds); wall time is measured by the evaluator. Values are converted to floats, and the mappings are
    copied and made read-only.
    """

    objectives: Mapping[str, float]
    constraints: Mapping[str, float] = field(default_factory=dict[str, float])
    descriptors: Mapping[str, float] = field(default_factory=dict[str, float])
    cost: Mapping[str, float] = field(default_factory=dict[str, float])

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError("a Result needs at least one objective")
        for kind, values in (
            ("objective", self.objectives),
            ("constraint", self.constraints),
            ("descriptor", self.descriptors),
            ("cost unit", self.cost),
        ):
            _check_names(values, kind)
            object.__setattr__(
                self, {"cost unit": "cost"}.get(kind, kind + "s"), MappingProxyType({n: _number(v, kind, n) for n, v in values.items()})
            )
        for name, violation in self.constraints.items():
            if not (math.isfinite(violation) and violation >= 0):
                raise ValueError(f"constraint {name!r} must be a finite violation amount of at least 0, got {violation}")
        for name, amount in self.cost.items():
            if not (math.isfinite(amount) and amount >= 0):
                raise ValueError(f"cost unit {name!r} must be finite and non-negative, got {amount}")


def _column(values: Array, kind: str, name: str) -> npt.NDArray[np.float64]:
    array: npt.NDArray[np.float64] = np.asarray(Backend().to_numpy(values), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{kind} {name!r} must be a 1-D array of one value per candidate, got shape {array.shape}")
    return array


@dataclass(frozen=True)
class BatchResult:
    """What a vectorised function returns for a whole batch, when a bare array is not enough.

    The same fields as `Result`, but each name maps to an array of shape `(n,)`, with one value per candidate in batch
    order. The arrays may be on any supported backend; they are copied to host float64 arrays, because evaluation
    records are plain Python values. `cost` holds optional per-candidate arrays of user-defined cost units.
    """

    objectives: Mapping[str, Array]
    constraints: Mapping[str, Array] = field(default_factory=dict[str, Array])
    descriptors: Mapping[str, Array] = field(default_factory=dict[str, Array])
    cost: Mapping[str, Array] = field(default_factory=dict[str, Array])

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError("a BatchResult needs at least one objective")
        sizes: dict[str, int] = {}
        for kind, attribute, values in (
            ("objective", "objectives", self.objectives),
            ("constraint", "constraints", self.constraints),
            ("descriptor", "descriptors", self.descriptors),
            ("cost unit", "cost", self.cost),
        ):
            _check_names(values, kind)
            columns = {name: _column(v, kind, name) for name, v in values.items()}
            for name, column in columns.items():
                sizes[f"{kind} {name!r}"] = int(column.shape[0])
            object.__setattr__(self, attribute, MappingProxyType(columns))
        if len(set(sizes.values())) > 1:
            raise ValueError(f"all the arrays of a BatchResult must have the same length, got {sizes}")
        for name, column in self.constraints.items():
            if not (np.isfinite(column).all() and (column >= 0).all()):
                raise ValueError(f"constraint {name!r} must be finite violation amounts of at least 0")
        for name, column in self.cost.items():
            if not (np.isfinite(column).all() and (column >= 0).all()):
                raise ValueError(f"cost unit {name!r} must be finite and non-negative")

    @property
    def size(self) -> int:
        """The number of candidates the result covers."""
        return int(next(iter(self.objectives.values())).shape[0])

    def host_columns(self, mapping: Mapping[str, Array]) -> dict[str, npt.NDArray[np.float64]]:
        """The arrays of one of the mappings (`objectives`, `constraints`, ...), as host float64 arrays."""
        return {name: cast("npt.NDArray[np.float64]", column) for name, column in mapping.items()}
