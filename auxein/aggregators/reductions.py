"""Reductions: how the measurements of many scenarios become one number per candidate (design doc §6.5).

A reduction is a *source* (a measurement name, or a function of the dict of measurement arrays) and a *reducer* applied across
scenarios, axis 1 of an `(n, s)` array with `n` candidates and `s` scenarios. Everything is written against the array
namespace of its input, so the same code reduces numpy arrays and torch tensors on any device.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

from auxein.backend import Array, ArrayNamespace

Source: TypeAlias = "str | Callable[[Mapping[str, Array]], Array]"
"""A measurement name, or a function from the dict of `(n, s)` measurement arrays to one `(n, s)` array."""

Reducer: TypeAlias = Callable[[Array, ArrayNamespace], Array]
"""`(values of shape (n, s), array namespace) -> array of shape (n,)`."""


def _name_of(function: object) -> str:
    return str(getattr(function, "__qualname__", type(function).__name__))


@dataclass(frozen=True)
class Reduction:
    """A source and a reducer, with a name that describes them (it goes into the evaluator's description)."""

    source: Source
    reducer: Reducer
    label: str

    def values(self, measurements: Mapping[str, Array]) -> Array:
        """The `(n, s)` array this reduction reads."""
        if callable(self.source):
            return self.source(measurements)
        try:
            return measurements[self.source]
        except KeyError:
            raise KeyError(
                f"the aggregator reads the measurement {self.source!r}, but the environment reported {sorted(measurements)}"
            ) from None

    def reduce(self, measurements: Mapping[str, Array], xp: ArrayNamespace) -> Array:
        """One value per candidate, shape `(n,)`."""
        values = self.values(measurements)
        if len(values.shape) != 2:
            raise ValueError(f"{self.label} needs an array of shape (candidates, scenarios), got shape {tuple(values.shape)}")
        return self.reducer(values, xp)

    def __repr__(self) -> str:
        return self.label


def _label(name: str, source: Source, *extra: object) -> str:
    shown = repr(source) if isinstance(source, str) else f"<{_name_of(source)}>"
    return f"{name}({', '.join([shown, *(repr(e) for e in extra)])})"


def mean(source: Source) -> Reduction:
    """The mean over scenarios."""
    return Reduction(source, lambda v, xp: xp.mean(v, axis=1), _label("mean", source))


def minimum(source: Source) -> Reduction:
    """The smallest value over scenarios (the worst case of something where lower is worse)."""
    return Reduction(source, lambda v, xp: xp.min(v, axis=1), _label("minimum", source))


def maximum(source: Source) -> Reduction:
    """The largest value over scenarios (the worst case of something where higher is worse, such as a constraint violation)."""
    return Reduction(source, lambda v, xp: xp.max(v, axis=1), _label("maximum", source))


def total(source: Source) -> Reduction:
    """The sum over scenarios."""
    return Reduction(source, lambda v, xp: xp.sum(v, axis=1), _label("total", source))


def quantile(source: Source, q: float) -> Reduction:
    """The `q`-quantile over scenarios, `q` in [0, 1], by linear interpolation between the sorted values (as numpy's default)."""
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")

    def reduce(values: Array, xp: ArrayNamespace) -> Array:
        ordered = xp.sort(values, axis=1)
        position = q * (ordered.shape[1] - 1)
        low, high = math.floor(position), math.ceil(position)
        fraction = position - low
        return ordered[:, low] * (1.0 - fraction) + ordered[:, high] * fraction

    return Reduction(source, reduce, _label("quantile", source, q))


def _tail(alpha: float) -> None:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")


def _tail_size(alpha: float, scenarios: int) -> int:
    """How many scenarios the tail holds: `ceil(alpha * s)`, at least 1 and at most `s`.

    `alpha * s` is computed in binary floating point, where `0.07 * 100` is `7.000000000000001`, so a plain `ceil` would put
    one scenario too many in the tail. A hair of tolerance (far below any `alpha` anyone writes, far above rounding) makes
    the count the one the decimal arithmetic gives.
    """
    return min(scenarios, max(1, math.ceil(alpha * scenarios - 1e-9)))


def cvar_upper(source: Source, alpha: float) -> Reduction:
    """CVaR of the **upper** tail: the mean of the largest `ceil(alpha * s)` values over `s` scenarios.

    This is the "worst α fraction" of something where **higher is worse** (fuel, time, a constraint violation, cost).
    `alpha=1` is the mean, and a very small `alpha` is the maximum.
    """
    _tail(alpha)

    def reduce(values: Array, xp: ArrayNamespace) -> Array:
        s = values.shape[1]
        k = _tail_size(alpha, s)
        return xp.mean(xp.sort(values, axis=1)[:, s - k :], axis=1)

    return Reduction(source, reduce, _label("cvar_upper", source, alpha))


def cvar_lower(source: Source, alpha: float) -> Reduction:
    """CVaR of the **lower** tail: the mean of the smallest `ceil(alpha * s)` values over `s` scenarios.

    This is the "worst α fraction" of something where **lower is worse** (a reward, a success rate, a safety margin).
    `alpha=1` is the mean, and a very small `alpha` is the minimum.
    """
    _tail(alpha)

    def reduce(values: Array, xp: ArrayNamespace) -> Array:
        s = values.shape[1]
        k = _tail_size(alpha, s)
        return xp.mean(xp.sort(values, axis=1)[:, :k], axis=1)

    return Reduction(source, reduce, _label("cvar_lower", source, alpha))
