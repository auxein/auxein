"""Reductions: how the measurements of many scenarios become one number per candidate (design doc §6.5).

A reduction is a *source* (a measurement name, or a function of the dict of measurement arrays) and a *reducer* applied across
scenarios, axis 1 of an `(n, s)` array with `n` candidates and `s` scenarios. Its label names both, and a parameterised source
(a callable object with a `describe()` or a `__repr__`, or a `functools.partial`) shows its parameters in it, so that resuming
a run with a changed parameter is refused like any other change of the evaluator. Everything is written against the array
namespace of its input, so the same code reduces numpy arrays and torch tensors on any device.
"""

import functools
import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from auxein.backend import Array, ArrayNamespace

Source: TypeAlias = "str | Callable[[Mapping[str, Array]], Array]"
"""A measurement name, or a function from the dict of `(n, s)` measurement arrays to one `(n, s)` array."""

Reducer: TypeAlias = Callable[[Array, ArrayNamespace], Array]
"""`(values of shape (n, s), array namespace) -> array of shape (n,)`."""


def _name_of(function: object) -> str:
    return str(getattr(function, "__qualname__", type(function).__name__))


def _describe_source(source: Callable[..., object]) -> str:
    """How a callable source appears in a reduction's label, which goes into the evaluator's description and so into what
    resume compares (design doc §10.4): it must show everything that changes the values, and nothing that changes from one
    process to the next (a memory address).

    - A function, a method, a builtin or anything else with a qualified name of its own (a numpy ufunc): that name, as always.
    - A `functools.partial`: the function's name with the bound arguments, `weighted(0.5, scale=2)`.
    - A callable object with a `describe()`: what it returns; else with a `__repr__` of its own class: that repr. A
      parameterised source (fitness weights, a threshold) shows its parameters this way.
    - Any other callable object: its class name.
    """
    # a partial first: from Python 3.14 it is a method descriptor, which `inspect.isroutine` counts as a routine
    if isinstance(source, functools.partial):
        bound = cast("functools.partial[object]", source)
        arguments = [repr(a) for a in bound.args] + [f"{k}={v!r}" for k, v in bound.keywords.items()]
        return f"{_describe_source(bound.func)}({', '.join(arguments)})"
    if inspect.isroutine(source) or hasattr(source, "__qualname__"):
        return _name_of(source)
    describe = getattr(source, "describe", None)
    if callable(describe):
        return str(describe())
    if type(source).__repr__ is not object.__repr__:
        return repr(source)
    return _name_of(source)


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
    shown = repr(source) if isinstance(source, str) else f"<{_describe_source(source)}>"
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
    """The `q`-quantile over scenarios, `q` in [0, 1], by linear interpolation between the sorted values (as numpy's default).

    References:
        Hyndman, R. J. and Fan, Y. (1996). Sample quantiles in statistical packages. The American Statistician, 50(4),
            361-365 (definition 7). https://doi.org/10.2307/2684934
    """
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

    References:
        Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional value-at-risk. The Journal of Risk,
            2(3), 21-41.
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

    References:
        Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional value-at-risk. The Journal of Risk,
            2(3), 21-41.
    """
    _tail(alpha)

    def reduce(values: Array, xp: ArrayNamespace) -> Array:
        s = values.shape[1]
        k = _tail_size(alpha, s)
        return xp.mean(xp.sort(values, axis=1)[:, :k], axis=1)

    return Reduction(source, reduce, _label("cvar_lower", source, alpha))
