"""`Box`: a bounded real vector, with an optional log scale per dimension."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from auxein.backend import Array, Backend, backend_of
from auxein.random import RandomStream

_HOST = Backend()
FloatVector = npt.NDArray[np.float64]


def _bounds_array(values: float | Sequence[float] | Array, name: str) -> FloatVector:
    array: FloatVector = np.asarray(_HOST.to_numpy(values), dtype=np.float64)
    if array.ndim > 1:
        raise ValueError(f"{name} must be a scalar or a 1-D sequence, got {array.ndim} dimensions")
    return array


def _expand(values: FloatVector, size: int) -> FloatVector:
    """A scalar broadcast to `size` values, or a copy of an array that already has them."""
    return np.full(size, values.item(), dtype=np.float64) if values.ndim == 0 else values.copy()


def _expand_flags(values: npt.NDArray[np.bool_], size: int) -> npt.NDArray[np.bool_]:
    return np.full(size, values.item(), dtype=bool) if values.ndim == 0 else values.copy()


def _masked(values: FloatVector, mask: npt.NDArray[np.bool_]) -> FloatVector:
    """`values` where `mask` is set and 1 elsewhere, so that a logarithm of it is always defined."""
    result = np.ones(values.shape, dtype=np.float64)
    result[mask] = values[mask]
    return result


class Box:
    """A bounded real vector: `lower[i] <= genome[i] <= upper[i]`, uniform on each dimension, log-uniform on the
    dimensions flagged in `log_scale` (useful for scale parameters such as learning rates).

    Scalar bounds are broadcast to `dim`; bounds given per dimension define `dim` themselves. The bounds are kept in
    float64 on the host and are converted when sampling.

    Samples are guaranteed to lie within `[lower, upper]` in float32 as well as float64: a float32 bound is rounded
    *inward* (a bound that does not round-trip exactly is moved to the next float32 inside the box) and samples are
    clipped to it, which also guards against rounding at the upper bound. A box too narrow for float32 to represent is
    an error at sampling time.
    """

    def __init__(
        self,
        lower: float | Sequence[float] | Array,
        upper: float | Sequence[float] | Array,
        dim: int | None = None,
        log_scale: bool | Sequence[bool] = False,
    ) -> None:
        low, high = _bounds_array(lower, "lower"), _bounds_array(upper, "upper")
        lengths = {int(a.shape[0]) for a in (low, high) if a.ndim == 1}
        if len(lengths) > 1:
            raise ValueError(f"lower and upper have different lengths: {low.shape[0]} and {high.shape[0]}")
        if lengths:
            (size,) = lengths
            if dim is not None and dim != size:
                raise ValueError(f"dim={dim} does not match the bounds, which have {size} dimensions")
        elif dim is None:
            raise ValueError("dim is required when both bounds are scalars")
        else:
            size = dim
        if size < 1:
            raise ValueError(f"a box needs at least one dimension, got {size}")

        low, high = _expand(low, size), _expand(high, size)
        if not (np.isfinite(low).all() and np.isfinite(high).all()):
            raise ValueError("bounds must be finite")
        if not (low < high).all():
            raise ValueError("lower must be strictly below upper in every dimension")
        with np.errstate(over="ignore"):
            finite_range = bool(np.isfinite(high - low).all())
        if not finite_range:
            raise ValueError("the range upper - lower must be finite in every dimension")

        logs: npt.NDArray[np.bool_] = np.asarray(log_scale, dtype=bool)
        if logs.ndim > 1 or (logs.ndim == 1 and logs.shape[0] != size):
            raise ValueError(f"log_scale must be a bool or a sequence of {size} bools, got shape {logs.shape}")
        logs = _expand_flags(logs, size)
        if (logs & (low <= 0)).any():
            raise ValueError("log-scale dimensions need lower > 0")

        for array in (low, high, logs):
            array.setflags(write=False)
        self._lower, self._upper, self._log = low, high, logs
        self._inward: dict[str, tuple[FloatVector, FloatVector]] = {}

    @property
    def dim(self) -> int:
        return int(self._lower.shape[0])

    @property
    def lower(self) -> FloatVector:
        """The lower bounds, float64, read-only."""
        return self._lower

    @property
    def upper(self) -> FloatVector:
        """The upper bounds, float64, read-only."""
        return self._upper

    @property
    def log_scale(self) -> npt.NDArray[np.bool_]:
        """Which dimensions are log-scale, read-only."""
        return self._log

    def __repr__(self) -> str:
        return f"Box(lower={self._lower.tolist()}, upper={self._upper.tolist()}, log_scale={self._log.tolist()})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Box)
            and self.dim == other.dim
            and bool((self._lower == other._lower).all())
            and bool((self._upper == other._upper).all())
            and bool((self._log == other._log).all())
        )

    def __hash__(self) -> int:
        return hash((self._lower.tobytes(), self._upper.tobytes(), self._log.tobytes()))

    def _bounds(self, precision: str) -> tuple[FloatVector, FloatVector]:
        """The bounds as floats of `precision`, rounded inward so that they never leave `[lower, upper]`."""
        cached = self._inward.get(precision)
        if cached is not None:
            return cached
        dtype = np.dtype(precision)
        with np.errstate(over="ignore"):
            low, high = self._lower.astype(dtype), self._upper.astype(dtype)
            too_low = low.astype(np.float64) < self._lower
            too_high = high.astype(np.float64) > self._upper
            low[too_low] = np.nextafter(low[too_low], np.array(np.inf, dtype))
            high[too_high] = np.nextafter(high[too_high], np.array(-np.inf, dtype))
            usable = bool((low <= high).all() and np.isfinite(high - low).all())
        if not usable:
            raise ValueError(f"the box is too narrow or too wide to be represented in {precision}")
        result = (low, high)
        self._inward[precision] = result
        return result

    def sample_genomes(self, n: int, rng: RandomStream, backend: Backend) -> Array:
        """An `(n, d)` array on `backend`: uniform per dimension, log-uniform on the log-scale dimensions.

        Every value lies within `[lower, upper]`, in float32 as well as float64.
        """
        if n < 0:
            raise ValueError(f"n must not be negative, got {n}")
        if rng.backend != backend:
            raise ValueError(f"the random stream is on {rng.backend} but samples were requested on {backend}")
        xp = backend.xp
        low_host, high_host = self._bounds(backend.precision)
        low, high = backend.asarray(low_host), backend.asarray(high_host)

        unit = rng.uniform((n, self.dim))
        genomes = low + unit * (high - low)
        if self._log.any():
            log_low = np.log(_masked(self._lower, self._log))
            log_high = np.log(_masked(self._upper, self._log))
            log_low_a, log_high_a = backend.asarray(log_low), backend.asarray(log_high)
            logged = xp.exp(log_low_a + unit * (log_high_a - log_low_a))
            genomes = xp.where(backend.asarray(self._log, dtype=backend.bool_dtype), logged, genomes)
        return xp.minimum(xp.maximum(genomes, low), high)

    def contains(self, genome: Array) -> bool:
        """Whether a genome (a vector of length `dim`) is finite and within the bounds."""
        try:
            host = _HOST.to_numpy(genome)
            x = np.asarray(host, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if x.shape != (self.dim,) or not bool(np.isfinite(x).all()):
            return False
        return bool(((x >= self._lower) & (x <= self._upper)).all())

    def clip(self, genomes: Array) -> Array:
        """Clip genomes of shape `(d,)` or `(n, d)` into the box, in the dtype and on the device they already have.

        This repairs out-of-bounds values for operators that need it. The bounds are rounded inward for float32, so a
        clipped genome is always `contains`-valid. NaNs pass through unchanged.
        """
        backend = backend_of(genomes)
        if not backend.matches(genomes):
            raise TypeError("clip expects floating-point genomes (float32 or float64)")
        if genomes.shape[-1] != self.dim:
            raise ValueError(f"genomes must have {self.dim} values in their last dimension, got shape {tuple(genomes.shape)}")
        low_host, high_host = self._bounds(backend.precision)
        xp = backend.xp
        return xp.minimum(xp.maximum(genomes, backend.asarray(low_host)), backend.asarray(high_host))
