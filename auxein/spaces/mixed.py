"""`MixedSpace`: real, integer, binary and categorical dimensions in one array genome (design doc §4.5).

**Why one array.** A mixed genome is a single numeric array of the backend's float dtype, exactly like a `Box` genome, so
that everything built for arrays keeps working unchanged: the array fast path and `VectorisedEvaluator`, GPUs, recording as
raw bytes, the genome store, replay. The price is that discrete values are stored as integral floats: an integer as itself,
a binary as 0 or 1, a categorical as the *index* of its choice. Decoding to named Python values is the space's job
(`values`, `columns`), done by user code when it needs them; the reader of a recorded run returns arrays.

**Float32.** A float32 holds every integer up to 2**24 exactly, so integer bounds beyond that cannot be represented in a
float32 run: that is an error when the run starts (`check_backend`, called by the strategies' `bind`). Sampling an integer
range wider than 2**23 in float32 is uniform only to float32's resolution of the unit interval.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from auxein.backend import Array, Backend
from auxein.random import RandomStream
from auxein.spaces.codec import CanonicalEncodingError, canonical_json

_HOST = Backend()
FloatVector = npt.NDArray[np.float64]

REAL, INTEGER, BINARY, CATEGORICAL = 0, 1, 2, 3
"""The kind codes of the dimensions, as stored in `MixedSpace.kinds`."""

_KIND_NAMES = {REAL: "real", INTEGER: "integer", BINARY: "binary", CATEGORICAL: "categorical"}
_EXACT_INTEGERS = {"float32": 2**24, "float64": 2**53}
"""The largest magnitude up to which every integer is a float of the given precision."""


@dataclass(frozen=True)
class Real:
    """A real dimension in `[lower, upper]`, uniform, or log-uniform with `log=True` (then `lower` must be positive)."""

    lower: float
    upper: float
    log: bool = False

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
            raise ValueError(f"the bounds of a real dimension must be finite, got [{self.lower}, {self.upper}]")
        if not self.lower < self.upper:
            raise ValueError(f"a real dimension needs lower < upper, got [{self.lower}, {self.upper}]")
        if not math.isfinite(self.upper - self.lower):
            raise ValueError("the range upper - lower of a real dimension must be finite")
        if self.log and self.lower <= 0:
            raise ValueError(f"a log-scale dimension needs lower > 0, got {self.lower}")


@dataclass(frozen=True)
class Integer:
    """An integer dimension with inclusive bounds, `lower <= value <= upper`."""

    lower: int
    upper: int

    def __post_init__(self) -> None:
        for name, bound in (("lower", self.lower), ("upper", self.upper)):
            if isinstance(bound, bool) or not isinstance(bound, (int, np.integer)):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(f"the {name} bound of an integer dimension must be an int, got {bound!r}")
        if not self.lower < self.upper:
            raise ValueError(f"an integer dimension needs lower < upper, got [{self.lower}, {self.upper}]")


@dataclass(frozen=True)
class Binary:
    """A binary dimension: 0 or 1 (`False` or `True` when decoded)."""


@dataclass(frozen=True)
class Categorical:
    """A categorical dimension: one of a finite list of at least two distinct JSON-serialisable choices.

    The genome stores the *index* of the choice. Two choices are the same when their canonical encodings are.
    """

    choices: tuple[object, ...] = field()

    def __init__(self, choices: Sequence[object]) -> None:
        if isinstance(choices, str):
            raise TypeError("the choices of a categorical dimension are a list, not a string; wrap it: Categorical([...])")
        items = tuple(choices)
        if len(items) < 2:
            raise ValueError(f"a categorical dimension needs at least two choices, got {len(items)}")
        try:
            encoded = [canonical_json(item) for item in items]
        except CanonicalEncodingError as error:
            raise ValueError(f"the choices of a categorical dimension must be JSON-serialisable: {error}") from error
        if len(set(encoded)) != len(encoded):
            raise ValueError(f"the choices of a categorical dimension must be distinct, got {list(items)!r}")
        object.__setattr__(self, "choices", items)


Dimension: TypeAlias = Real | Integer | Binary | Categorical


def _describe_dimension(name: str, dimension: Dimension) -> dict[str, object]:
    if isinstance(dimension, Real):
        return {"name": name, "type": "real", "lower": dimension.lower, "upper": dimension.upper, "log": dimension.log}
    if isinstance(dimension, Integer):
        return {"name": name, "type": "integer", "lower": int(dimension.lower), "upper": int(dimension.upper)}
    if isinstance(dimension, Binary):
        return {"name": name, "type": "binary"}
    return {"name": name, "type": "categorical", "choices": list(dimension.choices)}


@dataclass(frozen=True)
class _OnDevice:
    """The space's constants as arrays of one backend, built once: bounds, and the masks of the kinds."""

    low: Array
    high: Array
    count: Array
    log_low: Array
    log_high: Array
    is_log: Array
    is_discrete: Array


class MixedSpace:
    """A search space of named dimensions of different types, whose genome is one float array (see the module docstring)::

        MixedSpace({
            "lr": Real(1e-5, 1e-1, log=True),
            "layers": Integer(1, 8),
            "dropout": Binary(),
            "optimiser": Categorical(["sgd", "adam"]),
        })

    Dimensions keep their declared order, which is the order of the genome's columns. Besides sampling and membership it has
    the decoding helpers for user code: `values(genome)` (a dict of Python values) and `columns(genomes)` (one array per
    dimension, for vectorised objectives). `Box` is the all-real case and stays a separate class.
    """

    def __init__(self, dimensions: Mapping[str, Dimension] | Sequence[tuple[str, Dimension]]) -> None:
        pairs = list(dimensions.items()) if isinstance(dimensions, Mapping) else list(dimensions)
        if not pairs:
            raise ValueError("a mixed space needs at least one dimension")
        names = [name for name, _ in pairs]
        for name in names:
            if not isinstance(name, str) or not name:  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ValueError(f"dimension names must be non-empty strings, got {name!r}")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate dimension names: {duplicates}")
        for name, dimension in pairs:
            if not isinstance(dimension, (Real, Integer, Binary, Categorical)):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(f"dimension {name!r} must be a Real, Integer, Binary or Categorical, got {type(dimension).__name__}")

        self._names = tuple(names)
        self._dimensions = tuple(dimension for _, dimension in pairs)
        kinds = np.empty(len(pairs), dtype=np.int8)
        lower = np.empty(len(pairs), dtype=np.float64)
        upper = np.empty(len(pairs), dtype=np.float64)
        log = np.zeros(len(pairs), dtype=bool)
        for i, dimension in enumerate(self._dimensions):
            if isinstance(dimension, Real):
                kinds[i], lower[i], upper[i], log[i] = REAL, dimension.lower, dimension.upper, dimension.log
            elif isinstance(dimension, Integer):
                kinds[i], lower[i], upper[i] = INTEGER, dimension.lower, dimension.upper
            elif isinstance(dimension, Binary):
                kinds[i], lower[i], upper[i] = BINARY, 0, 1
            else:
                kinds[i], lower[i], upper[i] = CATEGORICAL, 0, len(dimension.choices) - 1
        for array in (kinds, lower, upper, log):
            array.setflags(write=False)
        self._kinds, self._lower, self._upper, self._log = kinds, lower, upper, log
        self._inward: dict[str, tuple[FloatVector, FloatVector]] = {}
        self._on_device: dict[Backend, _OnDevice] = {}

    # --- the declaration ---

    @property
    def dim(self) -> int:
        """The number of dimensions, which is the length of a genome."""
        return len(self._names)

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def dimensions(self) -> dict[str, Dimension]:
        """The dimensions by name, in declared order (a copy)."""
        return dict(zip(self._names, self._dimensions, strict=True))

    @property
    def kinds(self) -> npt.NDArray[np.int8]:
        """The kind code of each dimension (`REAL`, `INTEGER`, `BINARY`, `CATEGORICAL`), read-only."""
        return self._kinds

    @property
    def lower(self) -> FloatVector:
        """The lower bound of each column, float64, read-only: a categorical or binary column starts at 0."""
        return self._lower

    @property
    def upper(self) -> FloatVector:
        """The upper bound of each column, float64, read-only: the last index for a categorical, 1 for a binary."""
        return self._upper

    @property
    def log_scale(self) -> npt.NDArray[np.bool_]:
        return self._log

    def indices(self, kind: int) -> tuple[int, ...]:
        """The columns of the dimensions of one kind, in order."""
        return tuple(i for i, k in enumerate(self._kinds.tolist()) if k == kind)

    def categories(self, name: str) -> tuple[object, ...]:
        """The choices of a categorical dimension, in index order."""
        dimension = self.dimensions[name]
        if not isinstance(dimension, Categorical):
            raise TypeError(f"dimension {name!r} is not categorical")
        return dimension.choices

    def describe(self) -> dict[str, object]:
        """A JSON-serialisable description, for run metadata and resume validation: the names, types, bounds and choices are
        part of a run's identity, so a changed space is refused on resume."""
        return {
            "type": "MixedSpace",
            "dim": self.dim,
            "dimensions": [_describe_dimension(name, d) for name, d in zip(self._names, self._dimensions, strict=True)],
        }

    def __repr__(self) -> str:
        inner = ", ".join(f"{name}={d!r}" for name, d in zip(self._names, self._dimensions, strict=True))
        return f"MixedSpace({inner})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MixedSpace) and self.describe() == other.describe()

    def __hash__(self) -> int:
        return hash(repr(self.describe()))

    # --- precision ---

    def check_backend(self, backend: Backend) -> None:
        """Raise `ValueError` unless every integer bound is exactly representable in the backend's precision.

        In float32 that is magnitudes up to 2**24. The strategies call it when they are bound to a problem, so a space that
        cannot work is refused when the run starts, not on the first sample. It also checks that the real bounds can be
        represented (a box too narrow or too wide for float32).
        """
        limit = _EXACT_INTEGERS[backend.precision]
        for name, dimension in zip(self._names, self._dimensions, strict=True):
            if isinstance(dimension, Integer) and max(abs(int(dimension.lower)), abs(int(dimension.upper))) > limit:
                raise ValueError(
                    f"the integer dimension {name!r} has bounds [{dimension.lower}, {dimension.upper}], but {backend.precision} holds "
                    f"integers exactly only up to {limit}: use a narrower range or precision='float64'"
                )
        self._bounds(backend.precision)

    def _bounds(self, precision: str) -> tuple[FloatVector, FloatVector]:
        """The bounds as floats of `precision`, rounded inward so that they never leave `[lower, upper]` (as `Box` does)."""
        cached = self._inward.get(precision)
        if cached is not None:
            return cached
        dtype = np.dtype(precision)
        with np.errstate(over="ignore"):
            low, high = self._lower.astype(dtype), self._upper.astype(dtype)
            too_low, too_high = low.astype(np.float64) < self._lower, high.astype(np.float64) > self._upper
            low[too_low] = np.nextafter(low[too_low], np.array(np.inf, dtype))
            high[too_high] = np.nextafter(high[too_high], np.array(-np.inf, dtype))
            usable = bool((low <= high).all() and np.isfinite(high - low).all())
        if not usable:
            raise ValueError(f"the space is too narrow or too wide to be represented in {precision}")
        self._inward[precision] = (low, high)
        return low, high

    def _device(self, backend: Backend) -> _OnDevice:
        cached = self._on_device.get(backend)
        if cached is not None:
            return cached
        low_host, high_host = self._bounds(backend.precision)
        discrete = self._kinds != REAL
        masked_low, masked_high = self._lower.copy(), self._upper.copy()
        masked_low[~self._log] = masked_high[~self._log] = 1.0  # so that a logarithm of every column is defined
        count = self._upper - self._lower + 1.0
        count[~discrete] = 1.0
        built = _OnDevice(
            low=backend.asarray(low_host),
            high=backend.asarray(high_host),
            count=backend.asarray(count),
            log_low=backend.asarray(np.log(masked_low)),
            log_high=backend.asarray(np.log(masked_high)),
            is_log=backend.asarray(self._log, dtype=backend.bool_dtype),
            is_discrete=backend.asarray(discrete, dtype=backend.bool_dtype),
        )
        self._on_device[backend] = built
        return built

    # --- sampling and membership ---

    def sample_genomes(self, n: int, rng: RandomStream, backend: Backend) -> Array:
        """An `(n, d)` array on `backend`: uniform per dimension (log-uniform on log-scale reals, uniform over the integers of
        an integer range, over {0, 1} for a binary and over the indices of a categorical).

        Every sample is valid in both precisions: real values are within the (inward-rounded) bounds, discrete values are
        integral and in range. Check with `contains`.
        """
        if n < 0:
            raise ValueError(f"n must not be negative, got {n}")
        if rng.backend != backend:
            raise ValueError(f"the random stream is on {rng.backend} but samples were requested on {backend}")
        self.check_backend(backend)
        xp = backend.xp
        c = self._device(backend)
        unit = rng.uniform((n, self.dim))
        genomes = c.low + unit * (c.high - c.low)
        if self._log.any():
            logged = xp.exp(c.log_low + unit * (c.log_high - c.log_low))
            genomes = xp.where(c.is_log, logged, genomes)
        if self._kinds.any():  # some dimension is not real
            index = xp.minimum(xp.floor(unit * c.count), c.count - 1.0)
            genomes = xp.where(c.is_discrete, c.low + index, genomes)
        return xp.minimum(xp.maximum(genomes, c.low), c.high)

    def contains(self, genome: Array) -> bool:
        """Whether a genome is valid: a finite vector of length `dim`, real values within their bounds, integers integral and
        within theirs, binaries 0 or 1 and categoricals the index of one of their choices."""
        try:
            x = np.asarray(_HOST.to_numpy(genome), dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if x.shape != (self.dim,) or not bool(np.isfinite(x).all()):
            return False
        if not bool(((x >= self._lower) & (x <= self._upper)).all()):
            return False
        discrete = self._kinds != REAL
        return bool((x[discrete] == np.rint(x[discrete])).all())

    # --- decoding, for user code ---

    def values(self, genome: Array) -> dict[str, object]:
        """The named Python values of one genome: a float for a real dimension, an `int` for an integer, a `bool` for a
        binary and the choice itself for a categorical. Raises `ValueError` for a genome that is not a member of the space."""
        if not self.contains(genome):
            raise ValueError(f"the genome is not a member of this space: {_HOST.to_numpy(genome).tolist()}")
        x = np.asarray(_HOST.to_numpy(genome), dtype=np.float64)
        decoded: dict[str, object] = {}
        for name, dimension, value in zip(self._names, self._dimensions, x.tolist(), strict=True):
            if isinstance(dimension, Real):
                decoded[name] = float(value)
            elif isinstance(dimension, Integer):
                decoded[name] = int(value)
            elif isinstance(dimension, Binary):
                decoded[name] = bool(value)
            else:
                decoded[name] = dimension.choices[int(value)]
        return decoded

    def columns(self, genomes: Array) -> dict[str, Array]:
        """One array of shape `(n,)` per dimension, on the genomes' backend, for vectorised objectives: floats for a real
        dimension, int64 for an integer, bool for a binary, and the int64 **index** for a categorical (the choices are not
        arrays: use `categories(name)` to map indices to them, or compare indices). Rows are not validated."""
        from auxein.backend import backend_of

        backend = backend_of(genomes)
        if genomes.ndim != 2 or genomes.shape[1] != self.dim:
            raise ValueError(f"genomes must have shape (n, {self.dim}), got {tuple(genomes.shape)}")
        xp = backend.xp
        out: dict[str, Array] = {}
        for i, (name, dimension) in enumerate(zip(self._names, self._dimensions, strict=True)):
            column = genomes[:, i]
            if isinstance(dimension, Real):
                out[name] = column
            elif isinstance(dimension, Binary):
                out[name] = column > 0.5
            else:
                out[name] = xp.astype(xp.round(column), backend.int_dtype)
        return out


class IntegerSpace(MixedSpace):
    """`dim` integer dimensions `x0 .. x{dim-1}`, each in `[lower, upper]`: a thin wrapper over `MixedSpace`."""

    def __init__(self, lower: int, upper: int, dim: int) -> None:
        if dim < 1:
            raise ValueError(f"dim must be at least 1, got {dim}")
        super().__init__({f"x{i}": Integer(lower, upper) for i in range(dim)})


class BinarySpace(MixedSpace):
    """`dim` binary dimensions `x0 .. x{dim-1}`: a thin wrapper over `MixedSpace`."""

    def __init__(self, dim: int) -> None:
        if dim < 1:
            raise ValueError(f"dim must be at least 1, got {dim}")
        super().__init__({f"x{i}": Binary() for i in range(dim)})
