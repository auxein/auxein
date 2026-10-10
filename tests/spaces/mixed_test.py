import json

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.random import RunSeed
from auxein.spaces import Binary, BinarySpace, Categorical, Integer, IntegerSpace, MixedSpace, Real
from auxein.spaces.mixed import BINARY, CATEGORICAL, INTEGER, REAL
from tests.support.fixtures import assert_on_backend

MIXED = MixedSpace(
    {
        "lr": Real(1e-5, 1e-1, log=True),
        "w": Real(-2.0, 3.0),
        "layers": Integer(1, 8),
        "offset": Integer(-5, 5),
        "dropout": Binary(),
        "optimiser": Categorical(["sgd", "adam", "rmsprop"]),
    }
)


def stream(backend: Backend, seed: int = 1):
    return RunSeed(seed).stream("strategy", backend=backend)


def all_valid(space: MixedSpace, samples: np.ndarray) -> bool:
    """Every row is finite, within the bounds, and integral on the discrete columns: checked on the whole array at once."""
    discrete = space.kinds != REAL
    inside = ((samples >= space.lower) & (samples <= space.upper)).all()
    integral = (samples[:, discrete] == np.rint(samples[:, discrete])).all()
    return bool(np.isfinite(samples).all() and inside and integral)


# --- the dimension types ---


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: Real(1.0, 1.0), "lower < upper"),
        (lambda: Real(float("nan"), 1.0), "finite"),
        (lambda: Real(0.0, float("inf")), "finite"),
        (lambda: Real(0.0, 1.0, log=True), "lower > 0"),
        (lambda: Real(-1e308, 1e308), "finite"),
        (lambda: Integer(3, 3), "lower < upper"),
        (lambda: Integer(5, 1), "lower < upper"),
        (lambda: Integer(0.5, 3), "must be an int"),  # type: ignore[arg-type]
        (lambda: Integer(True, 3), "must be an int"),
        (lambda: Categorical([]), "at least two"),
        (lambda: Categorical(["only"]), "at least two"),
        (lambda: Categorical(["a", "a"]), "distinct"),
        (lambda: Categorical([1, 1.0, True][:1] + [1]), "distinct"),
        (lambda: Categorical([object(), "a"]), "JSON-serialisable"),
        (lambda: Categorical("abc"), "not a string"),
    ],
)
def test_dimension_validation(make, message: str):
    with pytest.raises((ValueError, TypeError), match=message):
        make()


def test_categorical_choices_keep_their_json_types_and_order():
    choices = Categorical(["sgd", 3, 2.5, None, True, ["a", 1], {"k": "v"}])
    assert choices.choices == ("sgd", 3, 2.5, None, True, ["a", 1], {"k": "v"})
    Categorical([1, 1.0])  # 1 and 1.0 encode differently in canonical JSON, so they are two choices


# --- the space ---


@pytest.mark.parametrize(
    ("dimensions", "message"),
    [
        ({}, "at least one dimension"),
        ([("a", Binary()), ("a", Binary())], "duplicate dimension names"),
        ({"": Binary()}, "non-empty"),
        ({"a": "binary"}, "must be a Real"),
        ({1: Binary()}, "non-empty strings"),
    ],
)
def test_space_validation(dimensions, message: str):
    with pytest.raises((ValueError, TypeError), match=message):
        MixedSpace(dimensions)


def test_dimensions_keep_their_declared_order_and_kinds():
    assert MIXED.names == ("lr", "w", "layers", "offset", "dropout", "optimiser") and MIXED.dim == 6
    assert MIXED.kinds.tolist() == [REAL, REAL, INTEGER, INTEGER, BINARY, CATEGORICAL]
    assert MIXED.lower.tolist() == [1e-5, -2.0, 1.0, -5.0, 0.0, 0.0] and MIXED.upper.tolist() == [0.1, 3.0, 8.0, 5.0, 1.0, 2.0]
    assert MIXED.indices(INTEGER) == (2, 3) and MIXED.categories("optimiser") == ("sgd", "adam", "rmsprop")
    with pytest.raises(TypeError, match="not categorical"):
        MIXED.categories("lr")


def test_float32_refuses_integer_bounds_it_cannot_hold_when_the_run_starts():
    wide = MixedSpace({"n": Integer(0, 2**24 + 1)})
    wide.check_backend(Backend())  # float64 holds them
    with pytest.raises(ValueError, match="holds integers exactly only up to 16777216"):
        wide.check_backend(Backend("numpy", "cpu", "float32"))
    MixedSpace({"n": Integer(-(2**24), 2**24)}).check_backend(Backend("numpy", "cpu", "float32"))  # the limit itself is fine
    with pytest.raises(ValueError, match="'n'"):
        wide.sample_genomes(3, stream(Backend("numpy", "cpu", "float32")), Backend("numpy", "cpu", "float32"))


def test_a_real_dimension_too_narrow_for_float32_is_refused_like_a_box():
    narrow = MixedSpace({"x": Real(0.1, 0.1 + 1e-9)})
    narrow.check_backend(Backend())
    with pytest.raises(ValueError, match="too narrow or too wide"):
        narrow.check_backend(Backend("numpy", "cpu", "float32"))
    with pytest.raises(ValueError, match="too narrow or too wide"):
        MixedSpace({"x": Real(-3e38, 3e38)}).check_backend(Backend("numpy", "cpu", "float32"))


# --- sampling ---


def test_one_hundred_thousand_samples_are_all_valid_on_every_backend(backend: Backend):
    samples = MIXED.sample_genomes(100_000, stream(backend), backend)
    assert_on_backend(samples, backend)
    assert samples.shape == (100_000, 6)
    assert all_valid(MIXED, backend.to_numpy(samples).astype(np.float64))
    assert MIXED.contains(samples[0]) and MIXED.contains(samples[-1])


def test_sampling_is_uniform_per_dimension_type(backend: Backend):
    samples = backend.to_numpy(MIXED.sample_genomes(60_000, stream(backend, 5), backend)).astype(np.float64)
    for column, count in ((2, 8), (3, 11), (5, 3)):  # every integer and category appears equally often
        values, counts = np.unique(samples[:, column], return_counts=True)
        assert len(values) == count and counts.min() > 0.93 * len(samples) / count and counts.max() < 1.07 * len(samples) / count
    assert abs(samples[:, 4].mean() - 0.5) < 0.01  # bits
    assert abs(samples[:, 1].mean() - 0.5) < 0.03 and abs(samples[:, 1].std() - 5 / np.sqrt(12)) < 0.03  # real, uniform on [-2, 3]
    logs = np.log10(samples[:, 0])  # log-uniform: uniform over the four decades
    assert abs(logs.mean() + 3.0) < 0.03 and abs(logs.std() - 4 / np.sqrt(12)) < 0.03


def test_the_extremes_are_reachable(backend: Backend):
    samples = backend.to_numpy(MIXED.sample_genomes(60_000, stream(backend, 9), backend))
    for column in (2, 3, 4, 5):
        assert samples[:, column].min() == MIXED.lower[column] and samples[:, column].max() == MIXED.upper[column]


def test_sampling_is_reproducible_and_needs_a_stream_on_the_same_backend(backend: Backend):
    one = backend.to_numpy(MIXED.sample_genomes(50, stream(backend), backend))
    two = backend.to_numpy(MIXED.sample_genomes(50, stream(backend), backend))
    np.testing.assert_array_equal(one, two)
    assert MIXED.sample_genomes(0, stream(backend), backend).shape == (0, 6)
    with pytest.raises(ValueError, match="must not be negative"):
        MIXED.sample_genomes(-1, stream(backend), backend)
    other = Backend("numpy", "cpu", "float32" if backend.precision == "float64" else "float64")
    with pytest.raises(ValueError, match="random stream is on"):
        MIXED.sample_genomes(3, stream(backend), other)


def test_a_degenerate_range_still_samples_validly_in_float32():
    backend = Backend("numpy", "cpu", "float32")
    space = MixedSpace({"a": Integer(0, 1), "b": Real(0.1, 0.3), "c": Categorical(["x", "y"])})
    samples = backend.to_numpy(space.sample_genomes(50_000, stream(backend), backend)).astype(np.float64)
    assert all_valid(space, samples) and (samples[:, 1] >= 0.1).all() and (samples[:, 1] <= 0.3).all()


# --- membership ---


def valid_genome(backend: Backend):
    return backend.asarray([0.01, 0.5, 3.0, -2.0, 1.0, 2.0])


def test_contains_accepts_valid_genomes(backend: Backend):
    assert MIXED.contains(valid_genome(backend))
    assert MIXED.contains(np.array([0.01, 0.5, 3.0, -2.0, 1.0, 2.0]))


@pytest.mark.parametrize(
    "bad",
    [
        [0.01, 0.5, 3.5, -2.0, 1.0, 2.0],  # a non-integral integer
        [0.01, 0.5, 3.0, -2.0, 0.5, 2.0],  # a non-binary bit
        [0.01, 0.5, 3.0, -2.0, 1.0, 1.5],  # a fractional category index
        [0.01, 0.5, 9.0, -2.0, 1.0, 2.0],  # an integer above its bound
        [0.01, 0.5, 0.0, -2.0, 1.0, 2.0],  # an integer below its bound
        [0.01, 0.5, 3.0, -2.0, 1.0, 3.0],  # an index past the last category
        [0.01, 0.5, 3.0, -2.0, 1.0, -1.0],  # a negative index
        [0.01, 0.5, 3.0, -2.0, 2.0, 2.0],  # a bit of 2
        [1e-6, 0.5, 3.0, -2.0, 1.0, 2.0],  # a real below its bound
        [0.01, 3.5, 3.0, -2.0, 1.0, 2.0],  # a real above its bound
        [float("nan"), 0.5, 3.0, -2.0, 1.0, 2.0],
        [0.01, float("inf"), 3.0, -2.0, 1.0, 2.0],
        [0.01, 0.5, 3.0, -2.0, 1.0],  # too short
        [0.01, 0.5, 3.0, -2.0, 1.0, 2.0, 0.0],  # too long
    ],
)
def test_contains_rejects_invalid_genomes(backend: Backend, bad: list[float]):
    assert not MIXED.contains(backend.asarray(bad))


def test_contains_rejects_things_that_are_not_genomes():
    assert not MIXED.contains("genome")  # type: ignore[arg-type]
    assert not MIXED.contains(np.zeros((2, 6)))


# --- decoding ---


def test_values_decode_a_genome_to_named_python_values(backend: Backend):
    decoded = MIXED.values(valid_genome(backend))
    assert list(decoded) == list(MIXED.names)
    assert decoded["layers"] == 3 and type(decoded["layers"]) is int
    assert decoded["offset"] == -2 and decoded["dropout"] is True and decoded["optimiser"] == "rmsprop"
    assert type(decoded["lr"]) is float and decoded["w"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="not a member"):
        MIXED.values(backend.asarray([0.01, 0.5, 3.5, -2.0, 1.0, 2.0]))


def encode(space: MixedSpace, values: dict[str, object], backend: Backend):
    """The inverse of `values`, written independently of the library: what a user would write to build a genome by hand."""
    row = []
    for name, dimension in space.dimensions.items():
        value = values[name]
        row.append(float(dimension.choices.index(value)) if isinstance(dimension, Categorical) else float(value))  # type: ignore[arg-type]
    return backend.asarray(row)


def test_values_and_columns_round_trip_and_agree(backend: Backend):
    samples = MIXED.sample_genomes(200, stream(backend, 3), backend)
    columns = MIXED.columns(samples)
    assert list(columns) == list(MIXED.names)
    xp = backend.xp
    assert columns["layers"].dtype == backend.int_dtype and columns["dropout"].dtype == backend.bool_dtype
    assert columns["optimiser"].dtype == backend.int_dtype and columns["lr"].dtype == backend.dtype
    for row in range(0, 200, 7):
        decoded = MIXED.values(samples[row])
        np.testing.assert_array_equal(backend.to_numpy(encode(MIXED, decoded, backend)), backend.to_numpy(samples[row]))  # round trip
        assert decoded["layers"] == int(backend.to_numpy(columns["layers"])[row])
        assert decoded["dropout"] is bool(backend.to_numpy(columns["dropout"])[row])
        assert MIXED.categories("optimiser")[int(backend.to_numpy(columns["optimiser"])[row])] == decoded["optimiser"]
    assert_on_backend(columns["lr"], backend)
    assert float(backend.to_numpy(xp.sum(columns["layers"]))) == backend.to_numpy(samples[:, 2]).sum()
    with pytest.raises(ValueError, match="shape"):
        MIXED.columns(samples[:, :3])


# --- description ---


def test_describe_is_stable_json_and_identifies_the_space():
    description = MIXED.describe()
    assert json.loads(json.dumps(description)) == description
    assert description == MIXED.describe() == json.loads(json.dumps(MIXED.describe()))
    assert description["dim"] == 6
    kinds = [d["type"] for d in description["dimensions"]]  # type: ignore[index]
    assert kinds == ["real", "real", "integer", "integer", "binary", "categorical"]
    same = MixedSpace(dict(MIXED.dimensions))
    assert same == MIXED and hash(same) == hash(MIXED)
    changed = MixedSpace({**MIXED.dimensions, "layers": Integer(1, 9)})
    other_order = MixedSpace(dict(reversed(list(MIXED.dimensions.items()))))
    assert changed != MIXED and other_order != MIXED
    assert changed.describe() != description


# --- the convenience spaces ---


def test_integer_and_binary_spaces_are_thin_wrappers(backend: Backend):
    integers, bits = IntegerSpace(-3, 4, 5), BinarySpace(7)
    assert integers.names == ("x0", "x1", "x2", "x3", "x4") and isinstance(integers, MixedSpace)
    assert integers.describe()["dimensions"][0] == {"name": "x0", "type": "integer", "lower": -3, "upper": 4}  # type: ignore[index]
    assert bits.dim == 7 and set(bits.kinds.tolist()) == {BINARY}
    samples = backend.to_numpy(integers.sample_genomes(5000, stream(backend), backend)).astype(np.float64)
    assert samples.min() == -3 and samples.max() == 4 and all_valid(integers, samples)
    flags = backend.to_numpy(bits.sample_genomes(5000, stream(backend), backend)).astype(np.float64)
    assert set(np.unique(flags).tolist()) == {0.0, 1.0}
    with pytest.raises(ValueError, match="at least 1"):
        IntegerSpace(0, 1, 0)
    with pytest.raises(ValueError, match="at least 1"):
        BinarySpace(0)
