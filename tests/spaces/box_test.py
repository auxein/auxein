import numpy as np
import pytest

from auxein.backend import Backend
from auxein.random import RunSeed
from auxein.spaces import Box
from tests.support.fixtures import assert_on_backend


def rng(backend: Backend, name: str = "space"):
    return RunSeed(7).stream(name, backend=backend)


def sample(box: Box, backend: Backend, n: int, name: str = "space"):
    return box.sample_genomes(n, rng(backend, name), backend)


# --- construction and validation ---


def test_scalar_bounds_broadcast_to_dim():
    box = Box(-5.0, 5.0, dim=4)
    assert box.dim == 4
    np.testing.assert_array_equal(box.lower, [-5.0] * 4)
    np.testing.assert_array_equal(box.upper, [5.0] * 4)
    assert not box.log_scale.any()


def test_per_dimension_bounds_define_dim():
    box = Box([0, -1, 2], [1, 1, 10])
    assert box.dim == 3
    np.testing.assert_array_equal(box.lower, [0.0, -1.0, 2.0])
    assert Box([0, -1, 2], [1, 1, 10], dim=3).dim == 3


def test_a_scalar_bound_broadcasts_against_an_array_bound():
    box = Box(0.0, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(box.lower, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(box.upper, [1.0, 2.0, 3.0])
    assert Box([-1.0, -2.0], 5).dim == 2


def test_bounds_may_be_arrays_or_tensors(backend: Backend):
    box = Box(backend.asarray([0.0, 1.0]), backend.asarray([2.0, 3.0]))
    np.testing.assert_array_equal(box.lower, [0.0, 1.0])
    assert box.lower.dtype == np.float64


def test_log_scale_as_bool_or_per_dimension():
    assert Box(1e-3, 1.0, dim=3, log_scale=True).log_scale.tolist() == [True] * 3
    assert Box(1e-3, 1.0, dim=3, log_scale=False).log_scale.tolist() == [False] * 3
    assert Box([-1, 1e-3, 1e-3], [1, 1, 1], log_scale=[False, True, True]).log_scale.tolist() == [False, True, True]
    assert Box(-1.0, 1.0, dim=2, log_scale=np.array([False, False])).log_scale.tolist() == [False, False]


def test_properties_are_read_only_and_boxes_are_values():
    box = Box(-1.0, 1.0, dim=2)
    with pytest.raises(ValueError, match="read-only"):
        box.lower[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        box.upper[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        box.log_scale[0] = True
    assert box == Box([-1, -1], [1, 1])
    assert hash(box) == hash(Box([-1, -1], [1, 1]))
    assert box != Box(-1.0, 2.0, dim=2)
    assert box != Box(-1.0, 1.0, dim=3)
    assert "Box(" in repr(box)


def test_the_box_does_not_alias_the_bounds_it_was_given():
    lower = np.array([0.0, 0.0])
    box = Box(lower, [1.0, 1.0])
    lower[0] = 0.5
    assert box.lower[0] == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lower": 0.0, "upper": 1.0}, "dim is required"),
        ({"lower": 1.0, "upper": 1.0, "dim": 2}, "strictly below"),
        ({"lower": 2.0, "upper": 1.0, "dim": 2}, "strictly below"),
        ({"lower": [0.0, 2.0], "upper": [1.0, 1.0]}, "strictly below"),
        ({"lower": float("nan"), "upper": 1.0, "dim": 2}, "finite"),
        ({"lower": 0.0, "upper": float("inf"), "dim": 2}, "finite"),
        ({"lower": float("-inf"), "upper": 1.0, "dim": 2}, "finite"),
        ({"lower": [0.0, 0.0], "upper": [1.0, 1.0, 1.0]}, "different lengths"),
        ({"lower": [0.0, 0.0], "upper": [1.0, 1.0], "dim": 3}, "does not match"),
        ({"lower": 0.0, "upper": 1.0, "dim": 0}, "at least one dimension"),
        ({"lower": [[0.0, 0.0]], "upper": 1.0, "dim": 2}, "scalar or a 1-D"),
        ({"lower": -1e308, "upper": 1e308, "dim": 2}, "range"),
        ({"lower": 0.0, "upper": 1.0, "dim": 2, "log_scale": True}, "lower > 0"),
        ({"lower": [1.0, -1.0], "upper": [2.0, 1.0], "log_scale": [False, True]}, "lower > 0"),
        ({"lower": 1.0, "upper": 2.0, "dim": 3, "log_scale": [True, False]}, "log_scale"),
        ({"lower": 1.0, "upper": 2.0, "dim": 2, "log_scale": [[True, False]]}, "log_scale"),
    ],
)
def test_validation(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        Box(**kwargs)


def test_non_log_dimensions_may_have_non_positive_bounds_next_to_log_dimensions():
    assert Box([-5.0, 1e-3], [5.0, 1.0], log_scale=[False, True]).dim == 2


# --- sampling ---


def test_sample_genomes_returns_an_n_by_d_array_on_the_backend(backend: Backend):
    x = sample(Box(-5.0, 5.0, dim=7), backend, 11)
    assert_on_backend(x, backend)
    assert tuple(x.shape) == (11, 7)
    assert tuple(sample(Box(-5.0, 5.0, dim=7), backend, 0).shape) == (0, 7)


def test_sampling_is_deterministic_per_stream(backend: Backend):
    box = Box([-1.0, 1e-3], [1.0, 10.0], log_scale=[False, True])
    a, b = sample(box, backend, 50, "x"), sample(box, backend, 50, "x")
    np.testing.assert_array_equal(backend.to_numpy(a), backend.to_numpy(b))
    c = sample(box, backend, 50, "y")
    assert not np.array_equal(backend.to_numpy(a), backend.to_numpy(c))


def test_the_stream_and_the_requested_backend_must_agree():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="random stream is on"):
        Box(0.0, 1.0, dim=2).sample_genomes(3, rng(Backend("numpy")), Backend("torch"))
    with pytest.raises(ValueError, match="random stream is on"):
        Box(0.0, 1.0, dim=2).sample_genomes(3, rng(Backend("numpy", "cpu", "float32")), Backend("numpy"))


def test_a_negative_sample_size_is_an_error(backend: Backend):
    with pytest.raises(ValueError, match="n must not be negative"):
        sample(Box(0.0, 1.0, dim=2), backend, -1)


AWKWARD = [
    Box(-5.0, 5.0, dim=10),
    Box(0.1, 0.7, dim=4),  # 0.1 and 0.7 are not float32 numbers
    Box([1 / 3, -1 / 3, 1e-9], [2 / 3, 1 / 3, 3e-9]),
    Box(-1e30, 1e30, dim=3),
    Box([1e-5, 0.01, 1e-3], [1e-1, 100.0, 1.0], log_scale=True),
    Box([-3.0, 1e-6, 0.123456789], [3.0, 0.5, 0.123456799], log_scale=[False, True, False]),
]


@pytest.mark.parametrize("box", AWKWARD, ids=range(len(AWKWARD)))
def test_a_hundred_thousand_samples_lie_within_the_bounds(backend: Backend, box: Box):
    x = backend.to_numpy(sample(box, backend, 100_000)).astype(np.float64)  # compared in float64, exactly
    assert x.shape == (100_000, box.dim)
    assert np.isfinite(x).all()
    assert (x >= box.lower).all(), (x.min(axis=0), box.lower)
    assert (x <= box.upper).all(), (x.max(axis=0), box.upper)


def test_samples_cover_the_box(backend: Backend):
    box = Box(-2.0, 6.0, dim=3)
    x = backend.to_numpy(sample(box, backend, 100_000))
    assert (x.min(axis=0) < -1.99).all() and (x.max(axis=0) > 5.99).all()
    np.testing.assert_allclose(x.mean(axis=0), 2.0, atol=0.05)
    np.testing.assert_allclose(x.std(axis=0), 8 / np.sqrt(12), atol=0.05)


def test_float32_bounds_are_rounded_inward_so_that_the_upper_bound_is_reachable_but_never_exceeded():
    box = Box(0.1, 0.7, dim=1)
    low, high = box._bounds("float32")
    assert float(low[0]) >= 0.1 and float(high[0]) <= 0.7
    assert float(low[0]) - 0.1 < 1e-7 and 0.7 - float(high[0]) < 1e-7
    low64, high64 = box._bounds("float64")
    np.testing.assert_array_equal(low64, [0.1])
    np.testing.assert_array_equal(high64, [0.7])


def test_a_box_too_narrow_for_float32_is_an_error_when_sampling_in_float32_only():
    box = Box(0.1, 0.1 + 1e-9, dim=1)
    assert box.dim == 1
    assert sample(box, Backend("numpy", "cpu", "float64"), 5).shape == (5, 1)
    with pytest.raises(ValueError, match="too narrow or too wide to be represented in float32"):
        sample(box, Backend("numpy", "cpu", "float32"), 5)


def test_float32_boxes_wider_than_float32_are_an_error():
    box = Box(-3e38, 3e38, dim=1)  # float32 can hold the bounds, but not their difference
    with pytest.raises(ValueError, match="too narrow or too wide"):
        sample(box, Backend("numpy", "cpu", "float32"), 5)


def test_log_scale_samples_are_roughly_uniform_in_log_space(backend: Backend):
    box = Box([1e-4, 1e-4], [1e2, 1e2], log_scale=[True, False])
    x = backend.to_numpy(sample(box, backend, 100_000)).astype(np.float64)
    logged = np.log10(x[:, 0])
    assert logged.min() >= -4 and logged.max() <= 2
    histogram, _ = np.histogram(logged, bins=12, range=(-4, 2))
    np.testing.assert_allclose(histogram / 100_000, 1 / 12, atol=0.01)  # each decade-half holds 1/12 of the draws
    assert abs(np.median(x[:, 0]) - 1e-1) / 1e-1 < 0.15  # the median is the geometric mean, not the arithmetic one
    assert np.median(x[:, 1]) > 40  # the linear dimension is not log-uniform


def test_linear_and_log_dimensions_do_not_interfere(backend: Backend):
    box = Box([-1.0, 1.0], [1.0, 1000.0], log_scale=[False, True])
    x = backend.to_numpy(sample(box, backend, 50_000)).astype(np.float64)
    assert abs(x[:, 0].mean()) < 0.02
    assert abs(np.log10(x[:, 1]).mean() - 1.5) < 0.03


# --- contains and clip ---


def test_contains():
    box = Box([0.0, -1.0], [1.0, 1.0])
    assert box.contains(np.array([0.5, 0.0]))
    assert box.contains(np.array([0.0, 1.0]))  # the bounds are inclusive
    assert box.contains([1.0, -1.0])
    assert not box.contains(np.array([1.5, 0.0]))
    assert not box.contains(np.array([0.5, -1.1]))
    assert not box.contains(np.array([np.nan, 0.0]))
    assert not box.contains(np.array([np.inf, 0.0]))
    assert not box.contains(np.array([0.5]))  # wrong length
    assert not box.contains(np.zeros((2, 2)))  # not a single genome
    assert not box.contains("genome")
    assert not box.contains([[0.5, 0.0], [0.5]])  # ragged


def test_contains_accepts_arrays_of_every_backend(backend: Backend):
    box = Box(0.0, 1.0, dim=3)
    assert box.contains(backend.asarray([0.2, 0.3, 0.4]))
    assert not box.contains(backend.asarray([0.2, 0.3, 1.4]))


def test_sampled_genomes_are_contained(backend: Backend):
    box = Box([0.1, 1e-3, -0.3], [0.7, 1.0, 0.3], log_scale=[False, True, False])
    x = sample(box, backend, 2000)
    assert all(box.contains(x[i]) for i in range(2000))


def test_clip_repairs_out_of_bounds_values(backend: Backend):
    box = Box([0.0, -1.0], [1.0, 1.0])
    x = backend.asarray([[-0.5, 0.5], [0.5, 2.0], [0.25, -0.25], [3.0, -3.0]])
    clipped = box.clip(x)
    assert_on_backend(clipped, backend)
    np.testing.assert_allclose(backend.to_numpy(clipped), [[0.0, 0.5], [0.5, 1.0], [0.25, -0.25], [1.0, -1.0]])


def test_clip_accepts_a_single_genome_and_leaves_valid_values_alone(backend: Backend):
    box = Box(0.0, 1.0, dim=3)
    inside = backend.asarray([0.25, 0.5, 0.75])
    np.testing.assert_array_equal(backend.to_numpy(box.clip(inside)), backend.to_numpy(inside))
    assert tuple(box.clip(backend.asarray([2.0, -2.0, 0.5])).shape) == (3,)


def test_clip_does_not_modify_its_input(backend: Backend):
    box = Box(0.0, 1.0, dim=2)
    x = backend.asarray([[2.0, -1.0]])
    box.clip(x)
    np.testing.assert_array_equal(backend.to_numpy(x), [[2.0, -1.0]])


def test_clipped_genomes_are_contained_in_every_precision(backend: Backend):
    box = Box([0.1, 1e-3], [0.7, 0.9])
    wild = backend.asarray(np.random.default_rng(0).uniform(-3, 3, (500, 2)))
    clipped = box.clip(wild)
    assert all(box.contains(clipped[i]) for i in range(500))


def test_clip_passes_nan_through(backend: Backend):
    clipped = backend.to_numpy(Box(0.0, 1.0, dim=2).clip(backend.asarray([float("nan"), 5.0])))
    assert np.isnan(clipped[0]) and clipped[1] == 1.0


def test_clip_validation(backend: Backend):
    box = Box(0.0, 1.0, dim=2)
    with pytest.raises(ValueError, match="2 values in their last dimension"):
        box.clip(backend.asarray([1.0, 2.0, 3.0]))
    with pytest.raises(TypeError, match="floating-point"):
        box.clip(backend.asarray([1, 2], dtype=backend.int_dtype))
