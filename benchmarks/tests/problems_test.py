import numpy as np
import pytest

from benchmarks.problems import PROBLEMS, Ellipsoid, make_problem, random_rotation, random_shift

NAMES = sorted(PROBLEMS)
DIMS = [2, 10, 30]


def optimum(problem):
    return problem.optimum


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dim", DIMS)
def test_error_is_zero_at_the_optimum_and_positive_elsewhere(name, dim):
    problem = make_problem(name, dim, instance=3)
    assert problem.true_error(optimum(problem)) == pytest.approx(0.0, abs=1e-9)

    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.uniform(problem.lower, problem.upper, dim)
        assert problem.true_error(x) > 0
    assert problem.true_error(optimum(problem) + 1e-3) > 0


@pytest.mark.parametrize("name", NAMES)
def test_optimum_is_inside_the_domain(name):
    problem = make_problem(name, 10, instance=0)
    assert np.all(np.abs(optimum(problem)) <= 4)


def test_ellipsoid_rotation_is_orthogonal():
    for dim in DIMS:
        rotation = random_rotation(7, dim)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(dim), atol=1e-12)
        assert abs(abs(np.linalg.det(rotation)) - 1) < 1e-9


def test_ellipsoid_has_condition_number_1e6():
    problem = Ellipsoid(10, 0)
    assert problem.weights.max() / problem.weights.min() == pytest.approx(1e6)
    assert Ellipsoid(1, 0).weights.tolist() == [1.0]


def test_ellipsoid_is_rotated():
    problem = make_problem("ellipsoid", 10, 0)
    step = np.zeros(10)
    step[0] = 1.0
    # an axis-aligned step in the first coordinate would cost exactly weights[0] = 1 if there were no rotation
    assert problem.true_error(optimum(problem) + step) != pytest.approx(1.0)


def test_rosenbrock_needs_two_dimensions():
    with pytest.raises(ValueError):
        make_problem("rosenbrock", 1, 0)


def test_known_values_away_from_the_optimum():
    sphere = make_problem("sphere", 3, 0)
    assert sphere.true_error(optimum(sphere) + np.array([1.0, 2.0, 2.0])) == pytest.approx(9.0)

    rastrigin = make_problem("rastrigin", 2, 0)
    # at integer offsets cos(2 pi z) = 1, so the error is just the sum of squares
    assert rastrigin.true_error(optimum(rastrigin) + np.array([1.0, 2.0])) == pytest.approx(5.0)

    rosenbrock = make_problem("rosenbrock", 2, 0)
    # z = (0, 0): 100 * (0 - 0)^2 + (1 - 0)^2
    assert rosenbrock.true_error(optimum(rosenbrock) - 1.0) == pytest.approx(1.0)


@pytest.mark.parametrize("name", NAMES)
def test_instances_are_deterministic_per_id_and_differ_between_ids(name):
    a, b, other = make_problem(name, 10, 4), make_problem(name, 10, 4), make_problem(name, 10, 5)
    np.testing.assert_array_equal(optimum(a), optimum(b))
    assert not np.array_equal(optimum(a), optimum(other))

    x = np.linspace(-3, 3, 10)
    assert a.true_error(x) == b.true_error(x)
    assert a.true_error(x) != other.true_error(x)


def test_instances_are_independent_of_global_random_state():
    np.random.seed(0)
    first = random_shift(2, 10), random_rotation(2, 10)
    np.random.seed(123)
    np.random.uniform(size=1000)
    second = random_shift(2, 10), random_rotation(2, 10)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_ellipsoid_rotation_differs_between_ids():
    assert not np.allclose(random_rotation(0, 10), random_rotation(1, 10))


def test_noisy_sphere_true_error_is_noise_free():
    noisy, clean = make_problem("noisy_sphere", 5, 1), make_problem("sphere", 5, 1)
    x = np.full(5, 1.5)
    assert noisy.noisy and not clean.noisy
    values = {noisy.true_error(x) for _ in range(20)}
    assert values == {clean.true_error(x)}


def test_noisy_sphere_evaluate_is_noisy_with_the_right_level():
    noisy = make_problem("noisy_sphere", 5, 1)
    x = np.full(5, 1.5)
    true = noisy.true_error(x)
    ratios = np.array([noisy.evaluate(x) / true for _ in range(4000)])
    assert len(set(ratios.round(12))) > 3000
    assert ratios.mean() == pytest.approx(1.0, abs=0.01)
    assert ratios.std() == pytest.approx(0.1, abs=0.01)


def test_noise_is_reproducible_per_instance():
    x = np.full(5, 1.5)
    a, b = make_problem("noisy_sphere", 5, 1), make_problem("noisy_sphere", 5, 1)
    assert [a.evaluate(x) for _ in range(5)] == [b.evaluate(x) for _ in range(5)]


def test_noise_free_problems_evaluate_to_the_true_error():
    x = np.full(5, 1.5)
    for name in ("sphere", "ellipsoid", "rosenbrock", "rastrigin"):
        problem = make_problem(name, 5, 0)
        assert problem.evaluate(x) == problem.true_error(x)


def test_unknown_problem():
    with pytest.raises(ValueError, match="unknown problem"):
        make_problem("nope", 2, 0)
