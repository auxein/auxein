"""Benchmark problems.

Every problem is minimised on the search domain [-5, 5]^d, has its optimum value at 0, and is reported as the
error f(x) - f*. An instance (a random shift x*, and a random rotation for the ellipsoid) is a pure function of
its instance id and the dimension, drawn from generators that are separate from any algorithm's randomness.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

LOWER = -5.0
UPPER = 5.0
SHIFT_RANGE = 4.0
NOISE_LEVEL = 0.1

# streams of the instance random generator
_SHIFT, _ROTATION, _NOISE = 0, 1, 2


def instance_rng(instance: int, dim: int, stream: int) -> np.random.Generator:
    """Random generator for one aspect of an instance, independent of every algorithm's randomness."""
    return np.random.default_rng(np.random.SeedSequence([instance, dim, stream]))


def random_shift(instance: int, dim: int) -> np.ndarray:
    return instance_rng(instance, dim, _SHIFT).uniform(-SHIFT_RANGE, SHIFT_RANGE, dim)


def random_rotation(instance: int, dim: int) -> np.ndarray:
    """Uniform random orthogonal matrix: QR of a Gaussian matrix, with the signs of R's diagonal corrected."""
    q, r = np.linalg.qr(instance_rng(instance, dim, _ROTATION).standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))


class Problem(ABC):
    """A minimisation problem. To add one, subclass it and register it with `register`."""

    name: str
    noisy: bool = False  # whether evaluate() differs from true_error()

    def __init__(self, dim: int, instance: int) -> None:
        self.dim = dim
        self.instance = instance
        self.lower = LOWER
        self.upper = UPPER

    @abstractmethod
    def true_error(self, x: np.ndarray) -> float:
        """The noise-free error f(x) - f*, which is always >= 0 and is 0 at the optimum."""

    def evaluate(self, x: np.ndarray) -> float:
        """The value the algorithm sees. It may be noisy; by default it is the true error."""
        return self.true_error(x)


class _Shifted(Problem):
    def __init__(self, dim: int, instance: int) -> None:
        super().__init__(dim, instance)
        self.optimum = random_shift(instance, dim)

    def _z(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) - self.optimum


class Sphere(_Shifted):
    name = "sphere"

    def true_error(self, x: np.ndarray) -> float:
        z = self._z(x)
        return float(np.dot(z, z))


class Ellipsoid(_Shifted):
    """Condition number 1e6, rotated, so that neither a per-axis step size nor separability helps."""

    name = "ellipsoid"

    def __init__(self, dim: int, instance: int) -> None:
        super().__init__(dim, instance)
        self.rotation = random_rotation(instance, dim)
        exponents = np.arange(dim) / (dim - 1) if dim > 1 else np.zeros(dim)
        self.weights = 10.0 ** (6.0 * exponents)

    def true_error(self, x: np.ndarray) -> float:
        z = self.rotation @ self._z(x)
        return float(np.dot(self.weights, z * z))


class Rosenbrock(_Shifted):
    name = "rosenbrock"

    def __init__(self, dim: int, instance: int) -> None:
        if dim < 2:
            raise ValueError("Rosenbrock needs at least 2 dimensions")
        super().__init__(dim, instance)

    def true_error(self, x: np.ndarray) -> float:
        z = self._z(x) + 1.0
        return float(np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (1.0 - z[:-1]) ** 2))


class Rastrigin(_Shifted):
    name = "rastrigin"

    def true_error(self, x: np.ndarray) -> float:
        z = self._z(x)
        return float(10.0 * self.dim + np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z)))


class NoisySphere(Sphere):
    """Sphere multiplied by (1 + 0.1 * eps), eps ~ N(0, 1), drawn per call. true_error() is noise-free."""

    name = "noisy_sphere"
    noisy = True

    def __init__(self, dim: int, instance: int) -> None:
        super().__init__(dim, instance)
        self._noise = instance_rng(instance, dim, _NOISE)

    def evaluate(self, x: np.ndarray) -> float:
        return self.true_error(x) * (1.0 + NOISE_LEVEL * float(self._noise.standard_normal()))


PROBLEMS: dict[str, Callable[[int, int], Problem]] = {}


def register(factory: Callable[[int, int], Problem], name: str | None = None) -> None:
    """Make a problem available by name in the benchmark configs: `factory(dim, instance)` builds an instance."""
    key = name or getattr(factory, "name")  # noqa: B009
    PROBLEMS[key] = factory


for _problem in (Sphere, Ellipsoid, Rosenbrock, Rastrigin, NoisySphere):
    register(_problem)


def make_problem(name: str, dim: int, instance: int) -> Problem:
    try:
        factory = PROBLEMS[name]
    except KeyError:
        raise ValueError(f"unknown problem {name!r}, available: {sorted(PROBLEMS)}") from None
    return factory(dim, instance)
