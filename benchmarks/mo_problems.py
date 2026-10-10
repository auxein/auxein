"""Multi-objective benchmark problems: ZDT1, ZDT2, ZDT3 (two objectives, 30 variables) and DTLZ2 (three objectives, 12 variables).

All variables are in [0, 1] and all objectives are minimised. Unlike the single-objective problems there are no instances
(the problems are fixed, so runs differ by the seed of the algorithm alone) and no shift: what an algorithm is judged on is
how well the set of non-dominated points it has found approximates the **known Pareto front**, which every problem provides as
a dense reference set, together with the reference point at which hypervolume is measured (1.1 times the nadir of the front).
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

REFERENCE_FACTOR = 1.1


class MOProblem(ABC):
    """A multi-objective minimisation problem on [0, 1]^dim."""

    name: str
    dim: int
    n_obj: int
    lower = 0.0
    upper = 1.0

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """The objective vector of one point, shape `(n_obj,)`."""

    @abstractmethod
    def pareto_front(self) -> np.ndarray:
        """A dense sample of the true Pareto front, shape `(n, n_obj)`: the reference set of IGD+."""

    @property
    def reference_point(self) -> np.ndarray:
        """The point hypervolume is measured against: 1.1 times the nadir of the true front, componentwise."""
        return REFERENCE_FACTOR * self.pareto_front().max(axis=0)


class _ZDT(MOProblem):
    n_obj = 2

    def __init__(self, dim: int = 30) -> None:
        self.dim = dim

    def _g(self, x: np.ndarray) -> float:
        return float(1.0 + 9.0 * np.mean(x[1:]))


class ZDT1(_ZDT):
    """Convex front `f2 = 1 - sqrt(f1)`."""

    name = "zdt1"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = float(x[0])
        g = self._g(x)
        return np.array([f1, g * (1.0 - math.sqrt(f1 / g))])

    def pareto_front(self) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, 1000)
        return np.stack([f1, 1.0 - np.sqrt(f1)], axis=1)


class ZDT2(_ZDT):
    """Concave front `f2 = 1 - f1^2`."""

    name = "zdt2"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = float(x[0])
        g = self._g(x)
        return np.array([f1, g * (1.0 - (f1 / g) ** 2)])

    def pareto_front(self) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, 1000)
        return np.stack([f1, 1.0 - f1**2], axis=1)


class ZDT3(_ZDT):
    """Disconnected front: five separate pieces of `f2 = 1 - sqrt(f1) - f1 sin(10 pi f1)`."""

    name = "zdt3"
    SEGMENTS = (
        (0.0, 0.0830015349),
        (0.1822287280, 0.2577623634),
        (0.4093136748, 0.4538821041),
        (0.6183967944, 0.6525117038),
        (0.8233317983, 0.8518328654),
    )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = float(x[0])
        g = self._g(x)
        return np.array([f1, g * (1.0 - math.sqrt(f1 / g) - (f1 / g) * math.sin(10.0 * math.pi * f1))])

    def pareto_front(self) -> np.ndarray:
        f1 = np.concatenate([np.linspace(low, high, 200) for low, high in self.SEGMENTS])
        return np.stack([f1, 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)], axis=1)


class DTLZ2(MOProblem):
    """Three objectives (the positive octant of the unit sphere is the front); 12 variables, 10 of which only add distance."""

    name = "dtlz2"
    n_obj = 3

    def __init__(self, dim: int = 12) -> None:
        self.dim = dim

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        g = float(np.sum((x[2:] - 0.5) ** 2))
        a, b = x[0] * math.pi / 2.0, x[1] * math.pi / 2.0
        return np.array([(1.0 + g) * math.cos(a) * math.cos(b), (1.0 + g) * math.cos(a) * math.sin(b), (1.0 + g) * math.sin(a)])

    def pareto_front(self) -> np.ndarray:
        angles = np.linspace(0.0, math.pi / 2.0, 41)
        a, b = np.meshgrid(angles, angles, indexing="ij")
        return np.stack([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)], axis=-1).reshape(-1, 3)


MO_PROBLEMS: dict[str, Callable[[], MOProblem]] = {"zdt1": ZDT1, "zdt2": ZDT2, "zdt3": ZDT3, "dtlz2": DTLZ2}


def make_mo_problem(name: str) -> MOProblem:
    try:
        return MO_PROBLEMS[name]()
    except KeyError:
        raise ValueError(f"unknown multi-objective problem {name!r}, available: {sorted(MO_PROBLEMS)}") from None
