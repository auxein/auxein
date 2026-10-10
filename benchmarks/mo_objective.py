"""Budget enforcement and quality traces for multi-objective runs, outside the algorithms.

The counterpart of `CountingObjective`: every algorithm gets the same number of evaluations, counted here, and the quality of
a run is measured on the **non-dominated set of everything it has evaluated so far** (an archive kept here, not the
algorithm's own population), so that algorithms are compared on what they found and not on what they kept. Two indicators
are recorded at log-spaced evaluation counts: the **hypervolume** of that set against a fixed reference point (higher is
better) and **IGD+** to the true front (lower is better). Both are computed with pymoo, which only the benchmarks use.
"""

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from benchmarks.mo_problems import MOProblem
from benchmarks.objective import BudgetExhausted, log_checkpoints

__all__ = ["BudgetExhausted", "MOCountingObjective"]


class MOCountingObjective:
    """The objective an algorithm optimises: `objective(x)` returns the vector of objective values of `problem`."""

    def __init__(self, problem: MOProblem, budget: int) -> None:
        self.problem = problem
        self.budget = budget
        self.evals = 0
        self._archive = np.empty((0, problem.n_obj))
        self._hypervolume = HV(ref_point=problem.reference_point)
        self._igd_plus = IGDPlus(problem.pareto_front())
        self._checkpoints = set(log_checkpoints(budget))
        self._trace: list[tuple[int, float, float]] = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.evals >= self.budget:
            raise BudgetExhausted
        self.evals += 1
        value = self.problem.evaluate(np.asarray(x, dtype=float))
        self._archive_add(value)
        if self.evals in self._checkpoints:
            self._trace.append((self.evals, *self._indicators()))
        return value

    def _archive_add(self, point: np.ndarray) -> None:
        """Keep the archive non-dominated: drop the point if something covers it, else drop what it dominates."""
        archive = self._archive
        if archive.shape[0]:
            if bool((archive <= point).all(axis=1).any()):  # an archived point dominates it or equals it
                return
            dominated = (point <= archive).all(axis=1) & (point < archive).any(axis=1)
            archive = archive[~dominated]
        self._archive = np.vstack([archive, point[None, :]])

    def _indicators(self) -> tuple[float, float]:
        hypervolume, igd_plus = self._hypervolume(self._archive), self._igd_plus(self._archive)
        assert hypervolume is not None and igd_plus is not None
        return float(hypervolume), float(igd_plus)

    @property
    def remaining(self) -> int:
        return self.budget - self.evals

    @property
    def front(self) -> np.ndarray:
        """The non-dominated set of everything evaluated so far."""
        return self._archive.copy()

    @property
    def trace(self) -> list[tuple[int, float, float]]:
        """(evaluations, hypervolume, IGD+) at the checkpoints, ending at the last evaluation made."""
        if self.evals and (not self._trace or self._trace[-1][0] != self.evals):
            return [*self._trace, (self.evals, *self._indicators())]
        return list(self._trace)

    @property
    def final_hypervolume(self) -> float:
        return self._indicators()[0] if self.evals else 0.0

    @property
    def final_igd_plus(self) -> float:
        return self._indicators()[1] if self.evals else float("inf")
