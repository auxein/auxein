"""Budget enforcement and trace recording, outside the algorithms.

Every algorithm gets the same number of fitness evaluations. A `CountingObjective` wraps a problem, counts every
call, and refuses to evaluate once the budget is spent, so no algorithm can overspend. It also records the best
*true* error seen so far (never the noisy value the algorithm sees), at log-spaced evaluation counts.
"""

import math
from collections.abc import Iterable, Sequence

import numpy as np

from benchmarks.problems import Problem

CHECKPOINTS_PER_DECADE = 20


class BudgetExhausted(Exception):
    """Raised by the call that would exceed the evaluation budget. Adapters catch it and finish the run cleanly."""


def log_checkpoints(budget: int, per_decade: int = CHECKPOINTS_PER_DECADE) -> list[int]:
    """Log-spaced evaluation counts (about `per_decade` per decade) including the first and the last evaluation."""
    if budget < 1:
        raise ValueError("the budget must be at least one evaluation")
    steps = math.ceil(per_decade * math.log10(budget)) + 1
    counts = {1, budget}
    counts.update(int(round(10 ** (k / per_decade))) for k in range(steps))
    return sorted(c for c in counts if 1 <= c <= budget)


class CountingObjective:
    """The objective an algorithm optimises: `objective(x)` returns the (possibly noisy) value of `problem`."""

    def __init__(self, problem: Problem, budget: int, targets: Iterable[float] = (), checkpoints: Sequence[int] | None = None) -> None:
        self.problem = problem
        self.budget = budget
        self.evals = 0
        self.best_error = math.inf
        self.hits: dict[float, int | None] = {float(t): None for t in targets}  # evaluations needed to reach each target
        self._checkpoints = set(log_checkpoints(budget) if checkpoints is None else checkpoints)
        self._trace: list[tuple[int, float]] = []

    def __call__(self, x: np.ndarray) -> float:
        if self.evals >= self.budget:
            raise BudgetExhausted
        self.evals += 1
        error = self.problem.true_error(x)
        value = self.problem.evaluate(x) if self.problem.noisy else error
        if error < self.best_error:
            self.best_error = error
            for target, hit in self.hits.items():
                if hit is None and error <= target:
                    self.hits[target] = self.evals
        if self.evals in self._checkpoints:
            self._trace.append((self.evals, self.best_error))
        return value

    @property
    def remaining(self) -> int:
        return self.budget - self.evals

    @property
    def trace(self) -> list[tuple[int, float]]:
        """Best true error so far at the checkpoints, ending at the last evaluation made."""
        if self.evals and (not self._trace or self._trace[-1][0] != self.evals):
            return [*self._trace, (self.evals, self.best_error)]
        return list(self._trace)
