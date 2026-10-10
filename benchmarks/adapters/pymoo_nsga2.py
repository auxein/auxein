"""pymoo's NSGA-II, the external reference: the same population, operators and parameters as `auxein_nsga2`.

Population and offspring 100, simulated binary crossover (probability 0.9, eta 15), polynomial mutation (eta 20, each variable
with probability 1/d), binary tournament on the crowded comparison, random sampling. Everything else is pymoo's default
(notably duplicate elimination, which Auxein does not do, and the *bounded* form of SBX; Auxein's is the unbounded form and
relies on its bounds repair). `pymoo` is a benchmark dependency only: `auxein` never imports it.
"""

from typing import Any

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from benchmarks.adapters.base import RunInfo
from benchmarks.mo_objective import BudgetExhausted, MOCountingObjective


class _Wrapped(Problem):
    """The harness's counting objective as a pymoo problem (vectorised over the rows that pymoo hands in)."""

    def __init__(self, objective: MOCountingObjective, dim: int) -> None:
        problem = objective.problem
        super().__init__(n_var=dim, n_obj=problem.n_obj, xl=problem.lower, xu=problem.upper)
        self._objective = objective

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        out["F"] = np.stack([self._objective(row) for row in x])


def run(objective: MOCountingObjective, dim: int, seed: int, params: dict[str, Any]) -> RunInfo:
    population = int(params.get("population_size", 100))
    algorithm = NSGA2(
        pop_size=population,
        n_offsprings=int(params.get("offspring_size", population)),
        crossover=SBX(prob=float(params.get("crossover_probability", 0.9)), eta=params.get("sbx_eta", 15)),
        mutation=PM(eta=params.get("mutation_eta", 20)),
    )
    try:
        minimize(_Wrapped(objective, dim), algorithm, ("n_eval", objective.remaining), seed=seed, verbose=False)
    except BudgetExhausted:  # the last generation is cut where the budget ends
        pass
    return RunInfo(stop_reason="budget", extra={"population_size": population})
