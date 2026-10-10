"""Uniform random search on a multi-objective problem: the floor that any real algorithm has to beat."""

from typing import Any

import numpy as np

from benchmarks.adapters.base import RunInfo
from benchmarks.mo_objective import BudgetExhausted, MOCountingObjective

BATCH = 256


def run(objective: MOCountingObjective, dim: int, seed: int, params: dict[str, Any]) -> RunInfo:
    rng = np.random.default_rng(seed)
    problem = objective.problem
    try:
        while True:
            for x in rng.uniform(problem.lower, problem.upper, (BATCH, dim)):
                objective(x)
    except BudgetExhausted:
        pass
    return RunInfo(stop_reason="budget")
