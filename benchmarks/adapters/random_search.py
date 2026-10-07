"""Uniform random search: the floor that any real algorithm has to beat."""

from typing import Any

import numpy as np

from benchmarks.adapters.base import RunInfo
from benchmarks.objective import BudgetExhausted, CountingObjective

BATCH = 256


def run(objective: CountingObjective, dim: int, seed: int, params: dict[str, Any]) -> RunInfo:
    rng = np.random.default_rng(seed)
    lower, upper = objective.problem.lower, objective.problem.upper
    try:
        while True:
            # successive rows of one draw are the same stream as successive single draws
            for x in rng.uniform(lower, upper, (BATCH, dim)):
                objective(x)
    except BudgetExhausted:
        pass
    return RunInfo(stop_reason="budget")
