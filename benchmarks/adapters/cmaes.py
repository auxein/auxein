"""CMA-ES (pycma) through its ask/tell interface: the strong reference."""

from typing import Any

import cma
import numpy as np

from benchmarks.adapters.base import RunInfo
from benchmarks.objective import BudgetExhausted, CountingObjective

DEFAULTS: dict[str, Any] = {"sigma0": 2.0, "x0_range": 4.0}


def run(objective: CountingObjective, dim: int, seed: int, params: dict[str, Any]) -> RunInfo:
    options = {**DEFAULTS, **params}
    x0 = np.random.default_rng(seed).uniform(-options["x0_range"], options["x0_range"], dim)
    cma_options: dict[str, Any] = {"seed": seed + 1, "verbose": -9, "maxfevals": objective.budget}
    if "popsize" in options:
        cma_options["popsize"] = options["popsize"]
    strategy = cma.CMAEvolutionStrategy(x0, options["sigma0"], cma_options)

    generations = 0
    stop_reason = "budget"
    try:
        while True:
            reasons = strategy.stop()
            if reasons:
                stop_reason = ", ".join(sorted(reasons))
                break
            candidates = strategy.ask()
            strategy.tell(candidates, [objective(x) for x in candidates])
            generations += 1
    except BudgetExhausted:
        pass
    return RunInfo(
        generations=generations,
        stop_reason=stop_reason,
        evals_per_generation=objective.evals / generations if generations else None,
        extra={"popsize": int(strategy.popsize)},
    )
