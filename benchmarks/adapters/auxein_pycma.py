"""The new Auxein core: the driver with `PycmaStrategy`, on a `Box`, evaluating the harness's counting objective.

It exists to cross-check the wrapper against raw pycma (the `cmaes` adapter): with the same start and step size they are the
same algorithm with different random streams, so they must be statistically indistinguishable (see
benchmarks/tests/crosscheck_test.py).

Parameters (as for the `cmaes` adapter): `sigma0` (default 2.0), `x0_range` (the start is drawn uniformly from
[-x0_range, x0_range]^d, default 4.0, from the run's seed), and `popsize`. The objective is evaluated through a
`VectorisedEvaluator` whose function loops over the rows of the batch.
"""

import warnings
from typing import Any

import numpy as np

from auxein.backend import Array, Backend
from auxein.driver import Budget, RecordingDisabledWarning
from auxein.driver import run as run_driver
from auxein.evaluators import EvaluationError, VectorisedEvaluator
from auxein.spaces import Box
from auxein.strategies import PycmaStrategy
from benchmarks.adapters.base import RunInfo, backend_from
from benchmarks.objective import BudgetExhausted, CountingObjective

DEFAULTS: dict[str, Any] = {"sigma0": 2.0, "x0_range": 4.0}


def run(objective: CountingObjective, dim: int, seed: int, params: dict[str, Any]) -> RunInfo:
    options = {**DEFAULTS, **params}
    problem = objective.problem
    x0 = np.random.default_rng(seed).uniform(-options["x0_range"], options["x0_range"], dim)
    strategy = PycmaStrategy(options.get("popsize"), x0=x0, sigma0=options["sigma0"])
    host = Backend()

    def evaluate(batch: Array) -> Array:
        return np.array([objective(row) for row in (batch if isinstance(batch, np.ndarray) else host.to_numpy(batch))])

    evaluations = 0
    stop_reason = "budget"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RecordingDisabledWarning)  # the harness records through its own traces
            result = run_driver(
                strategy=strategy,
                evaluator=VectorisedEvaluator(evaluate),
                space=Box(problem.lower, problem.upper, dim=dim),
                budget=Budget(evaluations=objective.remaining),
                seed=seed,
                batch_size=int(options.get("popsize") or 4 + int(3 * np.log(dim))),
                backend=backend_from(params),
            )
        evaluations = result.evaluations_used
        stop_reason = "budget" if result.stop_reason == "budget:evaluations" else result.stop_reason
    except EvaluationError as error:  # defensive: the driver's budget is exact, but the harness has the final word
        if not isinstance(error.__cause__, BudgetExhausted):
            raise
    popsize = strategy.population_size or 4 + int(3 * np.log(dim))
    return RunInfo(
        generations=-(-evaluations // popsize) if evaluations else 0,
        stop_reason=stop_reason,
        evals_per_generation=float(popsize),
        extra={"popsize": popsize},
    )
