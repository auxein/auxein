"""The new Auxein core: the driver with `RandomSearch`, on a `Box`, evaluating the harness's counting objective.

It exists to cross-check the new driver against the harness: with `RandomSearch` it must be statistically
indistinguishable from the harness's own random search (see benchmarks/tests/crosscheck_test.py).
"""

import warnings
from typing import Any

import numpy as np

from auxein.driver import Budget, RecordingDisabledWarning
from auxein.driver import run as run_driver
from auxein.evaluators import EvaluationError, FunctionEvaluator
from auxein.spaces import Box
from auxein.strategies import RandomSearch
from benchmarks.adapters.base import RunInfo
from benchmarks.objective import BudgetExhausted, CountingObjective

DEFAULT_BATCH_SIZE = 64


def run(objective: CountingObjective, dim: int, seed: int, params: dict[str, Any]) -> RunInfo:
    batch_size = int(params.get("batch_size", DEFAULT_BATCH_SIZE))
    problem = objective.problem
    stop_reason = "budget"
    evaluations = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RecordingDisabledWarning)  # the harness records through its own traces
            result = run_driver(
                strategy=RandomSearch(),
                evaluator=FunctionEvaluator(lambda genome: objective(np.asarray(genome))),
                space=Box(problem.lower, problem.upper, dim=dim),
                budget=Budget(evaluations=objective.remaining),
                seed=seed,
                batch_size=batch_size,
            )
        evaluations = result.evaluations_used
        stop_reason = "budget" if result.stop_reason == "budget:evaluations" else result.stop_reason
    except EvaluationError as error:  # defensive: the driver's budget is exact, but the harness has the final word
        if not isinstance(error.__cause__, BudgetExhausted):
            raise
    return RunInfo(generations=-(-evaluations // batch_size), stop_reason=stop_reason, evals_per_generation=float(batch_size))
