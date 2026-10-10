"""The new Auxein core: `NSGA2` with its defaults (population 100, SBX and polynomial mutation), on a `Box`.

Parameters (all optional, in the benchmark config): `population_size`, `offspring_size`, `crossover_probability`, `sbx_eta`,
`mutation_eta`, and `backend` / `precision` / `device`. The objective is evaluated through a `VectorisedEvaluator` whose
function loops over the rows of the batch, since the harness's `MOCountingObjective` is per-point and counts every call.
"""

import warnings
from typing import Any

import numpy as np

from auxein.backend import Array, Backend
from auxein.core import Objective
from auxein.driver import Budget, RecordingDisabledWarning
from auxein.driver import run as run_driver
from auxein.evaluators import EvaluationError, VectorisedEvaluator
from auxein.spaces import Box
from auxein.strategies import NSGA2
from auxein.strategies.ga import PolynomialMutation, SimulatedBinaryCrossover
from benchmarks.adapters.base import RunInfo, backend_from
from benchmarks.mo_objective import BudgetExhausted, MOCountingObjective


def run(objective: MOCountingObjective, dim: int, seed: int, params: dict[str, Any]) -> RunInfo:
    problem = objective.problem
    population = int(params.get("population_size", 100))
    strategy = NSGA2(
        population,
        int(params.get("offspring_size", population)),
        recombination=SimulatedBinaryCrossover(eta=float(params.get("sbx_eta", 15.0))),
        mutation=PolynomialMutation(eta=float(params.get("mutation_eta", 20.0))),
        crossover_probability=float(params.get("crossover_probability", 0.9)),
    )
    host = Backend()

    def evaluate(batch: Array) -> Array:
        return np.stack([objective(row) for row in (batch if isinstance(batch, np.ndarray) else host.to_numpy(batch))])

    evaluations = 0
    stop_reason = "budget"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RecordingDisabledWarning)  # the harness records through its own traces
            result = run_driver(
                strategy=strategy,
                evaluator=VectorisedEvaluator(evaluate),
                space=Box(problem.lower, problem.upper, dim=dim),
                objectives=[Objective(f"f{i + 1}") for i in range(problem.n_obj)],
                budget=Budget(evaluations=objective.remaining),
                seed=seed,
                batch_size=population,
                backend=backend_from(params),
            )
        evaluations = result.evaluations_used
        stop_reason = "budget" if result.stop_reason == "budget:evaluations" else result.stop_reason
    except EvaluationError as error:  # defensive: the driver's budget is exact, but the harness has the final word
        if not isinstance(error.__cause__, BudgetExhausted):
            raise
    return RunInfo(stop_reason=stop_reason, extra={"population_size": population, "evaluations": evaluations})
