"""The new Auxein core: the driver with `GeneticAlgorithm`, on a `Box`, evaluating the harness's counting objective.

The objective is evaluated through a `VectorisedEvaluator` (the way a numeric user would), whose function loops over
the rows of the batch because the harness's `CountingObjective` is per-point. The driver's evaluation budget is exact,
so the harness's counting semantics (every call counted, the trace at the checkpoints) are untouched.

Parameters (all optional, in the benchmark config): `population_size` (mu, the name the overhead benchmark varies),
`offspring_size` (lambda, or absent for the default), `selection`, `recombination`, `mutation`, `repair`,
`crossover_probability` and `batch_size`. Operators are tables with a `type` and its parameters.
"""

import warnings
from typing import Any

import numpy as np

from auxein.backend import Array
from auxein.driver import Budget, RecordingDisabledWarning
from auxein.driver import run as run_driver
from auxein.evaluators import EvaluationError, VectorisedEvaluator
from auxein.spaces import Box
from auxein.strategies import GeneticAlgorithm
from auxein.strategies.ga import (
    BoundsRepair,
    ClipRepair,
    GaussianMutation,
    IntermediateRecombination,
    Mutation,
    NoRecombination,
    ParentSelection,
    Recombination,
    ReflectRepair,
    SelfAdaptiveMutation,
    SigmaScalingSUS,
    TournamentSelection,
    UniformRecombination,
)
from benchmarks.adapters.base import RunInfo
from benchmarks.objective import BudgetExhausted, CountingObjective


def _table(spec: dict[str, Any] | str, kind: str) -> tuple[str, dict[str, Any]]:
    if isinstance(spec, str):
        return spec, {}
    options = dict(spec)
    if "type" not in options:
        raise ValueError(f"the {kind} needs a 'type', got {spec!r}")
    return str(options.pop("type")), options


def build_selection(spec: dict[str, Any] | str) -> ParentSelection:
    kind, options = _table(spec, "selection")
    if kind == "tournament":
        return TournamentSelection(**options)
    if kind == "sus":
        return SigmaScalingSUS(**options)
    raise ValueError(f"unknown selection {kind!r}")


def build_recombination(spec: dict[str, Any] | str) -> Recombination:
    kind, options = _table(spec, "recombination")
    if kind == "intermediate":
        return IntermediateRecombination(**options)
    if kind == "uniform":
        return UniformRecombination(**options)
    if kind == "none":
        return NoRecombination(**options)
    raise ValueError(f"unknown recombination {kind!r}")


def build_mutation(spec: dict[str, Any] | str) -> Mutation:
    kind, options = _table(spec, "mutation")
    if kind == "self_adaptive":
        return SelfAdaptiveMutation(**options)
    if kind == "gaussian":
        return GaussianMutation(**options)
    raise ValueError(f"unknown mutation {kind!r}")


def build_repair(spec: dict[str, Any] | str) -> BoundsRepair:
    kind, _ = _table(spec, "repair")
    if kind == "clip":
        return ClipRepair()
    if kind == "reflect":
        return ReflectRepair()
    raise ValueError(f"unknown repair {kind!r}")


def build_strategy(params: dict[str, Any]) -> GeneticAlgorithm:
    keywords: dict[str, Any] = {}
    for name, builder in (
        ("selection", build_selection),
        ("recombination", build_recombination),
        ("mutation", build_mutation),
        ("repair", build_repair),
    ):
        if name in params:
            keywords[name] = builder(params[name])
    if "crossover_probability" in params:
        keywords["crossover_probability"] = params["crossover_probability"]
    sizes = {k: params[k] for k in ("population_size", "offspring_size") if k in params}
    return GeneticAlgorithm(**sizes, **keywords)


def run(objective: CountingObjective, dim: int, seed: int, params: dict[str, Any]) -> RunInfo:
    strategy = build_strategy(params)
    batch_size = int(params.get("batch_size", strategy.offspring_size or 64))
    problem = objective.problem

    def evaluate(batch: Array) -> Array:
        return np.array([objective(np.asarray(row)) for row in batch])

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
                batch_size=batch_size,
            )
        evaluations = result.evaluations_used
        stop_reason = "budget" if result.stop_reason == "budget:evaluations" else result.stop_reason
    except EvaluationError as error:  # defensive: the driver's budget is exact, but the harness has the final word
        if not isinstance(error.__cause__, BudgetExhausted):
            raise

    mu = strategy.population_size
    lam = strategy.offspring_size or batch_size
    generations = max(0, -(-(evaluations - mu) // lam)) if evaluations > mu else 0
    return RunInfo(
        generations=generations,
        stop_reason=stop_reason,
        evals_per_generation=float(lam),  # lambda children: nothing is ever re-scored
        extra={"initial_evals": min(mu, evaluations), "population_size": mu, "offspring_size": lam},
    )
