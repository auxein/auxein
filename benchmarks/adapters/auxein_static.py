"""Auxein: the `Static` playground with a configurable operator set.

The evaluation budget is what stops the run: `max_generations` is set far beyond anything reachable.
"""

from typing import Any

import numpy as np

from auxein.fitness.kernel_based import GlobalMinimum
from auxein.mutations import FixedVariance, Mutation, SelfAdaptiveSingleStep
from auxein.parents.distributions import Distribution, Fps, FpsWithWindowing, SigmaScaling
from auxein.parents.selections import StochasticUniversalSampling
from auxein.playgrounds import Static
from auxein.population import Population, UniformRandomDnaBuilder, build_fixed_dimension_population
from auxein.recombinations import SimpleArithmetic
from auxein.replacements import ReplaceWorst
from benchmarks.adapters.base import RunInfo
from benchmarks.objective import BudgetExhausted, CountingObjective

MAX_GENERATIONS = 10**9

DEFAULTS: dict[str, Any] = {
    "population_size": 100,
    "mutation": {"type": "self_adaptive", "tau": 0.1},
    "distribution": "sigma_scaling",
    "offspring_size": 4,
    "alpha": 0.5,
}

DISTRIBUTIONS: dict[str, type[Distribution]] = {"sigma_scaling": SigmaScaling, "fps_windowing": FpsWithWindowing, "fps": Fps}


def build_mutation(spec: dict[str, Any]) -> Mutation:
    kind = spec["type"]
    if kind == "self_adaptive":
        return SelfAdaptiveSingleStep(tau=spec["tau"])
    if kind == "fixed_variance":
        return FixedVariance(sigma=spec["sigma"])
    raise ValueError(f"unknown mutation {kind!r}")


def run(objective: CountingObjective, dim: int, seed: int, params: dict[str, Any]) -> RunInfo:
    options = {**DEFAULTS, **params}
    np.random.seed(seed)  # Auxein still draws from the global numpy random state

    # the generation boundaries are noticed when the first evaluation of the next generation arrives
    boundaries: list[int] = []
    state: dict[str, Any] = {"population": None, "generation": 0}

    def kernel(x: np.ndarray) -> float:
        population: Population | None = state["population"]
        if population is not None and population.generation_count != state["generation"]:
            state["generation"] = population.generation_count
            boundaries.append(objective.evals)
        return objective(x)

    fitness = GlobalMinimum(kernel)
    offspring_size = options["offspring_size"]
    initial_evals = 0
    generations = 0
    stop_reason = "budget"
    try:
        population = build_fixed_dimension_population(
            dim,
            options["population_size"],
            fitness,
            UniformRandomDnaBuilder((objective.problem.lower, objective.problem.upper)),
        )
        initial_evals = objective.evals
        state["population"] = population
        playground = Static(
            population=population,
            fitness=fitness,
            mutation=build_mutation(options["mutation"]),
            distribution=DISTRIBUTIONS[options["distribution"]](),
            selection=StochasticUniversalSampling(offspring_size=offspring_size),
            recombination=SimpleArithmetic(alpha=options["alpha"]),
            replacement=ReplaceWorst(offspring_size=offspring_size),
        )
        playground.train(MAX_GENERATIONS)
        stop_reason = "population_size"  # the playground's own stop condition: the population is too small to breed
        generations = population.generation_count
    except BudgetExhausted:
        completed: Population | None = state["population"]
        generations = completed.generation_count if completed is not None else 0

    costs = np.diff([initial_evals, *boundaries])
    return RunInfo(
        generations=generations,
        stop_reason=stop_reason,
        evals_per_generation=float(costs.mean()) if len(costs) else None,
        extra={"initial_evals": initial_evals, "population_size": options["population_size"]},
    )
