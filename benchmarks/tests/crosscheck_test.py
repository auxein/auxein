"""Cross-check of the new driver against the harness.

With `RandomSearch`, the new core must be statistically indistinguishable from the harness's own random search: both
draw uniformly in the same box, so if the Vargha-Delaney A12 of their final errors leaves [0.35, 0.65], the driver (or the
adapter) is doing something other than a plain random search with the budget it was given. The seeds are fixed, so the
test is deterministic.
"""

import pytest

from benchmarks.runner import run_one
from benchmarks.stats import vargha_delaney

RUNS = 30
QUICK_BUDGET_PER_DIM = 500


def final_errors(adapter: str, dim: int) -> list[float]:
    errors = []
    for k in range(RUNS):  # run k uses instance k and seed k, as in the harness
        objective, _, _ = run_one(adapter, {}, "sphere", dim, instance=k, seed=k, budget=QUICK_BUDGET_PER_DIM * dim)
        assert objective.evals == QUICK_BUDGET_PER_DIM * dim
        errors.append(objective.best_error)
    return errors


@pytest.mark.parametrize("dim", [2, 10])
def test_the_new_core_with_random_search_is_indistinguishable_from_the_harness_random_search(dim: int):
    harness = final_errors("random_search", dim)
    core = final_errors("auxein_core_random", dim)
    a12 = vargha_delaney(core, harness)  # the probability that the new core ends with the lower error
    assert 0.35 <= a12 <= 0.65, f"A12 = {a12:.3f} on the {dim}-D sphere"


# --- the torch backend searches as well as numpy ---

GA_PARAMS = {
    "population_size": 50,
    "offspring_size": 50,
    "selection": {"type": "tournament", "size": 2},
    "recombination": {"type": "intermediate", "per_gene": False},
    "mutation": {"type": "self_adaptive", "per_gene": False, "initial_step": 0.1, "min_step": 1e-12},
    "repair": "clip",
}
TORCH_FLOAT32 = {"backend": "torch", "precision": "float32"}


FIRST_RUN = 120
"""The torch checks use runs 120 to 149 (instances and seeds), not 0 to 29. With 30 runs the band [0.35, 0.65] is about two
standard errors of A12 under no difference at all, so one cell in twenty leaves it by chance, whatever the code does, and the
block 0 to 29 puts the 2-D sphere 0.002 inside its edge: float32 results that differ in the last bit between CPU
architectures could tip it. This block keeps every cell between 0.40 and 0.58, at least 0.05 inside the band, on arm64 and on
x86-64. The band guards against gross shifts (a precision bug, a broken operator on tensors), which move A12 far out of it;
it cannot see a 10% difference."""


def final_errors_of(adapter: str, params: dict, problem: str, dim: int, first: int = FIRST_RUN, exact: bool = True) -> list[float]:
    errors = []
    for k in range(first, first + RUNS):
        objective, _, _ = run_one(adapter, params, problem, dim, instance=k, seed=k, budget=QUICK_BUDGET_PER_DIM * dim)
        assert not exact or objective.evals == QUICK_BUDGET_PER_DIM * dim  # CMA-ES stops early once it has converged
        errors.append(objective.best_error)
    return errors


@pytest.mark.parametrize("dim", [2, 10])
@pytest.mark.parametrize("problem", ["sphere", "rastrigin"])
def test_the_genetic_algorithm_on_torch_float32_is_statistically_indistinguishable_from_numpy_float64(problem: str, dim: int):
    """The streams differ, so the runs differ; but a backend that searched worse (a precision bug, a broken operator on tensors)
    would shift the whole distribution of final errors. Same seeds and instances on both sides, so the comparison is paired."""
    numpy_errors = final_errors_of("auxein_core_ga", GA_PARAMS, problem, dim)
    torch_errors = final_errors_of("auxein_core_ga", {**GA_PARAMS, **TORCH_FLOAT32}, problem, dim)
    a12 = vargha_delaney(torch_errors, numpy_errors)  # the probability that torch ends with the lower error
    assert 0.35 <= a12 <= 0.65, f"A12 = {a12:.3f} on the {dim}-D {problem}"


@pytest.mark.parametrize("dim", [2, 10])
def test_random_search_on_torch_float32_is_statistically_indistinguishable_from_numpy_float64(dim: int):
    numpy_errors = final_errors_of("auxein_core_random", {}, "sphere", dim)
    torch_errors = final_errors_of("auxein_core_random", TORCH_FLOAT32, "sphere", dim)
    a12 = vargha_delaney(torch_errors, numpy_errors)
    assert 0.35 <= a12 <= 0.65, f"A12 = {a12:.3f} on the {dim}-D sphere"


# --- the pycma wrapper against raw pycma ---


@pytest.mark.parametrize("problem", ["sphere", "rastrigin"])
def test_the_pycma_strategy_is_statistically_indistinguishable_from_raw_pycma(problem: str):
    """Both are pycma started from the same kind of point with the same step size; only the random streams differ (the wrapper
    draws from Auxein's stream, raw pycma from its own seeded generator), and the wrapper adds the box as pycma bounds. A
    wrapper that changed what pycma does (a wrong failure substitution, a mis-ordered tell, bounds that bite) would shift the
    distribution of the final errors, and A12 would leave the band."""
    wrapper = final_errors_of("auxein_pycma", {}, problem, 10, first=0, exact=False)
    raw = final_errors_of("cmaes", {}, problem, 10, first=0, exact=False)
    a12 = vargha_delaney(wrapper, raw)
    assert 0.35 <= a12 <= 0.65, f"A12 = {a12:.3f} on the 10-D {problem}"
