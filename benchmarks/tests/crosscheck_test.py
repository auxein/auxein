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
