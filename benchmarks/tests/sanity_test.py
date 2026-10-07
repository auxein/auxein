"""Checks on the harness itself. If one of these fails, suspect the harness before the algorithms."""

import pytest

from benchmarks.runner import run_one

FULL_BUDGET_10D = 2000 * 10
RUNS = 10


def hit_rate(adapter, params, problem, target):
    hits = 0
    for k in range(RUNS):
        objective, _, _ = run_one(adapter, params, problem, 10, k, k, FULL_BUDGET_10D, targets=(target,))
        hits += objective.hits[target] is not None
    return hits / RUNS


@pytest.mark.parametrize("problem", ["sphere", "ellipsoid"])
def test_cma_es_reaches_1e_6_within_the_full_budget_in_at_least_90_percent_of_runs(problem):
    assert hit_rate("cmaes", {}, problem, 1e-6) >= 0.9


def test_random_search_does_not_reach_1e_3_on_the_10d_sphere():
    assert hit_rate("random_search", {}, "sphere", 1e-3) == 0.0
