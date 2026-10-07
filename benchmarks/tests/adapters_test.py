import pytest

from benchmarks.adapters import load_adapter
from benchmarks.objective import CountingObjective
from benchmarks.problems import make_problem

AUXEIN = {
    "auxein-default": {},
    "auxein-fixedvar": {"mutation": {"type": "fixed_variance", "sigma": 0.1}},
    "auxein-windowing": {"distribution": "fps_windowing"},
}
CASES = [("auxein_static", params) for params in AUXEIN.values()] + [("random_search", {}), ("cmaes", {})]
IDS = [*AUXEIN, "random-search", "cma-es"]


def run_once(adapter, params, budget=700, dim=3, seed=5, problem="sphere"):
    objective = CountingObjective(make_problem(problem, dim, 0), budget, targets=[1e-1])
    info = load_adapter(adapter)(objective, dim, seed, params)
    return objective, info


@pytest.mark.parametrize(("adapter", "params"), CASES, ids=IDS)
def test_adapter_respects_the_budget(adapter, params):
    for budget in (30, 700):
        objective, info = run_once(adapter, params, budget=budget)
        assert objective.evals <= budget
        if info.stop_reason == "budget":  # CMA-ES may stop earlier, once it has converged
            assert objective.evals == budget
        assert objective.trace[-1][0] == objective.evals


def test_cma_stops_early_once_converged_and_says_so():
    objective, info = run_once("cmaes", {}, budget=700)
    assert objective.evals < 700
    assert info.stop_reason == "tolfun"


@pytest.mark.parametrize(("adapter", "params"), CASES, ids=IDS)
def test_adapter_is_deterministic_for_a_given_seed(adapter, params):
    a, info_a = run_once(adapter, params)
    b, info_b = run_once(adapter, params)
    assert a.trace == b.trace
    assert a.hits == b.hits
    assert info_a == info_b
    c, _ = run_once(adapter, params, seed=6)
    assert c.trace != a.trace


@pytest.mark.parametrize(("adapter", "params"), CASES, ids=IDS)
def test_adapter_is_deterministic_on_a_noisy_problem(adapter, params):
    a, _ = run_once(adapter, params, problem="noisy_sphere")
    b, _ = run_once(adapter, params, problem="noisy_sphere")
    assert a.trace == b.trace


@pytest.mark.parametrize(("adapter", "params"), CASES, ids=IDS)
def test_adapter_improves_on_the_sphere(adapter, params):
    objective, _ = run_once(adapter, params, budget=2000)
    assert objective.trace[-1][1] < objective.trace[0][1]


@pytest.mark.parametrize("params", AUXEIN.values(), ids=AUXEIN)
def test_auxein_reports_generations_and_cost_per_generation(params):
    objective, info = run_once("auxein_static", params, budget=1000)
    # 100 initial evaluations, then per generation 4 children plus the re-scoring of all 100 individuals
    assert info.extra["initial_evals"] == 100
    assert info.evals_per_generation == 104
    assert info.generations == (1000 - 100) // 104
    assert info.stop_reason == "budget"


def test_auxein_with_a_budget_smaller_than_the_population_finishes_cleanly():
    objective, info = run_once("auxein_static", {}, budget=50)
    assert objective.evals == 50
    assert info.generations == 0
    assert info.evals_per_generation is None


def test_auxein_stops_on_the_playground_condition_when_the_population_is_too_small():
    objective, info = run_once("auxein_static", {"population_size": 4}, budget=1000)
    assert info.stop_reason == "population_size"
    assert info.generations == 0
    assert objective.evals == 4


def test_cma_reports_its_population_size_and_generations():
    objective, info = run_once("cmaes", {}, budget=700, dim=10)
    assert info.extra["popsize"] == 10
    assert info.generations == 70


def test_adapters_can_be_loaded_by_module_path():
    assert load_adapter("benchmarks.adapters.random_search") is load_adapter("random_search")
