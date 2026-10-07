import pytest

from benchmarks.adapters import load_adapter
from benchmarks.objective import CountingObjective
from benchmarks.problems import make_problem

AUXEIN = {
    "auxein-default": {},
    "auxein-fixedvar": {"mutation": {"type": "fixed_variance", "sigma": 0.1}},
    "auxein-windowing": {"distribution": "fps_windowing"},
}
CASES = [("auxein_static", params) for params in AUXEIN.values()] + [
    ("random_search", {}),
    ("cmaes", {}),
    ("auxein_core_random", {}),
    ("auxein_core_ga", {}),
]
IDS = [*AUXEIN, "random-search", "cma-es", "auxein-core-random", "auxein-core-ga"]


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


def test_the_new_core_adapter_reports_its_batches():
    objective, info = run_once("auxein_core_random", {"batch_size": 10}, budget=95)
    assert objective.evals == 95 and info.stop_reason == "budget"
    assert info.generations == 10 and info.evals_per_generation == 10.0  # 9 full batches and one of 5


def test_the_new_core_adapter_does_not_depend_on_the_batch_size_for_the_budget():
    for batch_size in (1, 7, 64, 500):
        objective, _ = run_once("auxein_core_random", {"batch_size": batch_size}, budget=130)
        assert objective.evals == 130


GA_VARIANTS = [
    {"population_size": 20, "offspring_size": 20, "selection": {"type": "sus", "scaling": 2.0}},
    {"mutation": {"type": "self_adaptive", "per_gene": True}, "repair": "reflect"},
    {
        "selection": "tournament",
        "recombination": {"type": "uniform"},
        "mutation": {"type": "gaussian", "step": 0.05},
        "crossover_probability": 0.8,
    },
    {"population_size": 10, "offspring_size": 7, "recombination": "none"},
]


@pytest.mark.parametrize("params", GA_VARIANTS, ids=range(len(GA_VARIANTS)))
def test_the_new_core_ga_adapter_accepts_operator_parameters(params):
    for budget in (30, 400):
        objective, info = run_once("auxein_core_ga", params, budget=budget)
        assert objective.evals == budget and info.stop_reason == "budget"
    again, _ = run_once("auxein_core_ga", params, budget=400)
    assert again.trace == objective.trace


def test_the_new_core_ga_adapter_reports_what_the_population_costs():
    objective, info = run_once("auxein_core_ga", {"population_size": 20, "offspring_size": 10}, budget=420, dim=4)
    assert objective.evals == 420 and info.extra["initial_evals"] == 20 and info.extra["population_size"] == 20
    assert info.evals_per_generation == 10.0 and info.generations == 40  # (420 - 20) / 10: nothing is re-scored


def test_the_new_core_ga_adapter_rejects_unknown_operators():
    with pytest.raises(ValueError, match="unknown selection 'roulette'"):
        run_once("auxein_core_ga", {"selection": "roulette"}, budget=10)
    with pytest.raises(ValueError, match="needs a 'type'"):
        run_once("auxein_core_ga", {"mutation": {"step": 0.1}}, budget=10)


def test_the_new_core_ga_beats_random_search_on_the_sphere():
    ga, _ = run_once("auxein_core_ga", {}, budget=4000, dim=5)
    rs, _ = run_once("random_search", {}, budget=4000, dim=5)
    assert ga.best_error < rs.best_error / 100
