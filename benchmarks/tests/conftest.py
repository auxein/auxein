import pytest

from benchmarks.config import Config, parse_config

TINY = {
    "name": "tiny",
    "base_seed": 3,
    "runs": 4,
    "budget_per_dim": 150,
    "dims": [2, 3],
    "problems": ["sphere", "rastrigin", "noisy_sphere"],
    "targets": [1e-1, 1e-3, 1e-6],
    "algorithms": [
        {
            "name": "auxein-default",
            "adapter": "auxein_static",
            "params": {"population_size": 20, "mutation": {"type": "self_adaptive", "tau": 0.1}, "distribution": "sigma_scaling"},
        },
        {"name": "random-search", "adapter": "random_search"},
        {"name": "cma-es", "adapter": "cmaes", "params": {"sigma0": 2.0}},
    ],
    "overhead": {"budget": 400, "repeats": 2, "dims": [2, 3], "population_sizes": [10, 30]},
}


@pytest.fixture
def tiny_config() -> Config:
    return parse_config(TINY)


@pytest.fixture(scope="session")
def tiny_results(tmp_path_factory: pytest.TempPathFactory):
    """A small but complete results directory: runs, overhead and metadata."""
    from benchmarks.runner import run_benchmark

    return run_benchmark(parse_config(TINY), tmp_path_factory.mktemp("results"), workers=2, progress=lambda message: None)
