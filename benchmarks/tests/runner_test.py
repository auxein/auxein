import json
from pathlib import Path

import pytest

from benchmarks.config import load_config, parse_config
from benchmarks.runner import build_tasks, overhead_cases, read_jsonl, run_benchmark

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def stable(records):
    """Records without the wall-clock measurements, which are the only part that varies between runs."""
    return [{k: v for k, v in r.items() if k != "wall_time"} for r in records]


def test_tasks_pair_runs_with_instances_and_seeds(tiny_config):
    tasks = build_tasks(tiny_config)
    assert len(tasks) == 3 * 2 * 3 * 4
    assert [t.index for t in tasks] == list(range(len(tasks)))
    for task in tasks:
        assert task.instance < 4
        assert task.seed == 3 + task.instance
        assert task.budget == 150 * task.dim
    # every algorithm sees the same instances and seeds
    by_algorithm = {}
    for task in tasks:
        by_algorithm.setdefault(task.algorithm, []).append((task.problem, task.dim, task.instance, task.seed))
    assert by_algorithm["auxein-default"] == by_algorithm["random-search"] == by_algorithm["cma-es"]


def test_instance_offset_moves_the_instances_but_not_the_seeds(tiny_config):
    from dataclasses import replace

    shifted = replace(tiny_config, instance_offset=500)
    tasks, moved = build_tasks(tiny_config), build_tasks(shifted)
    assert [t.instance for t in moved] == [500 + t.instance for t in tasks]
    assert [t.seed for t in moved] == [t.seed for t in tasks]


def test_the_selection_config_uses_other_instances_than_the_full_one():
    selection, full = load_config(CONFIGS / "ga-default-selection.toml"), load_config(CONFIGS / "full.toml")
    assert (
        selection.instance_offset >= full.runs and selection.base_seed != full.base_seed
    )  # no run of the two shares an instance or a seed
    assert [a.name for a in selection.algorithms] == ["ga-a", "ga-b", "ga-c"] and selection.runs == 15 and selection.dims == (10, 30)
    assert selection.overhead is None and set(selection.problems) == set(full.problems)


def test_results_directory_layout(tiny_results):
    assert {"metadata.json", "overhead.jsonl", "runs.jsonl"} <= {p.name for p in tiny_results.iterdir()}
    assert "-" in tiny_results.name and tiny_results.name[0].isdigit()  # <timestamp>-<short-sha>


def test_run_records(tiny_results, tiny_config):
    records = read_jsonl(tiny_results / "runs.jsonl")
    assert [r["task"] for r in records] == list(range(len(records)))
    assert len(records) == 72
    for r in records:
        assert {
            "algorithm",
            "params",
            "problem",
            "dim",
            "instance",
            "seed",
            "budget",
            "evals",
            "wall_time",
            "trace",
            "hits",
            "info",
        } <= set(r)
        assert r["evals"] <= r["budget"]
        assert r["trace"][-1][0] == r["evals"]
        assert r["trace"][-1][1] == r["final_error"]
        assert set(r["hits"]) == {"0.1", "0.001", "1e-06"}
        assert {"generations", "stop_reason", "evals_per_generation", "extra"} <= set(r["info"])


def test_metadata(tiny_results, tiny_config):
    metadata = json.loads((tiny_results / "metadata.json").read_text())
    assert {"git_sha", "git_dirty", "versions", "cpu", "config", "timestamp", "workers"} <= set(metadata)
    assert {"python", "auxein", "numpy", "cma"} <= set(metadata["versions"])
    assert metadata["versions"]["auxein"]
    assert metadata["config"]["name"] == "tiny"
    assert metadata["workers"] == 2


def test_overhead_records(tiny_results, tiny_config):
    records = read_jsonl(tiny_results / "overhead.jsonl")
    cases = list(overhead_cases(tiny_config))
    assert len(records) == len(cases) * 2  # two repeats per case
    assert {r["population_size"] for r in records if r["algorithm"] == "auxein-default"} == {10, 30}
    assert {r["population_size"] for r in records if r["algorithm"] != "auxein-default"} == {None}
    for r in records:
        assert r["us_per_eval"] > 0
    auxein = [r for r in records if r["algorithm"] == "auxein-default"]
    assert all(r["evals_per_generation"] == r["population_size"] + 4 for r in auxein)


def test_same_config_and_seeds_reproduce_identical_results_whatever_the_workers(tiny_results, tiny_config, tmp_path):
    again = run_benchmark(tiny_config, tmp_path, workers=1, skip_overhead=True, progress=lambda message: None)
    assert stable(read_jsonl(again / "runs.jsonl")) == stable(read_jsonl(tiny_results / "runs.jsonl"))


def test_the_shipped_configs_are_valid():
    full, quick = load_config(CONFIGS / "full.toml"), load_config(CONFIGS / "quick.toml")

    assert full.runs == 25 and full.dims == (2, 10, 30) and len(full.problems) == 5
    assert [a.name for a in full.algorithms] == [
        "auxein-default",
        "auxein-fixedvar",
        "auxein-windowing",
        "random-search",
        "cma-es",
        "auxein-core-random",
    ]
    assert full.budget(10) == 20000
    assert full.targets == (1e-1, 1e-3, 1e-6)
    assert full.overhead is not None and full.overhead.dims == (2, 10, 100) and full.overhead.population_sizes == (50, 200, 800)

    assert quick.runs == 3 and quick.dims == (2, 10) and len(quick.problems) == 5
    assert [a.name for a in quick.algorithms] == ["auxein-default", "random-search", "cma-es", "auxein-core-random"]
    assert quick.budget(10) == 5000


def test_duplicate_algorithm_names_are_rejected(tiny_config):
    raw = dict(tiny_config.raw)
    raw["algorithms"] = [raw["algorithms"][1], raw["algorithms"][1]]
    with pytest.raises(ValueError, match="unique"):
        parse_config(raw)
