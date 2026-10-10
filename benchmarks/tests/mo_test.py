"""The multi-objective part of the harness: problems and their fronts, the counting objective, adapters, configs, runner, report."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pymoo.indicators.hv import HV
from pymoo.problems import get_problem

from benchmarks.__main__ import main
from benchmarks.adapters import load_adapter
from benchmarks.mo_config import KIND, MOConfig, config_kind, load_mo_config, parse_mo_config
from benchmarks.mo_objective import BudgetExhausted, MOCountingObjective
from benchmarks.mo_problems import MO_PROBLEMS, make_mo_problem
from benchmarks.mo_report import build_mo_report
from benchmarks.mo_runner import build_mo_tasks, run_mo_benchmark
from benchmarks.runner import read_jsonl

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
NAMES = ["zdt1", "zdt2", "zdt3", "dtlz2"]


# --- the problems ---


@pytest.mark.parametrize("name", NAMES)
def test_the_problems_agree_with_pymoos_implementations(name: str):
    problem = make_mo_problem(name)
    reference = get_problem(name, n_var=problem.dim, n_obj=problem.n_obj) if name == "dtlz2" else get_problem(name)
    x = np.random.default_rng(1).random((50, problem.dim))
    expected: np.ndarray = np.asarray(reference.evaluate(x))
    got = np.stack([problem.evaluate(row) for row in x])
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_the_known_front_is_on_the_problem_and_nothing_dominates_it(name: str):
    problem = make_mo_problem(name)
    front = problem.pareto_front()
    assert front.shape[1] == problem.n_obj and len(front) >= 500
    rng = np.random.default_rng(2)
    # a point whose distance variables are at their optimum lies on the front, and random points never dominate a front point
    values = np.stack([problem.evaluate(rng.random(problem.dim)) for _ in range(3000)])
    for point in front[:: max(1, len(front) // 40)]:
        assert not ((values <= point - 1e-9).all(axis=1)).any()
    if name.startswith("zdt"):
        x = np.zeros(problem.dim)
        for f1 in (0.0, 0.05, 0.2, 0.43, 0.64, 0.84):
            x[0] = f1
            on_front = problem.evaluate(x)
            assert np.min(np.linalg.norm(front - on_front, axis=1)) < 0.01 or name == "zdt3"
    else:
        point = problem.evaluate(np.array([0.3, 0.6, *([0.5] * 10)]))
        assert abs(float(np.sum(point**2)) - 1.0) < 1e-12  # on the unit sphere


def test_the_reference_points_are_one_tenth_beyond_the_nadir_of_the_true_front():
    np.testing.assert_allclose(make_mo_problem("zdt1").reference_point, [1.1, 1.1])
    np.testing.assert_allclose(make_mo_problem("zdt2").reference_point, [1.1, 1.1])
    np.testing.assert_allclose(make_mo_problem("zdt3").reference_point, [1.1 * 0.8518328654, 1.1], rtol=1e-6)
    np.testing.assert_allclose(make_mo_problem("dtlz2").reference_point, [1.1, 1.1, 1.1])
    assert sorted(MO_PROBLEMS) == sorted(NAMES)
    with pytest.raises(ValueError, match="unknown multi-objective problem"):
        make_mo_problem("zdt9")


# --- the counting objective ---


def test_the_budget_is_enforced_and_the_archive_is_the_non_dominated_set_of_everything_evaluated():
    problem = make_mo_problem("zdt1")
    objective = MOCountingObjective(problem, 300)
    rng = np.random.default_rng(3)
    seen = []
    for _ in range(300):
        seen.append(objective(rng.random(problem.dim)))
    with pytest.raises(BudgetExhausted):
        objective(rng.random(problem.dim))
    assert objective.evals == 300 and objective.remaining == 0
    points = np.array(seen)
    expected = [
        p
        for i, p in enumerate(points)
        if not any((q <= p).all() and (q < p).any() for q in points) and not any((points[j] == p).all() for j in range(i))
    ]
    assert sorted(map(tuple, objective.front.tolist())) == sorted(map(tuple, np.array(expected).tolist()))
    objective.front[0, 0] = -1.0  # the front is a copy
    assert objective.front[0, 0] != -1.0


def test_the_traces_are_monotone_end_at_the_last_evaluation_and_match_the_indicators():
    problem = make_mo_problem("zdt3")
    objective = MOCountingObjective(problem, 700)
    rng = np.random.default_rng(4)
    for _ in range(555):
        objective(rng.random(problem.dim))
    trace = objective.trace
    assert trace[-1][0] == 555 and [t[0] for t in trace] == sorted({t[0] for t in trace})
    hypervolumes = [hv for _, hv, _ in trace]
    assert hypervolumes == sorted(hypervolumes)  # the archive only improves
    igd = [value for _, _, value in trace]
    assert igd == sorted(igd, reverse=True)
    direct = HV(ref_point=problem.reference_point)(objective.front)
    assert direct is not None and objective.final_hypervolume == pytest.approx(float(direct))


# --- the adapters ---


@pytest.mark.parametrize("adapter", ["mo_random_search", "auxein_nsga2", "pymoo_nsga2"])
def test_the_adapters_spend_exactly_the_budget_and_are_deterministic(adapter: str):
    def go(seed: int) -> MOCountingObjective:
        problem = make_mo_problem("zdt2")
        objective = MOCountingObjective(problem, 777)
        info = load_adapter(adapter)(objective, problem.dim, seed, {})  # type: ignore[arg-type]
        assert objective.evals == 777 and info.stop_reason == "budget"
        return objective

    first, second, other = go(1), go(1), go(2)
    np.testing.assert_array_equal(first.front, second.front)
    assert first.trace == second.trace
    assert not np.array_equal(first.front, other.front) or len(first.front) != len(other.front)


def test_nsga2_and_its_reference_beat_random_search_even_at_a_small_budget():
    results: dict[str, float] = {}
    for adapter in ("mo_random_search", "auxein_nsga2", "pymoo_nsga2"):
        problem = make_mo_problem("dtlz2")
        objective = MOCountingObjective(problem, 3000)
        load_adapter(adapter)(objective, problem.dim, 0, {})  # type: ignore[arg-type]
        results[adapter] = objective.final_hypervolume
    assert results["auxein_nsga2"] > 1.5 * results["mo_random_search"] and results["pymoo_nsga2"] > 1.5 * results["mo_random_search"]


def test_the_auxein_adapter_runs_on_torch_too():
    pytest.importorskip("torch")
    problem = make_mo_problem("dtlz2")
    objective = MOCountingObjective(problem, 1000)
    load_adapter("auxein_nsga2")(objective, problem.dim, 0, {"backend": "torch", "precision": "float32"})  # type: ignore[arg-type]
    assert objective.evals == 1000 and objective.final_hypervolume > 0


# --- configs ---


def test_the_shipped_multi_objective_configs_are_valid():
    full, quick = load_mo_config(CONFIGS / "mo-full.toml"), load_mo_config(CONFIGS / "mo-quick.toml")
    assert full.runs == 25 and [(p.name, p.budget) for p in full.problems] == [
        ("zdt1", 25000),
        ("zdt2", 25000),
        ("zdt3", 25000),
        ("dtlz2", 30000),
    ]
    assert [a.name for a in full.algorithms] == ["auxein-nsga2", "pymoo-nsga2", "random-search"]
    assert full.raw["report"] == {"references": ["auxein-nsga2"]}
    assert quick.runs == 2 and all(p.budget == 2000 for p in quick.problems)
    # identical parameters for the two NSGA-IIs: that is what makes the comparison fair
    assert full.algorithm("auxein-nsga2").params == full.algorithm("pymoo-nsga2").params
    assert len(build_mo_tasks(full)) == 25 * 4 * 3
    assert config_kind(CONFIGS / "mo-full.toml") == KIND and config_kind(CONFIGS / "full.toml") == "single-objective"


def test_config_validation():
    base: dict[str, Any] = {
        "kind": KIND,
        "name": "t",
        "runs": 1,
        "problems": [{"name": "zdt1", "budget": 100}],
        "algorithms": [{"name": "a", "adapter": "mo_random_search"}],
    }
    assert parse_mo_config(base).problems[0].budget == 100
    with pytest.raises(ValueError, match="not a multi-objective config"):
        parse_mo_config({**base, "kind": "single-objective"})
    with pytest.raises(ValueError, match="unknown multi-objective problems"):
        parse_mo_config({**base, "problems": [{"name": "zdt9", "budget": 1}]})
    with pytest.raises(ValueError, match="unique"):
        parse_mo_config({**base, "algorithms": [base["algorithms"][0], base["algorithms"][0]]})


# --- the runner and the report ---


def tiny_config(runs: int = 2) -> MOConfig:
    return parse_mo_config(
        {
            "kind": KIND,
            "name": "tiny",
            "base_seed": 5,
            "runs": runs,
            "report": {"references": ["auxein-nsga2"]},
            "problems": [{"name": "zdt1", "budget": 500}, {"name": "dtlz2", "budget": 500}],
            "algorithms": [
                {"name": "auxein-nsga2", "adapter": "auxein_nsga2", "params": {"population_size": 20, "offspring_size": 20}},
                {"name": "pymoo-nsga2", "adapter": "pymoo_nsga2", "params": {"population_size": 20, "offspring_size": 20}},
                {"name": "random-search", "adapter": "mo_random_search"},
            ],
        }
    )


@pytest.fixture(scope="module")
def tiny_results(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return run_mo_benchmark(tiny_config(), tmp_path_factory.mktemp("mo"), workers=2, progress=lambda message: None)


def test_a_run_writes_the_records_and_the_metadata(tiny_results: Path):
    runs = read_jsonl(tiny_results / "runs.jsonl")
    assert len(runs) == 2 * 3 * 2 and [r["task"] for r in runs] == list(range(12))
    record = runs[0]
    assert {"algorithm", "problem", "seed", "budget", "evals", "final_hv", "final_igd_plus", "trace_hv", "trace_igd_plus", "front"} <= set(
        record
    )
    assert record["evals"] == 500 and record["trace_hv"][-1][0] == 500 and len(record["front"]) >= 1
    metadata = json.loads((tiny_results / "metadata.json").read_text())
    assert metadata["config"]["kind"] == KIND and metadata["versions"]["pymoo"]


def test_the_same_config_and_seeds_reproduce_identical_results(tiny_results: Path, tmp_path: Path):
    again = run_mo_benchmark(tiny_config(), tmp_path, workers=1, progress=lambda message: None)

    def stable(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in r.items() if k != "wall_time"} for r in records]

    assert stable(read_jsonl(again / "runs.jsonl")) == stable(read_jsonl(tiny_results / "runs.jsonl"))


def test_the_report_has_every_section_and_its_plots(tiny_results: Path):
    report = build_mo_report(tiny_results)
    text = report.read_text()
    for heading in (
        "## 1. Hypervolume against evaluations",
        "## 2. Final fronts",
        "## 3. Summary",
        "## 4. Statistical comparison",
        "## 5. Acceptance",
        "## 6. Methodology",
    ):
        assert heading in text
    for problem in ("zdt1", "dtlz2"):
        assert (tiny_results / f"hypervolume-{problem}.png").stat().st_size > 1000 and (
            tiny_results / f"fronts-{problem}.png"
        ).stat().st_size > 1000
    assert "not assessed" in text and "too few for a significance test" in text  # two runs cannot say anything


def test_the_report_decides_the_acceptance_criteria_from_enough_runs(tmp_path: Path):
    """With 12 runs the test has power: a clearly better algorithm is reported as better, and the criteria are judged."""
    results = run_mo_benchmark(tiny_config(runs=12), tmp_path, workers=4, progress=lambda message: None)
    text = build_mo_report(results).read_text()
    acceptance = text.split("## 5. Acceptance", 1)[1].split("## 6. Methodology", 1)[0]
    assert "PASS: dtlz2: `auxein-nsga2` is significantly better than random search" in acceptance
    assert "not assessed" not in acceptance and acceptance.count("- PASS") + acceptance.count("- FAIL") == 4  # every criterion is judged
    # at 500 evaluations nothing of ZDT1 is inside the reference point yet, so there the honest reading is a FAIL
    assert "FAIL: zdt1" in acceptance


def test_the_command_line_dispatches_on_the_kind_of_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config = tmp_path / "cli.toml"
    config.write_text(
        'kind = "multi-objective"\nname = "cli"\nruns = 1\n[[problems]]\nname = "zdt1"\nbudget = 300\n'
        '[[algorithms]]\nname = "random-search"\nadapter = "mo_random_search"\n'
    )
    main(["run", "--config", str(config), "--workers", "1", "--results-root", str(tmp_path / "out")])
    results_dir = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert (results_dir / "runs.jsonl").exists()
    main(["report", str(results_dir)])
    assert Path(capsys.readouterr().out.strip()).name == "report.md"
