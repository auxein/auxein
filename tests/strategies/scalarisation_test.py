"""Scalarisation: single-objective strategies on multi-objective problems, with every objective still recorded."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import auxein
from auxein.backend import Backend
from auxein.core import EvaluationBatch, IdIssuer, Objective, ProblemSpec, Status, StrategyContext
from auxein.driver import ConfigurationMismatchError
from auxein.random import RunSeed
from auxein.recording import open_run
from auxein.spaces import Box
from auxein.strategies import Chebyshev, Scalarised, WeightedSum, best_by_scalarisation
from tests.support import multiobjective as mo

pytestmark = pytest.mark.filterwarnings("ignore::auxein.RecordingDisabledWarning")

MIN_A, MAX_B = Objective("a"), Objective("b", "maximise")
BOTH = (MIN_A, MAX_B)


def apply(scalarisation: Any, rows: list[list[float]], backend: Backend, objectives: tuple[Objective, ...] = BOTH) -> list[float]:
    """The scalarised values of objective vectors given in natural units: converted to minimisation form, as a strategy gets them."""
    minimised = backend.asarray([[o.sign * v for o, v in zip(objectives, row, strict=True)] for row in rows])
    return backend.to_numpy(scalarisation.apply(minimised, objectives, backend)).astype(np.float64).tolist()


# --- the values, against hand-computed cases ---


def test_the_weighted_sum_of_minimisation_forms_with_mixed_directions(backend: Backend):
    # a = 2 (minimised) and b = 3 (maximised, so -3): 1·2 + 2·(-3) = -4; a = 0, b = 1: 0 + 2·(-1) = -2
    got = apply(WeightedSum({"a": 1.0, "b": 2.0}), [[2.0, 3.0], [0.0, 1.0], [5.0, 0.0]], backend)
    np.testing.assert_allclose(got, [-4.0, -2.0, 5.0], rtol=1e-6)


def test_the_chebyshev_scalarisation_with_and_without_a_reference_point(backend: Backend):
    # weights 1 and 1, reference (a: 1, b: 5) in natural units, so (1, -5) in minimisation form:
    #   (2, 3):  max(2 - 1, -3 + 5) = 2     (0, 1): max(0 - 1, -1 + 5) = 4     (1, 5): max(0, 0) = 0
    got = apply(Chebyshev({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 5.0}), [[2.0, 3.0], [0.0, 1.0], [1.0, 5.0]], backend)
    np.testing.assert_allclose(got, [2.0, 4.0, 0.0], atol=1e-6)
    # the default reference is zero: weights 2 and 0.5: (2, 3) -> max(2·2, 0.5·(-3)) = 4
    np.testing.assert_allclose(apply(Chebyshev({"a": 2.0, "b": 0.5}), [[2.0, 3.0], [0.0, 4.0]], backend), [4.0, 0.0], atol=1e-6)
    # a weight of 0 ignores an objective
    np.testing.assert_allclose(apply(Chebyshev({"a": 1.0, "b": 0.0}), [[2.0, 3.0]], backend), [2.0], atol=1e-6)


def test_a_reference_in_natural_units_means_the_same_for_a_maximised_objective(backend: Backend):
    """b is maximised: a reference of 5 is 'reach 5', and a point with b = 5 is exactly on it whatever the sign convention."""
    on_it = apply(Chebyshev({"a": 1.0, "b": 3.0}, {"a": 0.0, "b": 5.0}), [[0.0, 5.0]], backend)
    worse = apply(Chebyshev({"a": 1.0, "b": 3.0}, {"a": 0.0, "b": 5.0}), [[0.0, 4.0]], backend)
    assert on_it == [0.0] and worse[0] == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({}, "needs a weight"),
        ({"a": -1.0, "b": 1.0}, "not negative"),
        ({"a": float("nan"), "b": 1.0}, "finite"),
        ({"a": float("inf"), "b": 1.0}, "finite"),
        ({"a": 0.0, "b": 0.0}, "all zero"),
        ({"a": "1", "b": 1.0}, "finite number"),
    ],
)
def test_weight_validation(weights: Any, message: str):
    with pytest.raises(ValueError, match=message):
        WeightedSum(weights)
    with pytest.raises(ValueError, match=message):
        Chebyshev(weights)


def test_weights_and_references_are_checked_against_the_problems_objectives():
    with pytest.raises(ValueError, match=r"unknown \['c'\]"):
        WeightedSum({"a": 1.0, "b": 1.0, "c": 1.0}).validate(BOTH)
    with pytest.raises(ValueError, match=r"missing \['b'\].*give 0"):
        WeightedSum({"a": 1.0}).validate(BOTH)
    with pytest.raises(ValueError, match="reference values"):
        Chebyshev({"a": 1.0, "b": 1.0}, {"a": 1.0}).validate(BOTH)
    with pytest.raises(ValueError, match="finite"):
        Chebyshev({"a": 1.0, "b": 1.0}, {"a": float("nan"), "b": 0.0})
    WeightedSum({"a": 1.0, "b": 0.0}).validate(BOTH)


def test_the_description_names_the_weights_and_the_reference():
    assert repr(WeightedSum({"a": 1.0, "b": 2.0})) == "WeightedSum(weights={'a': 1.0, 'b': 2.0})"
    assert "reference={'a': 1.0}" in repr(Chebyshev({"a": 1.0}, {"a": 1}))
    assert repr(WeightedSum({"a": 1})) == repr(WeightedSum({"a": 1.0}))


# --- the wrapper ---


class Spy:
    """A single-objective strategy that records what it is bound to and told, and proposes random candidates."""

    capabilities = auxein.strategies.RandomSearch.capabilities

    def __init__(self) -> None:
        self.inner = auxein.RandomSearch()
        self.problem: ProblemSpec[Any] | None = None
        self.told: list[Any] = []

    def __repr__(self) -> str:
        return "Spy()"

    def bind(self, problem: Any, ctx: Any) -> None:
        self.problem = problem
        self.inner.bind(problem, ctx)

    def ask(self, n: int) -> Any:
        return self.inner.ask(n)

    def tell(self, results: Any) -> None:
        self.told.extend(results)
        self.inner.tell(results)

    def state_dict(self) -> Any:
        return self.inner.state_dict()

    def load_state_dict(self, state: Any) -> None:
        self.inner.load_state_dict(state)


def biobjective(backend: Backend) -> Any:
    def evaluate(genome: Any) -> auxein.Result:
        x = float(backend.to_numpy(genome)[0])
        return auxein.Result({"a": x * x, "b": 1.0 - (x - 1.0) ** 2}, {"c": max(0.0, 0.1 - x)})

    return evaluate


def run_scalarised(strategy: Any, backend: Backend, run_dir: Path | None = None, evaluations: int = 400, **over: Any) -> Any:
    options: dict[str, Any] = {
        "evaluator": auxein.FunctionEvaluator(biobjective(backend)),
        "space": Box(0.0, 1.0, dim=1),
        "objectives": list(BOTH),
        "constraints": ["c"],
        "budget": auxein.Budget(evaluations=evaluations),
        "seed": 3,
        "batch_size": 20,
        "backend": backend,
        "run_dir": run_dir,
    }
    options.update(over)
    return auxein.run(strategy=strategy, **options)


def test_the_inner_strategy_sees_one_objective_and_the_run_keeps_them_all(backend: Backend, tmp_path: Path):
    spy = Spy()
    scalarisation = WeightedSum({"a": 1.0, "b": 2.0})
    result = run_scalarised(Scalarised(spy, scalarisation), backend, tmp_path / "r")
    assert spy.problem is not None and [(o.name, o.direction) for o in spy.problem.objectives] == [("scalarised", "minimise")]
    assert spy.problem.constraints == ("c",)  # constraints reach the inner strategy
    assert len(spy.told) == 400 and all(set(e.objectives) == {"scalarised"} for e in spy.told)
    told = {int(e.candidate.id): e.objectives["scalarised"] for e in spy.told}
    with open_run(tmp_path / "r") as recorded:
        rows = list(recorded.evaluations())
    assert len(rows) == 400 and all(set(r.objectives) == {"a", "b"} for r in rows)  # the recording keeps the original objectives
    for row in rows:
        assert told[int(row.candidate_id)] == pytest.approx(row.objectives["a"] + 2.0 * -row.objectives["b"], rel=1e-5, abs=1e-6)
    front = [(e.objectives["a"], e.objectives["b"]) for e in result.pareto_front]
    assert len(front) > 5 and result.best is None
    assert all(set(e.objectives) == {"a", "b"} for e in result.pareto_front)  # and so does the Pareto archive


@pytest.mark.filterwarnings("ignore::auxein.driver.errors.EvaluationFailureWarning")
def test_failed_evaluations_reach_the_inner_strategy_as_failures(backend: Backend):
    spy = Spy()

    def sometimes(genome: Any) -> auxein.Result:
        x = float(backend.to_numpy(genome)[0])
        if x > 0.8:
            raise RuntimeError("boom")
        return biobjective(backend)(genome)

    strategy = Scalarised(spy, WeightedSum({"a": 1.0, "b": 1.0}))
    run_scalarised(strategy, backend, evaluator=auxein.FunctionEvaluator(sometimes), initial_failure_guard=None)
    failed = [e for e in spy.told if e.status is not Status.OK]
    assert failed and all(e.objectives == {} and e.error for e in failed)
    assert any(e.status is Status.OK for e in spy.told)


def test_the_wrapper_is_a_strategy_with_the_inner_strategys_state_and_description(backend: Backend):
    inner = auxein.GeneticAlgorithm(population_size=10, offspring_size=5)
    wrapped = Scalarised(inner, Chebyshev({"a": 1.0, "b": 1.0}, {"a": 0.0, "b": 1.0}))
    assert wrapped.capabilities.max_objectives is None and wrapped.capabilities.tell_mode == "both"
    assert wrapped.capabilities.supports_constraints
    assert repr(wrapped).startswith("Scalarised(GeneticAlgorithm(") and "Chebyshev(" in repr(wrapped)
    assert wrapped.strategy is inner
    issuer = IdIssuer()
    ctx = StrategyContext(RunSeed(1).stream("strategy", backend=backend), backend, issuer.next)
    with pytest.raises(RuntimeError, match="must be bound"):
        wrapped.tell(EvaluationBatch([]))
    wrapped.bind(ProblemSpec(Box(0.0, 1.0, dim=1), BOTH), ctx)
    wrapped.ask(1)
    assert set(wrapped.state_dict()) == set(inner.state_dict())
    with pytest.raises(TypeError, match="needs a strategy"):
        Scalarised(object(), WeightedSum({"a": 1.0}))
    mismatch = Scalarised(auxein.GeneticAlgorithm(), WeightedSum({"a": 1.0, "zzz": 1.0}))
    with pytest.raises(ValueError, match="do not match"):
        mismatch.bind(ProblemSpec(Box(0.0, 1.0, dim=1), BOTH), ctx)


def ga_under(scalarisation: Any) -> Scalarised[Any]:
    return Scalarised(auxein.GeneticAlgorithm(population_size=10, offspring_size=10), scalarisation)


def test_a_changed_weight_is_refused_on_resume_and_the_same_weights_resume(backend: Backend, tmp_path: Path):
    options: dict[str, Any] = {
        "evaluator": auxein.FunctionEvaluator(biobjective(backend)),
        "space": Box(0.0, 1.0, dim=1),
        "objectives": list(BOTH),
        "constraints": ["c"],
        "seed": 3,
        "batch_size": 20,
        "backend": backend,
        "run_dir": tmp_path / "r",
    }
    auxein.run(strategy=ga_under(WeightedSum({"a": 1.0, "b": 2.0})), budget=auxein.Budget(evaluations=60), **options)
    with pytest.raises(ConfigurationMismatchError, match="strategy"):
        auxein.resume(strategy=ga_under(WeightedSum({"a": 1.0, "b": 3.0})), budget=auxein.Budget(evaluations=120), **options)
    with pytest.raises(ConfigurationMismatchError, match="strategy"):
        auxein.resume(strategy=ga_under(Chebyshev({"a": 1.0, "b": 2.0})), budget=auxein.Budget(evaluations=120), **options)
    resumed = auxein.resume(strategy=ga_under(WeightedSum({"a": 1.0, "b": 2.0})), budget=auxein.Budget(evaluations=120), **options)
    assert resumed.evaluations_used == 120


# --- a genetic algorithm under scalarisation reaches the point its weights select ---


@pytest.mark.parametrize(
    ("scalarisation", "optimum"),
    [
        (WeightedSum({"a": 1.0, "b": 3.0}), 0.75),  # minimises x^2 + 3 (x - 1)^2, whose minimum is at x = 3/4
        (WeightedSum({"a": 3.0, "b": 1.0}), 0.25),
        (Chebyshev({"a": 1.0, "b": 1.0}, {"a": 0.0, "b": 1.0}), 0.5),  # x^2 against (x - 1)^2 cross at one half
        (Chebyshev({"a": 1.0, "b": 3.0}, {"a": 0.0, "b": 1.0}), np.sqrt(3.0) / (1.0 + np.sqrt(3.0))),  # x^2 = 3 (x - 1)^2
    ],
    ids=["sum-0.75", "sum-0.25", "chebyshev-half", "chebyshev-sqrt3"],
)
def test_a_genetic_algorithm_reaches_the_point_its_weights_select(backend: Backend, scalarisation: Any, optimum: float):
    strategy = Scalarised(auxein.GeneticAlgorithm(population_size=20, offspring_size=20), scalarisation)
    result = run_scalarised(strategy, backend, evaluations=2000)
    best = best_by_scalarisation(result, scalarisation, objectives=BOTH)
    assert best.constraints["c"] == 0.0
    assert float(backend.to_numpy(best.genome)[0]) == pytest.approx(optimum, abs=2e-3)  # type: ignore[arg-type]
    assert best.objectives["a"] == pytest.approx(optimum**2, abs=5e-3)


def test_the_best_by_scalarisation_of_a_run_directory_agrees_and_prefers_feasible_points(backend: Backend, tmp_path: Path):
    scalarisation = WeightedSum({"a": 1.0, "b": 3.0})
    strategy = Scalarised(auxein.GeneticAlgorithm(population_size=20, offspring_size=20), scalarisation)
    result = run_scalarised(strategy, backend, tmp_path / "r", 1200)
    from_result = best_by_scalarisation(result, scalarisation, objectives=BOTH)
    from_disk = best_by_scalarisation(tmp_path / "r", scalarisation)
    assert from_disk.candidate_id == from_result.candidate_id and from_disk.value == pytest.approx(from_result.value, rel=1e-6, abs=1e-9)
    with open_run(tmp_path / "r") as recorded:
        rows = [r for r in recorded.evaluations() if r.status is Status.OK and r.constraints["c"] == 0]
    manual = min(rows, key=lambda r: (r.objectives["a"] - 3.0 * r.objectives["b"], r.candidate_id))
    assert from_disk.candidate_id == manual.candidate_id
    with pytest.raises(ValueError, match="objectives="):
        best_by_scalarisation(result, scalarisation)
    with pytest.raises(TypeError, match="RunResult or a run directory"):
        best_by_scalarisation(42, scalarisation)


def test_scalarising_nsga2_is_pointless_but_a_single_objective_strategy_of_any_kind_works(backend: Backend):
    result = run_scalarised(Scalarised(auxein.RandomSearch(), WeightedSum({"a": 1.0, "b": 0.0})), backend, evaluations=200)
    assert result.evaluations_used == 200 and mo.F1.name == "f1"
