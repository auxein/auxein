"""`PycmaStrategy`: pycma as an Auxein strategy, with Auxein's randomness and without checkpoints."""

import importlib
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import auxein
from auxein.backend import Backend
from auxein.core import Evaluation, EvaluationBatch, IdIssuer, Objective, ProblemSpec, Status, StrategyContext
from auxein.random import RunSeed
from auxein.recording import open_run
from auxein.spaces import Binary, Box, MixedSpace, Real
from auxein.strategies import PycmaStrategy, Scalarised, WeightedSum
from auxein.strategies.external.pycma import FAILED_GENERATION_VALUE, substitute_failures
from tests.driver.resume_test import comparable
from tests.support.fixtures import assert_on_backend, integration_backend

pytestmark = pytest.mark.filterwarnings("ignore::auxein.RecordingDisabledWarning")


VALUE = (Objective("value"),)


def bind(strategy: PycmaStrategy, backend: Backend, space: Any = None, seed: int = 1, objectives: Any = VALUE, constraints: Any = ()):
    issuer = IdIssuer()
    ctx = StrategyContext(RunSeed(seed).stream("strategy", backend=backend), backend, issuer.next)
    strategy.bind(ProblemSpec(space or Box(-5.0, 5.0, dim=4), tuple(objectives), tuple(constraints)), ctx)
    return strategy


def evaluations(batch: Any, values: list[float | None]) -> EvaluationBatch[Any]:
    out: list[Evaluation[Any]] = []
    for c, v in zip(batch.candidates, values, strict=True):
        out.append(
            Evaluation.failed(c, Status.FAILED, "boom") if v is None else Evaluation(c, Status.OK, {"value": v}, {}, {}, auxein.core.Cost())
        )
    return EvaluationBatch(out)


def sphere(backend: Backend) -> Any:
    return auxein.VectorisedEvaluator(lambda X: backend.xp.sum(X * X, axis=1))


# --- binding ---


def test_bind_refuses_what_cma_es_cannot_do(backend: Backend):
    with pytest.raises(TypeError, match="needs a Box.*GeneticAlgorithm for a MixedSpace"):
        bind(PycmaStrategy(), backend, space=MixedSpace({"x": Real(0.0, 1.0), "b": Binary()}))
    with pytest.raises(ValueError, match="log-scale"):
        bind(PycmaStrategy(), backend, space=Box(1e-3, 1.0, dim=3, log_scale=True))
    with pytest.raises(ValueError, match="does not support constraints"):
        bind(PycmaStrategy(), backend, constraints=("c",))
    with pytest.raises(ValueError, match=r"single-objective.*Scalarised"):
        bind(PycmaStrategy(), backend, objectives=(Objective("a"), Objective("b")))
    with pytest.raises(ValueError, match="x0 lies outside"):
        bind(PycmaStrategy(x0=9.0), backend)
    with pytest.raises(ValueError, match="x0 must be"):
        bind(PycmaStrategy(x0=[1.0, 2.0]), backend)


def test_construction_validation_and_the_owned_options():
    for name in ("seed", "randn", "bounds", "popsize", "CMA_stds", "verbose"):
        with pytest.raises(ValueError, match=f"{name!r} cannot be set"):
            PycmaStrategy(options={name: 1})
    with pytest.raises(ValueError, match="global generator"):
        PycmaStrategy(options={"seed": 3})
    with pytest.raises(ValueError, match="population_size"):
        PycmaStrategy(1)
    with pytest.raises(ValueError, match="sigma0"):
        PycmaStrategy(sigma0=0.0)
    with pytest.raises(ValueError, match="sigma_fraction"):
        PycmaStrategy(sigma_fraction=2.0)
    capabilities = PycmaStrategy().capabilities
    assert capabilities.max_objectives == 1 and not capabilities.supports_constraints
    assert capabilities.tell_mode == "generation" and not capabilities.supports_checkpoints


def test_the_description_includes_the_options_so_that_a_change_is_caught_on_resume():
    assert repr(PycmaStrategy()) == repr(PycmaStrategy())
    assert repr(PycmaStrategy(options={"tolfun": 1e-3})) != repr(PycmaStrategy())
    assert repr(PycmaStrategy(sigma0=2.0)) != repr(PycmaStrategy(sigma0=3.0))
    assert "tolfun" in repr(PycmaStrategy(options={"tolfun": 1e-3}))


def test_importing_without_pycma_names_the_extra(monkeypatch: pytest.MonkeyPatch):
    real = importlib.import_module

    def missing(name: str, package: str | None = None) -> Any:
        if name == "cma":
            raise ModuleNotFoundError("No module named 'cma'", name="cma")
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ImportError, match=r"pip install auxein\[cma\]"):
        PycmaStrategy()


def test_an_unbound_strategy_and_a_misused_contract_are_clear_errors(backend: Backend):
    with pytest.raises(RuntimeError, match="must be bound"):
        PycmaStrategy().ask(1)
    strategy = bind(PycmaStrategy(population_size=6), backend)
    with pytest.raises(ValueError, match="nothing was asked"):
        strategy.tell(EvaluationBatch([]))
    batch = strategy.ask(1)
    with pytest.raises(RuntimeError, match="before the previous generation was told"):
        strategy.ask(1)
    with pytest.raises(ValueError, match="whole generation"):
        strategy.tell(EvaluationBatch(evaluations(batch, [1.0] * 6).evaluations[:3]))
    with pytest.raises(ValueError, match="at least 1"):
        bind(PycmaStrategy(), backend).ask(0)
    with pytest.raises(NotImplementedError, match="cannot be checkpointed"):
        strategy.state_dict()
    with pytest.raises(NotImplementedError):
        strategy.load_state_dict({})


# --- candidates ---


def test_candidates_are_arrays_of_the_backend_and_always_in_the_box(backend: Backend):
    box = Box([-5.0, 0.0, 1e-3], [5.0, 1.0, 10.0])  # widths that differ by orders of magnitude
    strategy = bind(PycmaStrategy(sigma0=50.0), backend, space=box)  # a step far larger than the box: the bounds must hold
    for _ in range(25):
        batch = strategy.ask(1)
        array = batch.as_array()
        assert array is not None
        assert_on_backend(array, backend)
        assert all(box.contains(array[i]) for i in range(array.shape[0]))
        assert set(batch.origins) == {"pycma"}
        strategy.tell(evaluations(batch, [float(backend.to_numpy(array[i]).sum()) for i in range(array.shape[0])]))


def test_the_generation_size_is_pycmas_and_the_defaults_start_at_the_centre_with_a_third_of_the_width(backend: Backend):
    strategy = bind(PycmaStrategy(), backend, space=Box(0.0, 10.0, dim=9))
    batch = strategy.ask(1000)  # the driver's suggestion is ignored
    assert len(batch.candidates) == 4 + int(3 * np.log(9))
    host = backend.to_numpy(batch.as_array()).astype(np.float64)
    assert abs(host.mean() - 5.0) < 3.5 and host.std() < 6.0  # around the centre, in the box
    assert len(bind(PycmaStrategy(population_size=12), backend).ask(1).candidates) == 12


def test_non_isotropic_boxes_scale_the_steps_to_the_widths(backend: Backend):
    box = Box([0.0, 0.0], [1000.0, 1.0])
    strategy = bind(PycmaStrategy(), backend, space=box, seed=3)
    host = np.concatenate([backend.to_numpy(strategy.ask(1).as_array()).astype(np.float64)])
    assert host[:, 0].std() > 50 * host[:, 1].std()  # the first variable moves in hundreds, the second in tenths


# --- failures ---


def test_failed_candidates_are_told_the_worst_finite_value_plus_the_spread():
    told = substitute_failures(np.array([3.0, np.nan, 1.0, np.inf, 5.0]))
    np.testing.assert_array_equal(told, [3.0, 9.0, 1.0, 9.0, 5.0])  # worst 5 + spread 4
    equal = substitute_failures(np.array([2.0, np.nan, 2.0]))
    np.testing.assert_array_equal(equal, [2.0, 4.0, 2.0])  # no spread: max(1, |worst|) = 2
    tiny = substitute_failures(np.array([0.0, np.nan, 0.0]))
    np.testing.assert_array_equal(tiny, [0.0, 1.0, 0.0])
    assert (substitute_failures(np.array([np.nan, np.nan, np.inf])) == FAILED_GENERATION_VALUE).all()
    clean = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(substitute_failures(clean), clean)


def test_the_strategy_tells_pycma_the_substituted_values_in_the_order_it_asked(backend: Backend):
    strategy = bind(PycmaStrategy(population_size=6), backend)
    seen: list[tuple[Any, list[float]]] = []
    es = strategy._es  # noqa: SLF001
    original = es.tell
    es.tell = lambda solutions, values: (seen.append((solutions, list(values))), original(solutions, values))[1]
    batch = strategy.ask(1)
    shuffled = evaluations(batch, [10.0, None, 30.0, 20.0, None, 40.0])
    strategy.tell(EvaluationBatch(list(reversed(shuffled.evaluations))))  # told in another order: matched by id
    assert seen[0][1] == [10.0, 70.0, 30.0, 20.0, 70.0, 40.0]  # worst 40 + spread 30 for the two failures


def test_a_generation_in_which_everything_failed_does_not_stop_the_strategy(backend: Backend):
    strategy = bind(PycmaStrategy(population_size=6), backend)
    batch = strategy.ask(1)
    strategy.tell(evaluations(batch, [None] * 6))
    again = strategy.ask(1)  # it carries on, asking for another generation
    assert len(again.candidates) == 6


@pytest.mark.filterwarnings("ignore::auxein.driver.errors.EvaluationFailureWarning")
def test_a_run_with_failures_still_converges(backend: Backend):
    def fn(genome: Any) -> auxein.Result:
        x = backend.to_numpy(genome).astype(np.float64)
        if x[0] > 3.0:
            raise RuntimeError("diverged")
        return auxein.Result({"value": float((x * x).sum())})

    result = auxein.run(
        strategy=PycmaStrategy(),
        evaluator=auxein.FunctionEvaluator(fn),
        space=Box(-5.0, 5.0, dim=5),
        budget=auxein.Budget(evaluations=3000),
        seed=2,
        backend=backend,
        batch_size=20,
        initial_failure_guard=None,
    )
    assert result.status_counts.get("failed", 0) > 0 and result.best is not None and result.best.objectives["value"] < 1e-6


# --- randomness ---


def run_sphere(backend: Backend, seed: int, evaluations_: int = 1500) -> auxein.RunResult[Any]:
    return auxein.run(
        strategy=PycmaStrategy(),
        evaluator=sphere(backend),
        space=Box(-5.0, 5.0, dim=10),
        budget=auxein.Budget(evaluations=evaluations_),
        seed=seed,
        backend=backend,
        batch_size=50,
    )


def test_pycma_never_touches_global_random_state_and_a_seed_reproduces_the_run(backend: Backend):
    np.random.seed(2024)
    random.seed(2024)
    numpy_before, python_before = np.random.get_state(), random.getstate()
    first, second, other = run_sphere(backend, 1), run_sphere(backend, 1), run_sphere(backend, 2)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and (numpy_before[1] == numpy_after[1]).all() and numpy_before[2:] == numpy_after[2:]
    assert random.getstate() == python_before
    assert first.trace == second.trace and first.best is not None and second.best is not None
    np.testing.assert_array_equal(backend.to_numpy(first.best.candidate.genome), backend.to_numpy(second.best.candidate.genome))
    assert other.trace != first.trace
    # and the result does not depend on what the global generator was seeded with
    np.random.seed(7)
    assert run_sphere(backend, 1).trace == first.trace


# --- quality ---


def test_it_solves_the_ten_dimensional_sphere_to_a_tight_tolerance_within_a_small_budget(backend: Backend):
    result = run_sphere(backend, 3, evaluations_=3000)
    assert result.best is not None
    assert result.best.objectives["value"] < (1e-8 if backend.precision == "float32" else 1e-12)
    assert result.stop_reason in ("strategy", "budget:evaluations")


def test_it_works_under_scalarised_on_a_two_objective_problem(backend: Backend):
    def objectives(X: Any) -> Any:
        return backend.xp.stack([backend.xp.sum(X * X, axis=1), backend.xp.sum((X - 1.0) ** 2, axis=1)], axis=1)

    result = auxein.run(
        strategy=Scalarised(PycmaStrategy(), WeightedSum({"f1": 1.0, "f2": 3.0})),
        evaluator=auxein.VectorisedEvaluator(objectives),
        space=Box(-5.0, 5.0, dim=3),
        objectives=[Objective("f1"), Objective("f2")],
        budget=auxein.Budget(evaluations=2000),
        seed=4,
        backend=backend,
        batch_size=20,
    )
    best = auxein.best_by_scalarisation(result, WeightedSum({"f1": 1.0, "f2": 3.0}), objectives=[Objective("f1"), Objective("f2")])
    np.testing.assert_allclose(backend.to_numpy(best.genome), 0.75, atol=1e-3)  # minimises x^2 + 3 (x - 1)^2: x = 3/4 in every variable


def test_pycma_options_pass_through(backend: Backend):
    quick = auxein.run(
        strategy=PycmaStrategy(options={"tolfun": 1e-2}),
        evaluator=sphere(backend),
        space=Box(-5.0, 5.0, dim=5),
        budget=auxein.Budget(evaluations=5000),
        seed=1,
        backend=backend,
        batch_size=20,
    )
    assert quick.stop_reason == "strategy" and quick.evaluations_used < 1500  # pycma's own tolfun stopped it


# --- no checkpoints, and a resume that replays ---


@pytest.mark.usefixtures("use_corner_backend")
def test_no_checkpoints_are_written_and_a_deterministic_resume_replays_from_the_start_without_evaluating_again(tmp_path: Path):
    backend = integration_backend()
    calls: list[int] = []

    def rastrigin(genome: Any) -> auxein.Result:
        calls.append(1)
        x = backend.to_numpy(genome).astype(np.float64)
        return auxein.Result({"value": float(10 * len(x) + (x * x - 10 * np.cos(2 * np.pi * x)).sum())})

    def arguments(run_dir: Path) -> dict[str, Any]:
        return {
            "strategy": PycmaStrategy(population_size=10),
            "evaluator": auxein.FunctionEvaluator(rastrigin),
            "space": Box(-5.0, 5.0, dim=6),
            "seed": 3,
            "batch_size": 10,
            "backend": backend,
            "run_dir": run_dir,
            "checkpoint_every_evaluations": 20,
        }

    auxein.run(budget=auxein.Budget(evaluations=300), **arguments(tmp_path / "a"))
    assert calls and len(calls) == 300
    with open_run(tmp_path / "a") as recorded:
        assert recorded.checkpoints() == []  # none were taken, though the run asked for one every 20 evaluations
    assert not any((tmp_path / "a" / "checkpoints").glob("*")) if (tmp_path / "a" / "checkpoints").exists() else True
    calls.clear()
    resumed = auxein.resume(budget=auxein.Budget(evaluations=500), **arguments(tmp_path / "a"))
    assert len(calls) == 200  # only the evaluations that were not recorded: the first 300 were replayed
    assert resumed.evaluations_used == 500
    reference = auxein.run(budget=auxein.Budget(evaluations=500), **arguments(tmp_path / "b"))
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")  # the event log of the run that was never stopped
    assert reference.trace == resumed.trace


def test_other_strategies_still_write_checkpoints(tmp_path: Path):
    auxein.run(
        strategy=auxein.GeneticAlgorithm(population_size=10, offspring_size=10),
        evaluator=auxein.FunctionEvaluator(lambda g: float((g * g).sum())),
        space=Box(-5.0, 5.0, dim=3),
        budget=auxein.Budget(evaluations=200),
        seed=1,
        batch_size=10,
        run_dir=tmp_path / "ga",
        checkpoint_every_evaluations=20,
    )
    with open_run(tmp_path / "ga") as recorded:
        assert len(recorded.checkpoints()) >= 1


@pytest.mark.usefixtures("use_corner_backend")
def test_a_throughput_resume_without_checkpoints_starts_again_from_the_beginning(tmp_path: Path):
    """The rule of design doc §10.4 for a run without a checkpoint in throughput mode: what was recorded is discarded and the
    run starts again (here with the same seed, so it ends where a fresh run would)."""
    backend = integration_backend()

    def arguments(run_dir: Path) -> dict[str, Any]:
        return {
            "strategy": PycmaStrategy(population_size=10),
            "evaluator": auxein.VectorisedEvaluator(lambda X: backend.xp.sum(X * X, axis=1)),
            "space": Box(-5.0, 5.0, dim=6),
            "seed": 3,
            "batch_size": 10,
            "backend": backend,
            "run_dir": run_dir,
            "deterministic": False,
        }

    auxein.run(budget=auxein.Budget(evaluations=200), **arguments(tmp_path / "a"))
    resumed = auxein.resume(budget=auxein.Budget(evaluations=400), **arguments(tmp_path / "a"))
    fresh = auxein.run(budget=auxein.Budget(evaluations=400), **arguments(tmp_path / "b"))
    assert resumed.evaluations_used == 400 and resumed.trace == fresh.trace
    with open_run(tmp_path / "a") as recorded:
        assert len(list(recorded.evaluations())) == 400


def test_auxein_imports_without_pycma_at_all():
    """Importing auxein and its strategies must not import pycma: it is an optional extra, needed only to build the strategy."""
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['cma'] = None; import auxein, auxein.strategies; "
        "assert 'cma' not in [m for m in sys.modules if sys.modules[m] is not None]; "
        "from auxein.strategies import PycmaStrategy"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
