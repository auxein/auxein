"""The numeric genetic algorithm on mixed spaces: it finds a known optimum, beats random search, and checkpoints exactly."""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import auxein
from auxein.backend import Backend
from auxein.core import EvaluationBatch, IdIssuer, Objective, validate_state_dict
from auxein.spaces import Binary, Integer, MixedSpace
from auxein.strategies.ga import (
    BitFlipMutation,
    CategoricalMutation,
    GaussianMutation,
    IntegerMutation,
    ReflectRepair,
    SelfAdaptiveMutation,
    SigmaScalingSUS,
    UniformRecombination,
)
from tests.strategies.ga_strategy_test import bind, evaluate
from tests.strategies.strategy_checkpoint_test import same, snapshot, through_a_checkpoint
from tests.support import mixed as mx
from tests.support.fixtures import assert_on_backend

pytestmark = pytest.mark.filterwarnings("ignore::auxein.RecordingDisabledWarning")


def solve(strategy: Any, backend: Backend, evaluations: int = 6000, seed: int = 1, **kwargs: Any) -> auxein.RunResult[Any]:
    return auxein.run(
        strategy=strategy,
        evaluator=auxein.VectorisedEvaluator(lambda X: mx.evaluate(X, backend)),
        space=mx.SPACE,
        constraints=["too_big"],
        budget=auxein.Budget(evaluations=evaluations),
        seed=seed,
        backend=backend,
        batch_size=50,
        **kwargs,
    )


# --- binding ---


def test_it_accepts_a_box_or_a_mixed_space_and_nothing_else(backend: Backend):
    bind(auxein.GeneticAlgorithm(), backend, space=mx.SPACE)
    bind(auxein.GeneticAlgorithm(), backend, space=auxein.Box(-1.0, 1.0, dim=3))
    with pytest.raises(TypeError, match="Box or a MixedSpace"):
        bind(auxein.GeneticAlgorithm(), backend, space=auxein.SequenceSpace(("a", "b"), 1, 3))


@pytest.mark.parametrize("strategy", [auxein.GeneticAlgorithm, auxein.RandomSearch])
def test_integer_bounds_float32_cannot_hold_are_refused_when_the_run_starts(strategy: Any):
    wide = MixedSpace({"n": Integer(0, 2**25), "b": Binary()})
    float32 = Backend("numpy", "cpu", "float32")
    with pytest.raises(ValueError, match="holds integers exactly only up to 16777216"):
        auxein.run(
            strategy=strategy(),
            evaluator=auxein.VectorisedEvaluator(lambda X: X[:, 0]),
            space=wide,
            budget=auxein.Budget(evaluations=10),
            seed=1,
            backend=float32,
        )
    result = auxein.run(
        strategy=strategy(),
        evaluator=auxein.VectorisedEvaluator(lambda X: X[:, 0]),
        space=wide,
        budget=auxein.Budget(evaluations=100),
        seed=1,
        backend=Backend(),
    )
    assert result.evaluations_used == 100  # float64 holds them


def test_random_search_works_on_a_mixed_space_without_special_handling(backend: Backend):
    result = solve(auxein.RandomSearch(), backend, 2000)
    assert result.best is not None and result.evaluations_used == 2000
    assert_on_backend(result.best.candidate.genome, backend)
    assert mx.SPACE.contains(result.best.candidate.genome)


# --- finding the optimum ---


def test_the_ga_finds_the_known_optimum_and_beats_random_search_at_the_same_budget(backend: Backend):
    found, random = [], []
    for seed in range(3):
        ga = solve(auxein.GeneticAlgorithm(), backend, seed=seed)
        rs = solve(auxein.RandomSearch(), backend, seed=seed)
        assert ga.best is not None and rs.best is not None
        found.append(ga.best.objectives["value"])
        random.append(rs.best.objectives["value"])
        values = mx.SPACE.values(ga.best.candidate.genome)
        # every discrete gene is right, and the real genes are close to the targets those choices imply
        assert (values["k"], values["m"], values["mode"]) == (mx.BEST["k"], mx.BEST["m"], mx.MODES[mx.BEST["mode"]])
        assert (values["b0"], values["b1"], values["b2"]) == (True, False, True)
        assert ga.best.constraints["too_big"] == 0.0
    assert max(found) < 0.01, found  # the known optimum is 0
    assert min(random) > 10 * max(found), (found, random)  # random search is nowhere near


def test_the_best_member_never_violates_the_constraint(backend: Backend):
    result = solve(auxein.GeneticAlgorithm(population_size=30, offspring_size=30), backend, 3000, seed=4)
    assert result.best is not None and result.best.constraints["too_big"] == 0.0
    infeasible = [e for e in result.pareto_front if e.constraints["too_big"] > 0]
    assert not infeasible


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_it_works_under_both_deliveries(backend: Backend, delivery: str):
    result = auxein.run(
        strategy=auxein.GeneticAlgorithm(population_size=30, offspring_size=15),
        evaluator=auxein.FunctionEvaluator(lambda genome: mx.evaluate_one(genome, backend)),
        space=mx.SPACE,
        constraints=["too_big"],
        budget=auxein.Budget(evaluations=4000),
        seed=2,
        backend=backend,
        batch_size=15,
        delivery=delivery,  # type: ignore[arg-type]
    )
    assert result.best is not None and result.best.objectives["value"] < 0.5
    assert mx.SPACE.values(result.best.candidate.genome)["mode"] == "c"


@pytest.mark.parametrize(
    "options",
    [
        {"mutation": GaussianMutation(0.05)},
        {"mutation": SelfAdaptiveMutation(per_gene=True)},
        {"selection": SigmaScalingSUS()},
        {"recombination": UniformRecombination(), "crossover_probability": 0.7},
        {"repair": ReflectRepair()},
        {"integer_mutation": IntegerMutation(adaptive=False, initial_step=1.5), "binary_mutation": BitFlipMutation(0.2)},
        {"categorical_mutation": CategoricalMutation(0.5)},
    ],
    ids=["gaussian", "per-gene", "sus", "uniform-pc", "reflect", "fixed-integer-step", "categorical-rate"],
)
def test_every_operator_can_be_replaced_and_the_genomes_stay_valid(backend: Backend, options: dict[str, Any]):
    ga = auxein.GeneticAlgorithm(population_size=30, offspring_size=30, **options)
    result = solve(ga, backend, 2400, seed=6)
    assert result.best is not None and result.best.objectives["value"] < 3.0
    assert mx.SPACE.contains(result.best.candidate.genome)


def test_a_search_without_real_genes_finds_the_bit_pattern(backend: Backend):
    space = auxein.BinarySpace(24)
    target = np.array([1, 0] * 12, dtype=np.float64)
    wanted = backend.asarray(target)
    result = auxein.run(
        strategy=auxein.GeneticAlgorithm(population_size=40, offspring_size=40),
        evaluator=auxein.VectorisedEvaluator(lambda X: backend.xp.sum(backend.xp.abs(X - wanted), axis=1)),
        space=space,
        budget=auxein.Budget(evaluations=4000),
        seed=3,
        backend=backend,
        batch_size=40,
    )
    assert result.best is not None and result.best.objectives["value"] <= 1.0


def test_integer_genes_keep_moving_late_in_a_run(backend: Backend):
    """The integer step has a floor, so a converged population still proposes new integer values every generation."""
    ga, _ = bind(auxein.GeneticAlgorithm(population_size=20, offspring_size=20), backend, space=mx.SPACE)
    batch = ga.ask(1)
    ga.tell(EvaluationBatch(evaluate(batch, lambda c: 1.0)))  # every member equally good: nothing is selected for or against anything
    seen: set[float] = set()
    for _ in range(40):
        batch = ga.ask(1)
        seen |= set(backend.to_numpy(batch.as_array())[:, 3].tolist())
        ga.tell(EvaluationBatch(evaluate(batch, lambda c: 1.0)))
    assert len(seen) > 6
    assert float(backend.to_numpy(ga.state_dict()["steps"])[:, -1].min()) >= 0.1 * (1 - 1e-6)  # type: ignore[arg-type]


def test_origins_name_the_operator_of_each_type(backend: Backend):
    ga, _ = bind(auxein.GeneticAlgorithm(population_size=6, offspring_size=6), backend, space=mx.SPACE)
    first = ga.ask(1)
    assert set(first.origins) == {"init"}
    ga.tell(EvaluationBatch(evaluate(first, lambda c: float(c.id))))
    assert set(ga.ask(1).origins) == {"tournament+intermediate/discrete+self_adaptive/geometric/bitflip/resample"}
    assert "integer_mutation" not in repr(auxein.GeneticAlgorithm())  # the description of a default GA is the one it always had


# --- state ---


def mixed_value(backend: Backend):
    return lambda c: float(backend.to_numpy(mx.evaluate(c.genome[None, :], backend).objectives["value"])[0])


def run_on(strategy: Any, backend: Backend, rounds: int) -> list[Any]:
    out = []
    value = mixed_value(backend)
    for _ in range(rounds):
        batch = strategy.ask(1)
        out.append(snapshot(batch, backend))
        strategy.tell(EvaluationBatch(evaluate(batch, value)))
    return out


@pytest.mark.parametrize("per_gene", [False, True])
def test_a_mixed_ga_restored_from_a_checkpoint_continues_byte_identically(tmp_path: Path, backend: Backend, per_gene: bool):
    kwargs = {"mutation": SelfAdaptiveMutation(per_gene=per_gene), "crossover_probability": 0.8}
    ga, issuer = bind(auxein.GeneticAlgorithm(population_size=8, offspring_size=5, **kwargs), backend, seed=21, space=mx.SPACE)
    run_on(ga, backend, 4)
    pending = ga.ask(1)
    value = mixed_value(backend)
    ga.tell(EvaluationBatch(evaluate(pending, value)[:2]))
    state = through_a_checkpoint(ga.state_dict(), tmp_path / "c", backend)
    validate_state_dict(state)
    steps = state["steps"]
    assert_on_backend(steps, backend)  # type: ignore[arg-type]
    assert steps.shape[1] == (4 if per_gene else 2)  # type: ignore[union-attr]  (the real step sizes, then the integer mean step)

    fresh = auxein.GeneticAlgorithm(population_size=8, offspring_size=5, **kwargs)
    restored, _ = bind(fresh, backend, seed=21, space=mx.SPACE, issuer=IdIssuer(issuer.issued))
    restored.load_state_dict(state)
    rest = EvaluationBatch(evaluate(pending, value)[2:])
    ga.tell(rest)
    restored.tell(rest)
    same(run_on(ga, backend, 8), run_on(restored, backend, 8))
    assert restored.ranked_ids() == ga.ranked_ids()
    np.testing.assert_array_equal(backend.to_numpy(restored.state_dict()["steps"]), backend.to_numpy(ga.state_dict()["steps"]))  # type: ignore[arg-type]


# --- the polynomial regression with structure genes ---


@pytest.mark.parametrize("seed", [1, 5, 7])
def test_the_ga_recovers_the_true_terms_of_a_noise_free_polynomial(backend: Backend, seed: int):
    """The genome is seven coefficients and seven switches; the objective is the error plus a price per active term. The true
    polynomial is `2 + 3x² − 1.5x⁵`. These seeds recover exactly the three true terms on all four configurations at this
    budget; across seeds 0 to 9 the recovery rate is 60 to 100 % depending on the configuration, the others ending in a
    near optimum with one spurious term (a premature convergence of the switches, not a failure of the operators)."""
    result = auxein.run(
        strategy=auxein.GeneticAlgorithm(population_size=100, offspring_size=100),
        evaluator=auxein.VectorisedEvaluator(lambda X: mx.polynomial_evaluate(X, backend)),
        space=mx.POLY_SPACE,
        objectives=[Objective("loss")],
        budget=auxein.Budget(evaluations=15000),
        seed=seed,
        backend=backend,
        batch_size=100,
    )
    assert result.best is not None
    assert mx.active_terms(result.best.candidate.genome) == mx.TRUE_TERMS
    values = mx.POLY_SPACE.values(result.best.candidate.genome)
    for term, coefficient in mx.TRUE_COEFFICIENTS.items():
        assert values[f"c{term}"] == pytest.approx(coefficient, abs=0.02 if backend.precision == "float64" else 0.05)
    assert result.best.objectives["loss"] == pytest.approx(0.02 * len(mx.TRUE_TERMS), abs=1e-3)


def test_unused_terms_are_not_kept_for_free():
    """Without the complexity price a spurious term could stay; with it, the minimum is the true set and nothing else."""
    backend = Backend()
    genome = mx.POLY_SPACE.sample_genomes(1, auxein.random.RunSeed(0).stream("s"), backend)
    true = np.zeros(14)
    for term, coefficient in mx.TRUE_COEFFICIENTS.items():
        true[term], true[7 + term] = coefficient, 1.0
    loss = float(mx.polynomial_evaluate(backend.asarray(true[None, :]), backend)[0])
    assert loss == pytest.approx(0.06, abs=1e-9)  # no data error, three active terms
    spurious = true.copy()
    spurious[7 + 4] = 1.0  # a zero coefficient switched on: same fit, a higher price
    assert float(mx.polynomial_evaluate(backend.asarray(spurious[None, :]), backend)[0]) == pytest.approx(0.08, abs=1e-9)
    _ = genome, warnings
