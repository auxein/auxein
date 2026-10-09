import warnings
from pathlib import Path

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import BatchResult, Objective, Result
from auxein.driver import Budget, RecordingDisabledWarning, run
from auxein.evaluators import FunctionEvaluator, VectorisedEvaluator
from auxein.recording import open_run
from auxein.spaces import Box
from auxein.strategies import GeneticAlgorithm
from auxein.strategies.ga import SelfAdaptiveMutation, SigmaScalingSUS


def sphere(X):
    return (X * X).sum(axis=1)


def go(strategy, evaluator=None, evaluations: int = 20000, dim: int = 10, seed: int = 1, backend=None, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RecordingDisabledWarning)
        return run(
            strategy=strategy,
            evaluator=evaluator or VectorisedEvaluator(sphere),
            space=kwargs.pop("space", Box(-5.0, 5.0, dim=dim)),
            budget=Budget(evaluations=evaluations),
            seed=seed,
            backend=backend,
            **kwargs,
        )


def test_the_default_configuration_solves_the_10d_sphere_within_20000_evaluations(backend: Backend):
    result = go(GeneticAlgorithm(), backend=backend)
    assert result.evaluations_used == 20000 and result.stop_reason == "budget:evaluations"
    assert result.best is not None and result.best.objectives["value"] < 1e-6  # 0.2.0's default ended around 2 (sphere benchmark)
    assert [v for _, v in result.trace] == sorted((v for _, v in result.trace), reverse=True)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GeneticAlgorithm(selection=SigmaScalingSUS()),
        lambda: GeneticAlgorithm(mutation=SelfAdaptiveMutation(per_gene=True)),
        lambda: GeneticAlgorithm(population_size=20, offspring_size=None),
    ],
    ids=["sus", "per-gene", "n-children"],
)
def test_other_configurations_also_solve_it(factory, backend: Backend):
    assert go(factory(), evaluations=20000, backend=backend).best.objectives["value"] < 1e-3  # type: ignore[union-attr]


def test_every_candidate_is_evaluated_exactly_once(tmp_path: Path, backend: Backend):
    rows: list[int] = []

    def spy(X):
        rows.append(int(X.shape[0]))
        return sphere(X)

    result = go(
        GeneticAlgorithm(population_size=20, offspring_size=15),
        VectorisedEvaluator(spy),
        evaluations=500,
        dim=4,
        run_dir=tmp_path / "r",
        backend=backend,
    )
    assert sum(rows) == result.evaluations_used == 500  # the population is never re-scored
    assert rows[0] == 20 and set(rows[1:-1]) == {15}  # the initial population, then exactly lambda children per generation
    with open_run(tmp_path / "r") as recorded:
        ids = [e.candidate_id for e in recorded.evaluations()]
    assert ids == list(range(500))


def test_an_oversized_final_generation_is_truncated_and_not_told(tmp_path: Path, backend: Backend):
    told_sizes: list[int] = []

    class Spy(GeneticAlgorithm):
        def tell(self, results):
            told_sizes.append(len(results))
            super().tell(results)

    result = go(Spy(population_size=10, offspring_size=8), evaluations=30, dim=3, run_dir=tmp_path / "r", backend=backend)
    assert result.evaluations_used == 30 and told_sizes == [
        10,
        8,
        8,
    ]  # 10 + 8 + 8 = 26, then only 4 of 8 fit: evaluated, recorded, not told
    with open_run(tmp_path / "r") as recorded:
        assert len(list(recorded.evaluations())) == 30


def test_lineage_in_the_event_log_matches_the_parents_in_the_batches(tmp_path: Path, backend: Backend):
    parents_asked: dict[int, tuple[int, ...]] = {}

    class Recording(GeneticAlgorithm):
        def ask(self, n):
            batch = super().ask(n)
            for c in batch.candidates:
                parents_asked[int(c.id)] = tuple(int(p) for p in c.parents)
            return batch

    result = go(Recording(population_size=12, offspring_size=9), evaluations=300, dim=3, run_dir=tmp_path / "r", backend=backend)
    with open_run(tmp_path / "r") as recorded:
        evaluations = list(recorded.evaluations())
        assert {int(e.candidate_id): tuple(int(p) for p in e.parents) for e in evaluations} == {i: parents_asked[i] for i in range(300)}
        init, children = [e for e in evaluations if e.origin == "init"], [e for e in evaluations if e.origin != "init"]
        assert len(init) == 12 and all(e.parents == () for e in init)
        assert all(
            len(e.parents) == 2 and e.parents[0] != e.parents[1] and e.origin == "tournament+intermediate+self_adaptive" for e in children
        )
        assert all(max(e.parents) < e.candidate_id for e in children)  # parents are older than their children
        best = int(result.best.candidate.id)  # type: ignore[union-attr]
        ancestors = recorded.ancestry(best)
        assert all(a < best for a in ancestors) and (ancestors or best < 12)
        if ancestors:
            assert set(ancestors) <= {e.candidate_id for e in evaluations}
            first_parent = recorded.ancestry(best)[0]
            assert first_parent in parents_asked[best]  # the nearest ancestor is a parent
        assert recorded.metadata["strategy"]["class"].endswith("Recording")  # type: ignore[index]


def test_the_same_seed_gives_the_same_run_and_different_seeds_differ(backend: Backend):
    a = go(GeneticAlgorithm(), evaluations=3000, backend=backend, seed=3)
    b = go(GeneticAlgorithm(), evaluations=3000, backend=backend, seed=3)
    c = go(GeneticAlgorithm(), evaluations=3000, backend=backend, seed=4)
    assert a.trace == b.trace and a.best.objectives == b.best.objectives and a.best.candidate.id == b.best.candidate.id  # type: ignore[union-attr]
    assert a.trace != c.trace


def test_constraints_are_respected_through_the_driver(backend: Backend):
    # minimise the sum of squares, but x0 must be at least 1: the optimum is (1, 0, ...)
    def fn(X):
        return BatchResult({"value": (X * X).sum(axis=1)}, {"cpa": np.maximum(0.0, 1.0 - np.asarray(X[:, 0]))})

    result = go(GeneticAlgorithm(), VectorisedEvaluator(fn), evaluations=10000, dim=4, constraints=["cpa"], backend=backend)
    assert result.best is not None and result.best.constraints["cpa"] == 0.0
    assert result.best.objectives["value"] == pytest.approx(1.0, abs=1e-2)
    assert all(e.constraints["cpa"] == 0.0 for e in result.pareto_front)


def test_function_evaluators_work_too(backend: Backend):
    result = go(
        GeneticAlgorithm(population_size=20), FunctionEvaluator(lambda g: float((g * g).sum())), evaluations=4000, dim=3, backend=backend
    )
    assert result.best is not None and result.best.objectives["value"] < 1e-3


def test_maximisation_through_the_declared_direction(backend: Backend):
    result = go(
        GeneticAlgorithm(),
        VectorisedEvaluator(lambda X: -sphere(X)),
        evaluations=6000,
        dim=4,
        objectives=[Objective("score", "maximise")],
        backend=backend,
    )
    assert result.best is not None and result.best.objectives["score"] > -1e-3  # the score is maximised towards its optimum, 0


def test_noisy_objectives_do_not_break_the_selection(backend: Backend):
    def noisy(X, rng):
        return sphere(X) * (1.0 + 0.1 * rng.normal(X.shape[0]))

    result = go(GeneticAlgorithm(), VectorisedEvaluator(noisy, uses_rng=True), evaluations=10000, dim=4, backend=backend)
    assert result.best is not None and np.isfinite(result.best.objectives["value"])


def test_a_result_object_with_cost_units_passes_through(backend: Backend):
    result = go(
        GeneticAlgorithm(population_size=10),
        FunctionEvaluator(lambda g: Result({"value": float((g * g).sum())}, cost={"tokens": 2.0})),
        evaluations=60,
        dim=2,
        backend=backend,
    )
    assert result.evaluations_used == 60
