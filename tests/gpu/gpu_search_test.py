"""Whole runs on the device: `GeneticAlgorithm`, `RandomSearch` and the structured GA, with a vectorised evaluator."""

import warnings
from typing import Any

import numpy as np

import auxein
from auxein.backend import Backend
from auxein.strategies.structured import ExternalMutation, OperatorNotRecordedWarning, SequenceMutation
from tests.support import sequences
from tests.support.fixtures import assert_on_backend


def sphere(X: Any) -> Any:
    return (X * X).sum(axis=1)


def go(strategy: Any, backend: Backend, evaluations: int, dim: int = 10, **kwargs: Any) -> auxein.RunResult[Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", auxein.RecordingDisabledWarning)
        return auxein.run(
            strategy=strategy,
            evaluator=auxein.VectorisedEvaluator(sphere),
            space=auxein.Box(-5.0, 5.0, dim=dim),
            budget=auxein.Budget(evaluations=evaluations),
            seed=1,
            backend=backend,
            batch_size=50,
            **kwargs,
        )


def test_the_genetic_algorithm_solves_the_sphere_on_the_device_and_its_best_stays_there(gpu_backend: Backend):
    result = go(auxein.GeneticAlgorithm(population_size=50, offspring_size=50), gpu_backend, 10_000)
    assert result.best is not None
    assert result.best.objectives["value"] < (1e-2 if gpu_backend.precision == "float32" else 1e-4)
    assert_on_backend(result.best.candidate.genome, gpu_backend)
    assert [v for _, v in result.trace] == sorted((v for _, v in result.trace), reverse=True)


def test_every_selection_and_mutation_runs_on_the_device(gpu_backend: Backend):
    from auxein.strategies.ga import GaussianMutation, SelfAdaptiveMutation, SigmaScalingSUS, TournamentSelection

    for selection in (TournamentSelection(3), SigmaScalingSUS()):
        for mutation in (GaussianMutation(0.05), SelfAdaptiveMutation(per_gene=True)):
            result = go(
                auxein.GeneticAlgorithm(population_size=40, offspring_size=40, selection=selection, mutation=mutation), gpu_backend, 2000
            )
            assert result.best is not None and np.isfinite(result.best.objectives["value"])
            assert result.status_counts.get("ok") == 2000


def test_random_search_runs_on_the_device_and_improves(gpu_backend: Backend):
    result = go(auxein.RandomSearch(), gpu_backend, 5000, dim=3)
    assert result.best is not None and result.best.objectives["value"] < 1.0
    assert_on_backend(result.best.candidate.genome, gpu_backend)


def test_constraints_and_failures_are_handled_with_device_arrays(gpu_backend: Backend):
    def with_constraint(X: Any) -> auxein.BatchResult:
        return auxein.BatchResult({"value": (X * X).sum(axis=1)}, {"c": gpu_backend.xp.clip(1.0 - X[:, 0], 0.0, None)})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", auxein.RecordingDisabledWarning)
        result = auxein.run(
            strategy=auxein.GeneticAlgorithm(population_size=40, offspring_size=40),
            evaluator=auxein.VectorisedEvaluator(with_constraint),
            space=auxein.Box(-5.0, 5.0, dim=3),
            constraints=["c"],
            budget=auxein.Budget(evaluations=4000),
            seed=2,
            backend=gpu_backend,
            batch_size=40,
        )
    assert result.best is not None and result.best.constraints["c"] == 0.0 and result.best.candidate.genome[0] >= 1.0 - 1e-5


def test_the_structured_genetic_algorithm_ranks_and_selects_on_the_device(float32_backend: Backend):
    """Its genomes are Python tuples; its ranking, its selection and the choice between mutation operators are device work
    (a draw read with `np.asarray` instead of `to_numpy` fails on CUDA and Metal, which this catches)."""
    strategy = auxein.StructuredGeneticAlgorithm(
        population_size=12,
        offspring_size=12,
        mutation=[(SequenceMutation(), 2.0), (ExternalMutation(sequences.FakeLLM()), 1.0)],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", auxein.RecordingDisabledWarning)
        warnings.simplefilter("ignore", OperatorNotRecordedWarning)
        result = auxein.run(
            strategy=strategy,
            evaluator=auxein.FunctionEvaluator(sequences.distance),
            space=sequences.SPACE,
            objectives=[auxein.Objective("distance")],
            constraints=["too_long"],
            budget=auxein.Budget(evaluations=240),
            seed=3,
            backend=float32_backend,
            batch_size=12,
        )
    assert result.best is not None and result.best.objectives["distance"] < 10
