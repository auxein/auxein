"""The numeric `GeneticAlgorithm` gives byte-identical results for fixed seeds: the digests were taken on the code before the
structured genomes step, and must never change (a change would silently alter every recorded run's reproducibility)."""

import hashlib
import warnings

import numpy as np
import pytest

import auxein

GOLDEN = {
    ("float64", "generation"): "33db3f5a354fb116",
    ("float64", "steady_state"): "4258c808cca54358",
    ("float32", "generation"): "0f995acd4a0a559d",
    ("float32", "steady_state"): "230eb304c2fe96d6",
}


@pytest.mark.parametrize(("precision", "delivery"), list(GOLDEN))
def test_the_numeric_genetic_algorithm_is_byte_identical_for_a_fixed_seed(precision: str, delivery: str):
    evaluator = (
        auxein.VectorisedEvaluator(lambda X: (X * X).sum(axis=1))
        if delivery == "generation"
        else auxein.FunctionEvaluator(lambda g: float((g * g).sum()))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = auxein.run(
            strategy=auxein.GeneticAlgorithm(population_size=20, offspring_size=20),
            evaluator=evaluator,
            space=auxein.Box(-5, 5, dim=6),
            budget=auxein.Budget(evaluations=500),
            seed=9,
            batch_size=20,
            backend=auxein.Backend("numpy", "cpu", precision),  # type: ignore[arg-type]
            delivery=delivery,  # type: ignore[arg-type]
        )
    assert result.best is not None
    digest = hashlib.sha256()
    for evaluation in [result.best, *result.pareto_front]:
        digest.update(np.asarray(evaluation.candidate.genome, dtype=np.float64).tobytes())
    digest.update(repr(result.trace).encode())
    assert digest.hexdigest()[:16] == GOLDEN[(precision, delivery)]
