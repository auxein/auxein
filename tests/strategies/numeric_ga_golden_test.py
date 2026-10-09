"""The numeric `GeneticAlgorithm` gives byte-identical results for fixed seeds: the digests were taken on the code before the
structured genomes step, and must not change by accident (a change would silently alter every recorded run's reproducibility).

Floating-point sums and dot products round differently on different CPU architectures (numpy picks different SIMD paths), so a
digest is only portable across machines of one family. The float64 digests are therefore pinned per architecture: `arm64`
(Apple silicon, aarch64 Linux) and `x86_64` (the CI runners). On another architecture the digests are not checked. float32
digests are not pinned at all, since they differ between machines of the same family too; a float32 run is only checked to be
reproducible on the machine it runs on. If a pinned digest changes, either the genetic algorithm changed (which this step
promised not to do) or the numerical libraries did: find out which before updating it.
"""

import hashlib
import platform
import warnings

import numpy as np
import pytest

import auxein

GOLDEN = {
    "arm64": {("float64", "generation"): "33db3f5a354fb116", ("float64", "steady_state"): "4258c808cca54358"},
    "x86_64": {("float64", "generation"): "39ca9c0d625c9eef", ("float64", "steady_state"): "6bb593843ba0c9a9"},
}
ARCHITECTURES = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x86_64", "amd64": "x86_64"}


def digest_of(precision: str, delivery: str) -> str:
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
    return digest.hexdigest()[:16]


@pytest.mark.parametrize(("precision", "delivery"), [("float64", "generation"), ("float64", "steady_state")])
def test_the_numeric_genetic_algorithm_is_byte_identical_for_a_fixed_seed(precision: str, delivery: str):
    family = ARCHITECTURES.get(platform.machine().lower())
    if family is None:
        pytest.skip(f"no pinned digests for the architecture {platform.machine()!r}")
    assert digest_of(precision, delivery) == GOLDEN[family][(precision, delivery)]


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_a_float32_run_is_reproducible_on_the_machine_it_runs_on(delivery: str):
    assert digest_of("float32", delivery) == digest_of("float32", delivery)
