"""The numeric `GeneticAlgorithm` follows the same trajectory for a fixed seed: the best candidate and the trace of improvements
below were recorded before the structured genomes step, and must not change by accident (a change would silently alter every
recorded run's reproducibility).

What is pinned is the *trajectory*, not a byte digest: floating-point sums and dot products round differently on different CPUs
(numpy picks different SIMD paths, even among x86-64 machines), so the last bits of a value differ from one machine to another
while the run itself is the same. The candidate ids and the evaluation counts of the trace must match exactly, and the values
to a relative 1e-9. A float32 run is only checked to be reproducible on the machine it runs on, since its rounding differs
more. If this test fails, either the genetic algorithm changed (which that step promised not to do) or the numerical
libraries did: find out which before updating the numbers.
"""

import warnings

import pytest

import auxein

GOLDEN: dict[str, tuple[int, list[tuple[int, float]]]] = {
    "generation": (
        488,
        [
            (1, 57.9725532674),
            (2, 43.4234363867),
            (4, 37.5045972155),
            (7, 34.956001132),
            (11, 25.3227559764),
            (26, 14.4253844799),
            (30, 8.1866055124),
            (46, 8.01793342196),
            (62, 6.15302755398),
            (63, 5.86599030552),
            (86, 3.46191994354),
            (93, 3.2812336929),
            (119, 1.47018137959),
            (156, 1.31764535693),
            (162, 0.970118386296),
            (164, 0.377939182315),
            (199, 0.268110344597),
            (204, 0.167613067225),
            (280, 0.0710264041798),
            (401, 0.0255159697228),
            (468, 0.0251750584188),
            (489, 0.0221139647468),
        ],
    ),
    "steady_state": (
        445,
        [
            (1, 57.9725532674),
            (2, 43.4234363867),
            (4, 37.5045972155),
            (7, 34.956001132),
            (11, 25.3227559764),
            (42, 10.8651856277),
            (56, 9.72280724473),
            (69, 4.14183847627),
            (82, 2.14593879235),
            (124, 1.07768842289),
            (141, 0.509924578509),
            (208, 0.404234944748),
            (277, 0.123200773792),
            (322, 0.116821370912),
            (323, 0.078954901079),
            (387, 0.0648662005623),
            (401, 0.0355461540737),
            (446, 0.025251480824),
        ],
    ),
}


def run(precision: str, delivery: str) -> auxein.RunResult[object]:
    evaluator = (
        auxein.VectorisedEvaluator(lambda X: (X * X).sum(axis=1))
        if delivery == "generation"
        else auxein.FunctionEvaluator(lambda g: float((g * g).sum()))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return auxein.run(  # type: ignore[return-value]
            strategy=auxein.GeneticAlgorithm(population_size=20, offspring_size=20),
            evaluator=evaluator,
            space=auxein.Box(-5, 5, dim=6),
            budget=auxein.Budget(evaluations=500),
            seed=9,
            batch_size=20,
            backend=auxein.Backend("numpy", "cpu", precision),  # type: ignore[arg-type]
            delivery=delivery,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_the_numeric_genetic_algorithm_follows_the_same_trajectory_for_a_fixed_seed(delivery: str):
    result = run("float64", delivery)
    best_id, trace = GOLDEN[delivery]
    assert result.best is not None and result.best.candidate.id == best_id
    assert [n for n, _ in result.trace] == [n for n, _ in trace]  # the same improvements, at the same evaluations
    assert [v for _, v in result.trace] == pytest.approx([v for _, v in trace], rel=1e-9)


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_a_float32_run_is_reproducible_on_the_machine_it_runs_on(delivery: str):
    first, second = run("float32", delivery), run("float32", delivery)
    assert first.trace == second.trace and first.best is not None and second.best is not None
    assert first.best.candidate.id == second.best.candidate.id
