"""Numerically delicate spots, on every backend and precision (design doc §7.2, rule 4: float32-safe).

Each test is about a place where float32 is not just float64 with fewer digits: a threshold written as a Python float that
the float32 array rounds across, a scale that vanishes next to the values, a sum that overflows. They pass in float64
by construction, so a failure here names a precision-dependent bug.
"""

import math

import numpy as np
import pytest

from auxein.aggregators import cvar_lower, cvar_upper, quantile
from auxein.backend import Backend
from auxein.core import EvaluationBatch, Objective, to_minimisation
from auxein.strategies import GeneticAlgorithm
from auxein.strategies.ga import GaussianMutation, SelfAdaptiveMutation, SigmaScalingSUS
from auxein.strategies.ga.base import PopulationView
from auxein.strategies.ga.ranking import view_of
from tests.strategies.ga_strategy_test import bind, evaluate
from tests.support.fixtures import assert_on_backend


def population(backend: Backend, values: list[float], violation: list[float] | None = None) -> PopulationView:
    v = backend.asarray(values)
    violation_array = backend.asarray(violation or [0.0] * len(values))
    return view_of(v, violation_array, backend.asarray(list(range(len(values))), dtype=backend.int_dtype), backend)


# --- step sizes at their floor ---


@pytest.mark.parametrize("min_step", [1e-12, 1e-6, 1e-3, 0.01, 0.05, 0.1])
def test_a_collapsed_population_is_recognised_whatever_float_the_floor_rounds_to(backend: Backend, min_step: float):
    """`float32(1e-3)` is slightly above `1e-3`: a floor compared as a Python float would never be 'reached' in float32."""
    ga, _ = bind(
        GeneticAlgorithm(
            population_size=4,
            offspring_size=4,
            convergence_tolerance=1e-9,
            mutation=SelfAdaptiveMutation(min_step=min_step, initial_step=min_step),
        ),
        backend,
    )
    ga.tell(EvaluationBatch(evaluate(ga.ask(1), lambda c: 1.0)))  # identical objectives, steps at the floor
    assert ga.should_stop()


@pytest.mark.parametrize("per_gene", [False, True])
def test_step_sizes_stay_finite_and_above_the_floor_over_many_mutations(backend: Backend, per_gene: bool):
    mutation = SelfAdaptiveMutation(per_gene=per_gene, initial_step=0.1, min_step=1e-6)
    from auxein.random import RunSeed

    rng = RunSeed(3).stream("strategy", backend=backend)
    genomes = backend.asarray(np.zeros((40, 5)))
    width = backend.asarray(np.full(5, 10.0))
    steps = mutation.initial_steps(40, 5, backend)
    for _ in range(300):
        genomes, steps = mutation.mutate(genomes, steps, width, rng, backend)
    host = backend.to_numpy(steps)
    assert np.isfinite(host).all() and (host >= 1e-6 * (1 - 1e-6)).all()
    assert_on_backend(steps, backend)
    assert np.isfinite(backend.to_numpy(genomes)).all()


def test_a_step_at_the_floor_does_not_break_a_float32_genome_that_cannot_move(backend: Backend):
    """At the floor the move (1e-12 of the width) is far below float32 resolution: the genome simply stays where it is."""
    from auxein.random import RunSeed

    mutation = SelfAdaptiveMutation(initial_step=1e-12, min_step=1e-12)
    rng = RunSeed(1).stream("strategy", backend=backend)
    genomes = backend.asarray(np.ones((6, 3)))
    child, steps = mutation.mutate(genomes, mutation.initial_steps(6, 3, backend), backend.asarray(np.full(3, 10.0)), rng, backend)
    assert np.isfinite(backend.to_numpy(child)).all() and np.isfinite(backend.to_numpy(steps)).all()
    np.testing.assert_allclose(backend.to_numpy(child), 1.0, atol=1e-9)


# --- sigma scaling when the spread is tiny or the values are huge ---


def test_sigma_scaling_still_prefers_the_best_when_the_spread_is_tiny_next_to_the_values(backend: Backend):
    base = 1000.0
    values = [base + 0.01 * k for k in range(8)]  # float32 resolves 6e-5 at 1000: the spread is visible, the scale is not
    weights = backend.to_numpy(SigmaScalingSUS().weights(population(backend, values))).astype(np.float64)
    assert np.isfinite(weights).all() and (weights >= 0).all() and weights.sum() > 0
    assert weights.argmax() == 0  # the lowest value is the best
    assert weights[0] > weights[-1]


def test_sigma_scaling_falls_back_to_uniform_when_all_values_are_equal(backend: Backend):
    weights = backend.to_numpy(SigmaScalingSUS().weights(population(backend, [7.0] * 6))).astype(np.float64)
    np.testing.assert_allclose(weights, 1.0)


@pytest.mark.parametrize("magnitude", [1e-30, 1e18, 1e30])
def test_sigma_scaling_is_invariant_to_the_scale_of_the_objective(backend: Backend, magnitude: float):
    """Squaring the deviations would overflow float32 at 1e19 or underflow it at 1e-23, so the goodness is normalised first."""
    reference = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]
    expected = backend.to_numpy(SigmaScalingSUS().weights(population(backend, reference))).astype(np.float64)
    scaled = backend.to_numpy(SigmaScalingSUS().weights(population(backend, [v * magnitude for v in reference]))).astype(np.float64)
    assert np.isfinite(scaled).all()
    np.testing.assert_allclose(scaled / scaled.sum(), expected / expected.sum(), atol=1e-4)


@pytest.mark.filterwarnings("ignore:overflow encountered in cast")  # numpy says so when 1e39 becomes infinity: that is the point
def test_an_objective_beyond_float32_ranks_last_without_breaking_the_ranking(backend: Backend):
    """1e39 is a finite objective in float64 and infinity in float32: it must rank last, not break the sort or the weights."""
    view = population(backend, [1.0, 1e39, 2.0, 1e39])
    order = backend.to_numpy(view.order).tolist()
    assert order[:2] == [0, 2] and sorted(order[2:]) == [1, 3]
    weights = backend.to_numpy(SigmaScalingSUS().weights(view)).astype(np.float64)
    assert np.isfinite(weights).all() and weights.sum() > 0
    if backend.precision == "float32":  # the overflowed members have no goodness to weigh, and must not flatten everyone else's
        assert weights[0] > weights[2] > weights[1] == weights[3] == 0


# --- the minimisation form ---


def test_the_minimisation_form_negates_maximised_objectives_exactly_in_the_backends_precision(backend: Backend):
    values = backend.asarray([[1.5, -2.25], [3.0, 1e-8]])
    converted = to_minimisation(values, (Objective("a"), Objective("b", "maximise")), backend)
    assert_on_backend(converted, backend)
    np.testing.assert_array_equal(backend.to_numpy(converted), backend.to_numpy(values) * np.array([1.0, -1.0]))


# --- reducers over scenarios ---


@pytest.mark.parametrize(("alpha", "scenarios"), [(0.07, 100), (0.14, 50), (0.28, 25), (0.55, 100), (0.1, 10), (0.3, 10), (0.7, 10)])
def test_cvar_takes_exactly_ceil_alpha_s_scenarios_even_when_alpha_times_s_is_a_hair_above_an_integer(
    backend: Backend, alpha: float, scenarios: int
):
    """`0.07 * 100` is 7.000000000000001 in binary floating point, and `ceil` of that is 8, not 7: one scenario too many."""
    expected_k = round(alpha * scenarios)
    values = np.arange(scenarios, dtype=np.float64)[None, :]  # 0 .. s-1: the tail's mean tells how many values it holds
    measured = {"m": backend.asarray(values)}
    upper = float(backend.to_numpy(cvar_upper("m", alpha).reduce(measured, backend.xp))[0])
    lower = float(backend.to_numpy(cvar_lower("m", alpha).reduce(measured, backend.xp))[0])
    assert upper == pytest.approx(np.mean(values[0, scenarios - expected_k :]), rel=1e-6)
    assert lower == pytest.approx(np.mean(values[0, :expected_k]), rel=1e-6)


def test_quantile_in_float32_agrees_with_float64(backend: Backend):
    rng = np.random.default_rng(0)
    values = rng.normal(size=(5, 37)) * 1e3
    got = backend.to_numpy(quantile("m", 0.9).reduce({"m": backend.asarray(values)}, backend.xp)).astype(np.float64)
    np.testing.assert_allclose(got, np.quantile(values, 0.9, axis=1), rtol=1e-5, atol=1e-2 if backend.precision == "float32" else 1e-9)


def test_a_mean_over_many_float32_scenarios_is_accurate_enough(backend: Backend):
    """A naive float32 running sum of 100,000 values near 1 drifts by about 1e-3; the backends' reductions must not."""
    from auxein.aggregators import mean

    values = np.full((2, 100_000), 1.1)
    got = backend.to_numpy(mean("m").reduce({"m": backend.asarray(values)}, backend.xp)).astype(np.float64)
    np.testing.assert_allclose(got, 1.1, rtol=1e-5)


def test_the_gaussian_mutation_keeps_the_dtype(backend: Backend):
    from auxein.random import RunSeed

    rng = RunSeed(1).stream("strategy", backend=backend)
    genomes = backend.asarray(np.zeros((4, 3)))
    child, steps = GaussianMutation(0.1).mutate(genomes, None, backend.asarray(np.full(3, 2.0)), rng, backend)
    assert steps is None
    assert_on_backend(child, backend)
    assert not math.isnan(float(backend.to_numpy(child).sum()))
