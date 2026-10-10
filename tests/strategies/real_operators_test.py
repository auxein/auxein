"""Simulated binary crossover and polynomial mutation: the standard real-coded operators of NSGA-II."""

import numpy as np
import pytest

import auxein
from auxein.backend import Backend
from auxein.random import RunSeed
from auxein.spaces import Box
from auxein.strategies.ga import GaussianMutation, PolynomialMutation, SimulatedBinaryCrossover
from auxein.strategies.ga.mutation import with_bounds
from auxein.strategies.ga.recombination import mix_genes
from tests.support.fixtures import assert_on_backend

N = 100_000


def stream(backend: Backend, seed: int = 1):
    return RunSeed(seed).stream("strategy", backend=backend)


def children(backend: Backend, a: float, b: float, count: int = N, dim: int = 1, **options: float) -> np.ndarray:
    weights = SimulatedBinaryCrossover(**options).weights(count, dim, stream(backend), backend)
    first, second = backend.asarray(np.full((count, dim), a)), backend.asarray(np.full((count, dim), b))
    return backend.to_numpy(mix_genes(weights, first, second)).astype(np.float64)


# --- SBX ---


def test_sbx_weights_are_arrays_of_the_backend_and_may_leave_the_unit_interval(backend: Backend):
    weights = SimulatedBinaryCrossover(variable_probability=1.0).weights(5000, 4, stream(backend), backend)
    assert_on_backend(weights, backend)
    assert tuple(weights.shape) == (5000, 4)
    host = backend.to_numpy(weights).astype(np.float64)
    assert host.min() < 0 and host.max() > 1  # extrapolation: how SBX goes beyond its parents
    assert np.isfinite(host).all()


def test_children_are_centred_on_the_midpoint_of_their_parents(backend: Backend):
    """For crossed genes the child is `mid ± beta/2 · |a − b|` with a fair coin for the sign: symmetric around the midpoint."""
    child = children(backend, 0.2, 0.6, variable_probability=1.0)[:, 0]
    assert abs(child.mean() - 0.4) < 0.003
    centred = child - 0.4
    assert abs((centred > 0).mean() - 0.5) < 0.006
    assert abs(np.mean(centred**3)) < 5e-4  # no skew
    # the quantiles of the positive and the negative side agree
    np.testing.assert_allclose(
        np.quantile(centred[centred > 0], [0.25, 0.5, 0.75]), -np.quantile(centred[centred < 0], [0.75, 0.5, 0.25]), rtol=0.03
    )


@pytest.mark.parametrize("eta", [2.0, 15.0])
def test_the_spread_follows_the_distribution_index(backend: Backend, eta: float):
    """`beta = |child − mid| / (|a − b| / 2)` has P(beta <= t) = t^(eta+1) / 2 below 1 and 1 − 1 / (2 t^(eta+1)) above it."""
    child = children(backend, 0.2, 0.6, variable_probability=1.0, eta=eta)[:, 0]
    beta = np.abs(child - 0.4) / 0.2
    for t in (0.5, 0.9, 1.0, 1.2, 2.0):
        expected = t ** (eta + 1) / 2 if t <= 1 else 1 - 1 / (2 * t ** (eta + 1))
        assert abs((beta <= t).mean() - expected) < 0.006, (t, expected)


def test_a_larger_index_keeps_children_closer_to_their_parents(backend: Backend):
    spreads = [children(backend, 0.2, 0.6, variable_probability=1.0, eta=eta)[:, 0].std() for eta in (1.0, 5.0, 15.0, 50.0)]
    assert spreads == sorted(spreads, reverse=True)


def test_the_probability_of_crossing_a_gene_is_respected_and_uncrossed_genes_come_from_one_parent(backend: Backend):
    a, b = (float(np.asarray(v, dtype=backend.precision)) for v in (0.2, 0.6))  # the parents' genes, as floats of the run
    child = children(backend, a, b, count=20_000, dim=40, variable_probability=0.3)
    from_a, from_b = child == a, child == b
    crossed = ~(from_a | from_b)
    assert abs(crossed.mean() - 0.3) < 0.004
    # the genes that were not crossed all come from the same parent within a child
    uncrossed_from_a = np.where(crossed, np.nan, from_a.astype(float))
    per_child = np.nanmean(uncrossed_from_a, axis=1)
    decided = ~np.isnan(per_child)
    # (a crossed gene whose weight rounds to exactly 0 or 1 in float32 looks uncrossed: a handful of rows at most)
    assert (~np.isin(per_child[decided], [0.0, 1.0])).sum() <= 5
    assert abs(per_child[decided].mean() - 0.5) < 0.02
    never = children(backend, a, b, count=2000, dim=10, variable_probability=0.0)
    assert np.isin(never, [a, b]).all()


def test_children_are_inside_the_bounds_after_the_bounds_repair(backend: Backend):
    box = Box(0.0, 1.0, dim=30)
    rng = stream(backend, 4)
    parents_a, parents_b = box.sample_genomes(3000, rng, backend), box.sample_genomes(3000, rng, backend)
    weights = SimulatedBinaryCrossover().weights(3000, 30, rng, backend)
    raw = mix_genes(weights, parents_a, parents_b)
    assert float(backend.to_numpy(raw).min()) < 0 or float(backend.to_numpy(raw).max()) > 1  # SBX does extrapolate
    repaired = backend.to_numpy(box.clip(raw)).astype(np.float64)
    assert (repaired >= 0).all() and (repaired <= 1).all()
    assert all(box.contains(box.clip(raw)[i]) for i in range(0, 3000, 331))


def test_sbx_validation():
    with pytest.raises(ValueError, match="eta"):
        SimulatedBinaryCrossover(eta=-1.0)
    with pytest.raises(ValueError, match="variable_probability"):
        SimulatedBinaryCrossover(variable_probability=1.5)
    assert "eta=15.0" in repr(SimulatedBinaryCrossover())


# --- polynomial mutation ---


def mutate(backend: Backend, start: float, *, eta: float = 20.0, probability: float | None = 1.0, count: int = N, dim: int = 1, seed=1):
    operator = with_bounds(PolynomialMutation(eta, probability), backend.asarray(np.zeros(dim)), backend.asarray(np.ones(dim)))
    genomes = backend.asarray(np.full((count, dim), start))
    mutated, steps = operator.mutate(genomes, None, backend.asarray(np.ones(dim)), stream(backend, seed), backend)
    assert steps is None
    assert_on_backend(mutated, backend)
    return backend.to_numpy(mutated).astype(np.float64)


def test_the_mutated_genes_never_leave_the_bounds_and_need_no_clipping(backend: Backend):
    for start in (0.0, 1e-6, 0.3, 0.999999, 1.0):
        out = mutate(backend, start, count=20_000)
        assert (out >= 0).all() and (out <= 1).all() and np.isfinite(out).all()


@pytest.mark.parametrize("eta", [5.0, 20.0, 50.0])
def test_the_perturbation_matches_the_distribution_index(backend: Backend, eta: float):
    """Away from the bounds the perturbation has the polynomial density (eta + 1) / 2 · (1 − |d|)^eta: E|d| = 1 / (eta + 2)
    and Var d = 2 / ((eta + 2)(eta + 3)), for a box of width 1."""
    delta = mutate(backend, 0.5, eta=eta)[:, 0] - 0.5
    if eta == 5.0:  # the density reaches the bounds from the middle: compare with the truncated density instead
        assert abs(delta.mean()) < 0.01 and 0.1 < np.abs(delta).mean() < 0.15
        return
    assert abs(delta.mean()) < 0.002
    assert abs(np.abs(delta).mean() - 1 / (eta + 2)) < 0.002
    assert abs(delta.var() - 2 / ((eta + 2) * (eta + 3))) < 0.0004


def test_a_gene_next_to_a_bound_moves_into_the_box(backend: Backend):
    low, high = mutate(backend, 0.0)[:, 0], mutate(backend, 1.0)[:, 0]
    assert (low >= 0).all() and (high <= 1).all()
    # the density is built from the distance to each wall: a gene on the wall has no room to move out, so the half of the
    # draws that point outwards leave it where it is, and the other half move it in
    assert abs((low == 0).mean() - 0.5) < 0.01 and abs((high == 1).mean() - 0.5) < 0.01
    assert (low[low > 0].mean() > 0.01) and (high[high < 1].mean() < 0.99)


def test_the_mutation_probability_is_per_gene_and_defaults_to_one_over_the_number_of_genes(backend: Backend):
    out = mutate(backend, 0.5, probability=0.25, count=20_000, dim=8)
    assert abs((out != 0.5).mean() - 0.25) < 0.005
    default = mutate(backend, 0.5, probability=None, count=20_000, dim=10)
    assert abs((default != 0.5).mean() - 0.1) < 0.004  # 1/d with d = 10
    assert (mutate(backend, 0.5, probability=1.0, count=2000) != 0.5).all()


def test_polynomial_mutation_is_reproducible_and_validated(backend: Backend):
    np.testing.assert_array_equal(mutate(backend, 0.4, count=1000), mutate(backend, 0.4, count=1000))
    assert not np.array_equal(mutate(backend, 0.4, count=1000), mutate(backend, 0.4, count=1000, seed=2))
    with pytest.raises(ValueError, match="eta"):
        PolynomialMutation(eta=-1.0)
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError, match="probability"):
            PolynomialMutation(probability=bad)
    operator = PolynomialMutation()
    assert not operator.adaptive and operator.min_step is None and operator.initial_steps(3, 2, backend) is None
    with pytest.raises(RuntimeError, match="bounds"):
        operator.mutate(backend.asarray(np.zeros((2, 2))), None, backend.asarray(np.ones(2)), stream(backend), backend)
    assert with_bounds(GaussianMutation(0.1), backend.asarray(np.zeros(2)), backend.asarray(np.ones(2))).name == "gaussian"


# --- in the genetic algorithm ---


def test_the_genetic_algorithm_can_use_them_and_keeps_its_own_defaults(backend: Backend):
    sphere = auxein.VectorisedEvaluator(lambda X: backend.xp.sum(X * X, axis=1))
    result = auxein.run(
        strategy=auxein.GeneticAlgorithm(
            population_size=60, offspring_size=60, recombination=SimulatedBinaryCrossover(), mutation=PolynomialMutation()
        ),
        evaluator=sphere,
        space=Box(0.0, 1.0, dim=10),
        budget=auxein.Budget(evaluations=12_000),
        seed=2,
        backend=backend,
        batch_size=60,
    )
    assert result.best is not None and result.best.objectives["value"] < 0.01
    assert_on_backend(result.best.candidate.genome, backend)
    assert "SimulatedBinaryCrossover" in str(auxein.GeneticAlgorithm(recombination=SimulatedBinaryCrossover()))
    defaults = auxein.GeneticAlgorithm()
    assert defaults.recombination.name == "intermediate" and defaults.mutation.name == "self_adaptive"


def test_on_a_mixed_space_they_apply_to_the_real_genes_only(backend: Backend):
    from auxein.spaces import Binary, Integer, MixedSpace, Real

    space = MixedSpace({"x": Real(0.0, 1.0), "n": Integer(0, 9), "b": Binary(), "y": Real(0.0, 1.0)})
    result = auxein.run(
        strategy=auxein.GeneticAlgorithm(
            population_size=40, offspring_size=40, recombination=SimulatedBinaryCrossover(), mutation=PolynomialMutation()
        ),
        evaluator=auxein.VectorisedEvaluator(lambda X: X[:, 0] ** 2 + X[:, 3] ** 2 + (X[:, 1] - 3.0) ** 2 * 0.1 + (1.0 - X[:, 2])),
        space=space,
        budget=auxein.Budget(evaluations=6000),
        seed=3,
        backend=backend,
        batch_size=40,
    )
    assert result.best is not None and result.best.objectives["value"] < 0.05
    assert space.contains(result.best.candidate.genome)
    assert "polynomial" in result.best.candidate.origin and "sbx" in result.best.candidate.origin or result.best.candidate.origin == "init"
