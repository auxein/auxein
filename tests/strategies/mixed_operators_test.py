"""The operators for integer, binary and categorical genes, and the glue that applies them to a mixed genome."""

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.random import RunSeed
from auxein.spaces import Binary, Categorical, Integer, MixedSpace, Real
from auxein.strategies.ga import (
    BitFlipMutation,
    CategoricalMutation,
    GaussianMutation,
    IntegerMutation,
    IntermediateRecombination,
    MixedVariation,
    NoRecombination,
    SelfAdaptiveMutation,
    UniformRecombination,
)
from auxein.strategies.ga.base import PopulationView  # noqa: F401  (documents where the operators plug in)
from auxein.strategies.ga.repair import ClipRepair
from tests.support.fixtures import assert_on_backend

N = 40_000


def stream(backend: Backend, seed: int = 1):
    return RunSeed(seed).stream("strategy", backend=backend)


def host(backend: Backend, array) -> np.ndarray:
    return backend.to_numpy(array).astype(np.float64)


# --- integer mutation ---


def integer_step(backend: Backend, mutation: IntegerMutation, start: float = 10.0, bounds=(0.0, 20.0), seed: int = 1, k: int = 1):
    values = backend.asarray(np.full((N, k), start))
    steps = mutation.initial_steps(N, backend)
    lower, upper = backend.asarray(np.full(k, bounds[0])), backend.asarray(np.full(k, bounds[1]))
    mutated, new_steps = mutation.mutate(values, steps, lower, upper, stream(backend, seed), backend)
    return host(backend, mutated) - start, new_steps


def test_integer_mutation_gives_integral_in_bounds_values_on_every_backend(backend: Backend):
    values = backend.asarray(np.tile(np.array([0.0, 3.0, 20.0, 7.0]), (N, 1)))
    lower, upper = backend.asarray(np.zeros(4)), backend.asarray(np.full(4, 20.0))
    mutation = IntegerMutation(probability=1.0, initial_step=5.0)
    mutated, steps = mutation.mutate(values, mutation.initial_steps(N, backend), lower, upper, stream(backend), backend)
    assert_on_backend(mutated, backend)
    assert steps is not None
    assert_on_backend(steps, backend)
    out = host(backend, mutated)
    assert (out == np.rint(out)).all() and out.min() >= 0 and out.max() <= 20
    assert (out[:, 0] >= 0).all() and out[:, 0].max() > 5 and out[:, 2].min() < 15  # it moves from either bound, but only inwards


def test_integer_steps_are_symmetric(backend: Backend):
    diff, _ = integer_step(backend, IntegerMutation(probability=1.0, adaptive=False, initial_step=2.0), bounds=(-1000.0, 1000.0))
    assert abs(diff.mean()) < 0.06 and abs((diff > 0).mean() - (diff < 0).mean()) < 0.01
    values, counts = np.unique(diff, return_counts=True)
    by_value = dict(zip(values.tolist(), counts.tolist(), strict=True))
    for step in (1, 2, 3):  # +s and -s are equally likely
        assert abs(by_value[float(step)] - by_value[float(-step)]) < 5 * np.sqrt(by_value[float(step)])


@pytest.mark.parametrize("mean", [0.2, 1.0, 3.0])
def test_the_difference_of_two_geometric_variables_has_the_stated_distribution(backend: Backend, mean: float):
    """Each variable has mean m, so P(0) = 1 / (1 + 2m) and the variance of the difference is 2·m·(1 + m)."""
    diff, _ = integer_step(backend, IntegerMutation(probability=1.0, adaptive=False, initial_step=mean), bounds=(-1e5, 1e5))
    assert abs((diff == 0).mean() - 1.0 / (1.0 + 2.0 * mean)) < 0.01
    assert abs(diff.var() - 2.0 * mean * (1.0 + mean)) < 0.1 * 2.0 * mean * (1.0 + mean)


def test_the_mutation_probability_is_per_gene_and_defaults_to_one_over_k(backend: Backend):
    mutation = IntegerMutation(probability=0.25, adaptive=False, initial_step=50.0)  # a huge step: a mutated gene almost surely moves
    diff, _ = integer_step(backend, mutation, bounds=(-1e5, 1e5), k=4)
    assert abs((diff != 0).mean() - 0.25) < 0.01
    diff, _ = integer_step(backend, IntegerMutation(adaptive=False, initial_step=50.0), bounds=(-1e5, 1e5), k=5)
    assert abs((diff != 0).mean() - 0.2) < 0.01  # 1/5


def test_adapted_mean_steps_stay_between_their_floor_and_the_range_over_many_generations(backend: Backend):
    mutation = IntegerMutation(initial_step=1.0, min_step=0.25)
    rng = stream(backend)
    values = backend.asarray(np.full((200, 3), 10.0))
    lower, upper = backend.asarray(np.zeros(3)), backend.asarray(np.full(3, 20.0))
    steps = mutation.initial_steps(200, backend)
    for _ in range(400):
        values, steps = mutation.mutate(values, steps, lower, upper, rng, backend)
    out = host(backend, steps)
    assert np.isfinite(out).all() and out.min() >= 0.25 * (1 - 1e-6) and out.max() <= 20.0 * (1 + 1e-6)
    assert (out == 0.25).any() or out.min() < 0.3  # selection-free drift does reach the floor, and stays on it
    moved = host(backend, values)
    assert (moved == np.rint(moved)).all() and moved.min() >= 0 and moved.max() <= 20


def test_a_gene_can_always_still_change_because_the_step_has_a_floor(backend: Backend):
    mutation = IntegerMutation(probability=1.0, initial_step=0.1, min_step=0.1)
    diff, steps = integer_step(backend, mutation, bounds=(-1e5, 1e5))
    assert host(backend, steps).min() >= 0.1 * (1 - 1e-6)
    assert (diff != 0).mean() > 0.1  # at the floor, 2m / (1 + 2m) = 0.17 of the mutated genes still move


def test_integer_mutation_validation():
    with pytest.raises(ValueError, match="min_step"):
        IntegerMutation(min_step=0.0)
    with pytest.raises(ValueError, match="min_step"):
        IntegerMutation(initial_step=0.5, min_step=1.0)
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="probability"):
            IntegerMutation(probability=bad)
        with pytest.raises(ValueError, match="probability"):
            BitFlipMutation(bad)
        with pytest.raises(ValueError, match="probability"):
            CategoricalMutation(bad)


def test_a_fixed_integer_step_has_no_strategy_state(backend: Backend):
    mutation = IntegerMutation(adaptive=False)
    assert mutation.initial_steps(5, backend) is None
    _, steps = integer_step(backend, mutation)
    assert steps is None


# --- binary and categorical mutation ---


def test_bit_flips_respect_their_probability(backend: Backend):
    values = backend.asarray(np.tile(np.array([0.0, 1.0, 0.0, 1.0]), (N, 1)))
    zeros, ones = backend.asarray(np.zeros(4)), backend.asarray(np.ones(4))
    flipped = host(backend, BitFlipMutation(0.1).mutate(values, zeros, ones, stream(backend), backend))
    assert set(np.unique(flipped).tolist()) == {0.0, 1.0}
    assert abs((flipped != host(backend, values)).mean() - 0.1) < 0.005
    default = host(backend, BitFlipMutation().mutate(values, zeros, ones, stream(backend, 2), backend))
    assert abs((default != host(backend, values)).mean() - 0.25) < 0.005  # 1/4


def test_a_single_bit_does_not_flip_every_time_by_default(backend: Backend):
    """With one binary gene the rate 1/k would be 1, and no child could keep its parent's value: the default is at most a half."""
    values = backend.asarray(np.ones((N, 1)))
    flipped = host(backend, BitFlipMutation().mutate(values, backend.asarray([0.0]), backend.asarray([1.0]), stream(backend), backend))
    assert abs((flipped == 0).mean() - 0.5) < 0.01


def test_categorical_mutation_changes_to_a_different_category_uniformly_and_at_its_rate(backend: Backend):
    choices = np.array([4.0, 2.0, 7.0])  # 5, 3 and 8 categories
    values = backend.asarray(np.tile(np.array([1.0, 0.0, 7.0]), (N, 1)))
    zeros, upper = backend.asarray(np.zeros(3)), backend.asarray(choices)
    mutated = host(backend, CategoricalMutation(probability=0.4).mutate(values, zeros, upper, stream(backend), backend))
    before = host(backend, values)
    changed = mutated != before
    assert abs(changed.mean() - 0.4) < 0.01
    assert (mutated == np.rint(mutated)).all() and (mutated >= 0).all() and (mutated <= choices).all()
    for column, start in enumerate((1.0, 0.0, 7.0)):
        new = mutated[changed[:, column], column]
        assert (new != start).all()  # never "mutates" to the category it already has
        _, counts = np.unique(new, return_counts=True)
        assert len(counts) == int(choices[column])  # every other category
        assert counts.max() < 1.12 * counts.min()  # and equally often


# --- the glue, on a mixed genome ---

SPACE = MixedSpace(
    {
        "a": Real(-2.0, 2.0),
        "n": Integer(0, 9),
        "b": Binary(),
        "c": Categorical(["x", "y", "z"]),
        "m": Integer(-3, 3),
        "d": Binary(),
        "r": Real(0.01, 10.0, log=True),
    }
)


def variation(backend: Backend, space: MixedSpace = SPACE, **kwargs) -> MixedVariation:
    options = {
        "real_mutation": SelfAdaptiveMutation(),
        "real_recombination": IntermediateRecombination(),
        "integer_mutation": IntegerMutation(),
        "binary_mutation": BitFlipMutation(),
        "categorical_mutation": CategoricalMutation(),
        "repair": ClipRepair(),
    }
    return MixedVariation(space, backend, **{**options, **kwargs})


def population(backend: Backend, count: int = 4000, seed: int = 1):
    return SPACE.sample_genomes(count, stream(backend, seed), backend)


def test_discrete_recombination_takes_every_discrete_gene_from_one_of_the_parents(backend: Backend):
    v = variation(backend)
    a, b = population(backend, seed=1), population(backend, seed=2)
    weights = v.weights(4000, stream(backend, 3))
    assert weights.shape == (4000, 7)
    w = host(backend, weights)
    discrete = SPACE.kinds != 0
    assert set(np.unique(w[:, discrete]).tolist()) == {0.0, 1.0}
    assert abs(w[:, discrete].mean() - 0.5) < 0.01
    real = ~discrete
    assert ((w[:, real] > 0) & (w[:, real] < 1)).all()  # the real genes are mixed by the real recombination
    child = host(backend, weights * a + (1.0 - weights) * b)
    pa, pb = host(backend, a), host(backend, b)
    from_a, from_b = child[:, discrete] == pa[:, discrete], child[:, discrete] == pb[:, discrete]
    assert (from_a | from_b).all()  # exactly a parent's value, never a blend
    assert from_a.mean() > 0.5 and from_b.mean() > 0.5
    assert SPACE.contains(backend.asarray(child[0]))


def test_an_asexual_recombination_copies_the_first_parent(backend: Backend):
    v = variation(backend, real_recombination=NoRecombination())
    weights = host(backend, v.weights(10, stream(backend)))
    assert (weights == 1.0).all()
    assert v.recombination_name == "none/discrete"


def test_a_uniform_real_recombination_is_used_for_the_real_genes_too(backend: Backend):
    w = host(backend, variation(backend, real_recombination=UniformRecombination()).weights(3000, stream(backend)))
    assert set(np.unique(w).tolist()) == {0.0, 1.0}


def test_mutation_and_repair_keep_every_genome_valid(backend: Backend):
    v = variation(backend)
    rng = stream(backend)
    genomes = population(backend, 2000)
    steps = v.initial_steps(2000)
    assert steps is not None and steps.shape == (2000, 2)  # one real step size, one integer mean step
    for _ in range(60):
        genomes, steps = v.mutate(genomes, steps, rng)
        genomes = v.repair(genomes)
        assert_on_backend(genomes, backend)
        assert steps is not None
        assert_on_backend(steps, backend)
    out = host(backend, genomes)
    assert np.isfinite(out).all() and ((out >= SPACE.lower) & (out <= SPACE.upper)).all()
    discrete = SPACE.kinds != 0
    assert (out[:, discrete] == np.rint(out[:, discrete])).all()
    assert all(SPACE.contains(genomes[i]) for i in range(0, 2000, 97))


def test_each_type_mutates_with_its_own_operator(backend: Backend):
    v = variation(backend, integer_mutation=IntegerMutation(probability=1.0, initial_step=3.0, adaptive=False),
                  binary_mutation=BitFlipMutation(probability=1.0), categorical_mutation=CategoricalMutation(probability=1.0),
                  real_mutation=GaussianMutation(0.1))  # fmt: skip
    genomes = population(backend, 3000)
    mutated, steps = v.mutate(genomes, None, stream(backend))
    assert steps is None and v.initial_steps(5) is None  # nothing is adapted: no strategy state
    before, after = host(backend, genomes), host(backend, mutated)
    assert (after[:, 2] != before[:, 2]).all() and (after[:, 5] != before[:, 5]).all()  # every bit flipped
    assert (after[:, 3] != before[:, 3]).all()  # every category replaced by another
    assert (after[:, 0] != before[:, 0]).all() and (after[:, 6] != before[:, 6]).all()  # and every real gene moved
    assert 0.5 < (after[:, 1] != before[:, 1]).mean() < 1.0


def test_the_packed_strategy_parameters_follow_the_real_operator(backend: Backend):
    per_gene = variation(backend, real_mutation=SelfAdaptiveMutation(per_gene=True))
    steps = per_gene.initial_steps(6)
    assert steps is not None and steps.shape == (6, 3)  # two real genes... and the integer step
    assert variation(backend, integer_mutation=IntegerMutation(adaptive=False)).initial_steps(6).shape == (6, 1)  # type: ignore[union-attr]
    no_real = MixedSpace({"n": Integer(0, 5), "b": Binary()})
    only_discrete = variation(backend, no_real, integer_mutation=IntegerMutation(adaptive=False))
    assert only_discrete.initial_steps(3) is None and not only_discrete.adaptive


def test_inherited_step_sizes_are_the_weighted_geometric_mean_of_the_parents(backend: Backend):
    v = variation(backend)
    first = backend.asarray(np.tile(np.array([0.04, 4.0]), (3, 1)))
    second = backend.asarray(np.tile(np.array([0.01, 1.0]), (3, 1)))
    weights = backend.asarray(np.ones((3, 7)) * 0.5)
    mixed = host(backend, v.mix_steps(weights, first, second))
    np.testing.assert_allclose(mixed, [[0.02, 2.0]] * 3, rtol=1e-5)  # the geometric mean when the weights are equal
    copy = host(backend, v.mix_steps(backend.asarray(np.ones((3, 1))), first, second))
    np.testing.assert_allclose(copy, host(backend, first), rtol=1e-6)  # weights of one: the first parent's, exactly


def test_the_floors_of_the_packed_parameters_tell_when_everything_has_stopped_adapting(backend: Backend):
    v = variation(backend, real_mutation=SelfAdaptiveMutation(min_step=1e-6), integer_mutation=IntegerMutation(min_step=0.2))
    floors = host(backend, v.floors())  # type: ignore[arg-type]
    np.testing.assert_allclose(floors, [1e-6, 0.2], rtol=1e-6)


def test_origins_name_the_operators_by_type(backend: Backend):
    assert variation(backend).mutation_name == "self_adaptive/geometric/bitflip/resample"
    assert variation(backend).recombination_name == "intermediate/discrete"
    only_real_and_bits = MixedSpace({"x": Real(0.0, 1.0), "b": Binary()})
    assert variation(backend, only_real_and_bits).mutation_name == "self_adaptive/bitflip"
    assert variation(backend, MixedSpace({"b": Binary(), "c": Binary()})).recombination_name == "discrete"


def test_a_space_without_real_genes_has_no_real_operators_and_still_works(backend: Backend):
    space = MixedSpace({"b0": Binary(), "n": Integer(0, 5), "b1": Binary(), "c": Categorical(["x", "y"])})
    v = variation(backend, space)
    genomes = space.sample_genomes(500, stream(backend), backend)
    mutated, steps = v.mutate(genomes, v.initial_steps(500), stream(backend))
    assert v.repair(mutated).shape == (500, 4) and steps is not None and steps.shape == (500, 1)
    assert all(space.contains(mutated[i]) for i in range(0, 500, 11))
