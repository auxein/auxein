import numpy as np

from auxein import Genotype
from auxein.population import build_individual
from auxein.mutations import Uniform, FixedVariance, SelfAdaptiveSingleStep


def test_uniform_mutate_one_gene():
    genotype = Genotype(np.zeros(5), np.zeros(5))
    mutation_function = Uniform(10000, 20000)
    mutated_genotype = mutation_function.mutate(genotype)
    assert genotype.dimension == mutated_genotype.dimension
    assert np.count_nonzero(genotype.dna == mutated_genotype.dna) == 4


def test_non_uniform_fixed_variance():
    genotype = Genotype(np.zeros(5), np.zeros(5))
    mutation_function = FixedVariance(1000)
    mutated_genotype = mutation_function.mutate(genotype)
    assert genotype.dimension == mutated_genotype.dimension
    assert np.count_nonzero(genotype.dna == mutated_genotype.dna) == 0


def test_uncorrelated_with_single_step_variance():
    genotype = Genotype(np.array([0.0, 0.0, 0.0, 0.0, 0.0]), np.ones(5))
    mutation_function = SelfAdaptiveSingleStep(0.05)
    mutated_genotype = mutation_function.mutate(genotype)
    assert genotype.dimension == mutated_genotype.dimension
    assert np.count_nonzero(genotype.mask == mutated_genotype.mask) == 0
    assert np.count_nonzero(genotype.dna == mutated_genotype.dna) == 0
    assert np.unique(mutated_genotype.mask).size == 1
    assert np.unique(mutated_genotype.dna).size != 1


def test_self_adaptive_single_step_on_individual_built_with_default_mask():
    individual = build_individual([0.0, 0.0, 0.0])
    mutated = individual.mutate(SelfAdaptiveSingleStep(0.05))
    assert mutated.genotype.dimension == 3
    assert mutated.genotype.mask.shape == (3,)
    assert np.unique(mutated.genotype.mask).size == 1


def test_self_adaptive_single_step_keeps_a_single_shared_step_size_when_extending():
    genotype = Genotype(np.zeros(3), np.full(3, 0.5))
    mutated = SelfAdaptiveSingleStep(0.05, extend_probability=1.0).mutate(genotype)
    assert mutated.dimension == 4
    assert mutated.mask.shape == (4,)
    assert np.unique(mutated.mask).size == 1


def test_fixed_variance_on_empty_genotype():
    mutated = FixedVariance(1.0).mutate(Genotype(np.array([]), np.array([])))
    assert mutated.dimension == 0
