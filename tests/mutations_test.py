import numpy as np

from auxein import Genotype
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
