import numpy as np

from auxein.fitness.kernel_based import GlobalMinimum
from auxein.population import build_individual


def test_global_minimum_fitness_is_minus_kernel():
    def kernel(x):
        return np.sum((x - 10) ** 2)

    individual = build_individual([13.0, 6.0])
    fitness = GlobalMinimum(kernel)
    assert kernel(individual.genotype.dna) == 25.0
    assert fitness.fitness(individual) == -kernel(individual.genotype.dna)


def test_global_minimum_value_is_kernel():
    def kernel(x):
        return np.sum((x - 10) ** 2)

    individual = build_individual([13.0, 6.0])
    fitness = GlobalMinimum(kernel)
    x = np.array([7.0, 10.0])
    assert kernel(x) == 9.0
    assert fitness.value(individual, x) == kernel(x)
