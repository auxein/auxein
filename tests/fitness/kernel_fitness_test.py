import numpy as np

from auxein.population import build_individual
from auxein.fitness.core import Fitness
from auxein.fitness.kernel_based import GlobalMinimum


def test_global_minimum_fitness_value_and_fitness():
    # for this fitness function the value and the fitness
    # are always the same but with opposite sign
    def kernel(x):
        return (x - 10)**2

    individual = build_individual([10])
    fitness = GlobalMinimum(kernel)
    assert fitness.value(0, 10) == fitness.fitness(individual)
