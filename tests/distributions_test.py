import numpy as np

from auxein.fitness import Fitness
from auxein.population import build_individual, Population
from auxein.parents.distributions import SigmaScaling


def init_population(dimension, size, fitness_function):
    population = Population()
    for _ in range(0, size):
        dna = np.random.uniform(-1, 1, dimension)
        i = build_individual(dna, [])
        population.add(i, fitness_function.fitness(i))
    return population


def build_fully_specified_population():
    population = Population()
    population.add(build_individual([0.1, 0.9], [], '3adee626-de78-4f83-84f9-ebde4e8ee64d'), 1.0)  # fitness = 1
    population.add(build_individual([0.1, 0.5], [], 'e2ee1fd8-7bb9-4556-9435-cd012b0f5403'), 0.6)  # fitness = 0.6
    population.add(build_individual([0.1, 0.1], [], '01f4eadc-e799-42d1-bc18-0fd85159bfb6'), 0.2)  # fitness = 0.2
    return population


def test_fps_sigma_scaling_with_known_fitness_function():
    from itertools import cycle
    ff_known_values = cycle([0.5, 1, 1, 1, 10])

    class TestFitnessFunction(Fitness):
        def fitness(self, individual):
            return next(ff_known_values)

        def value(self, individual, x):
            pass

    population = init_population(2, 5, TestFitnessFunction())
    distribution = SigmaScaling().get(population)
    assert len(distribution) == 5
    distribution_sum = sum(d[1] for d in distribution)
    assert np.isclose(distribution_sum, 1)

    distribution_values = list(map(lambda d : d[1], distribution))
    assert np.allclose(
        np.array(distribution_values),
        np.array([0.139, 0.153, 0.153, 0.153, 0.399]),
        rtol=0.001, atol=0.001
    )


def test_fps_sigma_scaling_with_known_values():
    population = build_fully_specified_population()
    distribution = SigmaScaling().get(population)
    assert len(distribution) == 3
    distribution_sum = sum(d[1] for d in distribution)
    assert np.isclose(distribution_sum, 1)
    assert (('3adee626-de78-4f83-84f9-ebde4e8ee64d', 0.5374574785652648) in distribution)

    assert (('e2ee1fd8-7bb9-4556-9435-cd012b0f5403', 0.3333333333333333) in distribution)
    assert (('01f4eadc-e799-42d1-bc18-0fd85159bfb6', 0.12920918810140183) in distribution)
