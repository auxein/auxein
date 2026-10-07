import pytest
import numpy as np

from auxein.fitness import Fitness
from auxein.population import build_individual, Population
from auxein.parents.distributions import Fps, FpsWithWindowing, SigmaScaling


def init_population(dimension, size, fitness_function):
    population = Population()
    for _ in range(0, size):
        dna = np.random.uniform(-1, 1, dimension)
        i = build_individual(dna)
        population.add(i, fitness_function.fitness(i))
    return population


def build_fully_specified_population():
    population = Population()
    population.add(build_individual([0.1, 0.9], id="3adee626-de78-4f83-84f9-ebde4e8ee64d"), 1.0)  # fitness = 1
    population.add(build_individual([0.1, 0.5], id="e2ee1fd8-7bb9-4556-9435-cd012b0f5403"), 0.6)  # fitness = 0.6
    population.add(build_individual([0.1, 0.1], id="01f4eadc-e799-42d1-bc18-0fd85159bfb6"), 0.2)  # fitness = 0.2
    return population


def test_fps_constant_fitness_function():
    class TestFitnessFunction(Fitness):
        def fitness(self, individual):
            return 1.0

        def value(self, individual, x):
            pass

    population = init_population(2, 5, TestFitnessFunction())
    distribution = Fps().get(population)
    assert len(distribution) == 5
    assert all(d[1] == 0.2 for d in distribution)
    distribution_sum = sum(d[1] for d in distribution)
    assert np.isclose(distribution_sum, 1)


def test_fps_non_constant_fitness_function():
    class TestFitnessFunction(Fitness):
        def fitness(self, individual):
            # dna genes are in [-1, 1], so shift to keep the fitness non-negative
            return individual.genotype.dna[0] + individual.genotype.dna[1] + 2

        def value(self, individual, x):
            pass

    population = init_population(2, 5, TestFitnessFunction())
    distribution = Fps().get(population)
    assert len(distribution) == 5
    distribution_sum = sum(d[1] for d in distribution)
    assert np.isclose(distribution_sum, 1)


def test_fps_known_fitness_function():
    from itertools import cycle

    ff_known_values = cycle([0.5, 1, 1.5, 2, 2.5])

    class TestFitnessFunction(Fitness):
        def fitness(self, individual):
            return next(ff_known_values)

        def value(self, individual, x):
            pass

    population = init_population(2, 5, TestFitnessFunction())
    distribution = Fps().get(population)
    assert len(distribution) == 5
    distribution_sum = sum(d[1] for d in distribution)
    assert np.isclose(distribution_sum, 1)

    distribution_values = list(map(lambda d: d[1], distribution))
    assert np.allclose(np.array(distribution_values), np.array([0.0666, 0.1333, 0.2, 0.2666, 0.333]), rtol=0.001, atol=0.001)


def test_fps_windowing_with_known_fitness_function():
    from itertools import cycle

    ff_known_values = cycle([0.5, 1, 1, 1, 10])

    class TestFitnessFunction(Fitness):
        def fitness(self, individual):
            return next(ff_known_values)

        def value(self, individual, x):
            pass

    population = init_population(2, 5, TestFitnessFunction())
    distribution = FpsWithWindowing().get(population)
    assert len(distribution) == 5
    distribution_sum = sum(d[1] for d in distribution)
    assert np.isclose(distribution_sum, 1)

    distribution_values = list(map(lambda d: d[1], distribution))
    assert np.allclose(np.array(distribution_values), np.array([0, 0.045, 0.045, 0.045, 0.863]), rtol=0.001, atol=0.001)


def test_fps_windowing_non_constant_fitness_function():

    class TestFitnessFunction(Fitness):
        def fitness(self, individual):
            return individual.genotype.dna[0] + individual.genotype.dna[1]

        def value(self, individual, x):
            pass

    population = init_population(2, 5, TestFitnessFunction())
    distribution = FpsWithWindowing().get(population)
    distribution_sum = sum(d[1] for d in distribution)
    assert np.isclose(distribution_sum, 1)


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

    distribution_values = list(map(lambda d: d[1], distribution))
    assert np.allclose(np.array(distribution_values), np.array([0.139, 0.153, 0.153, 0.153, 0.399]), rtol=0.001, atol=0.001)


def test_fps_sigma_scaling_with_known_values():
    population = build_fully_specified_population()
    distribution = SigmaScaling().get(population)
    assert len(distribution) == 3
    distribution_sum = sum(d[1] for d in distribution)
    assert np.isclose(distribution_sum, 1)
    assert ("3adee626-de78-4f83-84f9-ebde4e8ee64d", 0.5374574785652648) in distribution

    assert ("e2ee1fd8-7bb9-4556-9435-cd012b0f5403", 0.3333333333333333) in distribution
    assert ("01f4eadc-e799-42d1-bc18-0fd85159bfb6", 0.12920918810140183) in distribution


def build_population_with_fitnesses(fitnesses):
    population = Population()
    for fitness in fitnesses:
        population.add(build_individual([0.0, 0.0]), fitness)
    return population


def test_fps_rejects_negative_fitness():
    population = build_population_with_fitnesses([-1.0, -2.0, -3.0])
    with pytest.raises(ValueError, match="FpsWithWindowing"):
        Fps().get(population)


def test_fps_rejects_zero_total_fitness():
    population = build_population_with_fitnesses([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        Fps().get(population)


@pytest.mark.parametrize("distribution", [FpsWithWindowing(), SigmaScaling()], ids=["windowing", "sigma_scaling"])
def test_distribution_on_converged_population_is_uniform(distribution):
    population = build_population_with_fitnesses([-4.2, -4.2, -4.2, -4.2])
    probabilities = [p for _, p in distribution.get(population)]
    assert np.allclose(probabilities, [0.25] * 4)


def test_sigma_scaling_computes_population_statistics_once(monkeypatch):
    population = build_population_with_fitnesses([1.0, 2.0, 3.0, 4.0, 5.0])
    calls = {"mean": 0, "std": 0}
    mean_fitness, std_fitness = population.mean_fitness, population.std_fitness

    def counting_mean():
        calls["mean"] += 1
        return mean_fitness()

    def counting_std():
        calls["std"] += 1
        return std_fitness()

    monkeypatch.setattr(population, "mean_fitness", counting_mean)
    monkeypatch.setattr(population, "std_fitness", counting_std)
    SigmaScaling().get(population)
    assert calls["mean"] <= 1
    assert calls["std"] <= 1
