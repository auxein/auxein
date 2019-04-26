import numpy as np

from auxein.population import build_individual
from auxein.fitness import Fitness, MultipleLinearRegression


def test_multiple_linear_regression():
    xs = np.array([[23], [26], [30], [34], [43], [48], [52], [57], [58]])
    y = np.array([651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518])

    i = build_individual([23.42, 167.68], [])
    fitness_function = MultipleLinearRegression(xs, y)
    assert np.isclose(fitness_function.fitness(i), -18804)


def test_fitness_landscape():
    class TestFitnessFunction(Fitness):
        def fitness(self, individual):
            return individual.genotype.dna[0] + individual.genotype.dna[1]

        def value(self, individual, x):
            pass

    fitness_function = TestFitnessFunction()
    landscape = fitness_function.get_landscape([[-1, 1], [0, 1]], 3)
    assert len(landscape) == 9
    for e in landscape:
        assert len(e) == 3

    expected = [[-1, 0, -1], [0, 0, 0], [1, 0, 1], [-1, 0.5, -0.5], [0, 0.5, 0.5], [1, 0.5, 1.5], [-1, 1, 0], [0, 1, 1], [1, 1, 2]]
    assert np.array_equal(landscape, expected)
