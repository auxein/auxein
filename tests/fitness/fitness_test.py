import numpy as np

from auxein.population import build_individual
from auxein.fitness import Fitness, MultipleLinearRegression, MaximumLikelihood


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


# Classic example with students and time spent studying
# from: https://en.wikipedia.org/wiki/Logistic_regression
def test_maximum_likelihood_value():
    xs = np.array([[0.50], [0.75], [1.00], [1.25], [1.50], [1.75], [1.75], [2.00], [2.25], [2.50], [2.75], [3.00], [3.25], [3.50], [4.00], [4.25], [4.50], [4.75], [5.00], [5.50]])
    y = np.array([0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    fitness_function = MaximumLikelihood(xs, y)

    i = build_individual([-4.0777, 1.5046])

    assert np.isclose(fitness_function.value(i, [1]), 0.07, atol=0.01)
    assert np.isclose(fitness_function.value(i, [2]), 0.26, atol=0.01)
    assert np.isclose(fitness_function.value(i, [3]), 0.61, atol=0.01)
    assert np.isclose(fitness_function.value(i, [4]), 0.87, atol=0.01)
    assert np.isclose(fitness_function.value(i, [5]), 0.97, atol=0.01)
