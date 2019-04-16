import numpy as np

from auxein.population import build_individual
from auxein.fitness import LinearLeastSquares


def test_linear_least_squares():
    xs = np.array([[23], [26], [30], [34], [43], [48], [52], [57], [58]])
    y = np.array([651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518])

    i = build_individual([23.42, 167.68], [])
    fitness_function = LinearLeastSquares(xs, y)
    assert np.isclose(fitness_function.fitness(i), -18804)
