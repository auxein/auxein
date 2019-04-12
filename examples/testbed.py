from auxein.population import build_individual, build_population
from auxein.playgrounds import Static
from auxein.fitness import LinearLeastSquares
from auxein.mutations import Uniform
from auxein.recombinations import SimpleArithmetic
from auxein.parents.distributions import SigmaScaling
from auxein.parents.selections import StochasticUniversalSampling
from auxein.replacements import ReplaceWorst

import numpy as np


size = 100
x = np.arange(size)
delta = np.random.uniform(-5,5, size=(size,))
y = .4*x + 3 + delta

print(x)
print(y)

fitness_function = LinearLeastSquares(x.reshape(size, 1), y)
population = build_population(2, 10, fitness_function)
playground = Static(
    population = population,
    fitness = fitness_function,
    mutation = Uniform(lower_bound = -0.5, upper_bound = 0.5),
    distribution = SigmaScaling(),
    selection = StochasticUniversalSampling(offspring_size = 2),
    recombination = SimpleArithmetic(alpha = 0.5),
    replacement = ReplaceWorst(offspring_size = 2)
)
playground.train(2)
print(playground.predict(0))

# playground.train([], [], max_generations=200, validation=([], []))
# y_pred = playground.predict([])
# print(y_pred)