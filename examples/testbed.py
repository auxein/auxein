from auxein.population import build_fixed_dimension_population
from auxein.playgrounds import Static
from auxein.fitness import LinearLeastSquares
from auxein.mutations import SelfAdaptiveSingleStep
from auxein.recombinations import SimpleArithmetic
from auxein.parents.distributions import SigmaScaling
from auxein.parents.selections import StochasticUniversalSampling
from auxein.replacements import ReplaceWorst

import numpy as np


size = 100
x = np.arange(size)
delta = np.random.uniform(-5, 5, size=(size, ))
y = .4 * x + 3 + delta

fitness_function = LinearLeastSquares(x.reshape(size, 1), y)
population = build_fixed_dimension_population(2, 100, fitness_function)
playground = Static(
    population=population,
    fitness=fitness_function,
    mutation=SelfAdaptiveSingleStep(0.05),
    distribution=SigmaScaling(),
    selection=StochasticUniversalSampling(offspring_size=2),
    recombination=SimpleArithmetic(alpha=0.5),
    replacement=ReplaceWorst(offspring_size=2)
)
playground.train(10)
print(playground.predict(0))
print(playground.predict(10))
print(playground.predict(50))
print(playground.predict(99))
