from auxein.population import build_population
from auxein.playgrounds import Static
from auxein.fitness import LinearLeastSquares
from auxein.mutations import Uniform
from auxein.recombinations import SimpleArithmetic
from auxein.parents.distributions import SigmaScaling
from auxein.parents.selections import StochasticUniversalSampling
from auxein.replacements import ReplaceWorst

fitness_function = LinearLeastSquares([], [])
population = build_population(2, 100, fitness_function)
playground = Static(
    population = population,
    fitness = fitness_function,
    mutation = Uniform(lower_bound = -0.5, upper_bound = 0.5),
    distribution = SigmaScaling(),
    selection = StochasticUniversalSampling(offspring_size = 2),
    recombination = SimpleArithmetic(alpha = 0.5),
    replacement = ReplaceWorst(offspring_size = 2)
)

playground.train([], [], max_generations=200, validation=([], []))
y_pred = playground.predict([])
print(y_pred)