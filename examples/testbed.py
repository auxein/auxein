from auxein.playgrounds import Static
from auxein.mutations import Uniform
from auxein.recombinations import SimpleArithmetic

population = None
playground = Static(
    population = population,
    mutation = Uniform(lower_bound = -0.5, upper_bound = 0.5),
    distribution = None,
    selection = None,
    recombination = SimpleArithmetic(alpha = 0.5)
)

playground.train([], [], max_generations=200, validation=([], []))
y_pred = playground.predict([])
print(y_pred)