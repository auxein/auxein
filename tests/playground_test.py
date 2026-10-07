import numpy as np

from auxein.fitness.kernel_based import GlobalMinimum
from auxein.mutations import SelfAdaptiveSingleStep
from auxein.parents.distributions import SigmaScaling
from auxein.parents.selections import StochasticUniversalSampling
from auxein.playgrounds import Static
from auxein.population import NormalRandomDnaBuilder, build_fixed_dimension_population
from auxein.recombinations import SimpleArithmetic
from auxein.replacements import ReplaceWorst

OFFSPRING_SIZE = 4
POPULATION_SIZE = 50
MAX_GENERATIONS = 40


def rastrigin(x, a=10):
    return a * len(x) + sum(xi**2 - a * np.cos(2 * np.pi * xi) for xi in x)


def build_playground(pruning_function=None):
    fitness = GlobalMinimum(rastrigin)
    population = build_fixed_dimension_population(2, POPULATION_SIZE, fitness, NormalRandomDnaBuilder(0, 1.5))
    kwargs = {} if pruning_function is None else {"pruning_function": pruning_function}
    return Static(
        population=population,
        fitness=fitness,
        mutation=SelfAdaptiveSingleStep(0.1),
        distribution=SigmaScaling(),
        selection=StochasticUniversalSampling(offspring_size=OFFSPRING_SIZE),
        recombination=SimpleArithmetic(alpha=0.5),
        replacement=ReplaceWorst(offspring_size=OFFSPRING_SIZE),
        **kwargs,
    )


def test_train_rastrigin_end_to_end():
    playground = build_playground()
    stats = playground.train(MAX_GENERATIONS)

    generations = stats["generations"]
    assert list(generations.keys()) == list(range(MAX_GENERATIONS))
    for generation in generations.values():
        assert set(generation.keys()) == {"mean_fitness", "genome"}
        assert generation["genome"].shape == (POPULATION_SIZE, 2)
    assert playground.population.generation_count == MAX_GENERATIONS
    assert playground.population.size() == POPULATION_SIZE

    # Fitness is minus Rastrigin, so it is maximised at 0 and higher means better.
    assert playground.population.mean_fitness() > generations[0]["mean_fitness"]

    best = playground.get_most_performant()
    assert np.all(np.abs(best.genotype.dna) < 2.0)
    assert playground.fitness.fitness(best) == playground.population.max_fitness()

    # predict() evaluates the best individual's value function at x, i.e. the kernel itself.
    x = np.array([0.5, -0.5])
    assert playground.predict(x) == rastrigin(x)
    assert playground.get_most_performant(depth=1).id != best.id


def test_train_with_pruning_function_removes_offspring():
    seen = []

    def prune_all(individual):
        seen.append(individual.id)
        return True

    playground = build_playground(pruning_function=prune_all)
    before = set(item.individual.id for item in playground.population.pool)
    playground.train(3)

    # offspring were generated and all of them were pruned, so the population is unchanged
    assert len(seen) > 0
    assert set(item.individual.id for item in playground.population.pool) == before
    assert not set(seen) & before
