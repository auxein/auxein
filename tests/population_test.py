import pytest

import numpy as np

from auxein.fitness import Fitness
from auxein.population.individual import build_individual
from auxein.population import build_population, Population, Item


def test_build_population_dimension_and_size():
        class TestFitnessFunction(Fitness):
                def fitness(self, individual):
                        return 1.0
        pop = build_population(3, 10, TestFitnessFunction())
        assert pop.size() == 10
        for item in pop.pool:
                assert item[0].dimension() == 3


def init_population(dimension, size):
        population = Population()
        for _ in range(0, size):
                dna = np.random.uniform(-1, 1, dimension)
                population.add(build_individual(dna, []), 1.0)
        return population


def build_fully_specified_population():
    population = Population()
    population.add(build_individual([0.1, 0.9], [], '3adee626-de78-4f83-84f9-ebde4e8ee64d'), 1.0)
    population.add(build_individual([0.1, 0.5], [], 'e2ee1fd8-7bb9-4556-9435-cd012b0f5403'), 1.0)
    population.add(build_individual([0.1, 0.1], [], '01f4eadc-e799-42d1-bc18-0fd85159bfb6'), 1.0)
    return population


def test_kill():
    p = build_fully_specified_population()
    assert p.get('3adee626-de78-4f83-84f9-ebde4e8ee64d') is not None
    assert p.size() == 3
    p.kill('3adee626-de78-4f83-84f9-ebde4e8ee64d')
    assert p.size() == 2
    with pytest.raises(KeyError):
        assert p.get('3adee626-de78-4f83-84f9-ebde4e8ee64d')


def test_add():
    p = build_fully_specified_population()
    assert p.size() == 3
    with pytest.raises(KeyError):
        p.get('d5984693-5965-4534-9001-2616dfee90f9')

    p.add(build_individual([0.1, 0.9], [], 'd5984693-5965-4534-9001-2616dfee90f9'), 1.0)
    assert p.get('d5984693-5965-4534-9001-2616dfee90f9') is not None
    assert p.size() == 4


def test_generation_count():
    population = build_fully_specified_population()
    assert population.generation_count == 0

    class TestFitnessFunction(Fitness):
            def fitness(self, individual):
                    return 1.0

    fitness_function = TestFitnessFunction()

    population.update(fitness_function)
    assert population.generation_count == 1


def test_get_stats():
    p = build_fully_specified_population()
    stats = p.get_stats()
    assert stats['size'] == 3
    assert stats['mean_fitness'] == 1.0
    assert stats['min_fitness'] == 1.0
    assert stats['max_fitness'] == 1.0
    assert stats['std_fitness'] == 0.0


def test_update():
    population = Population()
    population.add(build_individual([0.1, 0.1], [], '3adee626-de78-4f83-84f9-ebde4e8ee64d'), 0.2)
    population.add(build_individual([0.1, 0.3], [], 'e2ee1fd8-7bb9-4556-9435-cd012b0f5403'), 0.4)
    population.add(build_individual([0.3, 0.2], [], '01f4eadc-e799-42d1-bc18-0fd85159bfb6'), 0.5)

    assert population.get('3adee626-de78-4f83-84f9-ebde4e8ee64d')[1] == 0.2
    assert population.get('e2ee1fd8-7bb9-4556-9435-cd012b0f5403')[1] == 0.4
    assert population.get('01f4eadc-e799-42d1-bc18-0fd85159bfb6')[1] == 0.5

    population.kill('3adee626-de78-4f83-84f9-ebde4e8ee64d')
    population.add(build_individual([0.5, 0.5], [], '4f5db033-896a-4521-ab41-48b2177d7cd7'), 1.0)
    
    class TestFitnessFunction(Fitness):
            def fitness(self, individual):
                    return individual.genotype.dna[0] + individual.genotype.dna[1]

    fitness_function = TestFitnessFunction()

    population.update(fitness_function)

    assert population.get('e2ee1fd8-7bb9-4556-9435-cd012b0f5403')[1] == 0.4
    assert population.get('01f4eadc-e799-42d1-bc18-0fd85159bfb6')[1] == 0.5
    assert population.get('4f5db033-896a-4521-ab41-48b2177d7cd7')[1] == 1.0
    assert population.total_fitness() == 1.9


def test_rank_by_fitness_desc():
    population = Population()
    population.add(build_individual([0.1, 0.1], [], '3adee626-de78-4f83-84f9-ebde4e8ee64d'), 0.2)
    population.add(build_individual([0.1, 0.3], [], 'e2ee1fd8-7bb9-4556-9435-cd012b0f5403'), 0.4)
    population.add(build_individual([0.3, 0.2], [], '01f4eadc-e799-42d1-bc18-0fd85159bfb6'), 0.5)
    rank = population.rank_by_fitness()

    assert rank[0] == ('01f4eadc-e799-42d1-bc18-0fd85159bfb6', 0.5)
    assert rank[1] == ('e2ee1fd8-7bb9-4556-9435-cd012b0f5403', 0.4)
    assert rank[2] == ('3adee626-de78-4f83-84f9-ebde4e8ee64d', 0.2)


def test_rank_by_fitness_asc():
    population = Population()
    population.add(build_individual([0.1, 0.1], [], '3adee626-de78-4f83-84f9-ebde4e8ee64d'), 0.2)
    population.add(build_individual([0.1, 0.3], [], 'e2ee1fd8-7bb9-4556-9435-cd012b0f5403'), 0.4)
    population.add(build_individual([0.3, 0.2], [], '01f4eadc-e799-42d1-bc18-0fd85159bfb6'), 0.5)
    rank = population.rank_by_fitness(reverse=False)

    assert rank[0] == Item('3adee626-de78-4f83-84f9-ebde4e8ee64d', 0.2)
    assert rank[1] == Item('e2ee1fd8-7bb9-4556-9435-cd012b0f5403', 0.4)
    assert rank[2] == Item('01f4eadc-e799-42d1-bc18-0fd85159bfb6', 0.5)
