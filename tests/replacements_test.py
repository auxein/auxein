# -*- coding: utf-8 -*-

import pytest

from auxein.fitness import Fitness
from auxein.population import build_individual, Population
from auxein.replacements import ReplaceWorst


def build_fully_specified_population():
    population = Population()
    population.add(build_individual([0.1, 0.9], id="3adee626-de78-4f83-84f9-ebde4e8ee64d"), 1.0)  # fitness = 1
    population.add(build_individual([0.1, 0.5], id="e2ee1fd8-7bb9-4556-9435-cd012b0f5403"), 0.6)  # fitness = 0.6
    population.add(build_individual([0.1, 0.1], id="01f4eadc-e799-42d1-bc18-0fd85159bfb6"), 0.2)  # fitness = 0.2
    return population


def test_replace_worst():
    population = build_fully_specified_population()
    offspring = [
        build_individual([0.1, 0.4], id="7fdbb922-6435-4ab1-87ec-3acccbf71da6"),
        build_individual([0.1, 0.3], id="45ae2513-4a81-4385-ad45-4c6d2e172c92"),
    ]

    class TestFitnessFunction(Fitness):
        def fitness(self, individual):
            return individual.genotype.dna[0] + individual.genotype.dna[1]

        def value(self, individual, x):
            pass

    replacement = ReplaceWorst(2)
    replacement.replace(offspring, population, TestFitnessFunction())

    assert population.size() == 3
    assert population.get("3adee626-de78-4f83-84f9-ebde4e8ee64d") is not None
    assert population.get("7fdbb922-6435-4ab1-87ec-3acccbf71da6") is not None
    assert population.get("45ae2513-4a81-4385-ad45-4c6d2e172c92") is not None


class SumFitness(Fitness):
    def fitness(self, individual):
        return individual.genotype.dna[0] + individual.genotype.dna[1]

    def value(self, individual, x):
        pass


def test_replace_worst_with_fewer_offspring_than_replacement_size():
    population = build_fully_specified_population()
    offspring = [
        build_individual([0.1, 0.4], id="7fdbb922-6435-4ab1-87ec-3acccbf71da6"),
        build_individual([0.1, 0.3], id="45ae2513-4a81-4385-ad45-4c6d2e172c92"),
    ]

    ReplaceWorst(5).replace(offspring, population, SumFitness())

    # only as many individuals as offspring are replaced: the two worst
    assert population.size() == 3
    assert population.get("3adee626-de78-4f83-84f9-ebde4e8ee64d") is not None
    with pytest.raises(KeyError):
        population.get("e2ee1fd8-7bb9-4556-9435-cd012b0f5403")
    with pytest.raises(KeyError):
        population.get("01f4eadc-e799-42d1-bc18-0fd85159bfb6")
    assert population.get("7fdbb922-6435-4ab1-87ec-3acccbf71da6") is not None
    assert population.get("45ae2513-4a81-4385-ad45-4c6d2e172c92") is not None


@pytest.mark.parametrize(
    "offspring_size",
    [2, 5],
    ids=["smaller_than_population", "larger_than_population"],
)
def test_replace_worst_with_no_offspring_is_a_noop(offspring_size):
    population = build_fully_specified_population()
    before = sorted(i.individual.id for i in population.pool)

    ReplaceWorst(offspring_size).replace([], population, SumFitness())

    assert sorted(i.individual.id for i in population.pool) == before
