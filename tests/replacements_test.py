# -*- coding: utf-8 -*-

from typing import Tuple, List

import numpy as np

from auxein.fitness import Fitness
from auxein.population import build_individual, Population, Individual
from auxein.replacements import ReplaceWorst

def build_fully_specified_population():
    population = Population()
    population.add(build_individual([0.1, 0.9], [], '3adee626-de78-4f83-84f9-ebde4e8ee64d'), 1.0)  # fitness = 1
    population.add(build_individual([0.1, 0.5], [], 'e2ee1fd8-7bb9-4556-9435-cd012b0f5403'), 0.6)  # fitness = 0.6
    population.add(build_individual([0.1, 0.1], [], '01f4eadc-e799-42d1-bc18-0fd85159bfb6'), 0.2)  # fitness = 0.2
    return population

def test_replace_worst():
    population = build_fully_specified_population()
    offspring = [
        build_individual([0.1, 0.4], [], '7fdbb922-6435-4ab1-87ec-3acccbf71da6'),
        build_individual([0.1, 0.3], [], '45ae2513-4a81-4385-ad45-4c6d2e172c92')
    ]
    class TestFitnessFunction(Fitness):
        def fitness(self, individual):
            return individual.genotype.dna[0] + individual.genotype.dna[1]

        def value(self, individual, x):
            pass

    replacement = ReplaceWorst(2)
    replacement.replace(offspring, population, TestFitnessFunction())

    assert population.size() == 3
    assert population.get('3adee626-de78-4f83-84f9-ebde4e8ee64d') is not None
    assert population.get('7fdbb922-6435-4ab1-87ec-3acccbf71da6') is not None
    assert population.get('45ae2513-4a81-4385-ad45-4c6d2e172c92') is not None
