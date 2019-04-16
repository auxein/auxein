# -*- coding: utf-8 -*-
"""Static playground.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from auxein.fitness import Fitness
from auxein.population import Population, Individual


class Replacement(ABC):

    def __init__(self, offspring_size: int) -> None:
        self.offspring_size = offspring_size

    def _replace(self, quantity: int, offspring: List[Individual], population: Population, individuals_to_kill: List[str], fitness_function: Fitness) -> None:
        for i in individuals_to_kill:
            population.kill(i)

        children: List[Individual] = np.random.choice(offspring, quantity, replace=False)
        for child in children:
            population.add(child, fitness_function.fitness(child))

    @abstractmethod
    def replace(self, offspring: List[Individual], population: Population, fitness_function: Fitness) -> None:
        pass


class ReplaceWorst(Replacement):

    def __init__(self, offspring_size: int) -> None:
        super().__init__(offspring_size=offspring_size)

    def replace(self, offspring: List[Individual], population: Population, fitness_function: Fitness) -> None:
        quantity = population.size() if self.offspring_size >= population.size() else min(self.offspring_size, len(offspring))
        individuals_to_kill: List[str] = list(map(lambda item : item[0], population.rank_by_fitness(quantity, reverse=False)))
        super()._replace(quantity, offspring, population, individuals_to_kill, fitness_function)
