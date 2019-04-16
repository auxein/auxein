# -*- coding: utf-8 -*-
"""Static playground.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABC

import numpy as np


class Replacement(ABC):

    def __init__(self, offspring_size):
        self.offspring_size = offspring_size

    def replace(self, quantity, offspring, population, individuals_to_kill, fitness_function):
        for i in individuals_to_kill:
            population.kill(i)

        for child in np.random.choice(offspring, quantity, replace=False):
            population.add(child, fitness_function.fitness(child))


class ReplaceWorst(Replacement):

    def __init__(self, offspring_size):
        super().__init__(offspring_size=offspring_size)

    def replace(self, offspring, population, fitness_function):
        quantity = population.size() if self.offspring_size >= population.size() else min(self.offspring_size, len(offspring))
        individuals_to_kill = map(lambda item : item[0], population.rank_by_fitness(quantity, reverse=False))
        super().replace(quantity, offspring, population, individuals_to_kill, fitness_function)
