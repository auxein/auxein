# -*- coding: utf-8 -*-
"""Core Auxein mutations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class Distribution:

    def get(self, opulation):
        return []


class SigmaScaling(Distribution):

    def __init__(self):
        super(SigmaScaling, self).__init__()
    

    def __scale_fitness_function(self, item, population):
        return max(item.fitness - (population.mean_fitness() - 2 * population.std_fitness()), 0)


    def get(self, population):
        total_fitness = sum(self.__scale_fitness_function(item, population) for item in population.pool)
        return list(map(lambda item: (item[0].id, self.__scale_fitness_function(item, population) / total_fitness), population.pool))
