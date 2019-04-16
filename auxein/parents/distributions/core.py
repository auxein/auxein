# -*- coding: utf-8 -*-
"""Core Auxein mutations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod
from typing import Tuple, List

from auxein.population import Item, Population


class Distribution(ABC):

    @abstractmethod
    def get(self, population: Population) -> List[Tuple[str, float]]:
        pass


class Fps(Distribution):

    def __init__(self) -> None:
        super().__init__()

    def get(self, population: Population) -> List[Tuple[str, float]]:
        return list(map(lambda item: (item.individual.id, item.fitness / population.total_fitness()), population.pool))


class FpsWithWindowing(Distribution):

    def __init__(self) -> None:
        super().__init__()

    def __scale_fitness_function(self, item: Item, minimum_fitness: float) -> float:
        return item.fitness - minimum_fitness

    def get(self, population: Population) -> List[Tuple[str, float]]:
        minimum_fitness = population.min_fitness()
        total_fitness = sum(self.__scale_fitness_function(item, minimum_fitness) for item in population.pool)
        return list(map(lambda item: (item.individual.id, self.__scale_fitness_function(item, minimum_fitness) / total_fitness), population.pool))


class SigmaScaling(Distribution):

    def __init__(self) -> None:
        super().__init__()

    def __scale_fitness_function(self, item: Item, population: Population) -> float:
        max_value: float = max(item.fitness - (population.mean_fitness() - 2 * population.std_fitness()), 0)
        return max_value

    def get(self, population: Population) -> List[Tuple[str, float]]:
        total_fitness = sum(self.__scale_fitness_function(item, population) for item in population.pool)
        return list(map(lambda item: (item[0].id, self.__scale_fitness_function(item, population) / total_fitness), population.pool))
