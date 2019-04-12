"""Contains the population class.
"""
from __future__ import absolute_import
from copy import deepcopy
from typing import NamedTuple

import numpy as np

from .individual import Individual, build_individual

Item = NamedTuple('Item', [('individual', Individual), ('fitness', float)])


class Population(object):

    def __init__(self):
        self.__pool = {}
        self.generation_count = 0
    
    def size(self):
        return len(self.__pool)
    
    def add(self, individual: Individual, fitness: float):
        self.__pool[str(individual.id)] = Item(deepcopy(individual), fitness)
    
    def get(self, individual_id: str):
        return deepcopy(self.__pool[individual_id])
    
    def update(self, fitness_function):
        for item in self.__pool.values():
            individual = item.individual
            new_fitness = fitness_function(individual)
            self.__pool[str(individual.id)] = Item(deepcopy(individual), new_fitness)
        self.generation_count += 1
    
    def kill(self, individual_id: str) -> None:
        try:
            del self.__pool[individual_id]
        except KeyError:
            pass
    
    def rank_by_fitness(self, k = None, reverse = True):
        sorted_values = sorted(self.__pool, key=lambda i: i.fitness, reverse=reverse)
        return list(map(lambda item : (item.individual.id, item.fitness), sorted_values))[:k]


def build_population(dimension: int, initial_size: int, fitness_function):
    population = Population()
    for _ in range(0, initial_size):
        mask = np.repeat(np.random.normal(0, 1), dimension)
        dna = np.random.uniform(-1.0, 1.0, dimension)
        individual = build_individual(dna, mask)
        population.add(individual, fitness_function(individual))
    return population
