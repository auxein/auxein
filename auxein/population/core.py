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
            self.__pool[str(individual.id)] = Item(
                deepcopy(individual), new_fitness)
        self.generation_count += 1

    def kill(self, individual_id: str):
        try:
            del self.__pool[individual_id]
        except KeyError:
            pass

    @property
    def pool(self):
        return self.__pool.values()

    def total_fitness(self):
        total_fitness: float = 0.0
        for item in self.__pool.values():
            total_fitness += item.fitness
        return total_fitness

    def rank_by_fitness(self, k=None, reverse=True):
        sorted_values = sorted(
            self.pool, key=lambda i: i.fitness, reverse=reverse)
        return list(map(lambda item: (item.individual.id, item.fitness), sorted_values))[:k]
    
    def __get_ages(self):
        return list(map(lambda item : item.individual.age(), list(self.__pool.values())))

    def __get_fitness(self):
        return list(map(lambda item : item.fitness, list(self.__pool.values())))
        
    def mean_age(self):
        pools_ages = self.__get_ages()
        value: float = np.mean(pools_ages)
        return value

    def std_age(self):
        pools_ages = self.__get_ages()
        value: float = np.std(pools_ages)
        return value

    def max_age(self):
        pools_ages = self.__get_ages()
        value: float = np.max(pools_ages)
        return value

    def min_age(self):
            pools_ages = self.__get_ages()
            value: float = np.min(pools_ages)
            return value

    def mean_fitness(self):
            return self.total_fitness() / self.size()

    def max_fitness(self):
        pools_fitness_values = self.__get_fitness()
        value: float = np.max(pools_fitness_values)
        return value

    def min_fitness(self):
        pools_fitness_values = self.__get_fitness()
        value: float = np.min(pools_fitness_values)
        return value

    def std_fitness(self):
        pools_fitness_values = self.__get_fitness()
        value: float = np.std(pools_fitness_values)
        return value
    
    def get_stats(self):
        return {
            'generation_count': self.generation_count,
            'size': self.size(),
            'mean_age': self.mean_age(),
            'std_age': self.std_age(),
            'max_age': self.max_age(),
            'min_age': self.min_age(),
            'mean_fitness': self.mean_fitness(),
            'min_fitness': self.min_fitness(),
            'max_fitness': self.max_fitness(),
            'std_fitness': self.std_fitness()
        }


def build_population(dimension: int, initial_size: int, fitness_function):
    population = Population()
    for _ in range(0, initial_size):
        mask = np.repeat(np.random.normal(0, 1), dimension)
        dna = np.random.uniform(-1.0, 1.0, dimension)
        individual = build_individual(dna, mask)
        population.add(individual, fitness_function(individual))
    return population
