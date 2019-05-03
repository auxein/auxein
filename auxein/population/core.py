"""Contains the population class.
"""
from __future__ import absolute_import
from copy import deepcopy
from typing import Tuple, NamedTuple, Iterable, Dict, Any, Optional, List

import numpy as np

from auxein.population.individual import Individual, build_individual
from auxein.fitness.core import Fitness

Item = NamedTuple('Item', [('individual', Individual), ('fitness', float)])


class Population:

    def __init__(self) -> None:
        self.__pool: Dict[str, Item] = {}
        self.__generation_count = 0

    def size(self) -> int:
        return len(self.__pool)

    def add(self, individual: Individual, fitness: float) -> None:
        self.__pool[str(individual.id)] = Item(deepcopy(individual), fitness)

    def get(self, individual_id: str) -> Item:
        return deepcopy(self.__pool[individual_id])

    def update(self, fitness_function: Fitness) -> None:
        for item in self.__pool.values():
            individual = item.individual
            new_fitness = fitness_function.fitness(individual)
            self.__pool[str(individual.id)] = Item(
                deepcopy(individual), new_fitness)
        self.__generation_count += 1

    def kill(self, individual_id: str) -> None:
        try:
            del self.__pool[individual_id]
        except KeyError:
            pass

    @property
    def pool(self) -> Iterable[Item]:
        return self.__pool.values()

    def total_fitness(self) -> float:
        total_fitness: float = 0.0
        for item in self.__pool.values():
            total_fitness += item.fitness
        return total_fitness

    @property
    def generation_count(self) -> int:
        return self.__generation_count

    def rank_by_fitness(self, k: Optional[int] = None, reverse: bool = True) -> List[Tuple[str, float]]:
        sorted_values = sorted(self.pool, key=lambda i: i.fitness, reverse=reverse)
        return list(map(lambda item : (item.individual.id, item.fitness), sorted_values))[:k]

    def __get_ages(self) -> List[float]:
        return list(map(lambda item : item.individual.age(), list(self.__pool.values())))

    def __get_fitness(self) -> List[float]:
        return list(map(lambda item : item.fitness, list(self.__pool.values())))

    def mean_age(self) -> float:
        pools_ages = self.__get_ages()
        value: float = np.mean(pools_ages)
        return value

    def std_age(self) -> float:
        pools_ages = self.__get_ages()
        value: float = np.std(pools_ages)
        return value

    def max_age(self) -> float:
        pools_ages = self.__get_ages()
        value: float = np.max(pools_ages)
        return value

    def min_age(self) -> float:
        pools_ages = self.__get_ages()
        value: float = np.min(pools_ages)
        return value

    def mean_fitness(self) -> float:
        return self.total_fitness() / self.size()

    def max_fitness(self) -> float:
        pools_fitness_values = self.__get_fitness()
        value: float = np.max(pools_fitness_values)
        return value

    def min_fitness(self) -> float:
        pools_fitness_values = self.__get_fitness()
        value: float = np.min(pools_fitness_values)
        return value

    def std_fitness(self) -> float:
        pools_fitness_values = self.__get_fitness()
        value: float = np.std(pools_fitness_values)
        return value

    def get_stats(self) -> Dict[str, Any]:
        return {
            'generation_count': self.__generation_count,
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

    def get_full_genome(self) -> np.ndarray:
        genome = []
        for item in self.pool:
            genome.append(item.individual.genotype.dna)
        return np.array(genome)


def __add_to_population(population: Population, dimension: int, fitness_function: Fitness, dna_interval: Tuple[float, float]) -> None:
    assert dimension > 0, 'dimension must be strictly positive'
    mask = np.repeat(np.random.normal(0, 1), dimension)
    dna = np.random.uniform(dna_interval[0], dna_interval[1], dimension)
    individual = build_individual(dna, mask)
    population.add(individual, fitness_function.fitness(individual))


def build_fixed_dimension_population(dimension: int, initial_size: int, fitness_function: Fitness, dna_interval: Tuple[float, float] = (-1.0, 1.0)) -> Population:
    population = Population()
    for _ in range(0, initial_size):
        __add_to_population(population, dimension, fitness_function, dna_interval)
    return population


def build_variable_dimension_population(initial_size: int, fitness_function: Fitness, dna_interval: Tuple[float, float] = (-1.0, 1.0)) -> Population:
    population = Population()
    for _ in range(0, initial_size):
        dimension = np.random.randint(1, 10)
        __add_to_population(population, dimension, fitness_function, dna_interval)
    return population
