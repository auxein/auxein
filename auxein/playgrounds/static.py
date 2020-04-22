# -*- coding: utf-8 -*-
"""Static playground.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

import logging
from itertools import permutations

import numpy as np

from auxein.fitness import Fitness
from auxein.population.individual import build_individual, Individual
from auxein.population import Population
from auxein.mutations import Mutation
from auxein.recombinations import Recombination
from auxein.parents.distributions import Distribution
from auxein.parents.selections import Selection
from auxein.replacements import Replacement

logging.basicConfig(level=logging.DEBUG)


class Playground(ABC):

    def __init__(self, population: Population, fitness: Fitness) -> None:
        self.population = population
        self.fitness = fitness

    @abstractmethod
    def train(self, max_generations: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> float:
        pass

    def _get_nth_top_performant(self, depth: int = 0) -> Individual:
        id = self.population.rank_by_fitness(depth + 1)[depth][0]
        return self.population.get(id).individual


class Static(Playground):

    def __init__(
        self,
        population: Population,
        fitness: Fitness,
        mutation: Mutation,
        distribution: Distribution,
        selection: Selection,
        recombination: Recombination,
        replacement: Replacement,
        verbose: bool = False
    ) -> None:
        super().__init__(population=population, fitness=fitness)
        self.mutation = mutation
        self.distribution = distribution
        self.selection = selection
        self.recombination = recombination
        self.replacement = replacement
        self.verbose = verbose

    def __mate(self, mating_pool: List[str]) -> List[Individual]:
        couples = permutations(mating_pool, 2)
        offspring = []
        for (parent1_id, parent2_id) in couples:
            (child1_genotype_dna, child1_genotype_mask, child2_genotype_dna, child2_genotype_mask) = self.__breed(
                parent1_id,
                parent2_id
            )

            offspring.append(build_individual(child1_genotype_dna, child1_genotype_mask))
            offspring.append(build_individual(child2_genotype_dna, child2_genotype_mask))

        return offspring

    def __breed(self, parent1_id: str, parent2_id: str) -> Tuple[List[float], List[float], List[float], List[float]]:
        parent_1 = self.population.get(parent1_id).individual.mutate(self.mutation)
        parent_2 = self.population.get(parent2_id).individual.mutate(self.mutation)

        parent1_genotype_dna = parent_1.genotype.dna
        parent2_genotype_dna = parent_2.genotype.dna

        (child1_genotype_dna, child2_genotype_dna) = self.recombination.recombine(
            parent1_genotype_dna,
            parent2_genotype_dna
        )
        return (child1_genotype_dna, parent_1.genotype.mask, child2_genotype_dna, parent_2.genotype.mask)

    def train(self, max_generations: int) -> Dict[str, Any]:
        logging.info(f'Starting evolution cycle with a maximum of {max_generations} generations')
        stats: Dict[str, Any] = {
            'generations': {}
        }
        while self.population.generation_count < max_generations and self.population.size() > self.selection.offspring_size:
            mean_fitness = self.population.mean_fitness()
            if self.verbose is True:
                logging.debug(f'Running generation: {self.population.generation_count}/{max_generations} -- average_fitness: {mean_fitness} -- population size: {self.population.size()}')
            else:
                logging.debug(f'Running generation: {self.population.generation_count}/{max_generations}')

            stats['generations'][self.population.generation_count] = {}
            stats['generations'][self.population.generation_count]['mean_fitness'] = mean_fitness
            stats['generations'][self.population.generation_count]['genome'] = self.population.get_full_genome()

            # Mating step
            distribution = self.distribution.get(self.population)
            individual_ids = list(map(lambda i : i[0], distribution))
            probabilities = list(map(lambda i : i[1], distribution))

            mating_pool: List[str] = self.selection.select(individual_ids, probabilities)
            offspring = self.__mate(mating_pool)

            # Replacement step
            self.replacement.replace(offspring, self.population, self.fitness)
            self.population.update(self.fitness)

        logging.info(f'Training ended with average_fitness: {self.population.mean_fitness()} and a population size of {self.population.size()}')
        return stats

    def predict(self, x: np.ndarray, depth: int = 0) -> float:
        i = super()._get_nth_top_performant(depth)
        return self.fitness.value(i, x)

    def get_most_performant(self, depth: int = 0) -> Individual:
        return super()._get_nth_top_performant(depth)
