# -*- coding: utf-8 -*-
"""Static playground.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from itertools import permutations

from auxein.population.individual import build_individual

logging.basicConfig(level=logging.DEBUG)

class Playground:

    def __init__(self, population, fitness):
        self.population = population
        self.fitness = fitness
    

    def train(self, x_train, y_train, max_generations, validations):
        pass

    
    def predict(self, x):
        return x


class Static(Playground):

    def __init__(self, population, fitness, mutation, distribution, selection, recombination, replacement):
        super(Static, self).__init__(population=population, fitness=fitness)
        self.mutation = mutation
        self.distribution = distribution
        self.selection = selection
        self.recombination = recombination
        self.replacement = replacement
    
    
    def __mate(self, mating_pool):
        couples = permutations(mating_pool, 2)
        offspring = []
        for (parent1_id, parent2_id) in couples:
            (child1_genotype_dna, child1_genotype_mask, child2_genotype_dna, child2_genotype_mask) = breed(
                parent1_id,
                parent2_id
            )

            offspring.append(build_individual(child1_genotype_dna, child1_genotype_mask))
            offspring.append(build_individual(child2_genotype_dna, child2_genotype_mask))

        return offspring


    def __breed(self, parent1_id, parent2_id):
        parent_1 = self.population.get(parent1_id).individual.mutate(self.mutation)
        parent_2 = self.population.get(parent2_id).individual.mutate(self.mutation)
        
        parent1_genotype_dna = parent_1.genotype.dna
        parent2_genotype_dna = parent_2.genotype.dna

        (child1_genotype_dna, child2_genotype_dna) = self.recombination.recombine(
            parent1_genotype_dna,
            parent2_genotype_dna
        )
        return (child1_genotype_dna, parent_1.genotype.mask, child2_genotype_dna, parent_2.genotype.mask)
    
    
    def train(self, x_train, y_train, max_generations, validation = None):
        logging.info(f'Starting evolution cycle with a maximum of {max_generations} generations')
        while self.population.generation_count < max_generations:
            mean_fitness = self.population.mean_fitness()
            logging.debug(f'{self.population.generation_count}/{max_generations} -- average_fitness: {mean_fitness}')

            # Mating step
            distribution = self.distribution.get(self.population)
            individual_ids = list(map(lambda i : i[0], distribution))
            probabilities = list(map(lambda i : i[1], distribution))

            mating_pool = self.selection.select(self, individual_ids, probabilities)
            offspring = self.__mate(mating_pool)

            self.replacement.replace(offspring, self.population)
            self.population.update(self.fitness)
        
        logging.info(f'Training ended with average_fitness: {self.population.mean_fitness()}')

    
    
    def predict(self, x, depth = 0):
        return 12.4



