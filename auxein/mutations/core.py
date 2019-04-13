# -*- coding: utf-8 -*-
"""Core Auxein mutations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from auxein.population.genotype import Genotype

import numpy as np

class Mutation:

    def mutate(self, genotype):
        return genotype


class Uniform(Mutation):

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def mutate(self, genotype):
        gene_index = np.random.randint(
            0,
            genotype.dimension
        )
        dna = genotype.dna
        dna[gene_index] = np.random.uniform(self.lower_bound, self.upper_bound)
        return Genotype(dna, genotype.mask)


class FixedVariance(Mutation):

    def __init__(self, sigma):
        self.sigma = sigma
    
    def mutate(self, genotype):
        move = np.vectorize(lambda g: g + np.random.normal(0, self.sigma))
        return Genotype(move(genotype.dna), genotype.mask)


class SelfAdaptiveSingleStep(Mutation):

    def __init__(self, tau):
        self.tau = tau
    
    def mutate(self, genotype):
        multiplier = np.exp(np.random.normal(0, self.tau))
        move_mask = np.vectorize(lambda g: g * multiplier)
        updated_mask = move_mask(genotype.mask)
        scalr = np.random.normal(0, 1, genotype.dimension)
        dna = genotype.dna + (updated_mask * scalr)
        return Genotype(dna, updated_mask)