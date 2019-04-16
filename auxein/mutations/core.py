# -*- coding: utf-8 -*-
"""Core Auxein mutations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod

from auxein.population.genotype import Genotype

import numpy as np


class Mutation(ABC):

    @abstractmethod
    def mutate(self, genotype: Genotype) -> Genotype:
        pass


class Uniform(Mutation):

    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def mutate(self, genotype: Genotype) -> Genotype:
        gene_index = np.random.randint(
            0,
            genotype.dimension
        )
        dna = genotype.dna
        dna[gene_index] = np.random.uniform(self.lower_bound, self.upper_bound)
        return Genotype(dna, genotype.mask)


class FixedVariance(Mutation):

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def mutate(self, genotype: Genotype) -> Genotype:
        move = np.vectorize(lambda g: g + np.random.normal(0, self.sigma))
        return Genotype(move(genotype.dna), genotype.mask)


class SelfAdaptiveSingleStep(Mutation):

    def __init__(self, tau: float) -> None:
        self.tau = tau

    def mutate(self, genotype: Genotype) -> Genotype:
        multiplier = np.exp(np.random.normal(0, self.tau))
        move_mask = np.vectorize(lambda g: g * multiplier)
        updated_mask = move_mask(genotype.mask)
        scalr = np.random.normal(0, 1, genotype.dimension)
        dna = genotype.dna + (updated_mask * scalr)
        return Genotype(dna, updated_mask)
