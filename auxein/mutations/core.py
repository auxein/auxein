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
    """Abstract class for mutations.

    Given an Individual, a Mutation is responsible for generating a new Individual with a mutated genotype.
    """

    def __init__(self, extend_probability: float = 0.0):
        assert 0 <= extend_probability <= 1, 'extend_probability must be within [0, 1]'
        self.extend_probability = extend_probability

    def _extend(self, genotype: Genotype, new_gene: float) -> Genotype:
        if np.random.uniform(0, 1) <= self.extend_probability:
            dna = genotype.dna
            mask = genotype.mask
            return Genotype(np.append(dna, new_gene), np.append(mask, np.random.normal(0, 1)))
        return genotype

    @abstractmethod
    def mutate(self, genotype: Genotype) -> Genotype:
        pass


class Uniform(Mutation):

    def __init__(self, lower_bound: float, upper_bound: float, extend_probability: float = 0.0) -> None:
        super().__init__(extend_probability=extend_probability)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def mutate(self, genotype: Genotype) -> Genotype:
        gene_index = np.random.randint(
            0,
            genotype.dimension
        )
        dna = genotype.dna
        dna[gene_index] = np.random.uniform(self.lower_bound, self.upper_bound)
        new_gene = np.random.uniform(self.lower_bound, self.upper_bound)
        return super()._extend(Genotype(dna, genotype.mask), new_gene)


class FixedVariance(Mutation):

    def __init__(self, sigma: float, extend_probability: float = 0.0) -> None:
        super().__init__(extend_probability=extend_probability)
        self.sigma = sigma

    def mutate(self, genotype: Genotype) -> Genotype:
        move = np.vectorize(lambda g: g + np.random.normal(0, self.sigma))
        new_gene = np.random.normal(0, self.sigma)
        return super()._extend(Genotype(move(genotype.dna), genotype.mask), new_gene)


class SelfAdaptiveSingleStep(Mutation):
    """Self-adaptive single step mutation as described in [back01].

    [back01]  T. Back, D.B. Fogel, and Z. Michalewicz, editors.
    "Evolutionary Computation 2:dvanced Algorithms and Operators. Institute of Physics Publishing", Bristol, 2000.
    """

    def __init__(self, tau: float, extend_probability: float = 0.0) -> None:
        super().__init__(extend_probability=extend_probability)
        self.tau = tau

    def mutate(self, genotype: Genotype) -> Genotype:
        multiplier = np.exp(np.random.normal(0, self.tau))
        move_mask = np.vectorize(lambda g: g * multiplier)
        updated_mask = move_mask(genotype.mask)
        scalr = np.random.normal(0, 1, genotype.dimension)
        dna = genotype.dna + (updated_mask * scalr)
        new_gene = np.random.normal(0, self.tau)
        return super()._extend(Genotype(dna, updated_mask), new_gene)
