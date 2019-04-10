# -*- coding: utf-8 -*-
"""Core Auxein mutations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

        