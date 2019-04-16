# -*- coding: utf-8 -*-
"""SimpleArithmetic.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod

import numpy as np


class Recombination(ABC):

    @abstractmethod
    def recombine(self, parent1_dna, parent2_dna):
        return ([], [])


class SimpleArithmetic(Recombination):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def __linear_combination(self, arr1, arr2):
        result = []
        for (i1, i2) in zip(arr1, arr2):
            result.append(self.alpha * i2 + (1 - self.alpha) * i1)
        return result

    def recombine(self, parent1_dna, parent2_dna):
        cross_over_point = np.random.randint(0, len(parent1_dna))
        child1_dna = np.concatenate(
            (
                parent1_dna[:cross_over_point],
                self.__linear_combination(
                    parent1_dna[cross_over_point:],
                    parent2_dna[cross_over_point:]
                )
            )
        )

        child2_dna = np.concatenate(
            (
                parent2_dna[:cross_over_point],
                self.__linear_combination(
                    parent2_dna[cross_over_point:],
                    parent1_dna[cross_over_point:]
                )
            )
        )
        return (child1_dna, child2_dna)
