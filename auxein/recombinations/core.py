# -*- coding: utf-8 -*-
"""SimpleArithmetic.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np


class Recombination(ABC):

    @abstractmethod
    def recombine(self, parent1_dna: np.ndarray, parent2_dna: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass


class SimpleArithmetic(Recombination):

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def __linear_combination(self, arr1: np.ndarray, arr2: np.ndarray) -> List[float]:
        result: List[float] = []
        for (i1, i2) in zip(arr1, arr2):
            result.append(self.alpha * i2 + (1 - self.alpha) * i1)
        return result

    def recombine(self, parent1_dna: np.ndarray, parent2_dna: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
