"""Fitness functions defined by a kernel function."""

from collections.abc import Callable

import numpy as np

from auxein.population import Individual

from .core import Fitness


class GlobalMinimum(Fitness):
    def __init__(self, kernel: Callable[[np.ndarray], float]) -> None:
        super().__init__()
        self.kernel = kernel

    def fitness(self, individual: Individual) -> float:
        dna = individual.genotype.dna
        return -1 * self.kernel(dna)

    def value(self, individual: Individual, x: np.ndarray) -> float:
        return self.kernel(x)
