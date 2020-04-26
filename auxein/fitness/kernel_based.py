# -*- coding: utf-8 -*-
"""Core Auxein mutations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable

import numpy as np

from .core import Fitness
from auxein.population import Individual


class GlobalMinimum(Fitness):

    def __init__(self, kernel: Callable[[np.ndarray], float]) -> None:
        super().__init__()
        self.kernel = kernel

    def fitness(self, individual: Individual) -> float:
        dna = individual.genotype.dna
        return -1 * self.kernel(dna)

    def value(self, individual: Individual, x: np.ndarray) -> float:
        return self.kernel(x)
