# -*- coding: utf-8 -*-
"""Core Auxein mutations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod

from typing import Callable

import numpy as np

from .utils import linear_fit, polynomial_fit, least_squares
from auxein.population import build_individual, Individual


class Fitness(ABC):

    @abstractmethod
    def fitness(self, individual: Individual) -> float:
        pass

    @abstractmethod
    def value(self, individual: Individual, x: np.ndarray) -> float:
        pass

    def __compute_f(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        F = []
        for v in np.transpose([np.tile(A, len(B)), np.repeat(B, len(A))]):
            F.append([v[0], v[1], self.fitness(build_individual([v[0], v[1]]))])
        return np.array(F)

    def get_landscape(self, specs: np.ndarray, size: int) -> np.ndarray:
        assert len(specs) == 2, 'Only 2-dimensional fitness landscapes are supported at the moment.'
        A = np.linspace(specs[0][0], specs[0][1], size)
        B = np.linspace(specs[1][0], specs[1][1], size)
        return self.__compute_f(A, B)


class MultipleLinearRegression(Fitness):

    def __init__(self, xs: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        assert xs.shape == (y.shape[0], xs.shape[1]), 'length of xs must be equal to length of y'
        self.xs = xs
        self.y = y

    def fitness(self, individual: Individual) -> float:
        dna = individual.genotype.dna
        return -1 * least_squares(self.xs, self.y, dna)

    def value(self, individual: Individual, x: np.ndarray) -> float:
        dna = individual.genotype.dna
        return linear_fit(dna, x)


class SimplePolynomialRegression(Fitness):

    def __init__(self, xs: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        assert xs.shape == (y.shape[0], xs.shape[1]), 'length of xs must be equal to length of y'
        self.xs = xs
        self.y = y

    def fitness(self, individual: Individual) -> float:
        dna = individual.genotype.dna
        return -1 * least_squares(self.xs, self.y, dna, fit=polynomial_fit)

    def value(self, individual: Individual, x: np.ndarray) -> float:
        dna = individual.genotype.dna
        return polynomial_fit(dna, x)


class GlobalMinumum(Fitness):

    def __init__(self, kernel: Callable[[np.ndarray], float]) -> None:
        super().__init__()
        self.kernel = kernel

    def fitness(self, individual: Individual) -> float:
        dna = individual.genotype.dna
        return -1 * self.kernel(dna)

    def value(self, individual: Individual, x: np.ndarray) -> float:
        return self.kernel(x)
