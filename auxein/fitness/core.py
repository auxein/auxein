# -*- coding: utf-8 -*-
"""Core Auxein mutations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod

import numpy as np

from .utils import linear_fit, polynomial_fit, least_squares
from auxein.population import Individual


class Fitness(ABC):

    @abstractmethod
    def fitness(self, individual: Individual) -> float:
        pass

    @abstractmethod
    def value(self, individual: Individual, x: np.ndarray) -> float:
        pass


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
