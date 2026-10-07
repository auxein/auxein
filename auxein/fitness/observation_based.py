"""Core Auxein mutations."""

from abc import abstractmethod

import numpy as np

from auxein.population import Individual, build_individual

from .core import Fitness
from .utils import least_squares, linear_fit, logit, polynomial_fit


class ObservationBasedFitness(Fitness):
    """Abstract class for observation-based fitness function."""

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
        assert len(specs) == 2, "Only 2-dimensional fitness landscapes are supported at the moment."
        A = np.linspace(specs[0][0], specs[0][1], size)
        B = np.linspace(specs[1][0], specs[1][1], size)
        return self.__compute_f(A, B)


class MultipleLinearRegression(ObservationBasedFitness):
    """Multiple linear regression fitness function.
    Given a set of observations (xi, yi), the fitness function will be computed as the
    sum of the squared residuals of the linear regression model.
    """

    def __init__(self, xs: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        assert xs.shape == (y.shape[0], xs.shape[1]), "length of xs must be equal to length of y"
        self.xs = xs
        self.y = y

    def fitness(self, individual: Individual) -> float:
        dna = individual.genotype.dna
        return -1 * least_squares(self.xs, self.y, dna)

    def value(self, individual: Individual, x: np.ndarray) -> float:
        """Compute the value of the linear regression model for a given x given an individual
        representing the a and b coefficients of the linear model ax + b.
        """
        dna = individual.genotype.dna
        return linear_fit(dna, x)


class SimplePolynomialRegression(ObservationBasedFitness):
    def __init__(self, xs: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        assert xs.shape == (y.shape[0], xs.shape[1]), "length of xs must be equal to length of y"
        self.xs = xs
        self.y = y

    def fitness(self, individual: Individual) -> float:
        dna = individual.genotype.dna
        return -1 * least_squares(self.xs, self.y, dna, fit=polynomial_fit)

    def value(self, individual: Individual, x: np.ndarray) -> float:
        dna = individual.genotype.dna
        return polynomial_fit(dna, x)


class MaximumLikelihood(ObservationBasedFitness):
    """Logistic regression fitness function: the log-likelihood of binary observations (y in {0, 1})."""

    def __init__(self, xs: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        assert xs.shape == (y.shape[0], xs.shape[1]), "length of xs must be equal to length of y"
        classes = np.unique(y)
        assert len(classes) == 2 and np.array_equal(classes, np.array([0, 1])), "y-values can only belong [0, 1] discrete interval"
        self.xs = xs
        self.y = y

    def fitness(self, individual: Individual) -> float:
        """Log-likelihood of the observations: sum(y * log(p) + (1 - y) * log(1 - p)), with p clipped to [eps, 1 - eps]."""
        alpha, *coeff = individual.genotype.dna
        eps = 1e-12
        log_likelihood: float = 0
        for x, y in zip(self.xs, self.y):
            p = min(max(logit(alpha, coeff, x), eps), 1 - eps)
            log_likelihood += y * np.log(p) + (1 - y) * np.log(1 - p)
        return log_likelihood

    def value(self, individual: Individual, x: np.ndarray) -> float:
        alpha, *coeff = individual.genotype.dna
        return logit(alpha, coeff, x)
