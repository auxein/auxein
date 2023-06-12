"""Contains few classes to build random dna sequences.
"""
from __future__ import absolute_import
from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np


class DnaBuilder(ABC):

    @abstractmethod
    def get(self, dimension: int) -> np.ndarray:
        pass


class RandomDnaBuilder(DnaBuilder):

    def __init__(self, distribution: str):
        super().__init__()
        self._distribution = distribution

    def get_distribution(self) -> str:
        return self._distribution

    @abstractmethod
    def get(self, dimension: int) -> np.ndarray:
        pass


class UniformRandomDnaBuilder(RandomDnaBuilder):

    def __init__(self, interval: Tuple[float, float] = (-1.0, 1.0)):
        super().__init__(distribution='uniform')
        self.interval = interval

    def get(self, dimension: int) -> np.ndarray:
        assert dimension > 0, 'dna dimension must be strictly positive.'
        return np.random.uniform(
            self.interval[0],
            self.interval[1],
            dimension
        )


class NormalRandomDnaBuilder(RandomDnaBuilder):

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        super().__init__(distribution='normal')
        self.mean = mean
        self.std = std

    def get(self, dimension: int) -> np.ndarray:
        assert dimension > 0, 'dna dimension must be strictly positive.'
        return np.random.normal(
            self.mean,
            self.std,
            dimension
        )


class CompositeDnaBuilder(RandomDnaBuilder):
    """Composite dna builder. It builds various dna sequences with
    different underlyng distributions and concatenates them.
    """

    def __init__(self, builders: List[Tuple[DnaBuilder, int]]):
        super().__init__(distribution='composite')
        self.builders = builders

    def get(self, dimension: int) -> np.ndarray:
        assert dimension > 0, 'dna dimension must be strictly positive.'
        assert dimension == sum(map(lambda item: item[1], self.builders)), 'dna dimension must be equal to the sum of the dimensions of the builders.'

        dna = []
        for (builder, dimension) in self.builders:
            dna.append(builder.get(dimension))

        return np.concatenate(dna)
