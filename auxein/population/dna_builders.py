"""Contains few classes to build random dna sequences.
"""
from __future__ import absolute_import
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Tuple, NamedTuple, Iterable, Dict, Any, Optional, List

import numpy as np

class DnaBuilder(ABC):

    def __init__(self, dimension: int):
        assert dimension > 0, 'dna dimension must be strictly positive.'
        self.dimension = dimension
    
    def get_dimension(self) -> int:
        return self.dimension

    @abstractmethod
    def get(self) -> np.ndarray:
        pass


class RandomDnaBuilder(DnaBuilder):

    def __init__(self, distribution: str, dimension: int):
        super().__init__(dimension = dimension)
        self._distribution = distribution
    
    def get_distribution(self):
        return self._distribution
    
    @abstractmethod
    def get(self) -> np.ndarray:
        pass


class UniformRandomDnaBuilder(RandomDnaBuilder):

    def __init__(self, dimension, interval: Tuple[float, float] = (-1.0, 1.0)):
        super().__init__(distribution = 'uniform', dimension = dimension)
        self.interval = interval
    
    def get(self) -> np.ndarray:
        return np.random.uniform(
            self.interval[0],
            self.interval[1],
            super().get_dimension()
        )