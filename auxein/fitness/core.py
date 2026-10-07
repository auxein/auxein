"""Core Auxein mutations."""

from abc import ABC, abstractmethod

import numpy as np

from auxein.population import Individual


class Fitness(ABC):
    @abstractmethod
    def fitness(self, individual: Individual) -> float:
        pass

    @abstractmethod
    def value(self, individual: Individual, x: np.ndarray) -> float:
        pass
