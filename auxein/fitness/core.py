# -*- coding: utf-8 -*-
"""Core Auxein mutations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod

from typing import Callable

import numpy as np

from .utils import linear_fit, polynomial_fit, least_squares, logit
from auxein.population import build_individual, Individual

class Fitness(ABC):

    @abstractmethod
    def fitness(self, individual: Individual) -> float:
        pass

    @abstractmethod
    def value(self, individual: Individual, x: np.ndarray) -> float:
        pass
