# -*- coding: utf-8 -*-
"""Core Auxein mutations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod
from typing import List

import numpy as np


def cumulative_probability_distribution(index: int, probabilities: List[float]) -> float:
    return sum(probabilities[: index + 1])


class Selection(ABC):
    def __init__(self, offspring_size: int) -> None:
        self.__offspring_size = offspring_size
        self.parents_to_select = np.around(np.roots([1, -1, -offspring_size / 2])[0])

    @property
    def offspring_size(self) -> int:
        return self.__offspring_size

    @abstractmethod
    def select(self, individual_ids: List[str], probabilities: List[float]) -> List[str]:
        pass


class StochasticUniversalSampling(Selection):
    def __init__(self, offspring_size: int) -> None:
        super().__init__(offspring_size=offspring_size)

    def select(self, individual_ids: List[str], probabilities: List[float]) -> List[str]:
        if len(individual_ids) != len(probabilities):
            raise ValueError("individual_ids and probabilities must have the same length")
        weights = np.asarray(probabilities, dtype=float)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("probabilities must be finite and non-negative")
        total = weights.sum()
        if not total > 0:
            raise ValueError("probabilities must have a strictly positive sum")

        cumulative = np.cumsum(weights / total)
        cumulative[-1] = 1.0  # rounding must not leave the last pointers beyond the end

        n = int(self.parents_to_select)
        step = 1 / n
        pointer = np.random.uniform(0, step)
        index = 0
        mating_pool: List[str] = []
        while len(mating_pool) < n:
            while pointer > cumulative[index] and index < len(cumulative) - 1:
                index += 1
            mating_pool.append(individual_ids[index])
            pointer += step

        return mating_pool
