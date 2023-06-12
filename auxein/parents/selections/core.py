# -*- coding: utf-8 -*-
"""Core Auxein mutations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod
from typing import List

import numpy as np


def cumulative_probability_distribution(index: int, probabilities: List[float]) -> float:
    return sum(probabilities[:index + 1])


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
        index = 0
        mating_pool: List[str] = []
        r = np.random.uniform(0, 1 / self.parents_to_select)
        while len(mating_pool) < self.parents_to_select:
            while r <= cumulative_probability_distribution(index, probabilities):
                mating_pool.append(individual_ids[index])
                r += 1 / self.parents_to_select

            index += 1

        return mating_pool
