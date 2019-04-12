# -*- coding: utf-8 -*-
"""Core Auxein mutations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def cumulative_probability_distribution(index: int, probabilities) -> float:
    return sum(probabilities[:index + 1])

class Selection:

    def __init__(self, offspring_size):
        self.parents_to_select = np.roots([1, -1, -offspring_size / 2])[0]

    def select(self, individual_ids, probabilities):
        return []

class StochasticUniversalSampling(Selection):
    
    def __init__(self, offspring_size):
        super(StochasticUniversalSampling, self).__init__(offspring_size = offspring_size)
    

    def select(self, individual_ids, probabilities):
        index = 0
        mating_pool = []
        r = np.random.uniform(0, 1 / self.parents_to_select)
        while len(mating_pool) < self.parents_to_select:
            while r <= cumulative_probability_distribution(index, probabilities):
                mating_pool.append(individual_ids[index])
                r += 1 / self.parents_to_select
            index += 1
        return mating_pool