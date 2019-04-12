# -*- coding: utf-8 -*-
"""Core Auxein mutations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .utils import least_squares

class Fitness(object):
    
    def get(self, individual):
        return 0.0


class LinearLeastSquares(Fitness):
    
    def __init__(self, xs, y):
        super(LinearLeastSquares, self).__init__()
        assert xs.shape == (y.shape[0], xs.shape[1]), 'length of xs must be equal to length of y'
        self.xs = xs
        self.y = y
    
    def get(self, individual):
        dna = individual.genotype.dna
        coeff = dna[:dna.size - 1]
        e = dna[dna.size - 1]
        return -1 * least_squares(self.xs, self.y, coeff, e)