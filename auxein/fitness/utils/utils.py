# -*- coding: utf-8 -*-
"""Core Auxein mutations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def linear_fit(coeff: np.ndarray, x: np.ndarray) -> float:
    assert type(coeff) == type(x) == np.ndarray, 'coefficients and variable must be np.ndarray'
    assert coeff.size - 1 == x.size, 'coefficients must be of the size of x+1'
    value: float = np.dot(x, coeff[:coeff.size - 1]) + coeff[coeff.size - 1]
    return value

def residual(coeff: np.array, x: np.array, yi: float) -> float:
    return (yi - linear_fit(coeff, x))**2

def least_squares(xs: np.ndarray, y: np.ndarray, coeff: np.ndarray) -> float:
    lsm: float = 0
    for x, yi in zip(xs, y):
        lsm = lsm + residual(coeff, x, yi)
    return lsm
