# -*- coding: utf-8 -*-
"""Core Auxein mutations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable

import numpy as np


def linear_fit(coeff: np.ndarray, x: np.ndarray) -> float:
    assert type(coeff) == type(x) == np.ndarray, 'coefficients and variable must be np.ndarray'
    assert coeff.size - 1 == x.size, 'coefficients must be of the size of x+1'
    value: float = np.dot(x, coeff[:coeff.size - 1]) + coeff[coeff.size - 1]
    return value


def polynomial_fit(coeff: np.ndarray, x: np.ndarray) -> float:
    assert type(coeff) and x.size == 1, 'Only simple polynomial fit is supported.'
    result: float = np.polyval(coeff, x[0])
    return result


def residual(coeff: np.array, x: np.array, yi: float, fit: Callable[[np.ndarray, np.ndarray], float] = linear_fit) -> float:
    return (yi - fit(coeff, x))**2


def least_squares(xs: np.ndarray, y: np.ndarray, coeff: np.ndarray, fit: Callable[[np.ndarray, np.ndarray], float] = linear_fit) -> float:
    lsm: float = 0
    for x, yi in zip(xs, y):
        lsm = lsm + residual(coeff, x, yi, fit)
    return lsm

def logit(alpha: float, coeff: np.ndarray, x: np.ndarray) -> float:
    kernel = 0
    for (bi, xi) in zip(coeff, x):
        kernel += bi*xi
    return 1 / (1 + np.exp(-(alpha + kernel)))
