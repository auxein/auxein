# -*- coding: utf-8 -*-
"""Core Auxein mutations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def linear_fit(coeff: np.array, e: float, x: np.array) -> float:
    return np.dot(x, coeff) + e


def residual(coeff: np.array, e: float, x: np.array, yi: float) -> float:
    return (yi - linear_fit(coeff, e, x))**2


def least_squares(xs: np.ndarray, y: np.ndarray, coeff: np.array, e: float) -> float:
    lsm = 0
    for x, yi in zip(xs, y):
        lsm = lsm + residual(coeff, e, x, yi)
    return lsm
