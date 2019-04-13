import pytest

import numpy as np

from auxein.fitness.utils import linear_fit, residual, least_squares

def test_linear_fit_2d():
    a = np.random.uniform(-10, 10)
    b = np.random.normal(5, 10)
    x = np.random.uniform(-100, 100)
    assert linear_fit([a], b, x) == (a * x) + b


def test_linear_fit_3d():
    a = np.random.uniform(-10, 10)
    b = np.random.normal(5, 10)
    c = np.random.normal(5, 10)
    x = np.random.uniform(-100, 100, 2)
    assert linear_fit([a, b], c, x) == (a * x[0]) + (b * x[1]) + c


def test_residual_2d():
    assert residual([1.0], 0.0, [5.0], 5.5) == 0.25
    assert residual([1.0], 0.0, [5.0], 5.0) == 0.0
    assert residual([1.0], 0.0, [5.0], 4.5) == 0.25


def test_residual_3d():
    assert residual([1.0, 1.0], 0.0, [5.0, 0.0], 5.5) == 0.25
    assert residual([1.0, 1.0], 0.0, [5.0, 0.0], 5.0) == 0.0
    assert residual([1.0, 1.0], 0.0, [5.0, 0.0], 4.5) == 0.25


def test_least_squares_2d():
    xs = np.array([[0.1], [0.2], [0.3]])
    y = np.array([10, 20, 30])

    assert least_squares(xs, y, [10.0], 0.0) == 1134.0
    assert least_squares(xs, y, [100.0], 0.0) == 0.0
    assert least_squares(xs, y, [100.0], 1.0) == 3.0


def test_least_squares_3d():
    xs = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([10, 20])

    assert least_squares(xs, y, [0.0, 1.0], 0.0) == 461.0
    assert least_squares(xs, y, [1.0, 0.0], 0.0) == 481.0
