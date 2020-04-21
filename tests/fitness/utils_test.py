import numpy as np

from auxein.fitness.utils import linear_fit, polynomial_fit, residual, least_squares, logit


def test_linear_fit_2d():
    a = np.random.uniform(-10, 10)
    b = np.random.normal(5, 10)
    x = np.random.uniform(-100, 100)
    assert linear_fit(np.array([a, b]), np.array(x)) == (a * x) + b


def test_linear_fit_3d():
    a = np.random.uniform(-10, 10)
    b = np.random.normal(5, 10)
    c = np.random.normal(5, 10)
    x = np.random.uniform(-100, 100, 2)
    assert linear_fit(np.array([a, b, c]), np.array(x)) == (a * x[0]) + (b * x[1]) + c


def test_residual_2d():
    assert residual(np.array([1.0, 0.0]), np.array([5.0]), 5.5) == 0.25
    assert residual(np.array([1.0, 0.0]), np.array([5.0]), 5.0) == 0.0
    assert residual(np.array([1.0, 0.0]), np.array([5.0]), 4.5) == 0.25


def test_residual_3d():
    assert residual(np.array([1.0, 1.0, 0.0]), np.array([5.0, 0.0]), 5.5) == 0.25
    assert residual(np.array([1.0, 1.0, 0.0]), np.array([5.0, 0.0]), 5.0) == 0.0
    assert residual(np.array([1.0, 1.0, 0.0]), np.array([5.0, 0.0]), 4.5) == 0.25


def test_least_squares_2d():
    xs = np.array([[0.1], [0.2], [0.3]])
    y = np.array([10, 20, 30])

    assert least_squares(xs, y, np.array([10.0, 0.0])) == 1134.0
    assert least_squares(xs, y, np.array([100.0, 0.0])) == 0.0
    assert least_squares(xs, y, np.array([100.0, 1.0])) == 3.0


def test_least_squares_3d():
    xs = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([10, 20])

    assert least_squares(xs, y, np.array([0.0, 1.0, 0.0])) == 461.0
    assert least_squares(xs, y, np.array([1.0, 0.0, 0.0])) == 481.0


def test_polynomial_fit():
    # 0.5*x^3 - 2.5*x^2 + x + 2
    value = polynomial_fit([0.5, -2.5, 1, 2], np.array([1.5]))
    assert np.isclose(value, -0.4375)


def test_residual_with_polynomial_fit():
    # 0.5*x^3 - 2.5*x^2 + x + 2
    assert residual(np.array([0.5, -2.5, 1, 2]), np.array([1.5]), -1.4375, fit=polynomial_fit) == 1.0
    assert residual(np.array([0.5, -2.5, 1, 2]), np.array([1.5]), -0.4375, fit=polynomial_fit) == 0.0
    assert residual(np.array([0.5, -2.5, 1, 2]), np.array([1.5]), 0.5625, fit=polynomial_fit) == 1.0


def test_logit_2d():
    x = np.array([0.1])
    alpha = 0.5
    coeff = np.array([2])
    # ~ 0.668
    value = logit(alpha, coeff, x)
    assert np.isclose(value, 0.66818, atol = 0.00001)

def test_logit_3d():
    x = np.array([0.1, 1.5])
    alpha = 0.5
    coeff = np.array([2, -0.5])
    # ~ 0.48750
    value = logit(alpha, coeff, x)
    assert np.isclose(value, 0.48750, atol = 0.00001)
    

    
