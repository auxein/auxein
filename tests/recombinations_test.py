import pytest
# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np

from auxein.recombinations import Recombination, SimpleArithmetic, MatrixRecombination


def test_simple_arithmetic_with_full_blending():
    dna1 = np.array([1, 2, 3, 4, 5])
    dna2 = np.array([0, 0, 0, 0, 0])
    recombination = SimpleArithmetic(1)
    (child1_dna, child2_dna) = recombination.recombine(dna1, dna2)

    assert sum(child1_dna) + sum(child2_dna) == sum(dna1)


def test_simple_arithmetic_with_no_blending():
    dna1 = np.array([1, 2, 3, 4, 5])
    dna2 = np.array([0, 0, 0, 0, 0])
    recombination = SimpleArithmetic(0)
    (child1_dna, child2_dna) = recombination.recombine(dna1, dna2)

    assert (child1_dna == [1, 2, 3, 4, 5]).all()
    assert (child2_dna == [0, 0, 0, 0, 0]).all()


def test_simple_arithmetic_with_half_blending():
    dna1 = np.array([1, 2, 3, 4, 5])
    dna2 = np.array([0, 0, 0, 0, 0])
    recombination = SimpleArithmetic(0.5)
    (child1_dna, child2_dna) = recombination.recombine(dna1, dna2)

    assert sum(child1_dna) + sum(child2_dna) == sum(dna1)


def test_simple_arithmetic_with_full_blending_with_uneven_dnas_left():
    dna1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dna2 = np.array([0, 0, 0, 0, 0])
    recombination = SimpleArithmetic(1, allow_uneven=True)
    (child1_dna, child2_dna) = recombination.recombine(dna1, dna2)

    assert sum(child1_dna) + sum(child2_dna) == sum(dna1)


def test_simple_arithmetic_with_full_blending_with_uneven_dnas_right():
    dna1 = np.array([0, 0, 0, 0, 0])
    dna2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    recombination = SimpleArithmetic(1, allow_uneven=True)
    (child1_dna, child2_dna) = recombination.recombine(dna1, dna2)

    assert sum(child1_dna) + sum(child2_dna) == sum(dna2)


def test_matrix_recombination():
    dna1 = np.array([[1, 2], [3, 4], [5, 6]])
    dna2 = np.array([[10, 20], [30, 40], [50, 60]])

    class Identity(Recombination):
        def recombine(self, parent1_dna: np.ndarray, parent2_dna: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return (parent1_dna, parent2_dna)

    recombination = MatrixRecombination((3, 2), Identity())
    (child1_dna, child2_dna) = recombination.recombine(dna1, dna2)
    assert np.array_equal(child1_dna, dna1)
    assert np.array_equal(child2_dna, dna2)


@pytest.mark.xfail(strict=True, reason="phase 3: MatrixRecombination flattens to shape (1, n), so the crossover point is always 0")
def test_matrix_recombination_with_simple_arithmetic_crosses_over(monkeypatch):
    dna1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    dna2 = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    monkeypatch.setattr(np.random, "randint", lambda low, high: 3)  # crossover point

    recombination = MatrixRecombination((3, 2), SimpleArithmetic(alpha=0.5))
    (child1_dna, child2_dna) = recombination.recombine(dna1, dna2)

    assert child1_dna.shape == (3, 2)
    assert child2_dna.shape == (3, 2)
    # the prefix before the crossover point is preserved from each parent...
    assert np.array_equal(child1_dna.reshape(-1)[:3], dna1.reshape(-1)[:3])
    assert np.array_equal(child2_dna.reshape(-1)[:3], dna2.reshape(-1)[:3])
    # ...and the rest is blended
    assert np.allclose(child1_dna.reshape(-1)[3:], [(4 + 40) / 2, (5 + 50) / 2, (6 + 60) / 2])
    assert np.allclose(child2_dna.reshape(-1)[3:], [(40 + 4) / 2, (50 + 5) / 2, (60 + 6) / 2])


@pytest.mark.xfail(strict=True, reason="phase 3: MatrixRecombination.__init__ does not call super().__init__()")
def test_matrix_recombination_has_allow_uneven():
    assert MatrixRecombination((3, 2), SimpleArithmetic(alpha=0.5)).allow_uneven is False
