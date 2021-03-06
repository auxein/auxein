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
