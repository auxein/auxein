"""
Contains various genotypes representations.
"""
from __future__ import absolute_import

import numpy as np


class Genotype:

    def __init__(self, dna: np.ndarray, mask: np.ndarray):
        self._dimension = len(dna)
        self._dna = dna.copy()
        self._mask = mask.copy()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def dna(self) -> np.ndarray:
        return self._dna.copy()

    @property
    def mask(self) -> np.ndarray:
        return self._mask.copy()

    def __repr__(self) -> str:
        return f'({self._dna}),({self._mask})'
