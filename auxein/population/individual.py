"""Contains the base Individual class.
"""
from __future__ import absolute_import
from typing import List, Optional
from uuid import uuid4, UUID
import time

import numpy as np

from auxein.mutations import Mutation
from auxein.population.genotype import Genotype


class Individual:

    def __init__(self, genotype: Genotype, id: Optional[str] = None) -> None:
        self._id = uuid4() if id is None else UUID(id)
        self._born_at = time.time()
        self._genotype = genotype

    @property
    def id(self) -> str:
        return str(self._id)

    def age(self) -> float:
        return time.time() - self._born_at

    def dimension(self) -> int:
        return self._genotype.dimension

    @property
    def genotype(self) -> Genotype:
        return self._genotype

    def mutate(self, mutation_function: Mutation) -> 'Individual':
        return Individual(mutation_function.mutate(self._genotype))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return False
        return self._id == other._id

    def __hash__(self) -> int:
        return hash(self._id)

    def __repr__(self) -> str:
        return f'[{self._id}],({self._genotype})'


def build_individual(dna: List[float], mask: List[float] = [], id: Optional[str] = None) -> Individual:
    """Utility function to build an Individual."""
    return Individual(Genotype(np.array(dna), np.array(mask)), id)
