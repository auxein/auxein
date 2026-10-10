"""Strategies (design doc §3): the algorithms. They propose candidates and learn from results; they never evaluate."""

from auxein.strategies.external import PycmaStrategy
from auxein.strategies.ga import GeneticAlgorithm
from auxein.strategies.nsga2 import NSGA2
from auxein.strategies.random_search import RandomSearch
from auxein.strategies.scalarisation import Chebyshev, Scalarisation, Scalarised, ScalarisedBest, WeightedSum, best_by_scalarisation
from auxein.strategies.structured import StructuredGeneticAlgorithm

__all__ = [
    "NSGA2",
    "PycmaStrategy",
    "Chebyshev",
    "GeneticAlgorithm",
    "RandomSearch",
    "Scalarisation",
    "Scalarised",
    "ScalarisedBest",
    "StructuredGeneticAlgorithm",
    "WeightedSum",
    "best_by_scalarisation",
]
