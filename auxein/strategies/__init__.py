"""Strategies (design doc §3): the algorithms. They propose candidates and learn from results; they never evaluate."""

from auxein.strategies.ga import GeneticAlgorithm
from auxein.strategies.random_search import RandomSearch
from auxein.strategies.structured import StructuredGeneticAlgorithm

__all__ = ["GeneticAlgorithm", "RandomSearch", "StructuredGeneticAlgorithm"]
