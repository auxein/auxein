"""Strategies (design doc §3): the algorithms. They propose candidates and learn from results; they never evaluate."""

from auxein.strategies.ga import GeneticAlgorithm
from auxein.strategies.random_search import RandomSearch

__all__ = ["GeneticAlgorithm", "RandomSearch"]
