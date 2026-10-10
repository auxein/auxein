"""NSGA-II (design doc §3.3): multi-objective optimisation with non-dominated sorting and crowding distance."""

from auxein.strategies.nsga2.nsga2 import NSGA2
from auxein.strategies.nsga2.sorting import crowded_order, crowding_distance, dominance_matrix, nondominated_ranks

__all__ = ["NSGA2", "crowded_order", "crowding_distance", "dominance_matrix", "nondominated_ranks"]
