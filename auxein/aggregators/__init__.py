"""Aggregators and their reductions (design doc §6.5)."""

from auxein.aggregators.aggregator import Aggregated, Aggregator
from auxein.aggregators.reductions import Reducer, Reduction, Source, cvar_lower, cvar_upper, maximum, mean, minimum, quantile, total

__all__ = [
    "Aggregated",
    "Aggregator",
    "Reducer",
    "Reduction",
    "Source",
    "cvar_lower",
    "cvar_upper",
    "maximum",
    "mean",
    "minimum",
    "quantile",
    "total",
]
