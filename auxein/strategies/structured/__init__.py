"""Strategies for structured (non-array) genomes (design doc §3.3, §3.5)."""

from auxein.strategies.structured.genetic_algorithm import StructuredGeneticAlgorithm
from auxein.strategies.structured.sequence import SequenceCrossover, SequenceMutation
from auxein.strategies.structured.variation import (
    ExternalMutation,
    ExternalRecombination,
    OperatorError,
    OperatorNotRecordedWarning,
    ReplayDivergence,
    StructuredMutation,
    StructuredRecombination,
    VariationContext,
    call_operator,
    operator_key,
)

__all__ = [
    "ExternalMutation",
    "ExternalRecombination",
    "OperatorError",
    "OperatorNotRecordedWarning",
    "ReplayDivergence",
    "SequenceCrossover",
    "SequenceMutation",
    "StructuredGeneticAlgorithm",
    "StructuredMutation",
    "StructuredRecombination",
    "VariationContext",
    "call_operator",
    "operator_key",
]
