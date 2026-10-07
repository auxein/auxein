"""Evaluators (design doc §5.3): they turn batches of candidates into evaluations."""

from auxein.evaluators.errors import EvaluationError
from auxein.evaluators.function import FunctionEvaluator
from auxein.evaluators.vectorised import VectorisedEvaluator

__all__ = ["EvaluationError", "FunctionEvaluator", "VectorisedEvaluator"]
