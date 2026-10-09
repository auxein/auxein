"""Evaluators (design doc §5.3): they turn batches of candidates into evaluations."""

from auxein.evaluators.episode import EpisodeEvaluator
from auxein.evaluators.errors import EvaluationError
from auxein.evaluators.function import FunctionEvaluator
from auxein.evaluators.vectorised import VectorisedEvaluator

__all__ = ["EpisodeEvaluator", "EvaluationError", "FunctionEvaluator", "VectorisedEvaluator"]
