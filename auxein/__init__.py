"""Auxein: a Python framework for evolving agents that act in environments.

The common case needs one import: a strategy, an evaluator, a space, a budget and `run`.

    import auxein

    result = auxein.run(
        strategy=auxein.GeneticAlgorithm(),
        evaluator=auxein.VectorisedEvaluator(lambda X: (X * X).sum(axis=1)),
        space=auxein.Box(-5.0, 5.0, dim=10),
        budget=auxein.Budget(evaluations=20_000),
        seed=42,
    )

Everything else is importable from its subpackage: `auxein.core` (candidates, batches, evaluations and the protocols),
`auxein.strategies.ga` (the genetic algorithm's operators), `auxein.backend`, `auxein.random`, `auxein.spaces`,
`auxein.driver`, `auxein.evaluators` and `auxein.recording`. The design is in `docs/design/core.md`.
"""

from importlib import metadata

from auxein.aggregators import Aggregator
from auxein.backend import Backend
from auxein.core import BatchResult, Objective, Result, Status
from auxein.driver import Budget, RecordingDisabledWarning, RunResult, aresume, arun, resume, run
from auxein.environments import EpisodeResult, Scenario, ScenarioSet
from auxein.evaluators import EpisodeEvaluator, FunctionEvaluator, VectorisedEvaluator
from auxein.recording import open_run
from auxein.spaces import Box, SequenceSpace
from auxein.strategies import GeneticAlgorithm, RandomSearch, StructuredGeneticAlgorithm

try:
    __version__ = metadata.version("auxein")
except metadata.PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "Aggregator",
    "Backend",
    "BatchResult",
    "Box",
    "Budget",
    "EpisodeEvaluator",
    "EpisodeResult",
    "FunctionEvaluator",
    "GeneticAlgorithm",
    "Objective",
    "RandomSearch",
    "RecordingDisabledWarning",
    "Result",
    "RunResult",
    "Scenario",
    "ScenarioSet",
    "SequenceSpace",
    "Status",
    "StructuredGeneticAlgorithm",
    "VectorisedEvaluator",
    "__version__",
    "aresume",
    "arun",
    "open_run",
    "resume",
    "run",
]
