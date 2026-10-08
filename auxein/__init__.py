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

from auxein.backend import Backend
from auxein.core import BatchResult, Objective, Result, Status
from auxein.driver import Budget, RecordingDisabledWarning, RunResult, arun, run
from auxein.evaluators import FunctionEvaluator, VectorisedEvaluator
from auxein.recording import open_run
from auxein.spaces import Box
from auxein.strategies import GeneticAlgorithm, RandomSearch

try:
    __version__ = metadata.version("auxein")
except metadata.PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "Backend",
    "BatchResult",
    "Box",
    "Budget",
    "FunctionEvaluator",
    "GeneticAlgorithm",
    "Objective",
    "RandomSearch",
    "RecordingDisabledWarning",
    "Result",
    "RunResult",
    "Status",
    "VectorisedEvaluator",
    "__version__",
    "arun",
    "open_run",
    "run",
]
