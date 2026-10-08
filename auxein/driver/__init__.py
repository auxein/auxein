"""The driver (design doc §9): it owns the loop that asks a strategy for candidates, evaluates them, records and tells."""

from auxein.driver.budget import Budget
from auxein.driver.driver import aresume, arun, resume, run
from auxein.driver.errors import (
    AllEvaluationsFailedError,
    ConfigurationMismatchError,
    DriverError,
    EvaluationFailureWarning,
    EvaluatorError,
    RecordingDisabledWarning,
    ReplayMismatchError,
    ResumeWarning,
    SteadyStateVectorisationWarning,
    StrategyError,
)
from auxein.driver.result import RunResult
from auxein.recording import ResumeError

__all__ = [
    "AllEvaluationsFailedError",
    "Budget",
    "ConfigurationMismatchError",
    "DriverError",
    "EvaluationFailureWarning",
    "EvaluatorError",
    "RecordingDisabledWarning",
    "ReplayMismatchError",
    "ResumeError",
    "ResumeWarning",
    "RunResult",
    "StrategyError",
    "SteadyStateVectorisationWarning",
    "aresume",
    "arun",
    "resume",
    "run",
]
