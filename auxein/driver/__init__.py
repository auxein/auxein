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
from auxein.driver.held_out import HeldOutCandidate, HeldOutReport, HeldOutScenario, aevaluate_held_out, evaluate_held_out
from auxein.driver.result import RunResult
from auxein.recording import ResumeError

__all__ = [
    "AllEvaluationsFailedError",
    "Budget",
    "ConfigurationMismatchError",
    "DriverError",
    "EvaluationFailureWarning",
    "HeldOutCandidate",
    "HeldOutReport",
    "HeldOutScenario",
    "EvaluatorError",
    "RecordingDisabledWarning",
    "ReplayMismatchError",
    "ResumeError",
    "ResumeWarning",
    "RunResult",
    "StrategyError",
    "SteadyStateVectorisationWarning",
    "aevaluate_held_out",
    "aresume",
    "arun",
    "evaluate_held_out",
    "resume",
    "run",
]
