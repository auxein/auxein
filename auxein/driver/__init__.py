"""The driver (design doc §9): it owns the loop that asks a strategy for candidates, evaluates them, records and tells."""

from auxein.driver.budget import Budget
from auxein.driver.driver import arun, run
from auxein.driver.errors import (
    DriverError,
    EvaluatorError,
    RecordingDisabledWarning,
    SteadyStateVectorisationWarning,
    StrategyError,
)
from auxein.driver.result import RunResult

__all__ = [
    "Budget",
    "DriverError",
    "EvaluatorError",
    "RecordingDisabledWarning",
    "RunResult",
    "StrategyError",
    "SteadyStateVectorisationWarning",
    "arun",
    "run",
]
