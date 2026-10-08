"""Where synchronous user functions run: inline, in threads or in processes (design doc §5.3)."""

from auxein.execution.errors import (
    AbandonedEvaluationWarning,
    EvaluationTimeout,
    ExecutorError,
    RemoteError,
    RemoteTraceback,
    WorkerCrashed,
)
from auxein.execution.executors import (
    Executor,
    ExecutorKind,
    ExecutorName,
    InlineExecutor,
    ProcessExecutor,
    ThreadExecutor,
    make_executor,
    resolve_executor,
)

__all__ = [
    "AbandonedEvaluationWarning",
    "EvaluationTimeout",
    "Executor",
    "ExecutorError",
    "ExecutorKind",
    "ExecutorName",
    "InlineExecutor",
    "ProcessExecutor",
    "RemoteError",
    "RemoteTraceback",
    "ThreadExecutor",
    "WorkerCrashed",
    "make_executor",
    "resolve_executor",
]
