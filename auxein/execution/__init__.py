"""Where synchronous user functions run: inline, in threads or in processes (design doc §5.3)."""

from auxein.execution.executors import (
    Executor,
    ExecutorError,
    ExecutorKind,
    ExecutorName,
    InlineExecutor,
    ProcessExecutor,
    ThreadExecutor,
    make_executor,
    resolve_executor,
)

__all__ = [
    "Executor",
    "ExecutorError",
    "ExecutorKind",
    "ExecutorName",
    "InlineExecutor",
    "ProcessExecutor",
    "ThreadExecutor",
    "make_executor",
    "resolve_executor",
]
