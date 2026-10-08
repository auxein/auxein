"""What can go wrong when a synchronous function is run somewhere else than the driver's thread."""


class ExecutorError(RuntimeError):
    """A function or its arguments could not be sent to a worker process, or its result could not be sent back, or a worker
    could not start. This is misconfiguration, not a result: it fails the run whatever the failure policy."""


class EvaluationTimeout(Exception):
    """An evaluation ran for longer than the run's `timeout`. Evaluators turn it into a `TIMEOUT` evaluation."""

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        super().__init__(f"the evaluation did not finish within {timeout:g} s")


class WorkerCrashed(RuntimeError):
    """The worker process died while it was evaluating (a segfault, `os._exit`, being killed by the OS for memory). It is
    the failure of the candidate it was evaluating, and the worker is replaced; evaluations on other workers go on."""


class RemoteTraceback(Exception):
    """The traceback of an exception that happened in a worker process, as text, chained as the cause of the exception
    re-raised in the parent: the real traceback objects do not cross a process boundary."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text

    def __str__(self) -> str:
        return f'\n"""\n{self.text}"""'


class RemoteError(Exception):
    """An exception of a worker process that could not be sent to the parent as an object (it is not picklable)."""


class AbandonedEvaluationWarning(UserWarning):
    """An evaluation running in a thread timed out. A thread cannot be stopped, so its result is discarded and the thread
    keeps running in the background until the function returns (it never blocks the interpreter from exiting). Use an
    `async def` function or `executor="process"` for timeouts that stop the evaluation."""
