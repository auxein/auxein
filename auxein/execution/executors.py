"""Executors: a small abstraction over "call this synchronous function somewhere, and let me await the result".

Evaluators are asynchronous so that many evaluations can be in progress at once, but most user functions are plain
synchronous code. An executor is where such a function runs: on the driver's own thread (`inline`, no hand-off at all),
in a thread (good for I/O, and for numeric code that releases the GIL) or in a worker process (for CPU-bound pure-Python
code, and the only place where a runaway evaluation can really be stopped). `async def` functions never go through an
executor: they run natively on the driver's event loop.

How hard a timeout is depends on the executor, because Python cannot stop a thread: a process is killed and replaced, a
thread is abandoned (its result is discarded and it keeps running, as a daemon, in the background), and an inline call
cannot be timed out at all, since it blocks the event loop.

Pools belong to one run: the driver creates the executor when the run starts and always calls `shutdown` when it ends.
"""

import asyncio
import pickle
import queue
import threading
import warnings
from collections.abc import Callable
from concurrent.futures import Future
from typing import Literal, Protocol, TypeVar

from auxein.execution.errors import AbandonedEvaluationWarning, EvaluationTimeout, ExecutorError
from auxein.execution.processes import ProcessPool

T = TypeVar("T")

ExecutorName = Literal["auto", "inline", "thread", "process"]
"""What a user asks for. `auto` resolves by concurrency and timeout; processes are never chosen automatically."""
ExecutorKind = Literal["inline", "thread", "process"]
"""What a run uses, once `auto` is resolved."""


class Executor(Protocol):
    """Runs synchronous callables for the driver. Obtained through `EvalContext.call`, not directly."""

    kind: ExecutorKind

    async def call(self, fn: Callable[..., T], /, *args: object, timeout: float | None = None) -> T:
        """Run `fn(*args)` and return its result. Exceptions of `fn` propagate unchanged. With a `timeout` (seconds), raise
        `EvaluationTimeout` if it takes longer; a worker that dies raises `WorkerCrashed`."""
        ...

    def abandoned_running(self) -> int:
        """How many timed-out evaluations are still running in the background (threads only; 0 for the others)."""
        ...

    def shutdown(self) -> None:
        """Stop the workers. Safe to call more than once."""
        ...


def resolve_executor(name: ExecutorName, concurrency: int, timeout: float | None = None) -> ExecutorKind:
    """`auto` is inline when `concurrency == 1` and threads otherwise, except that a `timeout` needs threads even for
    one worker: a synchronous function called inline blocks the event loop, so nothing could time it out. An explicit
    `inline` with a `timeout` is refused for the same reason. Processes are always an explicit choice, because they
    change what user code may do (arguments and results must be picklable, globals are not shared)."""
    if name not in ("auto", "inline", "thread", "process"):
        raise ValueError(f"unknown executor {name!r}: expected 'auto', 'inline', 'thread' or 'process'")
    if name == "auto":
        return "thread" if concurrency > 1 or timeout is not None else "inline"
    if name == "inline" and timeout is not None:
        raise ValueError(
            "executor='inline' cannot be combined with a timeout: a synchronous function called on the driver's thread "
            "blocks the event loop, so nothing could interrupt it. Use executor='auto' or 'thread' (the evaluation is "
            "abandoned on timeout: soft), executor='process' (the worker is killed: hard), or an async def function"
        )
    return name


class InlineExecutor:
    """Calls the function on the driver's thread: no hand-off, so the least overhead; it blocks the event loop meanwhile."""

    kind: ExecutorKind = "inline"

    async def call(self, fn: Callable[..., T], /, *args: object, timeout: float | None = None) -> T:
        return fn(*args)  # a timeout cannot be applied: the run refuses to combine it with this executor

    def abandoned_running(self) -> int:
        return 0

    def shutdown(self) -> None:
        pass


class _Worker:
    def __init__(self) -> None:
        self.busy = False
        self.thread: threading.Thread | None = None


class ThreadExecutor:
    """Runs functions in daemon threads, grown on demand and reused.

    Not a `ThreadPoolExecutor`: its threads are joined when the interpreter exits, so one evaluation stuck forever would
    stop the program from ending. Daemon threads never block exit. A thread is started only when none is idle, so the
    number of threads follows the number of evaluations in progress (which the run bounds by `concurrency`) plus the
    abandoned ones, which are temporary extras.

    On a timeout the evaluation is **abandoned**: the caller gets `EvaluationTimeout` straight away and the late result,
    if it ever comes, is discarded. The thread itself cannot be stopped and goes on until the function returns.
    """

    kind: ExecutorKind = "thread"

    def __init__(self) -> None:
        self._jobs: queue.SimpleQueue[tuple[Future[object], Callable[..., object], tuple[object, ...]] | None] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._idle = 0
        self._workers: list[_Worker] = []
        self._closed = False
        self._abandoned: list[Future[object]] = []
        self._warned = False

    def _loop(self, worker: _Worker) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            future, fn, args = job
            worker.busy = True
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(fn(*args))
                except BaseException as error:  # delivered to the awaiting task, which decides what it means
                    future.set_exception(error)
            worker.busy = False
            with self._lock:
                self._idle += 1

    def _submit(self, fn: Callable[..., object], args: tuple[object, ...]) -> Future[object]:
        future: Future[object] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("this executor has been shut down")
            self._jobs.put((future, fn, args))
            if self._idle > 0:
                self._idle -= 1
            else:
                worker = _Worker()
                worker.thread = threading.Thread(target=self._loop, args=(worker,), name="auxein-eval", daemon=True)
                self._workers.append(worker)
                worker.thread.start()
        return future

    async def call(self, fn: Callable[..., T], /, *args: object, timeout: float | None = None) -> T:
        future = self._submit(fn, args)
        wrapped = asyncio.wrap_future(future)
        if timeout is None:
            return await wrapped  # type: ignore[return-value]
        done, _ = await asyncio.wait({wrapped}, timeout=timeout)  # unlike wait_for, a TimeoutError of fn itself is not confused
        if wrapped in done:
            return wrapped.result()  # type: ignore[return-value]
        wrapped.cancel()  # a thread that is running cannot be cancelled: its late result is dropped by the wrapper
        self._abandoned.append(future)
        if not self._warned:
            self._warned = True
            warnings.warn(
                f"an evaluation in a thread took longer than the timeout ({timeout:g} s) and was abandoned: a thread cannot be "
                "stopped, so it keeps running in the background and its result is discarded. Use an async def function, or "
                "executor='process', for timeouts that stop the evaluation.",
                AbandonedEvaluationWarning,
                stacklevel=2,
            )
        raise EvaluationTimeout(timeout)

    def abandoned_running(self) -> int:
        return sum(1 for future in self._abandoned if not future.done())

    def shutdown(self) -> None:
        """Ask the threads to stop and wait for the idle ones. A thread still running a function (an abandoned one, or one
        interrupted by an error or Ctrl-C) is left to finish on its own: it is a daemon and does not block exit."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            workers = list(self._workers)
        for _ in workers:
            self._jobs.put(None)
        for worker in workers:
            if worker.thread is not None and not worker.busy:
                worker.thread.join(timeout=2.0)


class ProcessExecutor:
    """Runs functions in worker processes of Auxein's own pool, always started with `spawn` (see `ProcessPool`).

    `spawn` is what macOS and Windows use, and it does not copy the parent's state, so a run behaves the same on every
    platform. The price is that the function must be importable by name in the worker, and that its arguments and result
    must be picklable. Both are checked, and a failure explains the fixes. The call is pickled once, here, so that an
    unpicklable function fails with a clear error in the caller.
    """

    kind: ExecutorKind = "process"

    def __init__(self, max_workers: int) -> None:
        self._pool = ProcessPool(max_workers)

    async def call(self, fn: Callable[..., T], /, *args: object, timeout: float | None = None) -> T:
        name = getattr(fn, "__qualname__", type(fn).__name__)
        try:
            payload = pickle.dumps((fn, args))
        except Exception as error:
            raise ExecutorError(
                f"cannot send {name!r} or its arguments to a worker process ({type(error).__name__}: {error}). "
                "Lambdas, local functions and functions defined in a notebook cannot be pickled: "
                "define the function at the top level of a module, or use executor='thread'."
            ) from error
        return await self._pool.run(payload, timeout)  # type: ignore[no-any-return]

    def abandoned_running(self) -> int:
        return 0  # a timed-out worker is killed, not abandoned

    def shutdown(self) -> None:
        self._pool.shutdown()


def make_executor(kind: ExecutorKind, concurrency: int) -> Executor:
    """Builds the executor of a run. `concurrency` sizes the process pool; threads grow on demand, inline has no pool."""
    if kind == "inline":
        return InlineExecutor()
    if kind == "thread":
        return ThreadExecutor()
    return ProcessExecutor(concurrency)
