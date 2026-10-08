"""Executors: a small abstraction over "call this synchronous function somewhere, and let me await the result".

Evaluators are asynchronous so that many evaluations can be in progress at once, but most user functions are plain
synchronous code. An executor is where such a function runs: on the driver's own thread (`inline`, no hand-off at all),
in a thread pool (good for I/O, and for numeric code that releases the GIL) or in a process pool (for CPU-bound
pure-Python code). `async def` functions never go through an executor: they run natively on the driver's event loop.

Pools belong to one run: the driver creates the executor when the run starts and always calls `shutdown` when it ends.
"""

import asyncio
import multiprocessing
import pickle
from collections.abc import Callable
from concurrent.futures import Executor as _FuturesExecutor
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Literal, Protocol, TypeVar

T = TypeVar("T")

ExecutorName = Literal["auto", "inline", "thread", "process"]
"""What a user asks for. `auto` resolves by concurrency; processes are never chosen automatically."""
ExecutorKind = Literal["inline", "thread", "process"]
"""What a run uses, once `auto` is resolved."""


class ExecutorError(RuntimeError):
    """A function or its arguments could not be sent to a worker process (or its result could not be sent back)."""


class Executor(Protocol):
    """Runs synchronous callables for the driver. Obtained through `EvalContext.call`, not directly."""

    kind: ExecutorKind

    async def call(self, fn: Callable[..., T], /, *args: object) -> T:
        """Run `fn(*args)` and return its result. Exceptions of `fn` propagate unchanged."""
        ...

    def shutdown(self) -> None:
        """Release the pool, waiting for work already running. Safe to call more than once."""
        ...


def resolve_executor(name: ExecutorName, concurrency: int) -> ExecutorKind:
    """`auto` is inline when `concurrency == 1` and threads otherwise. Processes are always an explicit choice, because
    they change what user code may do (arguments and results must be picklable, globals are not shared)."""
    if name == "auto":
        return "inline" if concurrency == 1 else "thread"
    if name not in ("inline", "thread", "process"):
        raise ValueError(f"unknown executor {name!r}: expected 'auto', 'inline', 'thread' or 'process'")
    return name


class InlineExecutor:
    """Calls the function on the driver's thread: no hand-off, so the least overhead; it blocks the event loop meanwhile."""

    kind: ExecutorKind = "inline"

    async def call(self, fn: Callable[..., T], /, *args: object) -> T:
        return fn(*args)

    def shutdown(self) -> None:
        pass


class _PoolExecutor:
    kind: ExecutorKind
    _pool: _FuturesExecutor

    def shutdown(self) -> None:
        # cancel_futures drops work not yet started; work already running cannot be interrupted (Python has no way to
        # stop a thread or to cleanly stop a process mid-call), so the run waits for it before it ends.
        self._pool.shutdown(wait=True, cancel_futures=True)


class ThreadExecutor(_PoolExecutor):
    """A thread pool of `max_workers` threads."""

    kind: ExecutorKind = "thread"

    def __init__(self, max_workers: int) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="auxein-eval")

    async def call(self, fn: Callable[..., T], /, *args: object) -> T:
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)


def _run_payload(payload: bytes) -> bytes:
    """Runs in the worker: unpickle the call, make it, pickle the result. Failures to move data say why."""
    try:
        fn, args = pickle.loads(payload)
    except Exception as error:
        raise ExecutorError(
            f"a worker process could not rebuild the function or its arguments ({type(error).__name__}: {error}). "
            "Functions defined in a notebook or in a script's __main__ are not importable in a worker started with 'spawn' "
            "(the default on macOS and Windows): define the function in a module, or use executor='thread'."
        ) from None
    result = fn(*args)
    try:
        return pickle.dumps(result)
    except Exception as error:
        raise ExecutorError(
            f"the function returned a value that cannot be sent back to the parent process "
            f"({type(error).__name__}: {error}): return plain numbers or a Result, or use executor='thread'."
        ) from None


class ProcessExecutor(_PoolExecutor):
    """A pool of `max_workers` worker processes, always started with `spawn`.

    `spawn` is what macOS and Windows use, and it does not copy the parent's state, so a run behaves the same on every
    platform (and a fork cannot inherit a held lock or a half-initialised GPU context). The price is that the function
    must be importable by name in the worker, and that its arguments and result must be picklable. Both are checked, and
    a failure explains the fixes. The call is pickled once, here, so that an unpicklable function fails with a clear
    error in the caller instead of inside the pool's feeder thread.
    """

    kind: ExecutorKind = "process"

    def __init__(self, max_workers: int) -> None:
        self._pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn"))

    async def call(self, fn: Callable[..., T], /, *args: object) -> T:
        name = getattr(fn, "__qualname__", type(fn).__name__)
        try:
            payload = pickle.dumps((fn, args))
        except Exception as error:
            raise ExecutorError(
                f"cannot send {name!r} or its arguments to a worker process ({type(error).__name__}: {error}). "
                "Lambdas, local functions and functions defined in a notebook cannot be pickled: "
                "define the function at the top level of a module, or use executor='thread'."
            ) from error
        result = await asyncio.get_running_loop().run_in_executor(self._pool, _run_payload, payload)
        return pickle.loads(result)


def make_executor(kind: ExecutorKind, concurrency: int) -> Executor:
    """Builds the executor of a run. `concurrency` sizes the pools; inline has no pool."""
    if kind == "inline":
        return InlineExecutor()
    if kind == "thread":
        return ThreadExecutor(concurrency)
    return ProcessExecutor(concurrency)
