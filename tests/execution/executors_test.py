import asyncio
import multiprocessing
import threading

import pytest

from auxein.execution import (
    ExecutorError,
    InlineExecutor,
    ProcessExecutor,
    ThreadExecutor,
    make_executor,
    resolve_executor,
)
from tests.support import workers


def call(executor, fn, *args):
    return asyncio.run(executor.call(fn, *args))


def test_auto_is_inline_for_one_worker_and_threads_for_more_and_never_processes():
    assert resolve_executor("auto", 1) == "inline"
    assert resolve_executor("auto", 2) == "thread"
    assert resolve_executor("auto", 64) == "thread"
    for name in ("inline", "thread", "process"):
        assert resolve_executor(name, 1) == name and resolve_executor(name, 8) == name  # explicit choices are kept
    with pytest.raises(ValueError, match="unknown executor 'fork'"):
        resolve_executor("fork", 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["inline", "thread", "process"])
def test_each_executor_runs_a_function_and_returns_its_result(kind: str):
    executor = make_executor(kind, 2)  # type: ignore[arg-type]
    try:
        assert executor.kind == kind
        assert call(executor, workers.double, 21) == 42
    finally:
        executor.shutdown()


@pytest.mark.parametrize("kind", ["inline", "thread", "process"])
def test_exceptions_of_the_function_propagate_unchanged(kind: str):
    executor = make_executor(kind, 1)  # type: ignore[arg-type]
    try:
        with pytest.raises(ZeroDivisionError):
            call(executor, workers.divide_by_zero)
    finally:
        executor.shutdown()


def test_a_thread_executor_runs_off_the_calling_thread():
    where = []

    def record() -> None:
        where.append(threading.current_thread())

    executor = ThreadExecutor()
    try:
        call(executor, record)
        call(InlineExecutor(), record)
    finally:
        executor.shutdown()
    assert where[0] is not threading.main_thread() and where[0].name.startswith("auxein-eval")
    assert where[1] is threading.main_thread()


def test_a_process_executor_runs_in_another_process_started_with_spawn():
    import os

    executor = ProcessExecutor(1)
    try:
        assert call(executor, os.getpid) != os.getpid()
        assert executor._pool._context.get_start_method() == "spawn"  # pyright: ignore[reportPrivateUsage]
    finally:
        executor.shutdown()


def test_a_lambda_cannot_go_to_a_process_and_the_error_says_what_to_do():
    executor = ProcessExecutor(1)
    try:
        with pytest.raises(ExecutorError) as raised:
            call(executor, lambda: 0)
        message = str(raised.value)
        assert "worker process" in message and "module" in message and "executor='thread'" in message
        with pytest.raises(ExecutorError, match="cannot send"):
            call(executor, workers.identity, workers.make_lambda())  # an unpicklable argument
    finally:
        executor.shutdown()


def test_what_a_worker_cannot_rebuild_or_send_back_is_explained():
    async def main() -> None:
        executor = ProcessExecutor(1)  # one event loop for the whole session: a pool belongs to one run
        try:
            with pytest.raises(ExecutorError, match="could not rebuild"):
                await executor.call(workers.identity, workers.Unrebuildable())
            with pytest.raises(ExecutorError, match="cannot be sent back"):
                await executor.call(workers.make_lambda)  # returns a lambda
            assert await executor.call(workers.double, 1) == 2  # the pool is still healthy
        finally:
            executor.shutdown()

    asyncio.run(main())


@pytest.mark.parametrize("kind", ["thread", "process"])
def test_pools_are_shut_down_and_leave_nothing_behind(kind: str):
    before = {t for t in threading.enumerate() if t.name.startswith("auxein-eval")}
    executor = make_executor(kind, 2)  # type: ignore[arg-type]
    call(executor, workers.double, 1)
    executor.shutdown()
    executor.shutdown()  # twice is fine
    assert {t for t in threading.enumerate() if t.name.startswith("auxein-eval")} == before
    assert multiprocessing.active_children() == []


def test_shutdown_does_not_wait_for_a_thread_that_is_still_running():
    """Threads are daemons: an abandoned or interrupted function must never keep the program (or the run) from ending."""
    executor = ThreadExecutor()
    release = threading.Event()
    finished = []

    async def main() -> None:
        task = asyncio.ensure_future(executor.call(lambda: (release.wait(5), finished.append(1))))
        await asyncio.sleep(0.02)
        executor.shutdown()  # returns at once although the function has not returned
        assert finished == []
        release.set()
        await task

    asyncio.run(main())
    assert finished == [1]


def test_an_exception_in_the_function_does_not_cost_a_new_worker_process():
    import os

    async def main() -> tuple[int, int]:
        executor = ProcessExecutor(1)
        try:
            pid = await executor.call(os.getpid)
            for _ in range(4):
                with pytest.raises(ZeroDivisionError) as raised:
                    await executor.call(workers.divide_by_zero)
                assert "workers.py" in str(raised.value.__cause__)  # the worker's traceback is chained as text
            return pid, await executor.call(os.getpid)
        finally:
            executor.shutdown()

    before, after = asyncio.run(main())
    assert before == after  # the worker that raised was healthy: it served the next call too


def test_a_timed_out_worker_is_killed_and_replaced_while_another_keeps_its_work():
    import os

    from auxein.execution import EvaluationTimeout

    async def main() -> tuple[int, int, int]:
        executor = ProcessExecutor(2)
        try:
            first = await executor.call(os.getpid)
            slow = asyncio.ensure_future(executor.call(workers.pause_for, 0.4, timeout=None))
            with pytest.raises(EvaluationTimeout):
                await executor.call(workers.pause_for, 30.0, timeout=0.3)
            return first, await slow, await executor.call(os.getpid)
        finally:
            executor.shutdown()

    first, slow_pid, later = asyncio.run(main())
    assert slow_pid != 0 and later != 0
    assert multiprocessing.active_children() == []
