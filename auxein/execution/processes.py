"""Auxein's own pool of worker processes.

`concurrent.futures.ProcessPoolExecutor` cannot kill one task, and when a worker dies it breaks the whole pool, failing
every evaluation in progress. This pool has up to `max_workers` workers, each running **one evaluation at a time** and
talking to the parent over its own pipe, so that one can be killed (a timeout) or die (a crash) without touching the others.

Workers are started lazily, the first time work needs one, and a worker that is killed or dies is replaced the same way.
A worker says `ready` once it has started, so a worker that cannot start (typically a script without the
`if __name__ == "__main__":` guard that `spawn` needs) is reported as a misconfiguration, not as a failed candidate.
Workers are daemon processes and exit on end-of-file when the parent goes away, so they never outlive it.
"""

import asyncio
import multiprocessing
import multiprocessing.context
import pickle
import signal
import threading
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

from auxein.execution._worker import worker_main
from auxein.execution.errors import EvaluationTimeout, ExecutorError, RemoteError, RemoteTraceback, WorkerCrashed

START_TIMEOUT = 120.0
"""Seconds a worker may take to start (importing the program's modules can be slow on a loaded machine)."""
_GRACE = 0.5


def _describe_exit(code: int | None) -> str:
    if code is None:
        return "its exit status is unknown"
    if code < 0:
        try:
            return f"it was killed by signal {signal.Signals(-code).name} ({-code})"
        except ValueError:
            return f"it was killed by signal {-code}"
    return f"it exited with code {code}"


class _Worker:
    """One worker process, the pipe to it and the thread that reads its replies."""

    def __init__(self, context: multiprocessing.context.SpawnContext, loop: asyncio.AbstractEventLoop) -> None:
        parent, child = context.Pipe(duplex=True)
        self.process: BaseProcess = context.Process(target=worker_main, args=(child,), name="auxein-worker", daemon=True)
        self.process.start()
        child.close()  # only the worker holds this end, so that its death is seen as end-of-file
        self.connection: Connection = parent
        self._loop = loop
        self._replies: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.dead = False
        self.busy = False
        self._reader = threading.Thread(target=self._read, name="auxein-worker-reader", daemon=True)
        self._reader.start()

    def _read(self) -> None:
        while True:
            try:
                data: bytes | None = self.connection.recv_bytes()
            except (EOFError, OSError):
                data = None
            try:
                self._loop.call_soon_threadsafe(self._deliver, data)
            except RuntimeError:  # the loop is closed: the run is over
                return
            if data is None:
                return

    def _deliver(self, data: bytes | None) -> None:
        if data is None:
            self.dead = True
        self._replies.put_nowait(data)

    async def reply(self) -> bytes | None:
        """The next reply, or None if the worker died."""
        if self.dead and self._replies.empty():
            return None
        return await self._replies.get()

    def exit_description(self) -> str:
        self.process.join(timeout=2.0)
        return _describe_exit(self.process.exitcode)

    def stop(self, *, graceful: bool) -> None:
        """End the process: asked politely if `graceful`, else terminated; killed if it does not go."""
        if graceful and self.process.is_alive():
            try:
                self.connection.send_bytes(b"")
            except OSError:
                pass
            self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=_GRACE)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2.0)
        self.connection.close()
        self._reader.join(timeout=1.0)


class ProcessPool:
    """Up to `max_workers` spawned workers, handing each call to an idle one (see the module docstring)."""

    def __init__(self, max_workers: int) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        self._context = multiprocessing.get_context("spawn")
        self._slots = asyncio.Semaphore(max_workers)
        self._idle: list[_Worker] = []
        self._all: set[_Worker] = set()
        self._closed = False

    async def run(self, payload: bytes, timeout: float | None) -> object:
        if self._closed:
            raise RuntimeError("this executor has been shut down")
        await self._slots.acquire()
        worker: _Worker | None = None
        try:
            worker = await self._take()
            return await self._exchange(worker, payload, timeout)
        except BaseException:
            # Whatever went wrong (a timeout, a crash, a cancellation of this evaluation), a worker that may still be busy
            # or is gone must not serve the next call: its late reply would be taken for that call's.
            if worker is not None:
                self._discard(worker)
            raise
        finally:
            if worker is not None and worker in self._all:
                worker.busy = False
                self._idle.append(worker)
            self._slots.release()

    async def _take(self) -> _Worker:
        while self._idle:
            worker = self._idle.pop()
            if not worker.dead and worker.process.is_alive():
                return worker
            self._discard(worker)
        worker = _Worker(self._context, asyncio.get_running_loop())
        self._all.add(worker)
        try:
            ready = await asyncio.wait_for(worker.reply(), START_TIMEOUT)
        except TimeoutError:
            self._discard(worker)
            raise ExecutorError(f"a worker process did not start within {START_TIMEOUT:g} s") from None
        except BaseException:
            self._discard(worker)
            raise
        if ready is None:
            description = worker.exit_description()
            self._discard(worker)
            raise ExecutorError(
                f"a worker process failed to start ({description}). With the 'spawn' start method the program's main module is "
                "imported again in each worker, so a script must guard its entry point with `if __name__ == '__main__':`"
            )
        return worker

    async def _exchange(self, worker: _Worker, payload: bytes, timeout: float | None) -> object:
        worker.busy = True
        try:
            worker.connection.send_bytes(payload)
        except OSError:
            raise WorkerCrashed(f"the worker process died before the evaluation started ({worker.exit_description()})") from None
        try:
            raw = await asyncio.wait_for(worker.reply(), timeout)
        except TimeoutError:
            raise EvaluationTimeout(timeout if timeout is not None else 0.0) from None  # the worker is killed by `run`
        if raw is None:
            raise WorkerCrashed(f"the worker process died while evaluating this candidate: {worker.exit_description()}")
        return self._unpack(raw)

    @staticmethod
    def _unpack(raw: bytes) -> object:
        kind, *rest = pickle.loads(raw)
        if kind == "ok":
            return pickle.loads(rest[0])
        if kind == "misconfig":
            raise ExecutorError(rest[0])
        type_name, message, text, blob = rest
        error: BaseException = pickle.loads(blob) if blob is not None else RemoteError(f"{type_name}: {message}")
        raise error from RemoteTraceback(text)

    def _discard(self, worker: _Worker) -> None:
        """Remove a worker from the pool and end its process; the next call that needs one starts a replacement."""
        self._all.discard(worker)
        if worker in self._idle:
            self._idle.remove(worker)
        worker.stop(graceful=False)

    def shutdown(self) -> None:
        """Stop every worker: idle ones are asked to stop, busy ones (an error or Ctrl-C interrupted the run) are killed."""
        self._closed = True
        for worker in list(self._all):
            self._all.discard(worker)
            worker.stop(graceful=not worker.busy)
        self._idle.clear()
