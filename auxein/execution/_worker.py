"""The code that runs inside a worker process of `ProcessExecutor`.

Kept small and free of imports beyond the standard library, so that a spawned worker starts quickly; the user's function
brings in whatever else it needs when it is unpickled. A worker evaluates one call at a time.

Protocol, over a pipe, in `bytes`: the worker first sends `("ready",)`. Then, for each non-empty message received (a
pickled `(fn, args)`), it sends one reply: `("ok", result_bytes)`, `("raised", type, message, traceback, pickled_or_None)`
or `("misconfig", message)`. An empty message means "stop". EOF on the pipe (the parent died) also stops the worker, so
workers never outlive the run that started them.
"""

import pickle
import traceback
from multiprocessing.connection import Connection


def _serve(payload: bytes) -> bytes:
    try:
        fn, args = pickle.loads(payload)
    except Exception as error:
        return pickle.dumps(
            (
                "misconfig",
                f"a worker process could not rebuild the function or its arguments ({type(error).__name__}: {error}). "
                "Functions defined in a notebook or in a script's __main__ are not importable in a worker started with 'spawn' "
                "(the default on macOS and Windows): define the function in a module, or use executor='thread'.",
            )
        )
    try:
        result = fn(*args)
    except BaseException as error:
        text = "".join(traceback.format_exception(error)).rstrip()
        try:
            blob: bytes | None = pickle.dumps(error)
            pickle.loads(blob)  # an exception that pickles but cannot be rebuilt would break the parent
        except Exception:
            blob = None
        return pickle.dumps(("raised", type(error).__qualname__, str(error), text, blob))
    try:
        return pickle.dumps(("ok", pickle.dumps(result)))
    except Exception as error:
        return pickle.dumps(
            (
                "misconfig",
                f"the function returned a value that cannot be sent back to the parent process ({type(error).__name__}: "
                f"{error}): return plain numbers or a Result, or use executor='thread'.",
            )
        )


def worker_main(connection: Connection) -> None:
    connection.send_bytes(pickle.dumps(("ready",)))
    while True:
        try:
            payload = connection.recv_bytes()
        except (EOFError, OSError):
            return
        if not payload:
            return
        connection.send_bytes(_serve(payload))
