"""Functions that worker processes can import by name, and probes for what runs concurrently.

Only numpy is imported here, so that a spawned worker starts quickly.
"""

import asyncio
import threading
import time

import numpy as np


def double(x: float) -> float:
    return 2 * x


def divide_by_zero() -> float:
    return 1 / 0


def not_a_number(x: object) -> str:
    return "three"


def divide_by_zero_1(x: object) -> float:
    return 1 / 0


def pause(seconds: float) -> float:
    time.sleep(seconds)
    return seconds


def make_lambda() -> object:
    return lambda: 0


def sphere(genome: np.ndarray) -> float:
    return float((genome * genome).sum())


def jittery_sphere(genome: np.ndarray, rng: object) -> float:
    """A sphere that takes between 0 and 6 ms, as long as its own candidate's stream says, so that completion order varies."""
    time.sleep(float(rng.uniform(1)[0]) * 0.006)  # type: ignore[attr-defined]
    return float((genome * genome).sum())


async def async_jittery_sphere(genome: np.ndarray, rng: object) -> float:
    await asyncio.sleep(float(rng.uniform(1)[0]) * 0.006)  # type: ignore[attr-defined]
    return float((genome * genome).sum())


async def async_uneven(genome: np.ndarray, rng: object) -> float:
    """Takes between 1 and 20 ms, as its candidate's own stream says."""
    await asyncio.sleep(0.001 + 0.019 * float(rng.uniform(1)[0]))  # type: ignore[attr-defined]
    return float((genome * genome).sum())


class Unrebuildable:
    """Pickles fine, but cannot be rebuilt in the worker."""

    def __reduce__(self) -> tuple[object, tuple[()]]:
        return (_fail_to_rebuild, ())


def _fail_to_rebuild() -> None:
    raise ImportError("not importable in the worker")


def identity(x: object) -> object:
    return x


class Gauge:
    """Counts how many calls are in progress, and the most there ever were. Works across threads and tasks of one process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0
        self.calls = 0

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            self.calls += 1
            self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        with self._lock:
            self.current -= 1
