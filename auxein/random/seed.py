"""One seed per run, many named and deterministic streams (design doc §8)."""

import zlib
from dataclasses import dataclass

import numpy as np

from auxein.backend import Backend
from auxein.random.stream import RandomStream

MAX_KEY = 2**32 - 1


def stable_name_id(name: str) -> int:
    """Map a stream name to an integer with CRC-32 of its UTF-8 bytes.

    The mapping must be the same in every process on every machine, which rules out Python's built-in `hash()`
    (randomised per process). Two different names could in principle share a CRC-32 (about one chance in four billion);
    the names Auxein itself uses ("strategy", "scenarios", "evaluation", ...) are distinct, and a test pins them.
    """
    return zlib.crc32(name.encode("utf-8"))


@dataclass(frozen=True)
class RunSeed:
    """The user's integer seed. All of a run's randomness is derived from it, by name.

    `stream("strategy")` is the strategy's stream, `stream("scenarios")` generates scenarios, and
    `stream("evaluation", candidate_id)` is the stream of one candidate's evaluation, which follows the candidate
    and not the worker that evaluates it. The same seed, name and keys give the same stream in any process.
    """

    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"the seed must be an int, got {type(self.seed).__name__}")
        if self.seed < 0:
            raise ValueError(f"the seed must not be negative, got {self.seed}")

    @property
    def root(self) -> np.random.SeedSequence:
        """The root `SeedSequence`, from which no stream is drawn directly: streams use explicit spawn keys."""
        return np.random.SeedSequence(self.seed)

    def sequence(self, name: str, *keys: int) -> np.random.SeedSequence:
        """The `SeedSequence` of a named stream: `SeedSequence(entropy=seed, spawn_key=(hash(name), *keys))`.

        Keys are non-negative integers below 2**32, such as candidate ids (so a run can issue about four billion).
        """
        if not name:
            raise ValueError("a stream needs a non-empty name")
        for key in keys:
            if isinstance(key, bool) or not isinstance(key, int):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(f"stream keys must be ints, got {type(key).__name__}")
            if not 0 <= key <= MAX_KEY:
                raise ValueError(f"stream keys must be in [0, 2**32), got {key}")
        return np.random.SeedSequence(entropy=self.seed, spawn_key=(stable_name_id(name), *keys))

    def stream(self, name: str, *keys: int, backend: Backend | None = None) -> RandomStream:
        """A new `RandomStream` for the name and keys, on `backend` (numpy on the CPU by default)."""
        return RandomStream(self.sequence(name, *keys), backend)
