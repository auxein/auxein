"""What an episode returns: raw measurements, never scores (design doc §6.2)."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from auxein.backend import Array
from auxein.core import ArtifactRef, Status


@dataclass(frozen=True)
class EpisodeResult:
    """The outcome of one agent in one scenario: raw `measurements` (fuel used, time, closest approach, success, tokens...).

    An environment never scores: what counts as good is decided by the aggregator, so a run can be re-judged without
    re-simulating. A failed or timed-out episode has a `status` other than OK and an `error` that says what happened, and no
    measurements are needed. Measurement values are converted to floats; a NaN or infinite one in a successful episode is
    allowed here and shows up as a non-finite objective (a failed evaluation) if an objective is computed from it.
    `artifacts` is reserved for trajectories and transcripts, which are not stored yet.
    """

    measurements: Mapping[str, float] = field(default_factory=dict[str, float])
    status: Status = Status.OK
    error: str | None = None
    artifacts: ArtifactRef | None = None

    def __post_init__(self) -> None:
        values: dict[str, float] = {}
        for name, value in self.measurements.items():
            if not name:
                raise ValueError("measurement names must be non-empty")
            try:
                values[name] = float(value)
            except (TypeError, ValueError):
                raise TypeError(f"measurement {name!r} must be a number, got {type(value).__name__}") from None
        object.__setattr__(self, "measurements", values)
        if self.status is Status.OK and self.error is not None:
            raise ValueError("a successful episode has no error")
        if self.status is not Status.OK and not self.error:
            raise ValueError("a failed or timed-out episode needs an error that says what happened")

    @staticmethod
    def failed(error: str, status: Status = Status.FAILED) -> "EpisodeResult":
        """An episode that did not succeed: `status` is FAILED (an exception, a crash) or TIMEOUT."""
        return EpisodeResult({}, status, error)


@dataclass(frozen=True)
class EpisodeFailure:
    """Why one episode of a batch did not succeed."""

    status: Status
    error: str

    def __post_init__(self) -> None:
        if self.status is Status.OK or not self.error:
            raise ValueError("an episode failure needs the status FAILED or TIMEOUT and an error")


@dataclass(frozen=True)
class EpisodeBatchResult:
    """What a batched environment returns for `n` candidates on `s` scenarios.

    `measurements` maps each name to an array of shape `(n, s)` on the run's backend, so that the aggregator reduces them
    where they are (on a GPU, if that is where the simulator ran). `failures` is a sparse status and error grid: the episodes
    that did not succeed, keyed by `(candidate row, scenario index)`; every other episode succeeded. The values of a failed
    episode are ignored.
    """

    measurements: Mapping[str, Array]
    failures: Mapping[tuple[int, int], EpisodeFailure] = field(default_factory=dict[tuple[int, int], EpisodeFailure])

    def __post_init__(self) -> None:
        if not self.measurements:
            raise ValueError("an episode batch result needs at least one measurement")
        shapes = {name: tuple(int(d) for d in array.shape) for name, array in self.measurements.items()}
        if len(set(shapes.values())) != 1 or len(next(iter(shapes.values()))) != 2:
            raise ValueError(f"every measurement must be an array of the same shape (n_candidates, n_scenarios), got {shapes}")
        n, s = next(iter(shapes.values()))
        for (row, scenario), _ in self.failures.items():
            if not (0 <= row < n and 0 <= scenario < s):
                raise ValueError(f"failed episode ({row}, {scenario}) is outside the {n} x {s} batch")

    @property
    def shape(self) -> tuple[int, int]:
        array = next(iter(self.measurements.values()))
        return int(array.shape[0]), int(array.shape[1])
