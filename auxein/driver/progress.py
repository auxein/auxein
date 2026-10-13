"""Progress callbacks: what the driver hands to user code after each tell, and how that code may answer (design doc §9.6).

A long run needs two things the result alone does not give: to be **watched** while it goes (generation, best and typical
fitness, speed, time left) and to be **acted on** at chosen moments (render the current best every 25 generations, stop once
the agents succeed reliably). The driver calls the callbacks with a `Progress`, a read-only snapshot, and a callback may
answer only with `None` (carry on) or `StopRun` (finish like any other stop). It cannot change the strategy, the budget or
anything else the run computes, which is what keeps a run with callbacks the same run as one without.
"""

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, TypeAlias

import numpy as np

from auxein.core import Evaluation, Objective, Status
from auxein.core._typing import G
from auxein.driver.budget import Budget


@dataclass(frozen=True)
class StopRun:
    """What a callback returns to stop the run. The run ends like any other stop (a checkpoint, a `RunResult`, a final
    event) with the stop reason `callback:<reason>`.

    Whether the stop is **reproducible** depends on the callback: stopping on a fitness threshold is (the same seed stops at
    the same generation), stopping on wall-clock time or on a file appearing is not.
    """

    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"a StopRun needs a non-empty reason, got {self.reason!r}")


@dataclass(frozen=True)
class ObjectiveStats:
    """One objective over the `OK` evaluations of a batch, in natural units: `best` and `worst` follow the declared direction
    (for a maximised objective the best is the largest). All four are NaN when the batch has no `OK` evaluation."""

    best: float
    mean: float
    median: float
    worst: float


@dataclass(frozen=True)
class BatchStats:
    """The last batch that was told to the strategy (a generation, in generation delivery; a single candidate in steady state).

    `ok`, `failed` and `timeout` count its evaluations by status, and `feasible_fraction` is the share of **all** its
    evaluations that are `OK` with no constraint violated (a failed evaluation is infeasible). `objectives` are the
    statistics per objective, over the `OK` evaluations only.
    """

    size: int
    ok: int
    failed: int
    timeout: int
    feasible_fraction: float
    objectives: Mapping[str, ObjectiveStats]


@dataclass(frozen=True)
class Progress(Generic[G]):
    """A read-only snapshot of a run, handed to the callbacks after each tell and once more when the run ends.

    It holds **references, not copies**: `best` and the members of `pareto_front` are the run's own `Evaluation` objects,
    whose genomes may be arrays on a GPU device. Genomes are read-only for numpy arrays; for torch tensors they must be treated
    as read-only (the library cannot enforce it), and a callback that keeps one beyond the call should copy it. The snapshot
    itself is immutable, and building it costs the run nothing unless callbacks were given.
    """

    generation: int
    """The number of tells so far (across sessions of a resumed run): the generation count in generation delivery."""
    step: int | None
    """The strategy's ask round of the last batch that was told; None before the first tell."""
    evaluations_used: int
    budget: Budget
    remaining: int | None
    """Evaluations left before the evaluation budget, or None if the budget has no evaluation limit."""
    elapsed: float
    """Active wall-clock seconds of the whole run, as the wall-time budget counts them (time between sessions excluded)."""
    objectives: tuple[Objective, ...]
    best: Evaluation[G] | None
    """What `RunResult.best` would be now: single objective only, None for several objectives or before any `OK` evaluation."""
    pareto_front: tuple[Evaluation[G], ...]
    """What `RunResult.pareto_front` would be now (for a single objective, just the best feasible one)."""
    batch: BatchStats | None
    """The statistics of the last told batch; None before the first tell."""
    status_counts: Mapping[str, int]
    """Cumulative outcomes of the told evaluations: `ok`, `failed`, `timeout`."""
    evaluations_per_second: float | None
    """Evaluations used per active second over the whole run (None before any time has passed)."""
    recent_evaluations_per_second: float | None
    """The same over the last 30 seconds or so of the session, and None if the session has not run that long yet or is replaying."""
    eta: float | None
    """Estimated seconds to the evaluation budget at the recent (else the overall) speed; None without an evaluation budget or a speed."""
    delivery: str
    """`generation` or `steady_state`. In steady state a tell may be a single candidate, so throttle with `every_evaluations`."""
    run_dir: Path | None
    replaying: bool
    """True while a resumed run is replaying recorded evaluations: the callbacks fire so that displays catch up, but heavy work
    (rendering) should be skipped, and a `StopRun` returned now is ignored."""
    final: bool = False
    """True for the last call of a run that completed. A `StopRun` returned from it has nothing to stop and is ignored."""
    stop_reason: str | None = None
    """Why the run ended (`budget:*`, `strategy` or `callback:<reason>`); set only when `final`."""


Callback: TypeAlias = Callable[[Progress[Any]], "StopRun | None | Awaitable[StopRun | None]"]
"""Any callable taking a `Progress` and returning `None` or `StopRun`; an `async def` is awaited. `Any` is the genome type, which
callbacks are generic over."""

Callbacks: TypeAlias = Sequence[Callback]


def batch_stats(evaluations: Sequence[Evaluation[Any]], objectives: Sequence[Objective]) -> BatchStats:
    """The statistics of a told batch (see `BatchStats`): plain numpy on the host, since it is a few numbers per objective."""
    ok = [e for e in evaluations if e.status is Status.OK]
    failed = sum(1 for e in evaluations if e.status is Status.FAILED)
    timeout = sum(1 for e in evaluations if e.status is Status.TIMEOUT)
    feasible = sum(1 for e in ok if all(v == 0 for v in e.constraints.values()))
    per_objective: dict[str, ObjectiveStats] = {}
    for objective in objectives:
        if not ok:
            per_objective[objective.name] = ObjectiveStats(math.nan, math.nan, math.nan, math.nan)
            continue
        values = np.asarray([e.objectives[objective.name] for e in ok], dtype=np.float64)
        low, high = float(values.min()), float(values.max())
        best, worst = (low, high) if objective.direction == "minimise" else (high, low)
        per_objective[objective.name] = ObjectiveStats(best, float(values.mean()), float(np.median(values)), worst)
    return BatchStats(
        size=len(evaluations),
        ok=len(ok),
        failed=failed,
        timeout=timeout,
        feasible_fraction=feasible / len(evaluations) if evaluations else math.nan,
        objectives=MappingProxyType(per_objective),
    )
