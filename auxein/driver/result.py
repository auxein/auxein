"""Tracking results as they arrive, and the final `RunResult` (design doc §9.5)."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic

from auxein.core import Evaluation
from auxein.core._typing import G
from auxein.core.tracking import ResultTracker  # the ranking, shared with the recording's reader


@dataclass(frozen=True)
class RunResult(Generic[G]):
    """What a run found.

    `best` is, for a single-objective problem, the best `OK` evaluation: feasible beats infeasible, then a lower total
    violation, then a lower objective in minimisation form (the declared direction is respected), and ties go to the
    earliest id. It is None for several objectives, or when no `OK` evaluation exists.

    `pareto_front` holds the non-dominated feasible `OK` evaluations, in minimisation form, sorted by candidate id; for a
    single objective it is just the best feasible one.

    `trace` (single objective only) lists `(evaluations_used, best_value)` each time `best` changed, with the value in
    natural units, for quick plots. It follows `best`, so with constraints it can get worse when the first feasible
    candidate replaces an infeasible one with a better value.

    Failed and timed-out evaluations (design doc §6.6) never become `best` or enter the front; they are counted in
    `status_counts`.
    """

    stop_reason: str
    """`budget:evaluations`, `budget:wall_time`, `budget:cost:<unit>` or `strategy`."""
    evaluations_used: int
    wall_time: float
    run_dir: Path | None
    best: Evaluation[G] | None
    pareto_front: tuple[Evaluation[G], ...]
    trace: tuple[tuple[int, float], ...]
    status_counts: Mapping[str, int] = field(default_factory=dict[str, int])
    """How many evaluations ended in each status (`ok`, `failed`, `timeout`), counting only those that were told."""


__all__ = ["ResultTracker", "RunResult"]
