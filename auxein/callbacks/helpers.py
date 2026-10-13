"""The helpers behind `auxein.callbacks`.

A callback is any callable taking a `Progress` and returning `None` or `StopRun`. These are the few that nearly every long run
wants. The throttles (`every`, `every_evaluations`) wrap another callback, forward whatever it returns (so a wrapped
`StopRun`, or an awaitable from an `async def`, reaches the driver) and skip it while a resumed run is replaying recorded
evaluations: the work worth throttling (rendering a video) is the work not worth redoing for generations that already happened.
"""

import csv
import io
import math
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TextIO

from auxein.driver.progress import Callback, Progress, StopRun


class every:
    """Call `callback` every `n` generations, and always at the final event.

    `n` counts the driver's tells (`Progress.generation`), which are generations in generation delivery; in steady state use
    `every_evaluations`. While a resumed run replays recorded evaluations the callback is **not** called unless
    `during_replay=True` (the throttled callback is usually the heavy one). Returns what the callback returns.
    """

    def __init__(self, n: int, callback: Callback, *, during_replay: bool = False) -> None:
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        self._n, self._callback, self._during_replay = n, callback, during_replay

    def __call__(self, progress: Progress[Any]) -> StopRun | None | Awaitable[StopRun | None]:
        if progress.replaying and not self._during_replay:
            return None
        if progress.final or progress.generation % self._n == 0:
            return self._callback(progress)
        return None

    def __repr__(self) -> str:
        return f"every({self._n}, {self._callback!r})"


class every_evaluations:
    """Call `callback` whenever at least `k` more evaluations have been used since the last call, and always at the final event.

    This is the right throttle for steady-state delivery, where a tell can be a single candidate and a generation count means
    little. The instance remembers where it last fired, so give each run its own. Like `every`, it skips replayed tells unless
    `during_replay=True`, and returns what the callback returns.
    """

    def __init__(self, k: int, callback: Callback, *, during_replay: bool = False) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self._k, self._callback, self._during_replay = k, callback, during_replay
        self._last = 0

    def __call__(self, progress: Progress[Any]) -> StopRun | None | Awaitable[StopRun | None]:
        used = progress.evaluations_used
        if used < self._last:  # a new run: this instance was used by another one
            self._last = 0
        if progress.replaying and not self._during_replay:
            return None
        if progress.final or used - self._last >= self._k:
            self._last = used
            return self._callback(progress)
        return None

    def __repr__(self) -> str:
        return f"every_evaluations({self._k}, {self._callback!r})"


def stop_when(predicate: Callable[[Progress[Any]], bool], reason: str) -> Callback:
    """A callback that returns `StopRun(reason)` once `predicate(progress)` is true, e.g.
    `stop_when(lambda p: p.best is not None and p.best.objectives["error"] < 1e-6, "solved")`.

    The run then ends with the stop reason `callback:<reason>`. A predicate on the fitness is reproducible (the same seed stops
    at the same generation); one on the clock is not. It is not asked at the final event, where there is nothing left to stop.
    """
    if not reason:
        raise ValueError("stop_when needs a non-empty reason")

    def check(progress: Progress[Any]) -> StopRun | None:
        if progress.final:
            return None
        return StopRun(reason) if predicate(progress) else None

    check.__qualname__ = f"stop_when({reason!r})"
    return check


def _clock(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "    -:--"
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:>2d}:{minutes:02d}:{secs:02d}"


def _number(value: float | None, width: int = 11) -> str:
    if value is None or math.isnan(value):
        return f"{'-':>{width}}"
    return f"{value:>{width}.4g}"


class print_progress:
    """Print one compact line per call: generation, evaluations (and the share of the budget), the best objective, the batch mean,
    failures if any, evaluations per second, elapsed time and the estimated time left.

    `every` is in generations (the final line is always printed, with the reason the run stopped). The best is the run's best
    objective for a single-objective problem, and the size of the Pareto front otherwise; the batch mean is of the first
    objective. While a resumed run replays recorded evaluations nothing is printed except one `replaying…` line. `file` is
    `sys.stdout` looked up when printing, unless given.
    """

    def __init__(self, every: int = 1, file: TextIO | None = None) -> None:
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}")
        self._every, self._file = every, file
        self._announced = False

    def __call__(self, progress: Progress[Any]) -> None:
        out = self._file if self._file is not None else sys.stdout
        if progress.replaying:
            if not self._announced:
                self._announced = True
                print("replaying recorded evaluations…", file=out)
            return
        self._announced = False
        if not (progress.final or progress.generation % self._every == 0):
            return
        print(self.line(progress), file=out)

    @staticmethod
    def line(progress: Progress[Any]) -> str:
        budget = progress.budget.evaluations
        share = "" if budget is None else f"/{budget:<9,d} ({100.0 * progress.evaluations_used / budget:>3.0f}%)"
        if len(progress.objectives) == 1:
            best = progress.best.objectives[progress.objectives[0].name] if progress.best is not None else None
            leader = f"best {_number(best)}"
        else:
            leader = f"front {len(progress.pareto_front):>6d}"
        first = progress.objectives[0].name
        mean = progress.batch.objectives[first].mean if progress.batch is not None else None
        failed = progress.status_counts.get("failed", 0) + progress.status_counts.get("timeout", 0)
        speed = progress.recent_evaluations_per_second or progress.evaluations_per_second
        parts = [
            f"gen {progress.generation:>6d}",
            f"evals {progress.evaluations_used:>10,d}{share}",
            leader,
            f"mean {_number(mean)}",
        ]
        if failed:
            parts.append(f"failed {failed:>5d}")
        parts += [f"{_number(speed, 9)} ev/s", f"elapsed {_clock(progress.elapsed)}", f"ETA {_clock(progress.eta)}"]
        line = " | ".join(parts)
        return f"{line} | done: {progress.stop_reason}" if progress.final else line


class ProgressHistory:
    """A callback that keeps one row per call, for fitness-over-generations charts.

    `rows` is a list of dicts with the generation, the evaluations used, the elapsed seconds, the run's `best` (single
    objective, else None) and `front_size`, the cumulative `failed` and `timeout` counts, whether the call was `replaying` or
    `final`, and, for the last told batch, `batch_ok`, `batch_failed`, `batch_timeout`, `batch_feasible` and, per objective
    `<name>`, `<name>_best`, `<name>_mean`, `<name>_median` and `<name>_worst` (NaN when nothing was told yet or nothing was OK).
    The final event adds a last row, flagged `final`, with the same generation as the one before. `to_csv()` returns the rows
    as CSV text, and writes them to `path` if one is given. The rows of replayed generations are kept too: a resumed run's
    history covers the whole run.
    """

    def __init__(self) -> None:
        self._rows: list[dict[str, object]] = []

    def __call__(self, progress: Progress[Any]) -> None:
        batch = progress.batch
        row: dict[str, object] = {
            "generation": progress.generation,
            "evaluations": progress.evaluations_used,
            "elapsed": progress.elapsed,
            "best": progress.best.objectives[progress.objectives[0].name] if progress.best is not None else None,
            "front_size": len(progress.pareto_front),
            "failed": progress.status_counts.get("failed", 0),
            "timeout": progress.status_counts.get("timeout", 0),
            "replaying": progress.replaying,
            "final": progress.final,
            "batch_ok": None if batch is None else batch.ok,
            "batch_failed": None if batch is None else batch.failed,
            "batch_timeout": None if batch is None else batch.timeout,
            "batch_feasible": math.nan if batch is None else batch.feasible_fraction,
        }
        for objective in progress.objectives:
            stats = None if batch is None else batch.objectives[objective.name]
            for field in ("best", "mean", "median", "worst"):
                row[f"{objective.name}_{field}"] = math.nan if stats is None else getattr(stats, field)
        self._rows.append(row)

    @property
    def rows(self) -> list[dict[str, object]]:
        return [dict(row) for row in self._rows]

    def __len__(self) -> int:
        return len(self._rows)

    def to_csv(self, path: str | Path | None = None) -> str:
        buffer = io.StringIO()
        if self._rows:
            writer = csv.DictWriter(buffer, fieldnames=list(self._rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(self._rows)
        text = buffer.getvalue()
        if path is not None:
            Path(path).write_text(text)
        return text
