"""The driver: it owns the loop, the budget and the recording (design doc §9).

Strategies never call evaluators; the driver asks, evaluates, records and tells. It is built on asyncio from the start, so
that concurrency, steady-state delivery and timeouts can be added around `_step` without rewriting the loop; this
version delivers results by generation only, one batch at a time, with candidates evaluated sequentially.
"""

import asyncio
import time
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Generic, TypeVar, cast

from auxein.backend import Backend
from auxein.core import (
    ArrayBatch,
    Batch,
    EvalContext,
    EvaluationBatch,
    Evaluator,
    IdIssuer,
    Objective,
    ProblemSpec,
    Strategy,
    StrategyContext,
    take,
)
from auxein.core._typing import G
from auxein.driver.budget import Budget
from auxein.driver.errors import EvaluatorError, RecordingDisabledWarning, StrategyError
from auxein.driver.result import ResultTracker, RunResult
from auxein.random import RunSeed
from auxein.recording import NoopRecorder, Recorder, SQLiteRecorder
from auxein.spaces import Space

T = TypeVar("T")

DEFAULT_OBJECTIVES = (Objective("value"),)


def _describe_space(space: object) -> dict[str, object]:
    describe = getattr(space, "describe", None)
    if callable(describe):
        return cast("dict[str, object]", describe())
    return {"type": type(space).__name__, "repr": repr(space)}


def _describe_component(component: object) -> dict[str, str]:
    kind = type(component)
    return {"class": kind.__qualname__, "module": kind.__module__, "repr": repr(component)}


class Driver(Generic[G]):
    """One run: build it with the run's configuration, then `await driver.execute()`."""

    def __init__(
        self,
        *,
        strategy: Strategy[G],
        evaluator: Evaluator[G],
        problem: ProblemSpec[G],
        budget: Budget,
        seed: int,
        backend: Backend,
        batch_size: int,
        recorder: Recorder,
        run_dir: Path | None,
        name: str | None,
        clock: Callable[[], float],
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if strategy.capabilities.tell_mode == "steady_state":
            raise ValueError("this driver delivers results by generation only, but the strategy requires steady-state delivery")
        self._strategy, self._evaluator, self._problem = strategy, evaluator, problem
        self._budget, self._seed, self._backend, self._batch_size = budget, seed, backend, batch_size
        self._recorder, self._run_dir, self._name, self._clock = recorder, run_dir, name, clock

        run_seed = RunSeed(seed)
        self._issuer = IdIssuer()
        self._strategy_context = StrategyContext(run_seed.stream("strategy", backend=backend), backend, self._issuer.next)
        self._eval_context: EvalContext[G] = EvalContext(
            problem,
            backend,
            lambda cid: run_seed.stream("evaluation", cid, backend=backend),
            lambda cid: run_seed.stream("evaluation-batch", cid, backend=backend),
        )
        self._tracker: ResultTracker[G] = ResultTracker(problem.objectives, constrained=bool(problem.constraints))
        self._seen = bytearray()  # one byte per issued id: whether a batch has already carried it
        self._last_step = -1
        self._used = 0
        self._costs: dict[str, float] = {unit: 0.0 for unit in budget.cost}
        self._started = 0.0

    # --- the run ---

    async def execute(self) -> RunResult[G]:
        self._started = self._clock()
        self._recorder.on_start(self._metadata())
        status: str = "failed"
        stop_reason: str | None = None
        try:
            self._strategy.bind(self._problem, self._strategy_context)
            stop_reason = await self._loop()
            status = "completed"
        except (KeyboardInterrupt, asyncio.CancelledError):
            status = "interrupted"
            raise
        finally:
            self._recorder.on_end(status, stop_reason, self._summary())
        assert stop_reason is not None  # the run completed
        return RunResult(
            stop_reason=stop_reason,
            evaluations_used=self._used,
            wall_time=self._clock() - self._started,
            run_dir=self._run_dir,
            best=self._tracker.best,
            pareto_front=self._tracker.pareto_front,
            trace=self._tracker.trace,
        )

    async def _loop(self) -> str:
        while True:
            reason = self._stop_reason()
            if reason is not None:
                return reason
            if await self._step():
                return "budget:evaluations"

    def _stop_reason(self) -> str | None:
        """Why the run must stop before the next ask, if it must. Checked in this order."""
        budget = self._budget
        if budget.evaluations is not None and self._used >= budget.evaluations:
            return "budget:evaluations"
        if budget.wall_time is not None and self._clock() - self._started >= budget.wall_time:
            return "budget:wall_time"
        for unit, limit in budget.cost.items():
            if self._costs[unit] >= limit:
                return f"budget:cost:{unit}"
        if self._strategy.should_stop():
            return "strategy"
        return None

    async def _step(self) -> bool:
        """Ask, evaluate, record and tell one batch. Returns True when the batch was truncated, which ends the run."""
        remaining = None if self._budget.evaluations is None else self._budget.evaluations - self._used
        wanted = self._batch_size if remaining is None else min(self._batch_size, remaining)
        batch = self._strategy.ask(wanted)
        step = self._validate_batch(batch)

        truncated = remaining is not None and len(batch.candidates) > remaining
        if truncated:
            assert remaining is not None
            batch = take(batch, remaining)  # the budget is a hard limit: evaluate only what remains, in ask order

        results = await self._evaluator.evaluate(batch, self._eval_context)
        self._validate_results(batch, results)

        self._tracker.add(results.evaluations, self._used)
        self._used += len(results)
        for evaluation in results:
            for unit, amount in evaluation.cost.units.items():
                if unit in self._costs:
                    self._costs[unit] += amount
        self._recorder.on_batch(step, batch, results)

        if truncated:
            return True  # the strategy is not told about an incomplete batch
        self._strategy.tell(results)
        self._recorder.on_tell(step, len(results))
        return False

    # --- checks on what components hand back ---

    def _validate_batch(self, batch: Batch[G]) -> int:
        """Check an asked batch and return its step: non-empty, ids issued by this run and new, one consistent step."""
        who = type(self._strategy).__name__
        if isinstance(batch, ArrayBatch):
            ids, steps = batch.ids, {batch.step}
        else:
            candidates = batch.candidates
            ids, steps = [c.id for c in candidates], {c.step for c in candidates}
        if len(ids) == 0:
            raise StrategyError(f"{who}.ask() returned an empty batch")
        if len(steps) != 1:
            raise StrategyError(f"{who}.ask() returned candidates with different steps {sorted(steps)}: a batch is one ask round")
        (step,) = steps
        if step < self._last_step:
            raise StrategyError(f"{who}.ask() returned step {step} after step {self._last_step}: steps never go back")
        issued = self._issuer.issued
        if len(self._seen) < issued:
            self._seen.extend(bytes(issued - len(self._seen)))
        for candidate_id in ids:
            if not 0 <= candidate_id < issued:
                raise StrategyError(
                    f"{who}.ask() returned candidate id {candidate_id}, which this run never issued (use the context's new_id)"
                )
            if self._seen[candidate_id]:
                raise StrategyError(
                    f"{who}.ask() returned candidate id {candidate_id} again: ids are never reused, even for re-evaluations"
                )
            self._seen[candidate_id] = 1
        self._last_step = step
        return step

    def _validate_results(self, batch: Batch[G], results: EvaluationBatch[G]) -> None:
        who = type(self._evaluator).__name__
        if not isinstance(results, EvaluationBatch):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise EvaluatorError(f"{who}.evaluate() must return an EvaluationBatch, got {type(results).__name__}")
        asked = list(batch.ids) if isinstance(batch, ArrayBatch) else [c.id for c in batch.candidates]  # no candidates to build
        got = [e.candidate.id for e in results]
        if got != asked:
            raise EvaluatorError(
                f"{who}.evaluate() must return one evaluation per candidate, in ask order: "
                f"asked for {asked[:5]}..., got {got[:5]}... ({len(got)} of {len(asked)})"
            )

    # --- recording ---

    def _metadata(self) -> dict[str, object]:
        problem = self._problem
        return {
            "name": self._name,
            "seed": self._seed,
            "backend": {"name": self._backend.name, "device": self._backend.device},
            "precision": self._backend.precision,
            "problem": {
                "space": _describe_space(problem.space),
                "objectives": [{"name": o.name, "direction": o.direction} for o in problem.objectives],
                "constraints": list(problem.constraints),
                "descriptors": list(problem.descriptors),
            },
            "budget": self._budget.describe(),
            "batch_size": self._batch_size,
            "strategy": _describe_component(self._strategy),
            "evaluator": _describe_component(self._evaluator),
        }

    def _summary(self) -> dict[str, object]:
        best = self._tracker.best
        return {
            "evaluations_used": self._used,
            "wall_time": self._clock() - self._started,
            "best_candidate_id": None if best is None else int(best.candidate.id),
            "best_objectives": None if best is None else dict(best.objectives),
            "pareto_size": len(self._tracker.pareto_front),
        }


def _build(
    *,
    strategy: Strategy[G],
    evaluator: Evaluator[G],
    space: Space[G],
    objectives: Sequence[Objective],
    constraints: Sequence[str],
    descriptors: Sequence[str],
    budget: Budget,
    seed: int,
    backend: Backend | None,
    batch_size: int,
    run_dir: str | Path | None,
    name: str | None,
    clock: Callable[[], float],
) -> Driver[G]:
    problem = ProblemSpec(space, tuple(objectives), tuple(constraints), tuple(descriptors))
    recorder: Recorder = NoopRecorder() if run_dir is None else SQLiteRecorder(run_dir)
    return Driver(
        strategy=strategy,
        evaluator=evaluator,
        problem=problem,
        budget=budget,
        seed=seed,
        backend=Backend() if backend is None else backend,
        batch_size=batch_size,
        recorder=recorder,
        run_dir=None if run_dir is None else Path(run_dir),
        name=name if name is not None else (None if run_dir is None else Path(run_dir).name),
        clock=clock,
    )


def _warn_unrecorded(run_dir: str | Path | None, stacklevel: int) -> None:
    if run_dir is None:
        warnings.warn(
            "this run is not recorded: pass run_dir to keep its candidates, lineage and evaluations "
            "(silence this with warnings.filterwarnings('ignore', category=RecordingDisabledWarning))",
            RecordingDisabledWarning,
            stacklevel=stacklevel,
        )


def run(
    *,
    strategy: Strategy[G],
    evaluator: Evaluator[G],
    space: Space[G],
    objectives: Sequence[Objective] = DEFAULT_OBJECTIVES,
    constraints: Sequence[str] = (),
    descriptors: Sequence[str] = (),
    budget: Budget,
    seed: int,
    backend: Backend | None = None,
    batch_size: int = 64,
    run_dir: str | Path | None = None,
    name: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult[G]:
    """Run a strategy on a problem and return what it found.

    This is the synchronous entry point. It works in plain scripts, and inside an already running event loop (as in
    Jupyter), where it runs the driver on a fresh loop in a dedicated thread and waits for it.

    The run is recorded only if `run_dir` is given; otherwise nothing is written to disk and a
    `RecordingDisabledWarning` is emitted, once. `batch_size` is the number of candidates the driver asks for per round.
    `clock` is the time source of the wall-time budget (injectable for tests). See `Budget` for how limits are enforced.
    """
    _warn_unrecorded(run_dir, stacklevel=3)
    driver = _build(
        strategy=strategy, evaluator=evaluator, space=space, objectives=objectives, constraints=constraints, descriptors=descriptors,
        budget=budget, seed=seed, backend=backend, batch_size=batch_size, run_dir=run_dir, name=name, clock=clock,
    )  # fmt: skip
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(driver.execute())
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="auxein-driver") as pool:
        return pool.submit(asyncio.run, driver.execute()).result()


async def arun(
    *,
    strategy: Strategy[G],
    evaluator: Evaluator[G],
    space: Space[G],
    objectives: Sequence[Objective] = DEFAULT_OBJECTIVES,
    constraints: Sequence[str] = (),
    descriptors: Sequence[str] = (),
    budget: Budget,
    seed: int,
    backend: Backend | None = None,
    batch_size: int = 64,
    run_dir: str | Path | None = None,
    name: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult[G]:
    """The asynchronous entry point, for embedding Auxein in async applications. Same arguments as `run`."""
    _warn_unrecorded(run_dir, stacklevel=3)
    driver = _build(
        strategy=strategy, evaluator=evaluator, space=space, objectives=objectives, constraints=constraints, descriptors=descriptors,
        budget=budget, seed=seed, backend=backend, batch_size=batch_size, run_dir=run_dir, name=name, clock=clock,
    )  # fmt: skip
    return await driver.execute()
