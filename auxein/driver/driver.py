"""The driver: it owns the loop, the budget and the recording (design doc §9).

Strategies never call evaluators; the driver asks, evaluates, records and tells. It is built on asyncio so that many
evaluations can be in progress at once, and delivers results in one of two ways:

- **generation**: ask a batch, evaluate it (up to `concurrency` candidates at once), tell all its results together.
- **steady state**: keep a window of `W = batch_size` candidates asked but not yet told; evaluate up to `concurrency` of
  them at once, each as a one-candidate batch, and tell results one at a time.

`batch_size` (the window) is an algorithmic setting; `concurrency` is a resource setting. In deterministic mode the sequence
of asks and tells depends only on the seed and `batch_size`, never on `concurrency`, the executor or timing (design doc §8.1).
"""

import asyncio
import dataclasses
import time
import warnings
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar, cast

from auxein.backend import Backend
from auxein.core import (
    ArrayBatch,
    Batch,
    EvalContext,
    Evaluation,
    EvaluationBatch,
    Evaluator,
    FailurePolicy,
    IdIssuer,
    Objective,
    ProblemSpec,
    Status,
    Strategy,
    StrategyContext,
    single,
    take,
)
from auxein.core._typing import G
from auxein.driver.budget import Budget
from auxein.driver.errors import (
    AllEvaluationsFailedError,
    EvaluationFailureWarning,
    EvaluatorError,
    RecordingDisabledWarning,
    SteadyStateVectorisationWarning,
    StrategyError,
)
from auxein.driver.result import ResultTracker, RunResult
from auxein.evaluators import EvaluationError, VectorisedEvaluator
from auxein.execution import AbandonedEvaluationWarning, ExecutorKind, ExecutorName, make_executor, resolve_executor
from auxein.random import RunSeed
from auxein.recording import NoopRecorder, Recorder, SQLiteRecorder
from auxein.spaces import Space

T = TypeVar("T")

DEFAULT_OBJECTIVES = (Objective("value"),)

Delivery = Literal["generation", "steady_state"]


@dataclass
class _Slot(Generic[G]):
    """One candidate of a steady-state run, from the moment it is asked until it is told."""

    seq: int
    """Position in ask order across the whole run."""
    step: int
    batch: Batch[G]
    """The candidate as a one-candidate batch, so that every evaluator works unchanged."""
    task: "asyncio.Task[EvaluationBatch[G]] | None" = None
    result: EvaluationBatch[G] | None = None


@dataclass
class _Window(Generic[G]):
    """The state of a steady-state run."""

    queue: deque[_Slot[G]]
    """Asked, not yet started, in ask order."""
    inflight: dict[int, _Slot[G]]
    """Asked, not yet told, in ask order (a dict keeps insertion order). `len(inflight)` is what the window counts."""
    running: dict["asyncio.Task[EvaluationBatch[G]]", _Slot[G]]
    finished: list[_Slot[G]]
    """Throughput mode only: evaluated, not yet told, in completion order."""
    next_seq: int = 0
    stop: str | None = None
    """Why asking has stopped, once it has."""


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
        concurrency: int = 1,
        executor: ExecutorName = "auto",
        delivery: Delivery | None = None,
        deterministic: bool = True,
        failure_policy: FailurePolicy = "infeasible",
        timeout: float | None = None,
        initial_failure_guard: int | None = 10,
        recorder: Recorder,
        run_dir: Path | None,
        name: str | None,
        clock: Callable[[], float],
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {concurrency}")
        if failure_policy not in ("infeasible", "fail_fast"):
            raise ValueError(f"unknown failure_policy {failure_policy!r}: expected 'infeasible' or 'fail_fast'")
        if timeout is not None and not timeout > 0:
            raise ValueError(f"timeout must be a positive number of seconds or None, got {timeout!r}")
        if initial_failure_guard is not None and initial_failure_guard < 1:
            raise ValueError(f"initial_failure_guard must be at least 1 or None, got {initial_failure_guard}")
        if timeout is not None and isinstance(evaluator, VectorisedEvaluator):
            raise ValueError(
                "a timeout cannot be combined with a VectorisedEvaluator: it makes one call per batch on the driver's thread, "
                "which nothing can interrupt, and a vectorised call is normally cheap. Use FunctionEvaluator to time out evaluations"
            )
        self._executor_kind: ExecutorKind = resolve_executor(executor, concurrency, timeout)
        self._policy, self._timeout = failure_policy, timeout
        self._guard = initial_failure_guard
        self._guard_told = 0  # evaluations told while the guard is undecided
        self._guard_open = initial_failure_guard is not None
        self._first_failure: Evaluation[G] | None = None
        self._failures = {Status.FAILED: 0, Status.TIMEOUT: 0}
        self._abandoned = 0
        self._delivery: Delivery = _resolve_delivery(delivery, strategy)
        self._concurrency, self._deterministic = concurrency, deterministic
        self._strategy, self._evaluator, self._problem = strategy, evaluator, problem
        self._budget, self._seed, self._backend, self._batch_size = budget, seed, backend, batch_size
        self._recorder, self._run_dir, self._name, self._clock = recorder, run_dir, name, clock

        run_seed = RunSeed(seed)
        self._issuer = IdIssuer()
        self._strategy_context = StrategyContext(run_seed.stream("strategy", backend=backend), backend, self._issuer.next)
        host_backend = Backend("numpy", "cpu", backend.precision)
        self._base_context: EvalContext[G] = EvalContext(
            problem,
            backend,
            lambda cid: run_seed.stream("evaluation", cid, backend=host_backend),  # always numpy: see EvalContext.rng_for
            lambda cid: run_seed.stream("evaluation-batch", cid, backend=backend),
            concurrency=concurrency,
            timeout=timeout,
            failure_policy=failure_policy,
        )
        self._eval_context = self._base_context
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
        executor = make_executor(self._executor_kind, self._concurrency)
        self._eval_context = dataclasses.replace(self._base_context, executor=executor)
        try:
            self._strategy.bind(self._problem, self._strategy_context)
            if self._delivery == "steady_state":
                self._warn_if_vectorised()
                stop_reason = await self._steady_state()
            else:
                stop_reason = await self._loop()
            self._check_guard_at_end()
            status = "completed"
        except (KeyboardInterrupt, asyncio.CancelledError):
            status = "interrupted"
            raise
        finally:
            self._abandoned = executor.abandoned_running()
            if self._abandoned:
                warnings.warn(
                    f"{self._abandoned} timed-out evaluation(s) were still running in background threads when the run ended; "
                    "their results were discarded",
                    AbandonedEvaluationWarning,
                    stacklevel=2,
                )
            executor.shutdown()  # pools never outlive the run, whatever ended it
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
            status_counts=self._status_counts(),
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
        self._account(step, batch, results)  # records, and applies the failure policy and the guard before anyone is told

        if truncated:
            return True  # the strategy is not told about an incomplete batch
        self._strategy.tell(results)
        self._recorder.on_tell(step, len(results))
        return False

    def _account(self, step: int, batch: Batch[G], results: EvaluationBatch[G]) -> None:
        """Count, track and record evaluations that are about to be told (or, for a truncated batch, dropped)."""
        self._tracker.add(results.evaluations, self._used)
        self._used += len(results)
        failed: list[Evaluation[G]] = []
        for evaluation in results:
            if evaluation.status is not Status.OK:
                failed.append(evaluation)
            for unit, amount in evaluation.cost.units.items():
                if unit in self._costs:
                    self._costs[unit] += amount
        self._recorder.on_batch(step, batch, results)
        if failed or self._guard_open:
            self._apply_failure_rules(results, failed)

    # --- failures (design doc §6.6) ---

    def _apply_failure_rules(self, results: EvaluationBatch[G], failed: list[Evaluation[G]]) -> None:
        """The driver is the backstop for failures, whichever evaluator produced them: it counts them, stops the run under
        `fail_fast`, warns about the first one, and stops a run whose first evaluations all failed."""
        for evaluation in failed:
            self._failures[evaluation.status] += 1
        if failed and self._policy == "fail_fast":
            first = failed[0]
            raise EvaluationError([first.candidate.id], detail=f"{first.status.value}: {first.error}")
        if failed and self._first_failure is None:
            self._first_failure = failed[0]
            warnings.warn(self._describe_failure(failed[0], "first failure of this run"), EvaluationFailureWarning, stacklevel=2)
        if not self._guard_open:
            return
        assert self._guard is not None
        for evaluation in results:  # in the order they are told: ask order in deterministic mode
            if evaluation.status is Status.OK:
                self._guard_open = False  # at least one of the first evaluations worked: this is a result, not a bug
                return
            self._guard_told += 1
            if self._guard_told >= self._guard:
                raise self._all_failed(self._guard_told)

    def _check_guard_at_end(self) -> None:
        """A run too short to reach the guard's count, in which nothing ever succeeded, is just as broken."""
        if self._guard_open and self._guard_told > 0:
            raise self._all_failed(self._guard_told)

    def _all_failed(self, count: int) -> AllEvaluationsFailedError:
        assert self._first_failure is not None
        return AllEvaluationsFailedError(
            f"the first {count} evaluation(s) all failed, which almost certainly means the evaluation function has a bug: "
            "this is not a result worth optimising. The first failure was\n"
            f"{self._describe_failure(self._first_failure, None)}\n"
            "If failing at the start is expected, turn this check off with initial_failure_guard=None."
        )

    @staticmethod
    def _describe_failure(evaluation: Evaluation[G], note: str | None) -> str:
        head = f"candidate {evaluation.candidate.id} ended with status {evaluation.status.value!r}"
        return f"{head} ({note}):\n{evaluation.error}" if note else f"{head}:\n{evaluation.error}"

    def _status_counts(self) -> dict[str, int]:
        failed = sum(self._failures.values())
        return {"ok": self._used - failed, "failed": self._failures[Status.FAILED], "timeout": self._failures[Status.TIMEOUT]}

    # --- steady-state delivery ---

    def _warn_if_vectorised(self) -> None:
        if isinstance(self._evaluator, VectorisedEvaluator):
            warnings.warn(
                "steady-state delivery calls a VectorisedEvaluator with one candidate at a time, which defeats the purpose "
                "of vectorising: use delivery='generation' (the default for strategies that support it)",
                SteadyStateVectorisationWarning,
                stacklevel=2,
            )

    async def _steady_state(self) -> str:
        """Keep up to `batch_size` candidates asked-but-not-told, evaluate up to `concurrency` at once, tell one at a time.

        Deterministic mode tells in ask order and asks only right after a tell, one result at a time, so that the strategy
        sees the same calls whatever the timing and the number of workers. Throughput mode tells as results arrive and
        refills the window at once. See design doc §9.2.
        """
        window: _Window[G] = _Window(deque(), {}, {}, [])
        try:
            while True:
                self._refill(window)
                self._start(window)
                if not window.inflight:
                    assert window.stop is not None  # nothing in flight and nothing asked: the budget is spent or asking stopped
                    return window.stop
                if self._deliver(window):
                    continue
                await self._wait_for_one(window)
        finally:
            for task in window.running:
                task.cancel()
            await asyncio.gather(*window.running, return_exceptions=True)  # nothing keeps running after the run

    def _refill(self, window: _Window[G]) -> None:
        """Ask for as many candidates as the window and the budget allow, unless the run must stop asking."""
        if window.stop is None:
            window.stop = self._stop_reason()
            if window.stop is not None and window.stop != "budget:evaluations":
                for slot in window.queue:  # whatever has not started will never be evaluated, told or recorded
                    del window.inflight[slot.seq]
                window.queue.clear()
        if window.stop is not None:
            return
        room = self._batch_size - len(window.inflight)
        remaining = None if self._budget.evaluations is None else self._budget.evaluations - self._used - len(window.inflight)
        if room < 1 or (remaining is not None and remaining < 1):
            return
        batch = self._strategy.ask(room if remaining is None else min(room, remaining))
        step = self._validate_batch(batch)
        if remaining is not None and len(batch.candidates) > remaining:
            batch = take(batch, remaining)  # a hard limit: the surplus is dropped before it is ever queued
        for index in range(len(batch.candidates)):
            slot = _Slot(window.next_seq, step, single(batch, index))
            window.next_seq += 1
            window.queue.append(slot)
            window.inflight[slot.seq] = slot

    def _start(self, window: _Window[G]) -> None:
        while window.queue and len(window.running) < self._concurrency:
            slot = window.queue.popleft()
            slot.task = asyncio.ensure_future(self._evaluator.evaluate(slot.batch, self._eval_context))
            window.running[slot.task] = slot

    def _deliver(self, window: _Window[G]) -> bool:
        """Tell one finished result, if the delivery rules allow one now. Returns whether it did."""
        if self._deterministic:
            slot = next(iter(window.inflight.values()))  # the oldest: results are told in ask order
            if slot.result is None:
                return False
        else:
            if not window.finished:
                return False
            slot = window.finished.pop(0)
        assert slot.result is not None
        del window.inflight[slot.seq]
        self._validate_results(slot.batch, slot.result)
        self._account(slot.step, slot.batch, slot.result)
        self._strategy.tell(slot.result)
        self._recorder.on_tell(slot.step, 1)
        return True

    async def _wait_for_one(self, window: _Window[G]) -> None:
        done, _ = await asyncio.wait(window.running, return_when=asyncio.FIRST_COMPLETED)
        for task in sorted(done, key=lambda t: window.running[t].seq):  # simultaneous finishes are told in ask order
            slot = window.running.pop(task)
            slot.result = task.result()  # raises the evaluator's error; the run's `finally` cancels the rest
            if not self._deterministic:
                window.finished.append(slot)

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
            "concurrency": self._concurrency,
            "executor": self._executor_kind,
            "delivery": self._delivery,
            "deterministic": self._deterministic,
            "in_flight_window": self._batch_size,
            "failure_policy": self._policy,
            "timeout": self._timeout,
            "initial_failure_guard": self._guard,
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
            "status_counts": self._status_counts(),
            "abandoned_evaluations": self._abandoned,
        }


def _resolve_delivery(requested: Delivery | None, strategy: Strategy[G]) -> Delivery:
    """`None` is generation delivery unless the strategy can only be told steady-state. A mode the strategy does not
    support is an error at start-up, not a surprise in the middle of a run."""
    mode = strategy.capabilities.tell_mode
    if requested is None:
        return "steady_state" if mode == "steady_state" else "generation"
    if requested not in ("generation", "steady_state"):
        raise ValueError(f"unknown delivery {requested!r}: expected 'generation', 'steady_state' or None")
    if mode != "both" and mode != requested:
        raise ValueError(f"{type(strategy).__name__} supports tell_mode={mode!r} only, so delivery={requested!r} is not possible")
    return requested


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
    concurrency: int,
    executor: ExecutorName,
    delivery: Delivery | None,
    deterministic: bool,
    failure_policy: FailurePolicy,
    timeout: float | None,
    initial_failure_guard: int | None,
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
        concurrency=concurrency,
        executor=executor,
        delivery=delivery,
        deterministic=deterministic,
        failure_policy=failure_policy,
        timeout=timeout,
        initial_failure_guard=initial_failure_guard,
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
    concurrency: int = 1,
    executor: ExecutorName = "auto",
    delivery: Delivery | None = None,
    deterministic: bool = True,
    failure_policy: FailurePolicy = "infeasible",
    timeout: float | None = None,
    initial_failure_guard: int | None = 10,
    run_dir: str | Path | None = None,
    name: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult[G]:
    """Run a strategy on a problem and return what it found.

    This is the synchronous entry point. It works in plain scripts, and inside an already running event loop (as in
    Jupyter), where it runs the driver on a fresh loop in a dedicated thread and waits for it.

    The run is recorded only if `run_dir` is given; otherwise nothing is written to disk and a
    `RecordingDisabledWarning` is emitted, once. `clock` is the time source of the wall-time budget (injectable for
    tests). See `Budget` for how limits are enforced.

    **How evaluation is scheduled.** `batch_size` is the number of candidates the driver asks for per round, and in
    steady-state delivery the size of the in-flight window: candidates asked but not yet told. It is an algorithmic
    setting, fixed per run. `concurrency` is the most evaluations in progress at once, a resource setting that never
    changes what the strategy sees in deterministic mode. `executor` says where synchronous user functions run: `"inline"`
    (the driver's thread), `"thread"`, `"process"` (spawned workers; the function and its arguments must be picklable) or
    `"auto"` (inline when `concurrency == 1`, threads otherwise; never processes). `async def` functions always run on the
    driver's event loop. `delivery` is `"generation"` (tell a batch's results together), `"steady_state"` (tell results
    one at a time) or `None` (generation, unless the strategy needs steady-state). With `deterministic=True` (the default)
    the run is reproducible whatever `concurrency` and `executor`: steady-state results are told in ask order and the window
    is refilled only after a tell. With `deterministic=False` steady-state results are told as they finish, which is faster
    when evaluation times vary and not reproducible; generation delivery is unaffected.

    **Failures.** `failure_policy="infeasible"` (the default) turns an exception in user code, a timeout or a worker crash
    into a recorded `FAILED` or `TIMEOUT` evaluation that ranks below every feasible one, and the run goes on; it is never
    retried. `"fail_fast"` stops the run at the first one, naming the candidate and chaining the original exception.
    Misconfiguration (an unpicklable function, a return value that breaks the contract, a strategy or evaluator breaking
    its contract) always fails the run. Under `infeasible` the first failure is shown at once as an
    `EvaluationFailureWarning` with the full traceback, and a run whose first `initial_failure_guard` evaluations (10 by
    default; `None` turns it off) all failed is stopped with an `AllEvaluationsFailedError`: that is almost certainly a bug
    in the evaluation, not a result. `timeout` is the seconds one evaluation may take: an `async def` is cancelled, a worker
    process is killed and replaced, a function in a thread is abandoned (it keeps running in the background and its result
    is dropped), and it cannot be combined with `executor="inline"` or a `VectorisedEvaluator`. A timed-out evaluation
    counts towards the evaluation budget.
    """
    _warn_unrecorded(run_dir, stacklevel=3)
    driver = _build(
        strategy=strategy, evaluator=evaluator, space=space, objectives=objectives, constraints=constraints, descriptors=descriptors,
        budget=budget, seed=seed, backend=backend, batch_size=batch_size, concurrency=concurrency, executor=executor,
        delivery=delivery, deterministic=deterministic, failure_policy=failure_policy, timeout=timeout,
        initial_failure_guard=initial_failure_guard, run_dir=run_dir, name=name, clock=clock,
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
    concurrency: int = 1,
    executor: ExecutorName = "auto",
    delivery: Delivery | None = None,
    deterministic: bool = True,
    failure_policy: FailurePolicy = "infeasible",
    timeout: float | None = None,
    initial_failure_guard: int | None = 10,
    run_dir: str | Path | None = None,
    name: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult[G]:
    """The asynchronous entry point, for embedding Auxein in async applications. Same arguments as `run`."""
    _warn_unrecorded(run_dir, stacklevel=3)
    driver = _build(
        strategy=strategy, evaluator=evaluator, space=space, objectives=objectives, constraints=constraints, descriptors=descriptors,
        budget=budget, seed=seed, backend=backend, batch_size=batch_size, concurrency=concurrency, executor=executor,
        delivery=delivery, deterministic=deterministic, failure_policy=failure_policy, timeout=timeout,
        initial_failure_guard=initial_failure_guard, run_dir=run_dir, name=name, clock=clock,
    )  # fmt: skip
    return await driver.execute()
