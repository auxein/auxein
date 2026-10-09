"""The driver: it owns the loop, the budget and the recording (design doc §9).

Strategies never call evaluators; the driver asks, evaluates, records and tells. It is built on asyncio so that many
evaluations can be in progress at once, and delivers results in one of two ways:

- **generation**: ask a batch, evaluate it (up to `concurrency` candidates at once), tell all its results together.
- **steady state**: keep a window of `W = batch_size` candidates asked but not yet told; evaluate up to `concurrency` of
  them at once, each as a one-candidate batch, and tell results one at a time.

`batch_size` (the window) is an algorithmic setting; `concurrency` is a resource setting. In deterministic mode the sequence
of asks and tells depends only on the seed and `batch_size`, never on `concurrency`, the executor or timing (design doc §8.1).

A recorded run also writes checkpoints, and `resume` continues one (design doc §10.4). In deterministic mode it **replays**:
the strategy is restored and asked again, and each candidate that the recording holds an evaluation for is checked against the
recorded one and told that evaluation instead of being evaluated; in throughput mode it deletes what was recorded after the
latest checkpoint and redoes it. The decisions of a replay are those of a live run, which is why a resumed run has the log of one
that was never stopped.
"""

import asyncio
import dataclasses
import json
import time
import warnings
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, cast

from auxein.backend import Backend
from auxein.core import (
    ArrayBatch,
    Batch,
    Candidate,
    EvalContext,
    Evaluation,
    EvaluationBatch,
    Evaluator,
    FailurePolicy,
    IdIssuer,
    ListBatch,
    NoOperatorLog,
    Objective,
    OperatorLog,
    ProblemSpec,
    StateDict,
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
    ConfigurationMismatchError,
    EvaluationFailureWarning,
    EvaluatorError,
    RecordingDisabledWarning,
    ReplayMismatchError,
    ResumeWarning,
    SteadyStateVectorisationWarning,
    StrategyError,
)
from auxein.driver.result import ResultTracker, RunResult
from auxein.driver.window import Slot, Window, decode_window, encode_window
from auxein.evaluators import EvaluationError, VectorisedEvaluator
from auxein.execution import AbandonedEvaluationWarning, ExecutorKind, ExecutorName, make_executor, resolve_executor
from auxein.random import RunSeed
from auxein.recording import CheckpointInfo, ExistingRun, NoopRecorder, Recorder, ResumeError, SQLiteRecorder
from auxein.recording.genomes import EncodedGenome, encode_batch
from auxein.recording.replay import ReplayRecord, ReplayStream, iter_records
from auxein.recording.sqlite import DEFAULT_GENOME_THRESHOLD
from auxein.spaces import Space, codec_of
from auxein.spaces.codec import GenomeCodec

T = TypeVar("T")

DEFAULT_OBJECTIVES = (Objective("value"),)

Delivery = Literal["generation", "steady_state"]

DEFAULT_CHECKPOINT_EVERY = 300.0
"""Seconds of active run time between checkpoints, unless `checkpoint_every` says otherwise."""
DEFAULT_KEEP_CHECKPOINTS = 2

_REPLAY_YIELD = 256
"""A replay with nothing to wait for hands control back to the event loop this often, so that Ctrl-C still works."""

_REBUILD_CHUNK = 1024


@dataclass(frozen=True)
class _FailureNote:
    """The first failure of a run, as far as the guard's message needs it (so that it fits in a checkpoint)."""

    candidate_id: int
    status: Status
    error: str | None


@dataclass
class _Resumption:
    """What `execute` has worked out about a run it is resuming, before it replays or continues it."""

    mode: Literal["replay", "truncate"]
    loaded: tuple[CheckpointInfo, StateDict] | None
    replay: ReplayStream | None
    carried: float
    """Active seconds of earlier sessions, up to the point the run continues from."""


_COMPLETE = (
    "the run is already complete with this budget, so resume returns its result without evaluating anything; "
    "give a larger budget to extend it"
)


def _normalised(value: object) -> object:
    """A value as it reads back from JSON (tuples are lists, keys are strings), so that metadata compares equal after a round trip."""
    return json.loads(json.dumps(value, default=str))


def _show(value: object) -> str:
    return json.dumps(value, default=str)


def _after(batch: Batch[G], count: int) -> Batch[G]:
    """The candidates of a batch after its first `count`, in order."""
    if isinstance(batch, ArrayBatch):
        return cast("Batch[G]", batch.slice(count, len(batch)))
    return ListBatch(tuple(batch.candidates[count:]))


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
        checkpoint_every: float = DEFAULT_CHECKPOINT_EVERY,
        checkpoint_every_evaluations: int | None = None,
        keep_checkpoints: int = DEFAULT_KEEP_CHECKPOINTS,
        resume: bool = False,
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
        if not checkpoint_every > 0:
            raise ValueError(f"checkpoint_every must be a positive number of seconds (math.inf for never), got {checkpoint_every!r}")
        if checkpoint_every_evaluations is not None and checkpoint_every_evaluations < 1:
            raise ValueError(f"checkpoint_every_evaluations must be at least 1 or None, got {checkpoint_every_evaluations}")
        if keep_checkpoints < 0:
            raise ValueError(f"keep_checkpoints must be at least 0 (0 writes no checkpoints), got {keep_checkpoints}")
        if timeout is not None and getattr(evaluator, "batched", False):
            raise ValueError(
                "a timeout cannot be combined with a batched EpisodeEvaluator: it makes one call per batch on the driver's thread, "
                "which nothing can interrupt. Use an environment without run_batch to time episodes out"
            )
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
        self._first_failure: _FailureNote | None = None
        self._failures = {Status.FAILED: 0, Status.TIMEOUT: 0}
        self._abandoned = 0
        self._delivery: Delivery = _resolve_delivery(delivery, strategy)
        self._concurrency, self._deterministic = concurrency, deterministic
        self._strategy, self._evaluator, self._problem = strategy, evaluator, problem
        self._budget, self._seed, self._backend, self._batch_size = budget, seed, backend, batch_size
        self._recorder, self._run_dir, self._name, self._clock = recorder, run_dir, name, clock

        run_seed = RunSeed(seed)
        self._issuer = IdIssuer()
        self._codec: GenomeCodec[Any] | None = codec_of(problem.space)
        if isinstance(recorder, SQLiteRecorder):
            recorder.use_codec(self._codec)  # structured genomes are recorded through their space's codec
        operators: OperatorLog = recorder if isinstance(recorder, SQLiteRecorder) else NoOperatorLog()
        self._strategy_context = StrategyContext(run_seed.stream("strategy", backend=backend), backend, self._issuer.next, operators)
        host_backend = Backend("numpy", "cpu", backend.precision)
        self._base_context: EvalContext[G] = EvalContext(
            problem,
            backend,
            lambda cid: run_seed.stream("evaluation", cid, backend=host_backend),  # always numpy: see EvalContext.rng_for
            lambda cid: run_seed.stream("evaluation-batch", cid, backend=backend),
            lambda cid, scenario: run_seed.stream("episode", cid, scenario, backend=host_backend),
            lambda cid: run_seed.stream("episode-batch", cid, backend=backend),
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
        self._resume = resume
        # checkpoints and resuming (design doc §10.4)
        self._checkpoint_every, self._checkpoint_every_evaluations = checkpoint_every, checkpoint_every_evaluations
        self._keep_checkpoints = keep_checkpoints if run_dir is not None else 0
        self._last_checkpoint_time = 0.0
        self._last_checkpoint_used = 0
        self._consistent = True
        """Whether the strategy, the driver and the recording agree right now: false between an ask and the tell that follows."""
        self._shaped = False
        """Whether the evaluation budget has changed what the run did (a smaller ask, a dropped surplus). Once it has, a
        checkpoint could not be continued with a larger budget into the run a longer budget would have made, so none is
        taken any more; `_clean_checkpoint` saves the last state that could, just before."""
        self._clean_taken = False
        self._ask_high = batch_size
        """The most candidates one ask has returned so far: the budget is 'close' when what remains is not more than this."""
        self._window: Window[G] | None = None
        self._restored_window: Window[G] | None = None
        self._replay: ReplayStream | None = None
        self.already_complete = False
        """Set by a resume that found the run complete with this budget and returned its result without running."""

    # --- the run ---

    async def execute(self) -> RunResult[G]:
        self._started = self._clock()
        resumption: _Resumption | None = None
        if self._resume:
            opened = self._open_resume()
            if isinstance(opened, RunResult):
                return opened  # nothing to do: the run is complete with this budget
            resumption = opened
        else:
            self._recorder.on_start(self._metadata())
        status: str = "failed"
        stop_reason: str | None = None
        abandon = False
        executor = make_executor(self._executor_kind, self._concurrency)
        self._eval_context = dataclasses.replace(self._base_context, executor=executor)
        try:
            self._strategy.bind(self._problem, self._strategy_context)
            if resumption is not None:
                self._restore(resumption)
            if self._delivery == "steady_state":
                self._warn_if_vectorised()
                stop_reason = await self._steady_state()
            else:
                stop_reason = await self._loop()
            self._finish_replay()
            self._check_guard_at_end()
            self._checkpoint_at_end()
            status = "completed"
        except (KeyboardInterrupt, asyncio.CancelledError):
            status = "interrupted"
            self._checkpoint_on_interrupt()
            raise
        except ReplayMismatchError:
            abandon = True  # nothing new is recorded by an attempt that found the recording does not match
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
            if self._replay is not None:
                self._replay.close()
            if abandon and isinstance(self._recorder, SQLiteRecorder):
                self._recorder.abandon()
            else:
                self._recorder.on_end(status, stop_reason, self._summary())
        assert stop_reason is not None  # the run completed
        return RunResult(
            stop_reason=stop_reason,
            evaluations_used=self._used,
            wall_time=self._elapsed(),
            run_dir=self._run_dir,
            best=self._tracker.best,
            pareto_front=self._tracker.pareto_front,
            trace=self._tracker.trace,
            status_counts=self._status_counts(),
        )

    def _elapsed(self) -> float:
        """Active seconds of the whole run: this session, plus the earlier ones when the run was resumed."""
        return self._clock() - self._started

    async def _loop(self) -> str:
        while True:
            self._maybe_checkpoint()
            reason = self._stop_reason()
            if reason is not None:
                return reason
            if await self._step():
                return "budget:evaluations"

    def _replaying(self) -> bool:
        return self._replay is not None and self._replay.remaining > 0

    def _stop_reason(self) -> str | None:
        """Why the run must stop before the next ask, if it must. Checked in this order."""
        budget = self._budget
        if budget.evaluations is not None and self._used >= budget.evaluations:
            return "budget:evaluations"
        # while replaying, the clock says nothing: the recorded run got this far, so the wall-time budget did not stop it
        if budget.wall_time is not None and not self._replaying() and self._elapsed() >= budget.wall_time:
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
        if remaining is not None and not self._shaped:
            self._note_budget_pressure(remaining, self._batch_size)
        wanted = self._batch_size if remaining is None else min(self._batch_size, remaining)
        self._consistent = False
        batch = self._strategy.ask(wanted)
        step = self._validate_batch(batch)
        self._ask_high = max(self._ask_high, len(batch.candidates))

        truncated = remaining is not None and len(batch.candidates) > remaining
        if truncated:
            assert remaining is not None
            batch = take(batch, remaining)  # the budget is a hard limit: evaluate only what remains, in ask order
            self._shaped = True

        results, recorded, rerecord = await self._evaluate(batch)
        self._validate_results(batch, results)
        self._account(
            step, batch, results, recorded, rerecord
        )  # records, and applies the failure policy and the guard before anyone is told

        if truncated:
            self._consistent = True
            return True  # the strategy is not told about an incomplete batch
        self._strategy.tell(results)
        self._recorder.on_tell(step, len(results), recorded == len(results))
        self._consistent = True
        if recorded:
            await asyncio.sleep(0)  # replaying has nothing to wait for: let the event loop breathe
        return False

    async def _evaluate(self, batch: Batch[G]) -> tuple[EvaluationBatch[G], int, bool]:
        """The evaluations of a batch, how many of its first candidates came from the recording (a resumed run replaying),
        and whether the whole batch was evaluated again, replacing what was recorded.

        Replayed candidates are checked against the recorded ones and never evaluated again; if the recording holds only
        the first of them (a final batch that a larger budget has made longer), the others are evaluated now. A vectorised
        function draws its randomness for the whole batch, so it evaluates the whole batch again to give what a longer run
        would have given (design doc §10.4); so does any evaluator that says it is `batch_sensitive`."""
        replay = self._replay
        if replay is None or replay.remaining == 0:
            return await self._evaluator.evaluate(batch, self._eval_context), 0, False
        candidates = batch.candidates
        records = replay.take(len(candidates))
        self._verify(candidates, encode_batch(take(batch, len(records)), self._codec), records)
        done = [record.evaluation(candidate) for record, candidate in zip(records, candidates, strict=False)]
        if len(records) == len(candidates):
            return EvaluationBatch(done), len(records), False
        if getattr(self._evaluator, "batch_sensitive", False):
            return await self._evaluator.evaluate(batch, self._eval_context), len(records), True
        rest = _after(batch, len(records))
        live = await self._evaluator.evaluate(rest, self._eval_context)
        self._validate_results(rest, live)
        return EvaluationBatch([*done, *live], live.episodes), len(records), False

    def _account(self, step: int, batch: Batch[G], results: EvaluationBatch[G], recorded: int = 0, rerecord: bool = False) -> None:
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
        if failed:
            self._count_failures(failed)  # under fail_fast this raises, before the evaluation is recorded: it is evaluated again on resume
        self._recorder.on_batch(step, batch, results, recorded, rerecord)
        if failed or self._guard_open:
            self._apply_failure_rules(results, failed, quiet=recorded > 0)

    # --- failures (design doc §6.6) ---

    def _count_failures(self, failed: list[Evaluation[G]]) -> None:
        """Count failures, and stop the run at the first one under `fail_fast`."""
        for evaluation in failed:
            self._failures[evaluation.status] += 1
        if self._policy == "fail_fast":
            first = failed[0]
            raise EvaluationError([first.candidate.id], detail=f"{first.status.value}: {first.error}")

    def _apply_failure_rules(self, results: EvaluationBatch[G], failed: list[Evaluation[G]], quiet: bool) -> None:
        """The driver is the backstop for failures, whichever evaluator produced them: it warns about the first one and stops a
        run whose first evaluations all failed. A failure that comes from the recording was reported by the session that
        evaluated it, so replaying it is `quiet`."""
        if failed and self._first_failure is None:
            first = failed[0]
            self._first_failure = _FailureNote(int(first.candidate.id), first.status, first.error)
            if not quiet:
                warnings.warn(
                    self._describe_failure(self._first_failure, "first failure of this run"), EvaluationFailureWarning, stacklevel=2
                )
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
    def _describe_failure(note: _FailureNote, text: str | None) -> str:
        head = f"candidate {note.candidate_id} ended with status {note.status.value!r}"
        return f"{head} ({text}):\n{note.error}" if text else f"{head}:\n{note.error}"

    def _status_counts(self) -> dict[str, int]:
        failed = sum(self._failures.values())
        return {"ok": self._used - failed, "failed": self._failures[Status.FAILED], "timeout": self._failures[Status.TIMEOUT]}

    # --- steady-state delivery ---

    def _warn_if_vectorised(self) -> None:
        if isinstance(self._evaluator, VectorisedEvaluator) or getattr(self._evaluator, "batched", False):
            warnings.warn(
                "steady-state delivery calls a vectorised or batched evaluator with one candidate at a time, which defeats the "
                "purpose of batching: use delivery='generation' (the default for strategies that support it)",
                SteadyStateVectorisationWarning,
                stacklevel=2,
            )

    async def _steady_state(self) -> str:
        """Keep up to `batch_size` candidates asked-but-not-told, evaluate up to `concurrency` at once, tell one at a time.

        Deterministic mode tells in ask order and asks only right after a tell, one result at a time, so that the strategy
        sees the same calls whatever the timing and the number of workers. Throughput mode tells as results arrive and
        refills the window at once. See design doc §9.2.
        """
        window = self._restored_window if self._restored_window is not None else Window[G](deque(), {}, {}, [])
        self._window, self._restored_window = window, None
        self._attach_records(list(window.inflight.values()))  # a resumed run's window: candidates the recording may already hold
        replayed = 0
        try:
            while True:
                self._maybe_checkpoint()
                self._refill(window)
                self._start(window)
                if not window.inflight:
                    assert window.stop is not None  # nothing in flight and nothing asked: the budget is spent or asking stopped
                    return window.stop
                if self._deliver(window):
                    replayed += 1
                    if replayed % _REPLAY_YIELD == 0:
                        await asyncio.sleep(0)
                    continue
                await self._wait_for_one(window)
        finally:
            for task in window.running:
                task.cancel()
            await asyncio.gather(*window.running, return_exceptions=True)  # nothing keeps running after the run

    def _refill(self, window: Window[G]) -> None:
        """Ask for as many candidates as the window and the budget allow, unless the run must stop asking."""
        if window.stop is None:
            window.stop = self._stop_reason()
            if window.stop is not None and window.stop != "budget:evaluations":
                if window.queue:
                    self._shape()  # what has not started will never be evaluated: a longer run would have evaluated it
                for slot in window.queue:  # whatever has not started will never be evaluated, told or recorded
                    del window.inflight[slot.seq]
                window.queue.clear()
        if window.stop is not None:
            return
        room = self._batch_size - len(window.inflight)
        remaining = None if self._budget.evaluations is None else self._budget.evaluations - self._used - len(window.inflight)
        if remaining is not None and not self._shaped:
            self._note_budget_pressure(remaining, room)
        if room < 1 or (remaining is not None and remaining < 1):
            return
        self._consistent = False
        batch = self._strategy.ask(room if remaining is None else min(room, remaining))
        step = self._validate_batch(batch)
        self._ask_high = max(self._ask_high, len(batch.candidates))
        if remaining is not None and len(batch.candidates) > remaining:
            batch = take(batch, remaining)  # a hard limit: the surplus is dropped before it is ever queued
            self._shaped = True
        asked: list[Slot[G]] = []
        for index in range(len(batch.candidates)):
            slot = Slot(window.next_seq, step, single(batch, index))
            window.next_seq += 1
            window.queue.append(slot)
            window.inflight[slot.seq] = slot
            asked.append(slot)
        self._attach_records(asked)
        self._consistent = True

    def _start(self, window: Window[G]) -> None:
        while window.queue and len(window.running) < self._concurrency:
            slot = window.queue.popleft()
            if slot.result is not None:
                continue  # its result came from the recording: nothing to evaluate
            slot.task = asyncio.ensure_future(self._evaluator.evaluate(slot.batch, self._eval_context))
            window.running[slot.task] = slot

    def _deliver(self, window: Window[G]) -> bool:
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
        self._consistent = False
        del window.inflight[slot.seq]
        self._validate_results(slot.batch, slot.result)
        self._account(slot.step, slot.batch, slot.result, 1 if slot.recorded else 0)
        self._strategy.tell(slot.result)
        self._recorder.on_tell(slot.step, 1, slot.recorded)
        self._consistent = True
        return True

    async def _wait_for_one(self, window: Window[G]) -> None:
        done, _ = await asyncio.wait(window.running, return_when=asyncio.FIRST_COMPLETED)
        for task in sorted(done, key=lambda t: window.running[t].seq):  # simultaneous finishes are told in ask order
            slot = window.running.pop(task)
            slot.result = task.result()  # raises the evaluator's error; the run's `finally` cancels the rest
            if not self._deterministic:
                window.finished.append(slot)

    # --- checkpoints (design doc §10.4) ---

    def _note_budget_pressure(self, remaining: int, asking: int) -> None:
        """Called before an ask while the evaluation budget is running low.

        A checkpoint can only be continued with a larger budget into the run that budget would have made if the budget had
        not yet changed anything the run did. So the last such state is saved once the budget is close (what remains is not
        more than the largest ask so far), and when the next ask is smaller than it would be without the limit
        (`remaining < asking`) the run is `shaped` and takes no more checkpoints."""
        if not self._clean_taken and remaining <= self._ask_high:
            self._write_checkpoint()
            self._clean_taken = True
        if remaining < asking:
            self._shaped = True

    def _shape(self) -> None:
        """A stop is about to drop candidates the strategy has asked for: save the state before, if it can still be continued."""
        if not self._shaped:
            self._write_checkpoint()
            self._shaped = True

    def _maybe_checkpoint(self) -> None:
        """Take a periodic checkpoint if one is due. Called only where the strategy, the driver and the recording agree."""
        if self._keep_checkpoints < 1 or self._shaped:
            return
        due = self._elapsed() - self._last_checkpoint_time >= self._checkpoint_every
        every = self._checkpoint_every_evaluations
        if due or (every is not None and self._used - self._last_checkpoint_used >= every):
            self._write_checkpoint()

    def _write_checkpoint(self) -> None:
        if self._keep_checkpoints < 1:
            return
        self._recorder.checkpoint(self._state_dict(), self._used, self._keep_checkpoints)
        self._last_checkpoint_time = self._elapsed()
        self._last_checkpoint_used = self._used

    def _checkpoint_at_end(self) -> None:
        if not self._shaped:
            self._write_checkpoint()

    def _checkpoint_on_interrupt(self) -> None:
        """Save the state if the run was interrupted at a point where it is consistent: not between an ask and its tell."""
        if self._consistent and not self._shaped:
            try:
                self._write_checkpoint()
            except Exception as error:  # the interrupt is what matters; a checkpoint that fails must not hide it
                warnings.warn(f"could not write a checkpoint while interrupting: {error!r}", RuntimeWarning, stacklevel=2)

    def _state_dict(self) -> StateDict:
        """The state of the strategy and of the driver, for a checkpoint."""
        issued = self._issuer.issued
        unseen: list[int] = []  # ids issued but never carried by an asked batch: rare, so a few fast searches find them all
        position = self._seen.find(0)
        while position != -1:
            unseen.append(position)
            position = self._seen.find(0, position + 1)
        unseen.extend(range(len(self._seen), issued))
        first = self._first_failure
        driver: StateDict = {
            "issuer": dict(self._issuer.state_dict()),  # pyright: ignore[reportArgumentType]
            "unseen": unseen,
            "last_step": self._last_step,
            "used": self._used,
            "costs": dict(self._costs),
            "failures": {"failed": self._failures[Status.FAILED], "timeout": self._failures[Status.TIMEOUT]},
            "guard_told": self._guard_told,
            "guard_open": self._guard_open,
            "first_failure": None
            if first is None
            else {"candidate_id": first.candidate_id, "status": first.status.value, "error": first.error},
            "wall_time": self._elapsed(),
            "ask_high": self._ask_high,
            "window": None if self._window is None else encode_window(self._window, self._codec),
        }
        return {"delivery": self._delivery, "strategy": self._strategy.state_dict(), "driver": driver}

    def _restore(self, resumption: _Resumption) -> None:
        """Put the strategy and the driver back as the checkpoint had them (a fresh bound strategy if there is none)."""
        self._replay = resumption.replay
        self._last_checkpoint_time = self._elapsed()  # the next periodic checkpoint is one interval from here, not from time zero
        if resumption.loaded is None:
            return
        _, state = resumption.loaded
        if state["delivery"] != self._delivery:
            raise ResumeError(f"the checkpoint was taken with delivery {state['delivery']!r}, this run uses {self._delivery!r}")
        driver = cast("dict[str, object]", state["driver"])
        self._strategy.load_state_dict(cast("StateDict", state["strategy"]))
        self._issuer.load_state_dict(cast("dict[str, int]", driver["issuer"]))
        issued = self._issuer.issued
        self._seen = bytearray(b"\x01") * issued
        for candidate_id in cast("list[int]", driver["unseen"]):
            self._seen[candidate_id] = 0
        self._last_step = cast("int", driver["last_step"])
        self._used = cast("int", driver["used"])
        failures = cast("dict[str, int]", driver["failures"])
        self._failures = {Status.FAILED: failures["failed"], Status.TIMEOUT: failures["timeout"]}
        self._guard_told = cast("int", driver["guard_told"])
        self._guard_open = cast("bool", driver["guard_open"])
        first = cast("dict[str, object] | None", driver["first_failure"])
        if first is not None:
            self._first_failure = _FailureNote(
                cast("int", first["candidate_id"]), Status(cast("str", first["status"])), cast("str | None", first["error"])
            )
        self._last_checkpoint_used = self._used
        self._ask_high = max(self._batch_size, cast("int", driver["ask_high"]))
        window = cast("StateDict | None", driver["window"])
        if window is not None:
            self._restored_window = cast("Window[G]", decode_window(window, self._backend, self._codec))

    # --- resuming (design doc §10.4) ---

    def _open_resume(self) -> "_Resumption | RunResult[G]":
        """Open the recorded run, check that it can be resumed with these settings, and decide how.

        Returns what `execute` needs to restore and continue the run, or, when the run is already complete with this budget,
        its result: there is nothing to do."""
        recorder = self._recorder
        if not isinstance(recorder, SQLiteRecorder):  # pragma: no cover - resume() always builds one
            raise ResumeError("only a recorded run can be resumed")
        existing = recorder.open_existing()  # takes the lock
        replay: ReplayStream | None = None
        try:
            self._check_configuration(existing.metadata)
            budget = self._budget
            if budget.evaluations is not None and budget.evaluations < existing.evaluations_recorded:
                raise ResumeError(
                    f"the evaluation budget {budget.evaluations} is smaller than the {existing.evaluations_recorded} evaluations the "
                    "run has already recorded: when resuming, a budget can stay or grow"
                )
            sessions = cast("list[dict[str, object]]", existing.metadata.get("sessions", []))
            last = sessions[-1] if sessions else {}
            database = recorder.run_dir / "events.sqlite"
            if last.get("status") == "completed" and _normalised(last.get("budget")) == _normalised(budget.describe()):
                return self._finished_result(existing, recorder, database)
            loaded = recorder.load_checkpoint(existing.checkpoints, self._backend)
            seq = 0 if loaded is None else loaded[0].event_seq
            used_at_checkpoint = self._rebuild_tracker(database, seq)
            if loaded is not None:
                recorded_used = cast("int", cast("dict[str, object]", loaded[1]["driver"])["used"])
                if recorded_used != used_at_checkpoint:
                    raise ResumeError(
                        f"the checkpoint at event {seq} says {recorded_used} evaluations were used, but the recording holds "
                        f"{used_at_checkpoint} up to that event: the run directory was modified after the run"
                    )
            if self._deterministic:
                replay = ReplayStream(database, seq)
                mode: Literal["replay", "truncate"] = "replay"
                count = replay.total
            else:
                mode = "truncate"
                count = recorder.truncate_after(seq)
            # active time so far: what the checkpoint says, or what a session that ended says, whichever is more. A session
            # that was killed after its last checkpoint loses the time since, which nothing recorded
            ended = [cast("float", s["wall_time"]) for s in sessions if s.get("wall_time") is not None]
            at_checkpoint = cast("float", cast("dict[str, object]", loaded[1]["driver"])["wall_time"]) if loaded is not None else 0.0
            wall = max([at_checkpoint, *ended])
            self._started -= wall
            event = {
                "mode": mode,
                "checkpoint": seq if loaded is not None else None,
                "replayed" if mode == "replay" else "truncated": count,
                "budget_old": last.get("budget"),
                "budget_new": budget.describe(),
            }
            recorder.prepare_session(mode, budget.describe(), event, seq)
            return _Resumption(mode, loaded, replay, wall)
        except BaseException:
            if replay is not None:
                replay.close()
            recorder.abandon()
            raise

    def _check_configuration(self, recorded: dict[str, object]) -> None:
        """Only the budget may change when resuming: every other setting must match the recorded run, or the resumed run
        would not be the same run."""
        current = cast("dict[str, object]", _normalised(self._metadata()))
        problem, then = cast("dict[str, object]", current["problem"]), cast("dict[str, object]", recorded.get("problem", {}))
        differences: list[str] = []

        def compare(name: str, was: object, now: object) -> None:
            if was != now:
                differences.append(f"{name}: recorded {_show(was)}, given {_show(now)}")

        for key in ("seed", "backend", "precision", "batch_size", "concurrency", "executor", "delivery", "deterministic"):
            compare(key, recorded.get(key), current[key])
        for key in ("failure_policy", "timeout", "initial_failure_guard"):
            compare(key, recorded.get(key), current[key])
        for key in ("space", "objectives", "constraints", "descriptors"):
            compare(f"problem.{key}", then.get(key), problem[key])
        for key in ("strategy", "evaluator"):
            was, now = cast("dict[str, object]", recorded.get(key, {})), cast("dict[str, object]", current[key])
            if was.get("repr") != now["repr"]:
                compare(key, was.get("repr"), now["repr"])
            elif (was.get("class"), was.get("module")) != (now["class"], now["module"]):
                compare(key, f"{was.get('module')}.{was.get('class')}", f"{now['module']}.{now['class']}")
        if differences:
            listing = "\n".join(f"  - {line}" for line in differences)
            raise ConfigurationMismatchError(
                f"cannot resume {self._run_dir}: only the budget may change when resuming a run, but these settings differ from the "
                f"recorded run:\n{listing}"
            )

    def _finished_result(self, existing: ExistingRun, recorder: SQLiteRecorder, database: Path) -> "RunResult[G]":
        """The result of a run that is complete with the given budget, from its recording; nothing is evaluated or written."""
        self._rebuild_tracker(database, None)
        recorder.abandon()
        metadata = existing.metadata
        summary = cast("dict[str, object]", metadata.get("summary", {}))
        self.already_complete = True
        counts = cast("dict[str, int]", summary.get("status_counts", {}))
        return RunResult(
            stop_reason=cast("str", metadata["stop_reason"]),
            evaluations_used=existing.evaluations_recorded,
            wall_time=cast("float", summary.get("wall_time", 0.0)),
            run_dir=self._run_dir,
            best=self._tracker.best,
            pareto_front=self._tracker.pareto_front,
            trace=self._tracker.trace,
            status_counts=counts,
        )

    def _rebuild_tracker(self, database: Path, upto: int | None) -> int:
        """Fold the recorded evaluations (up to event `upto`, or all) back into the result tracker and the cost totals, in the
        order they were told. The recording is the one source of truth for the result, so the tracker is not checkpointed.
        Returns how many evaluations that was."""
        self._tracker = ResultTracker(self._problem.objectives, constrained=bool(self._problem.constraints))
        self._costs = {unit: 0.0 for unit in self._budget.cost}
        chunk: list[Evaluation[G]] = []
        total = 0
        for record in iter_records(database, upto=upto):
            evaluation = cast("Evaluation[G]", record.evaluation(record.decoded_candidate(self._backend, self._codec)))
            chunk.append(evaluation)
            for unit, amount in evaluation.cost.units.items():
                if unit in self._costs:
                    self._costs[unit] += amount
            if len(chunk) == _REBUILD_CHUNK:
                self._tracker.add(chunk, total)
                total += len(chunk)
                chunk = []
        if chunk:
            self._tracker.add(chunk, total)
            total += len(chunk)
        return total

    def _attach_records(self, slots: Sequence[Slot[G]]) -> None:
        """Give each slot that the recording has an evaluation for its recorded result, after checking the candidate is the
        one that was recorded. Records are in ask order, so they belong to the first slots."""
        replay = self._replay
        if replay is None or replay.remaining == 0 or not slots:
            return
        records = replay.take(len(slots))
        candidates = [slot.batch.candidates[0] for slot in slots[: len(records)]]
        genomes = [encode_batch(slot.batch, self._codec)[0] for slot in slots[: len(records)]]
        self._verify(candidates, genomes, records)
        for slot, record, candidate in zip(slots, records, candidates, strict=False):
            slot.result = EvaluationBatch([record.evaluation(candidate)])
            slot.recorded = True

    def _verify(self, candidates: Sequence[Candidate[G]], genomes: list[EncodedGenome], records: list[ReplayRecord]) -> None:
        """Replay's check: each regenerated candidate must be exactly the recorded one, genome included, byte for byte."""
        for candidate, genome, record in zip(candidates, genomes, records, strict=False):
            what: list[str] = []
            if candidate.id != record.candidate_id:
                what.append(f"its id (the recording holds candidate {record.candidate_id} here)")
            if candidate.origin != record.origin:
                what.append(f"its origin (recorded {record.origin!r}, regenerated {candidate.origin!r})")
            if candidate.step != record.step:
                what.append(f"its step (recorded {record.step}, regenerated {candidate.step})")
            if tuple(candidate.parents) != record.parents:
                what.append(f"its parents (recorded {list(record.parents)}, regenerated {list(candidate.parents)})")
            if (genome.kind, genome.data, genome.dtype, genome.shape) != (
                record.genome.kind,
                record.genome.data,
                record.genome.dtype,
                record.genome.shape,
            ):
                what.append("its genome")
            if what:
                raise ReplayMismatchError(
                    f"cannot resume {self._run_dir}: replaying the recording regenerated candidate {candidate.id} differently from the "
                    f"recorded one, in {' and '.join(what)}. Replay needs the strategy to propose exactly what it proposed in the "
                    "recorded run, so the likely causes are that the strategy's configuration or code changed, or that the seed is "
                    "different (or that the recording was edited). Nothing was recorded by this attempt"
                )

    def _finish_replay(self) -> None:
        replay = self._replay
        if replay is not None and replay.remaining > 0:
            raise ReplayMismatchError(
                f"cannot resume {self._run_dir}: the run ended with {replay.remaining} of its {replay.total} recorded evaluations "
                "never reached, so the restored run is not the recorded one (the strategy's configuration or code changed). "
                "Nothing was recorded by this attempt"
            )

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
            "wall_time": self._elapsed(),
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
    checkpoint_every: float | None,
    checkpoint_every_evaluations: int | None,
    keep_checkpoints: int | None,
    genome_store_threshold: int | None,
    run_dir: str | Path | None,
    name: str | None,
    clock: Callable[[], float],
    resume: bool = False,
) -> Driver[G]:
    if run_dir is None and (checkpoint_every is not None or checkpoint_every_evaluations is not None or keep_checkpoints is not None):
        raise ValueError(
            "checkpoint_every, checkpoint_every_evaluations and keep_checkpoints need run_dir: checkpoints are written into the "
            "run directory, and a run without one is not recorded"
        )
    problem = ProblemSpec(space, tuple(objectives), tuple(constraints), tuple(descriptors))
    recorder: Recorder = (
        NoopRecorder() if run_dir is None else SQLiteRecorder(run_dir, resume=resume, genome_threshold=genome_store_threshold)
    )
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
        checkpoint_every=DEFAULT_CHECKPOINT_EVERY if checkpoint_every is None else checkpoint_every,
        checkpoint_every_evaluations=checkpoint_every_evaluations,
        keep_checkpoints=DEFAULT_KEEP_CHECKPOINTS if keep_checkpoints is None else keep_checkpoints,
        resume=resume,
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


def _execute(driver: Driver[G]) -> RunResult[G]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(driver.execute())
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="auxein-driver") as pool:
        return pool.submit(asyncio.run, driver.execute()).result()


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
    checkpoint_every: float | None = None,
    checkpoint_every_evaluations: int | None = None,
    keep_checkpoints: int | None = None,
    genome_store_threshold: int | None = DEFAULT_GENOME_THRESHOLD,
    run_dir: str | Path | None = None,
    name: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult[G]:
    """Run a strategy on a problem and return what it found.

    This is the synchronous entry point. It works in plain scripts, and inside an already running event loop (as in
    Jupyter), where it runs the driver on a fresh loop in a dedicated thread and waits for it.

    The run is recorded only if `run_dir` is given; otherwise nothing is written to disk and a
    `RecordingDisabledWarning` is emitted, once. `clock` is the time source of the wall-time budget and of the checkpoint
    interval (injectable for tests). See `Budget` for how limits are enforced.

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

    **Checkpoints** (only with `run_dir`; see `resume`). A checkpoint of the strategy's and the driver's state is written
    every `checkpoint_every` seconds of run time (default 300; `math.inf` for never), optionally every
    `checkpoint_every_evaluations` evaluations, when the run ends and when it is interrupted at a consistent moment. The
    newest `keep_checkpoints` (default 2) are kept; 0 writes none, and then a resume replays the run from its start.
    Passing any of these without `run_dir` is an error.

    **The genome store** (design doc §10.3): a genome whose encoding is larger than `genome_store_threshold` bytes (default
    4096; `None` keeps every genome inline) is stored once in the run's database, by its SHA-256, and the candidates refer to
    it. It changes the size of the recording, not its content, and is ignored without `run_dir`.
    """
    _warn_unrecorded(run_dir, stacklevel=3)
    driver = _build(
        strategy=strategy, evaluator=evaluator, space=space, objectives=objectives, constraints=constraints, descriptors=descriptors,
        budget=budget, seed=seed, backend=backend, batch_size=batch_size, concurrency=concurrency, executor=executor,
        delivery=delivery, deterministic=deterministic, failure_policy=failure_policy, timeout=timeout,
        initial_failure_guard=initial_failure_guard, checkpoint_every=checkpoint_every,
        checkpoint_every_evaluations=checkpoint_every_evaluations, keep_checkpoints=keep_checkpoints,
        genome_store_threshold=genome_store_threshold, run_dir=run_dir, name=name,
        clock=clock,
    )  # fmt: skip
    return _execute(driver)


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
    checkpoint_every: float | None = None,
    checkpoint_every_evaluations: int | None = None,
    keep_checkpoints: int | None = None,
    genome_store_threshold: int | None = DEFAULT_GENOME_THRESHOLD,
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
        initial_failure_guard=initial_failure_guard, checkpoint_every=checkpoint_every,
        checkpoint_every_evaluations=checkpoint_every_evaluations, keep_checkpoints=keep_checkpoints,
        genome_store_threshold=genome_store_threshold, run_dir=run_dir, name=name,
        clock=clock,
    )  # fmt: skip
    return await driver.execute()


def resume(
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
    checkpoint_every: float | None = None,
    checkpoint_every_evaluations: int | None = None,
    keep_checkpoints: int | None = None,
    genome_store_threshold: int | None = DEFAULT_GENOME_THRESHOLD,
    run_dir: str | Path,
    name: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult[G]:
    """Continue a recorded run: one that was interrupted or killed, that failed, or that finished and now gets more budget.

    The intended workflow is to run the same script with `resume` instead of `run`: it takes the same arguments, with
    `run_dir` required. **Only the budget may change.** Everything else must match the recorded run, and any difference
    (the seed, the problem, `batch_size`, `delivery`, `deterministic`, `concurrency`, `executor`, `timeout`,
    `failure_policy`, the failure guard, the strategy's or the evaluator's class or `repr`) is a
    `ConfigurationMismatchError` listing each setting with its recorded and its given value. Auxein cannot check that the
    evaluator's *code* is unchanged: replay does not run it, so that is yours to keep true. The checkpoint options and
    `clock` are not part of the run and may differ.

    **How it continues** depends on the recorded `deterministic` setting. In deterministic mode the strategy is restored
    from the latest checkpoint (or built afresh if there is none) and asked again; every candidate the recording holds an
    evaluation for is checked against the recorded one, genome byte for byte, and its **recorded evaluation is used instead
    of evaluating it again**: no recorded evaluation is ever repeated. When the recorded evaluations run out the run
    continues live. A candidate that differs raises `ReplayMismatchError`, naming it, and records nothing. The resumed run
    is the run that never stopped: its event log equals an uninterrupted run's, and **extending a finished run with a
    larger evaluation budget gives the same result as one run with that budget**. In throughput mode
    (`deterministic=False`) the run restarts from its latest checkpoint, everything recorded after the checkpoint is
    deleted and that work is redone (with no checkpoint, the run starts again from the beginning).

    Budgets are cumulative: evaluations and cost units across sessions, and **wall time as active run time** (the time
    between sessions does not count; time that passed in a session killed after its last checkpoint is lost). A run that
    is already complete with exactly this budget returns its result without evaluating anything (with a `ResumeWarning`).
    A `fail_fast` run stopped at a candidate whose evaluation failed never recorded it, so resuming after fixing the
    evaluator evaluates it again. Two processes must never write to one run: a run being written by a live process is
    refused, and the lock of a killed one is taken over. The `RunResult` covers the whole run, not just this session.
    """
    driver = _build(
        strategy=strategy, evaluator=evaluator, space=space, objectives=objectives, constraints=constraints, descriptors=descriptors,
        budget=budget, seed=seed, backend=backend, batch_size=batch_size, concurrency=concurrency, executor=executor,
        delivery=delivery, deterministic=deterministic, failure_policy=failure_policy, timeout=timeout,
        initial_failure_guard=initial_failure_guard, checkpoint_every=checkpoint_every,
        checkpoint_every_evaluations=checkpoint_every_evaluations, keep_checkpoints=keep_checkpoints,
        genome_store_threshold=genome_store_threshold, run_dir=run_dir, name=name,
        clock=clock, resume=True,
    )  # fmt: skip
    result = _execute(driver)
    if driver.already_complete:
        warnings.warn(_COMPLETE, ResumeWarning, stacklevel=2)
    return result


async def aresume(
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
    checkpoint_every: float | None = None,
    checkpoint_every_evaluations: int | None = None,
    keep_checkpoints: int | None = None,
    genome_store_threshold: int | None = DEFAULT_GENOME_THRESHOLD,
    run_dir: str | Path,
    name: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult[G]:
    """The asynchronous entry point of `resume`. Same arguments as `resume`."""
    driver = _build(
        strategy=strategy, evaluator=evaluator, space=space, objectives=objectives, constraints=constraints, descriptors=descriptors,
        budget=budget, seed=seed, backend=backend, batch_size=batch_size, concurrency=concurrency, executor=executor,
        delivery=delivery, deterministic=deterministic, failure_policy=failure_policy, timeout=timeout,
        initial_failure_guard=initial_failure_guard, checkpoint_every=checkpoint_every,
        checkpoint_every_evaluations=checkpoint_every_evaluations, keep_checkpoints=keep_checkpoints,
        genome_store_threshold=genome_store_threshold, run_dir=run_dir, name=name,
        clock=clock, resume=True,
    )  # fmt: skip
    result = await driver.execute()
    if driver.already_complete:
        warnings.warn(_COMPLETE, ResumeWarning, stacklevel=2)
    return result
