"""`FunctionEvaluator`: an ordinary function of one genome as an evaluator."""

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Generic, Literal, cast, overload

from auxein.core import Batch, Candidate, EvalContext, Evaluation, EvaluationBatch, evaluation_from_return
from auxein.core._typing import G
from auxein.evaluators.errors import EvaluationError
from auxein.random import RandomStream


class FunctionEvaluator(Generic[G]):
    """Evaluates candidates one at a time with `fn(genome)`, or `fn(genome, rng)` when `uses_rng=True`.

    `fn` may be a plain function or an `async def`. It returns a bare number (the single objective of a problem with
    exactly one objective) or a `Result`; plain dicts are rejected (see `auxein.core.Result`).

    With `uses_rng=True` the function receives `ctx.rng_for(candidate.id)`: the evaluation stream of that candidate,
    which depends only on the run seed and the candidate's id, not on the order or the worker (design doc §8). The stream
    is always numpy-backed, whatever the run's backend, and picklable, so it reaches a worker process intact.

    **Concurrency.** With the defaults (`concurrency=1`, inline executor) candidates are evaluated sequentially, in batch
    order, and a synchronous function is called directly with no hand-off, which keeps the overhead per evaluation low.
    Otherwise up to `ctx.concurrency` candidates of a batch are evaluated at once: a synchronous function runs where
    `executor=` says (`ctx.call`), and an `async def` function always runs natively on the driver's event loop, whatever
    the executor (it is an error to ask for `executor="process"` with one). Results are returned in ask order whatever
    order the evaluations finish in. Each candidate's wall time is recorded in its `Cost`, next to any cost units of its
    `Result`; with a pool it includes the hand-off to the worker, not the wait for a free one.

    What the function returns is turned into an `Evaluation` here, in the driver's process and in ask order, so that
    validation errors look the same with every executor and always name the first bad candidate. An exception in `fn` is
    raised as an `EvaluationError` naming the candidate; the other evaluations of the batch are cancelled first (a function
    already running in a thread or process finishes first, as Python cannot interrupt it).
    """

    @overload
    def __init__(self, fn: Callable[[G], object], *, uses_rng: Literal[False] = False) -> None: ...
    @overload
    def __init__(self, fn: Callable[[G, RandomStream], object], *, uses_rng: Literal[True]) -> None: ...
    def __init__(self, fn: Callable[..., object], *, uses_rng: bool = False) -> None:
        self._fn = fn
        self._uses_rng = uses_rng
        self._is_async = inspect.iscoroutinefunction(fn)

    def __repr__(self) -> str:
        name = getattr(self._fn, "__qualname__", type(self._fn).__name__)
        return f"FunctionEvaluator(fn={name}, uses_rng={self._uses_rng})"

    async def evaluate(self, batch: Batch[G], ctx: EvalContext[G]) -> EvaluationBatch[G]:
        if self._is_async and ctx.executor.kind == "process":
            raise ValueError(
                f"{self!r} is an async function, which always runs on the driver's event loop and cannot be sent to worker "
                "processes: use executor='thread' (or 'auto'), or make the function synchronous"
            )
        if ctx.concurrency == 1 and ctx.executor.kind == "inline":
            return await self._sequential(batch, ctx)
        return await self._concurrent(batch, ctx)

    async def _sequential(self, batch: Batch[G], ctx: EvalContext[G]) -> EvaluationBatch[G]:
        evaluations: list[Evaluation[G]] = []
        for candidate in batch.candidates:
            start = time.perf_counter()
            try:
                value = self._fn(candidate.genome, ctx.rng_for(candidate.id)) if self._uses_rng else self._fn(candidate.genome)
                if inspect.isawaitable(value):
                    value = await value
            except Exception as error:
                raise EvaluationError([candidate.id], error) from error
            wall_time = time.perf_counter() - start
            evaluations.append(evaluation_from_return(value, candidate, ctx.problem, wall_time))
        return EvaluationBatch(evaluations)

    async def _concurrent(self, batch: Batch[G], ctx: EvalContext[G]) -> EvaluationBatch[G]:
        slots = asyncio.Semaphore(ctx.concurrency)

        async def evaluate_one(candidate: Candidate[G]) -> tuple[object, float]:
            async with slots:
                start = time.perf_counter()
                try:
                    args = (candidate.genome, ctx.rng_for(candidate.id)) if self._uses_rng else (candidate.genome,)
                    value: object
                    if self._is_async:
                        value = await cast("Awaitable[object]", self._fn(*args))
                    else:
                        value = await ctx.call(self._fn, *args)
                        if inspect.isawaitable(value):  # a synchronous callable that hands back an awaitable
                            value = await value
                except Exception as error:
                    raise EvaluationError([candidate.id], error) from error
                return value, time.perf_counter() - start

        candidates = batch.candidates
        tasks = [asyncio.ensure_future(evaluate_one(candidate)) for candidate in candidates]
        try:
            returned = await asyncio.gather(*tasks)  # in the order of the arguments, not of completion
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)  # leave nothing running behind us
            raise
        # Turned into evaluations here, in ask order and in the driver's process, so that a malformed return value is reported
        # for the first bad candidate whatever the executor, concurrency or timing.
        return EvaluationBatch(
            [
                evaluation_from_return(value, candidate, ctx.problem, wall_time)
                for candidate, (value, wall_time) in zip(candidates, returned, strict=True)
            ]
        )
