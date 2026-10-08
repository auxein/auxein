"""`FunctionEvaluator`: an ordinary function of one genome as an evaluator."""

import inspect
import time
from collections.abc import Callable
from typing import Generic, Literal, overload

from auxein.core import Batch, EvalContext, Evaluation, EvaluationBatch, evaluation_from_return
from auxein.core._typing import G
from auxein.evaluators.errors import EvaluationError
from auxein.random import RandomStream


class FunctionEvaluator(Generic[G]):
    """Evaluates candidates one at a time with `fn(genome)`, or `fn(genome, rng)` when `uses_rng=True`.

    `fn` may be a plain function or an `async def`. It returns a bare number (the single objective of a problem with
    exactly one objective) or a `Result`; plain dicts are rejected (see `auxein.core.Result`).

    With `uses_rng=True` the function receives `ctx.rng_for(candidate.id)`: the evaluation stream of that candidate,
    which depends only on the run seed and the candidate's id, not on the order or the worker (design doc §8). The stream
    is always numpy-backed, whatever the run's backend.

    Candidates are evaluated sequentially, in batch order. A synchronous function is called directly, with no thread
    hand-off, to keep the overhead per evaluation low. Each candidate's wall time is recorded in its `Cost`, next to
    any cost units of its `Result`. An exception in `fn` is raised as an `EvaluationError` naming the candidate.
    """

    @overload
    def __init__(self, fn: Callable[[G], object], *, uses_rng: Literal[False] = False) -> None: ...
    @overload
    def __init__(self, fn: Callable[[G, RandomStream], object], *, uses_rng: Literal[True]) -> None: ...
    def __init__(self, fn: Callable[..., object], *, uses_rng: bool = False) -> None:
        self._fn = fn
        self._uses_rng = uses_rng

    def __repr__(self) -> str:
        name = getattr(self._fn, "__qualname__", type(self._fn).__name__)
        return f"FunctionEvaluator(fn={name}, uses_rng={self._uses_rng})"

    async def evaluate(self, batch: Batch[G], ctx: EvalContext[G]) -> EvaluationBatch[G]:
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
