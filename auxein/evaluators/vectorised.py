"""`VectorisedEvaluator`: one function call for a whole batch of array genomes."""

import inspect
import time
from collections.abc import Callable
from typing import Literal, overload

from auxein.backend import Array
from auxein.core import Batch, EvalContext, EvaluationBatch, evaluations_from_batch_return
from auxein.evaluators.failures import failures_of
from auxein.random import RandomStream


class VectorisedEvaluator:
    """Evaluates a whole batch with one call, `fn(X)` or `fn(X, rng)`, where `X = batch.as_array()` is the `(n, d)` array.

    This is the fast path for numeric problems: the call can run on a GPU when `X` is on one. It needs an array-backed
    batch and fails clearly with any other. `fn` returns, on any supported backend:

    - an array of shape `(n,)`: the single objective of a problem with exactly one objective;
    - an array of shape `(n, k)`, with `k` the number of declared objectives, columns in declared order;
    - a `BatchResult`, for constraints, descriptors or cost units.

    Shapes and finiteness are validated. The batch's wall time is measured once and **split equally** across its
    candidates, since a vectorised call has no per-candidate time.

    **Failures.** An exception in `fn` fails **every candidate of the batch**, each with the same error, under the default
    `infeasible` policy (a non-finite objective fails only its own row); under `fail_fast` it raises an `EvaluationError`
    naming the batch. Timeouts are not supported: setting one is an error (see below).

    **Concurrency does not apply.** There is exactly one call per batch, made on the driver's thread (or awaited there if
    `fn` is an `async def`), so `concurrency` and `executor` have no effect on it: the parallelism is inside the array
    operation. Under steady-state delivery the driver hands over one candidate at a time, which works but defeats the
    purpose of vectorising; the driver warns once per run when that happens. Prefer generation delivery here.

    With `uses_rng=True`, `fn` receives **one stream per batch**, derived from the id of the batch's first candidate
    (`ctx.batch_rng_for`). That is deterministic because the composition of a batch is, and it is the documented
    exception to the per-candidate evaluation streams of design doc §8: a vectorised function draws its randomness for
    the whole batch at once, so per-candidate streams would be unusable.
    """

    @overload
    def __init__(self, fn: Callable[[Array], object], *, uses_rng: Literal[False] = False) -> None: ...
    @overload
    def __init__(self, fn: Callable[[Array, RandomStream], object], *, uses_rng: Literal[True]) -> None: ...
    def __init__(self, fn: Callable[..., object], *, uses_rng: bool = False) -> None:
        self._fn = fn
        self._uses_rng = uses_rng

    def __repr__(self) -> str:
        name = getattr(self._fn, "__qualname__", type(self._fn).__name__)
        return f"VectorisedEvaluator(fn={name}, uses_rng={self._uses_rng})"

    async def evaluate(self, batch: Batch[Array], ctx: EvalContext[Array]) -> EvaluationBatch[Array]:
        genomes = batch.as_array()
        if genomes is None:
            raise TypeError(
                "VectorisedEvaluator needs an array-backed batch (an ArrayBatch, from a strategy on an array space such as Box), "
                "but this batch has no array view: use FunctionEvaluator for structured genomes"
            )
        candidates = batch.candidates
        if len(candidates) == 0:
            return EvaluationBatch([])
        if ctx.timeout is not None:
            raise ValueError(
                "VectorisedEvaluator does not support a timeout: it makes one call per batch on the driver's thread, which "
                "nothing can interrupt, and a vectorised call is normally cheap. Use FunctionEvaluator to time evaluations out"
            )

        start = time.perf_counter()
        try:
            value = self._fn(genomes, ctx.batch_rng_for(candidates[0].id)) if self._uses_rng else self._fn(genomes)
            if inspect.isawaitable(value):
                value = await value
        except Exception as error:
            return EvaluationBatch(failures_of(list(candidates), error, time.perf_counter() - start, ctx))
        wall_time = time.perf_counter() - start
        return EvaluationBatch(evaluations_from_batch_return(value, candidates, ctx.problem, wall_time))
