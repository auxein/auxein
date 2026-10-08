"""`RandomSearch`: uniform sampling from the search space, the floor that any real strategy must beat."""

from typing import Generic, cast

from auxein.backend import is_array
from auxein.core import (
    ArrayBatch,
    Batch,
    Candidate,
    EvaluationBatch,
    ListBatch,
    ProblemSpec,
    StateDict,
    StrategyCapabilities,
    StrategyContext,
)
from auxein.core._typing import G


class RandomSearch(Generic[G]):
    """Proposes independent random candidates, sampled from the problem's space, and learns nothing from the results.

    It works with any number of objectives and with constraints (it ignores both), and accepts results in generations or
    one at a time. Tracking the best results is the driver's job, so `tell` only checks that the results are for
    candidates this strategy asked for. For array spaces such as `Box`, `ask` returns an array-backed batch; for other
    spaces, a general one.
    """

    capabilities = StrategyCapabilities(max_objectives=None, supports_constraints=True, tell_mode="both")

    def __init__(self) -> None:
        self._problem: ProblemSpec[G] | None = None
        self._ctx: StrategyContext | None = None
        self._step = 0
        self._max_id = -1

    def __repr__(self) -> str:
        return "RandomSearch()"

    def bind(self, problem: ProblemSpec[G], ctx: StrategyContext) -> None:
        self._problem, self._ctx = problem, ctx

    def _bound(self) -> tuple[ProblemSpec[G], StrategyContext]:
        if self._problem is None or self._ctx is None:
            raise RuntimeError("RandomSearch must be bound to a problem before use: the driver calls bind() first")
        return self._problem, self._ctx

    def ask(self, n: int) -> Batch[G]:
        problem, ctx = self._bound()
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        genomes = problem.space.sample_genomes(n, ctx.rng, ctx.backend)
        ids = [ctx.new_id() for _ in range(n)]
        step, self._step = self._step, self._step + 1
        self._max_id = max(self._max_id, ids[-1])
        if is_array(genomes):
            return cast("Batch[G]", ArrayBatch(genomes, ids, step, "random"))
        if len(genomes) != n:
            raise ValueError(f"the space returned {len(genomes)} genomes for a request of {n}")
        return ListBatch([Candidate(cid, genome, (), "random", step) for cid, genome in zip(ids, genomes, strict=True)])

    def tell(self, results: EvaluationBatch[G]) -> None:
        self._bound()
        ids = [e.candidate.id for e in results]
        if not ids:
            raise ValueError("tell() was called with no results")
        if len(set(ids)) != len(ids):
            raise ValueError("tell() was called with the same candidate more than once")
        unknown = [i for i in ids if not 0 <= i <= self._max_id]
        if unknown:
            raise ValueError(f"tell() was called with candidates this strategy never asked for: {unknown[:5]}")

    def should_stop(self) -> bool:
        return False

    def state_dict(self) -> StateDict:
        """The stream state and the counters: enough to continue the sequence of proposals exactly."""
        _, ctx = self._bound()
        return {"stream": ctx.rng.state_dict(), "step": self._step, "max_id": self._max_id}

    def load_state_dict(self, state: StateDict) -> None:
        _, ctx = self._bound()
        if set(state) != {"stream", "step", "max_id"}:
            raise ValueError(f"invalid RandomSearch state: expected keys 'stream', 'step' and 'max_id', got {sorted(state)}")
        ctx.rng.load_state_dict(cast("dict[str, object]", state["stream"]))
        self._step = cast("int", state["step"])
        self._max_id = cast("int", state["max_id"])


__all__ = ["RandomSearch"]
