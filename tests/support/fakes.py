"""Small fake strategies and evaluators for testing the driver."""

from collections.abc import Callable, Sequence

from auxein.core import (
    Batch,
    Candidate,
    CandidateId,
    EvaluationBatch,
    ListBatch,
    ProblemSpec,
    StateDict,
    StrategyCapabilities,
    StrategyContext,
)


class ScriptedStrategy:
    """Asks for scripted genomes: `genomes[i]` is the genome of the i-th candidate it ever proposes.

    `sizes[k]` is the size of its k-th batch (default: whatever the driver asks for), which lets a test ask for more than
    the driver wants. `reject` makes `bind` fail, `stop_after` makes `should_stop` true after that many asks.
    """

    def __init__(
        self,
        genomes: Sequence[object] | Callable[[int], object] = lambda i: float(i),
        sizes: Sequence[int] = (),
        *,
        stop_after: int | None = None,
        reject: str | None = None,
        tell_mode: str = "generation",
    ) -> None:
        self.capabilities = StrategyCapabilities(None, True, tell_mode)  # type: ignore[arg-type]
        self._genome = genomes if callable(genomes) else genomes.__getitem__
        self._sizes = list(sizes)
        self._stop_after = stop_after
        self._reject = reject
        self.asks = 0
        self.proposed = 0
        self.asked_n: list[int] = []
        self.told: list[EvaluationBatch] = []
        self.bound: tuple[ProblemSpec, StrategyContext] | None = None

    def bind(self, problem: ProblemSpec, ctx: StrategyContext) -> None:
        if self._reject:
            raise ValueError(self._reject)
        self.bound = (problem, ctx)

    def ask(self, n: int) -> Batch:
        assert self.bound is not None
        _, ctx = self.bound
        self.asked_n.append(n)
        size = self._sizes[self.asks] if self.asks < len(self._sizes) else n
        candidates = []
        for _ in range(size):
            candidates.append(Candidate(ctx.new_id(), self._genome(self.proposed), (), "scripted", self.asks))
            self.proposed += 1
        self.asks += 1
        return ListBatch(candidates)

    def tell(self, results: EvaluationBatch) -> None:
        self.told.append(results)

    def should_stop(self) -> bool:
        return self._stop_after is not None and self.asks >= self._stop_after

    def state_dict(self) -> StateDict:
        return {"asks": self.asks}

    def load_state_dict(self, state: StateDict) -> None:
        pass


class ManualClock:
    """A clock that only moves when told to, so that wall-time budgets are tested without sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def ids(batch: Batch) -> list[CandidateId]:
    return [c.id for c in batch.candidates]
