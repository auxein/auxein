"""The contracts between strategies, evaluators and the driver (design doc §3.2 and §5.3).

These are typing contracts: the driver, the strategies and the evaluators that implement them live in `auxein.driver`,
`auxein.strategies` and `auxein.evaluators`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, Protocol

from auxein.backend import Backend
from auxein.core._typing import G
from auxein.core.batch import Batch
from auxein.core.evaluation_batch import EvaluationBatch
from auxein.core.ids import CandidateId
from auxein.core.problem import ProblemSpec
from auxein.core.state import StateDict
from auxein.random import RandomStream

TellMode = Literal["generation", "steady_state", "both"]


@dataclass(frozen=True)
class StrategyCapabilities:
    """What a strategy supports, so that the driver can deliver results the way it needs and reject bad problems."""

    max_objectives: int | None
    """1 for a single-objective strategy, None for any number."""
    supports_constraints: bool
    tell_mode: TellMode
    """`generation`: all results of a batch together. `steady_state`: one at a time, as they arrive. `both`: either."""

    def __post_init__(self) -> None:
        if self.max_objectives is not None and self.max_objectives < 1:
            raise ValueError(f"max_objectives must be at least 1 or None, got {self.max_objectives}")
        if self.tell_mode not in ("generation", "steady_state", "both"):
            raise ValueError(f"unknown tell_mode {self.tell_mode!r}")


@dataclass(frozen=True)
class StrategyContext:
    """What the driver gives a strategy when it binds it to a problem."""

    rng: RandomStream
    """The strategy's own random stream (§8)."""
    backend: Backend
    """Array namespace, device and precision (§7)."""
    new_id: Callable[[], CandidateId]
    """The deterministic id issuer (§4.2)."""


@dataclass(frozen=True)
class EvalContext(Generic[G]):
    """What the driver gives an evaluator along with a batch."""

    problem: ProblemSpec[G]
    """What is being optimised: evaluators turn what user code returns into evaluations of this problem."""
    backend: Backend
    rng_for: Callable[[CandidateId], RandomStream]
    """Derives the evaluation stream of a candidate from its id, so that randomness follows the candidate, not the
    worker or the time of evaluation (§8). It is a factory rather than a list of streams, so that a batch of
    thousands of candidates does not create thousands of generators up front. **The stream is always numpy-backed,
    whatever the run's backend**: evaluating one candidate is host-side Python, per-candidate streams are the only
    kind a run creates by the tens of thousands (a torch CPU generator has only 32 bits of seed, which would make
    collisions likely), and a numpy stream can be pickled to a worker process."""
    batch_rng_for: Callable[[CandidateId], RandomStream]
    """Derives the one stream a vectorised evaluator receives per batch, from the id of the batch's first candidate
    (§8). It is deterministic because the composition of a batch is."""
    timeout: float | None = None
    """Seconds an evaluation may take, or None for no limit."""
    deadline: float | None = None
    """An absolute `time.monotonic()` deadline for the whole batch, or None."""


class Strategy(Protocol[G]):
    """An algorithm: it proposes candidates and learns from their evaluations, and never evaluates anything itself.

    Strategies are plain synchronous code: no I/O and no `async`. They own their internal state, which they save and
    restore through `state_dict` and `load_state_dict`.
    """

    capabilities: StrategyCapabilities

    def bind(self, problem: ProblemSpec[G], ctx: StrategyContext) -> None:
        """Called once by the driver before the first `ask`. Validates the problem, e.g. a single-objective strategy
        rejects two objectives."""
        ...

    def ask(self, n: int) -> Batch[G]:
        """Propose candidates. `n` is the driver's suggestion; generation-based strategies may return their own size."""
        ...

    def tell(self, results: EvaluationBatch[G]) -> None:
        """Receive the evaluations of previously asked candidates."""
        ...

    def should_stop(self) -> bool:
        """Strategy-specific termination, e.g. converged."""
        ...

    def state_dict(self) -> StateDict: ...

    def load_state_dict(self, state: StateDict) -> None: ...


class Evaluator(Protocol[G]):
    """Turns a batch of candidates into evaluations.

    The interface is asynchronous; users rarely implement it directly, because the built-in evaluators (a later step)
    wrap ordinary functions.
    """

    async def evaluate(self, batch: Batch[G], ctx: EvalContext[G]) -> EvaluationBatch[G]: ...
