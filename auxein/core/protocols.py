"""The contracts between strategies, evaluators and the driver (design doc §3.2 and §5.3).

These are typing contracts: the driver, the strategies and the evaluators that implement them live in `auxein.driver`,
`auxein.strategies` and `auxein.evaluators`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, Literal, Protocol, TypeVar

from auxein.backend import Backend
from auxein.core._typing import G
from auxein.core.batch import Batch
from auxein.core.evaluation_batch import EvaluationBatch
from auxein.core.ids import CandidateId
from auxein.core.operators import NoOperatorLog, OperatorLog
from auxein.core.problem import ProblemSpec
from auxein.core.state import StateDict
from auxein.execution import Executor, InlineExecutor
from auxein.random import RandomStream

T = TypeVar("T")

FailurePolicy = Literal["infeasible", "fail_fast"]

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
    operators: OperatorLog = field(default_factory=NoOperatorLog)
    """Where calls of external proposal operators are recorded and replayed (§3.5); a no-op log when the run is not recorded."""


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
    episode_rng_for: Callable[[CandidateId, int], RandomStream] | None = None
    """Derives the agent's stream for one episode, from the candidate's id and the scenario's index in its set (§8): the
    randomness that belongs to the agent (a stochastic policy), as opposed to the world's, which comes from the scenario's own
    seed so that every candidate faces the same realisation. Numpy-backed like `rng_for`, and picklable. The episode evaluator
    needs it; the driver always provides it."""
    episode_batch_rng_for: Callable[[CandidateId], RandomStream] | None = None
    """Derives the one stream a batched environment receives per batch, from the id of the batch's first candidate (§8)."""
    executor: Executor = field(default_factory=InlineExecutor)
    """Where synchronous user functions run (§5.3). Use `call` rather than this directly."""
    concurrency: int = 1
    """The most evaluations in progress at once in this run. An evaluator that evaluates candidates concurrently must
    not exceed it. `async def` user functions run natively on the driver's event loop, limited by this number."""
    timeout: float | None = None
    """Seconds one evaluation may take, or None for no limit. How hard it is depends on where the function runs (§5.3):
    an `async def` is cancelled, a worker process is killed, a thread is abandoned. `call` applies it."""
    failure_policy: FailurePolicy = "infeasible"
    """What an evaluator does with an exception in user code, a timeout or a worker crash: `infeasible` turns it into a
    `FAILED` or `TIMEOUT` evaluation, `fail_fast` raises an `EvaluationError` (§6.6). Misconfiguration is never a
    result: it raises whatever the policy."""

    async def call(self, fn: Callable[..., T], /, *args: object) -> T:
        """Run the synchronous `fn(*args)` where the run's executor says (inline, in a thread or in a process) and await
        its result, for at most `timeout` seconds if the run has one (then it raises `EvaluationTimeout`). User-written
        evaluators use this so that they obey `executor=` and `timeout=` like the built-in ones. With a process executor
        `fn` and `args` must be picklable, and a worker that dies raises `WorkerCrashed`. `async def` functions are
        awaited directly instead."""
        return await self.executor.call(fn, *args, timeout=self.timeout)


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
