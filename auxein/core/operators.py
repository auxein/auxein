"""External proposal operators: variation that is not a pure function of Auxein's random streams (design doc §3.5).

An LLM asked to rewrite a prompt is the typical case: its answer depends on a model, not on a seed. To keep runs that use such
operators reproducible and cheap to resume, every call is **recorded** (keyed by the operator, its inputs and a value drawn from
the strategy's stream) and **replayed**: a resumed or extended run gets the recorded answer without calling the operator again.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, Protocol

from auxein.core._typing import G
from auxein.random import RandomStream


@dataclass(frozen=True)
class Proposal(Generic[G]):
    """What an external operator returns: the new genome, and optionally what the call cost and anything worth keeping."""

    genome: G
    cost: Mapping[str, float] = field(default_factory=dict[str, float])
    """User-defined cost units of the call (tokens, money). Recorded, but not counted against the run's budget (yet)."""
    metadata: Mapping[str, object] = field(default_factory=dict[str, object])
    """JSON-serialisable notes about the call (the model used, a finish reason...), recorded with it."""


class ProposalOperator(Protocol[G]):
    """Proposes a new genome from its parents: one parent for a mutation, two for a recombination.

    `rng` is a stream derived from a value drawn from the strategy's stream, so an operator that uses it is deterministic;
    one that does not (a model behind an API) is made reproducible by recording. The call is **synchronous and blocks the
    driver** while it runs. An exception fails the run, naming the operator.
    """

    @property
    def name(self) -> str:
        """Names the operator in origins and in the record: changing it makes a replay diverge, on purpose."""
        ...

    def propose(self, parents: Sequence[G], rng: RandomStream) -> Proposal[G]: ...


@dataclass(frozen=True)
class OperatorRecord:
    """One recorded call of an external operator."""

    key: str
    operator: str
    output: object
    """The proposed genome, encoded by the space's codec."""
    cost: Mapping[str, float]
    wall_time: float
    metadata: Mapping[str, object]


class OperatorLog(Protocol):
    """Where a run keeps the calls of its external operators. The driver gives strategies one in `StrategyContext`."""

    recording: bool
    """Whether calls are recorded (and so replayable). Without a `run_dir` they are not, and a run using them cannot be reproduced."""

    def lookup(self, key: str) -> OperatorRecord | None: ...

    def store(self, record: OperatorRecord) -> None: ...

    def must_be_recorded(self, candidate_id: int) -> bool:
        """Whether the recording already holds this candidate, so that the call that makes it must be recorded too: a live
        call then means the run diverged from the recorded one."""
        ...


class NoOperatorLog:
    """The log of a run that is not recorded: every call is live, and nothing is kept."""

    recording = False

    def lookup(self, key: str) -> OperatorRecord | None:
        return None

    def store(self, record: OperatorRecord) -> None:
        pass

    def must_be_recorded(self, candidate_id: int) -> bool:
        return False
