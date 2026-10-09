"""Variation for structured genomes: the operator protocols, and how external operators are called (design doc §3.3, §3.5).

Structured operators work on one genome (or a pair) at a time, with Python values, because the genomes are not arrays. They
draw their randomness only from the stream they are given.
"""

import hashlib
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol

import numpy as np

from auxein.backend import Backend
from auxein.core import OperatorLog, OperatorRecord, Proposal, ProposalOperator
from auxein.core._typing import G
from auxein.random import RandomStream
from auxein.random.seed import stable_name_id
from auxein.spaces import GenomeCodec, canonical_json
from auxein.spaces.space import Space

_HOST = Backend()
_DRAW_LIMIT = 2**31 - 1


class OperatorError(RuntimeError):
    """An external proposal operator failed, or proposed something unusable. The run stops: failure policies for variation
    are future work. The message names the operator; the original exception is chained."""


class OperatorNotRecordedWarning(UserWarning):
    """A strategy uses external proposal operators in a run that is not recorded: every call is live, nothing is kept, and the
    run cannot be reproduced or resumed. Pass `run_dir` to record the calls."""


@dataclass(frozen=True)
class VariationContext(Generic[G]):
    """What an operator may need besides its parents and its stream."""

    space: Space[G]
    codec: GenomeCodec[G] | None
    operators: OperatorLog
    child_id: int
    """The id of the child being made, which lets the log tell a divergent replay from a legitimate live call."""


class StructuredMutation(Protocol[G]):
    """Changes one genome."""

    @property
    def name(self) -> str: ...

    def mutate(self, genome: G, rng: RandomStream, ctx: VariationContext[G]) -> G: ...


class StructuredRecombination(Protocol[G]):
    """Makes one child from two parents."""

    @property
    def name(self) -> str: ...

    def recombine(self, first: G, second: G, rng: RandomStream, ctx: VariationContext[G]) -> G: ...


def operator_key(name: str, inputs: Sequence[object], draw: int) -> str:
    """The key of an external call: the SHA-256 of the operator's name, the canonical encodings of its inputs and a value drawn
    from the strategy's stream at that point. Equal inputs asked for again give another draw, so another key."""
    return hashlib.sha256(canonical_json([name, list(inputs), draw])).hexdigest()


def call_operator(operator: ProposalOperator[Any], parents: Sequence[Any], rng: RandomStream, ctx: VariationContext[Any]) -> Any:
    """Call an external operator through the run's log: a recorded call is replayed, a new one is made live and recorded.

    The strategy's stream advances by exactly one draw whether the call is replayed or live, so a resumed run stays in step.
    """
    codec = ctx.codec
    if codec is None:
        raise OperatorError(f"operator {operator.name!r} needs a search space with a codec, to record what it proposes")
    draw = int(_HOST.to_numpy(rng.integers(0, _DRAW_LIMIT, (1,)))[0])
    inputs = [codec.encode(parent) for parent in parents]
    key = operator_key(operator.name, inputs, draw)
    record = ctx.operators.lookup(key)
    if record is not None:
        return codec.decode(record.output)
    if ctx.operators.must_be_recorded(ctx.child_id):
        raise ReplayDivergence(
            f"replaying the recording asked operator {operator.name!r} for candidate {ctx.child_id}, which is recorded, but the "
            "call (the operator's name, its inputs or the draw from the strategy's stream) is not in the recording: the run "
            "diverged from the recorded one, so the strategy's configuration or code changed. The operator was not called"
        )
    own = RandomStream(np.random.SeedSequence(entropy=draw, spawn_key=(stable_name_id("operator"),)))
    started = time.perf_counter()
    try:
        proposal = operator.propose(list(parents), own)
    except Exception as error:
        raise OperatorError(f"operator {operator.name!r} failed: {type(error).__name__}: {error}") from error
    wall = time.perf_counter() - started
    if not isinstance(proposal, Proposal):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise OperatorError(f"operator {operator.name!r} must return a Proposal, got {type(proposal).__name__}")
    genome = proposal.genome
    if not ctx.space.contains(genome):
        raise OperatorError(f"operator {operator.name!r} proposed a genome that is not in the search space: {genome!r}")
    try:
        output = codec.encode(genome)
        canonical_json(output)
    except TypeError as error:
        raise OperatorError(f"operator {operator.name!r} proposed a genome that cannot be encoded: {error}") from error
    ctx.operators.store(OperatorRecord(key, operator.name, output, dict(proposal.cost), wall, dict(proposal.metadata)))
    return genome


class ReplayDivergence(RuntimeError):
    """A replayed run asked an external operator for a call that the recording does not hold."""


class ExternalMutation(Generic[G]):
    """Uses an external proposal operator as a mutation: the child is what the operator proposes for its one parent."""

    def __init__(self, operator: ProposalOperator[G]) -> None:
        self.operator = operator

    @property
    def name(self) -> str:
        return f"external:{self.operator.name}"

    external = True

    def __repr__(self) -> str:
        return f"ExternalMutation({self.operator.name!r})"

    def mutate(self, genome: G, rng: RandomStream, ctx: VariationContext[G]) -> G:
        return call_operator(self.operator, [genome], rng, ctx)  # type: ignore[no-any-return]


class ExternalRecombination(Generic[G]):
    """Uses an external proposal operator as a recombination: the child is what it proposes for the two parents."""

    def __init__(self, operator: ProposalOperator[G]) -> None:
        self.operator = operator

    @property
    def name(self) -> str:
        return f"external:{self.operator.name}"

    external = True

    def __repr__(self) -> str:
        return f"ExternalRecombination({self.operator.name!r})"

    def recombine(self, first: G, second: G, rng: RandomStream, ctx: VariationContext[G]) -> G:
        return call_operator(self.operator, [first, second], rng, ctx)  # type: ignore[no-any-return]


def warn_if_unrecorded(log: OperatorLog, operators: Sequence[object], stacklevel: int = 3) -> None:
    if not log.recording and any(getattr(o, "external", False) for o in operators):
        warnings.warn(
            "this strategy calls external proposal operators in a run that is not recorded: every call is live and nothing is kept, "
            "so the run cannot be reproduced or resumed. Pass run_dir to record the calls",
            OperatorNotRecordedWarning,
            stacklevel=stacklevel,
        )
