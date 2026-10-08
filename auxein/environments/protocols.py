"""The interfaces of the agent layer (design doc §6.2): environments, decoders and agents."""

from collections.abc import Awaitable, Mapping, Sequence
from typing import Protocol, TypeAlias, TypeVar

from auxein.backend import Array
from auxein.environments.episode import EpisodeBatchResult, EpisodeResult
from auxein.environments.scenario import Scenario
from auxein.random import RandomStream

G_contra = TypeVar("G_contra", contravariant=True)

Agent: TypeAlias = object
"""What a decoder makes from a genome and an environment runs. Auxein never looks inside."""

AgentBatch: TypeAlias = object
"""What a batched decoder makes from a whole `(n, d)` array of genomes. Opaque to Auxein: it is passed to `run_batch`."""


class Environment(Protocol):
    """Runs one episode: agents (by role) in a scenario, returning raw measurements.

    `rng` is the **agent's** stream for this candidate and scenario, for a stochastic policy. The **world's** randomness
    (disturbances, noise) must come from `scenario.rng()`, which depends on the scenario alone, so that every candidate faces
    the same realisation (common random numbers, design doc §6.4). `run_episode` may be an `async def` for environments that
    do I/O. This version supports one evolved role per episode; other participants belong to the scenario. The agents are
    passed as a mapping by role so that several evolved roles need no change of interface.
    """

    roles: tuple[str, ...]

    def run_episode(
        self, agents: Mapping[str, Agent], scenario: Scenario, rng: RandomStream
    ) -> EpisodeResult | Awaitable[EpisodeResult]: ...


class BatchedEnvironment(Protocol):
    """Optional capability: runs many agents on many scenarios in one call, for simulators that batch on a GPU.

    `agents` is what the decoder's `decode_batch` made of the whole batch of genomes; `rng` is the one stream of the batch,
    derived from the id of its first candidate. It returns measurements as arrays of shape `(n_candidates, n_scenarios)`.
    """

    def run_batch(self, agents: AgentBatch, scenarios: Sequence[Scenario], rng: RandomStream) -> EpisodeBatchResult: ...


class Decoder(Protocol[G_contra]):
    """Turns a genome into the agent an environment runs. It lives on the evaluation side: the strategy never sees agents.

    A decoder may also offer `decode_batch(genomes: Array) -> AgentBatch` to decode a whole `(n, d)` array at once for a
    `BatchedEnvironment`.
    """

    def decode(self, genome: G_contra) -> Agent: ...


class BatchDecoder(Protocol):
    def decode_batch(self, genomes: Array) -> AgentBatch: ...
