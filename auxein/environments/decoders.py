"""Decoders (design doc §6.1): genome to agent."""

from typing import Generic

from auxein.backend import Array
from auxein.core._typing import G
from auxein.environments.protocols import Agent, AgentBatch


class IdentityDecoder(Generic[G]):
    """For environments that take the genome directly: the agent is the genome, and a batch of agents is the genome array."""

    def decode(self, genome: G) -> Agent:
        return genome

    def decode_batch(self, genomes: Array) -> AgentBatch:
        return genomes

    def __repr__(self) -> str:
        return "IdentityDecoder()"
