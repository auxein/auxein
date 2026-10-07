"""Candidates: a genome plus identity and lineage (design doc §4.2)."""

from dataclasses import dataclass
from typing import Generic

from auxein.core._typing import G
from auxein.core.ids import CandidateId


@dataclass(frozen=True)
class Candidate(Generic[G]):
    """A genome with its identity and lineage.

    The genome is opaque to the core and must be treated as immutable: operators create new genomes, never modify
    one (§4.1). Strategy parameters such as step sizes are not part of the genome; they live in the strategy's state,
    keyed by candidate id.
    """

    id: CandidateId
    """Deterministic: issued by the run's id counter."""
    genome: G
    parents: tuple[CandidateId, ...]
    """Empty for candidates sampled from the space."""
    origin: str
    """The operator that produced it, e.g. "init", "mutation:self_adaptive", "crossover:arithmetic", "reevaluation"."""
    step: int
    """The ask round it was proposed in."""

    def __post_init__(self) -> None:
        if not self.origin:
            raise ValueError("a candidate needs a non-empty origin")
        if self.step < 0:
            raise ValueError(f"step must not be negative, got {self.step}")
