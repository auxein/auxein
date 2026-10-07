"""The search space protocol (design doc §4.5)."""

from collections.abc import Sequence
from typing import Protocol, TypeVar

from auxein.backend import Array, Backend
from auxein.random import RandomStream

# G is invariant: it appears in a sampled sequence (covariant) and in `contains` (contravariant). Pyright only sees the
# second use, because the array alternative of the return type is `Any`.
G = TypeVar("G")


class Space(Protocol[G]):  # pyright: ignore[reportInvalidTypeVarUse]
    """Describes the genomes a problem admits, and samples them.

    Spaces return genomes, not batches: building a batch needs candidate ids, which are issued through the strategy
    context, so strategies wrap the sampled genomes into batches themselves. An array space such as `Box` returns an
    `(n, d)` array on the backend (the fast path); other spaces return a sequence of genomes.
    """

    def sample_genomes(self, n: int, rng: RandomStream, backend: Backend) -> Sequence[G] | Array:
        """`n` genomes, drawn from `rng`."""
        ...

    def contains(self, genome: G) -> bool:
        """Whether a genome is a member of the space."""
        ...
