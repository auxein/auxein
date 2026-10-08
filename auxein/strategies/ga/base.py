"""The operator protocols of the genetic algorithm, and the view of the population they work on.

Every operator works on whole arrays at once, through the array namespace of the backend, and draws its randomness only
from the stream it is given. There are no Python loops over individuals or genes.
"""

from dataclasses import dataclass
from typing import Protocol

from auxein.backend import Array, Backend
from auxein.random import RandomStream
from auxein.spaces import Box


@dataclass(frozen=True)
class PopulationView:
    """The current population, ranked: what a parent selection operator needs to know.

    The ranking puts feasible members before infeasible ones, then lower total violation, then lower objective in
    minimisation form, then lower candidate id (design doc §3.3). Failed members have NaN objectives and infinite
    violation, so they rank last, and parent selection never looks beyond `valid` of them.
    """

    values: Array
    """`(m,)` objective in minimisation form (NaN for failed members)."""
    violation: Array
    """`(m,)` total constraint violation (0 = feasible, infinite for failed members)."""
    order: Array
    """`(m,)` member indices, best first."""
    rank: Array
    """`(m,)` rank of each member (0 = best): the inverse permutation of `order`."""
    backend: Backend
    n_valid: int | None = None
    """How many members did not fail (their violation is finite), or None when the strategy knows none did. Failed members
    rank last, so the valid ones are exactly the first `valid` of the ranking."""

    @property
    def size(self) -> int:
        return int(self.order.shape[0])

    @property
    def valid(self) -> int:
        """The number of members that can be parents: those that did not fail. A failed member is never chosen while there
        is any alternative (design doc §6.6)."""
        return self.size if self.n_valid is None else self.n_valid


class ParentSelection(Protocol):
    """Chooses the parents of `count` children."""

    @property
    def name(self) -> str: ...

    def select(self, population: PopulationView, count: int, rng: RandomStream) -> tuple[Array, Array]:
        """Two int arrays of shape `(count,)` of member indices: the first and the second parent of each child.

        The two parents of a child are distinct members whenever the population has at least two.
        """
        ...


class Recombination(Protocol):
    """Decides how the genes of two parents are mixed."""

    @property
    def name(self) -> str: ...

    @property
    def sexual(self) -> bool:
        """False for asexual reproduction: a child then copies its first parent, and records one parent."""
        ...

    def weights(self, count: int, dim: int, rng: RandomStream, backend: Backend) -> Array:
        """The weight of the *first* parent in each child, of shape `(count, 1)` or `(count, dim)`.

        A child gene is `w * a + (1 - w) * b`, and the step sizes of a child are mixed with the same weights (see
        `mix_steps`), so that "the same recombination" applies to genes and to step sizes.
        """
        ...


class Mutation(Protocol):
    """Perturbs genomes, and owns the strategy parameters (step sizes) that go with it, if it adapts them."""

    @property
    def name(self) -> str: ...

    @property
    def adaptive(self) -> bool:
        """Whether the mutation keeps step sizes per individual, as strategy state (not in the genome)."""
        ...

    @property
    def min_step(self) -> float | None:
        """The lower bound of the step sizes, relative to the box width, or None if they are not adapted."""
        ...

    def initial_steps(self, count: int, dim: int, backend: Backend) -> Array | None:
        """The step sizes of `count` new random individuals, or None if the mutation has none."""
        ...

    def mutate(self, genomes: Array, steps: Array | None, width: Array, rng: RandomStream, backend: Backend) -> tuple[Array, Array | None]:
        """Mutated genomes and their new step sizes. `width` is the box width per dimension, `(dim,)`."""
        ...


class BoundsRepair(Protocol):
    """Brings out-of-bounds genomes back into the box."""

    @property
    def name(self) -> str: ...

    def repair(self, genomes: Array, box: Box) -> Array: ...
