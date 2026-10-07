"""Recombination: intermediate, uniform, and none (asexual)."""

from auxein.backend import Array, Backend
from auxein.random import RandomStream


def mix_genes(weights: Array, first: Array, second: Array) -> Array:
    """Child genes `w * a + (1 - w) * b`. With weights of exactly 0 or 1 a gene is copied from one parent exactly."""
    return weights * first + (1.0 - weights) * second


def mix_steps(weights: Array, first: Array, second: Array, backend: Backend) -> Array:
    """Child step sizes: the weighted geometric mean `a ** w * b ** (1 - w)`, with the same weights as the genes.

    Step sizes are scale parameters, so they are averaged geometrically: with equal weights this is the geometric mean
    of the parents' steps. Weights of 0 or 1 give one parent's step exactly. A single step size per individual uses the
    mean of the gene weights (the share of the first parent in the child).
    """
    xp = backend.xp
    if first.ndim == 1:
        weights = xp.mean(weights, axis=1)
    return xp.pow(first, weights) * xp.pow(second, 1.0 - weights)


class IntermediateRecombination:
    """Child = `a * parent1 + (1 - a) * parent2`, with `a ~ U(0, 1)` drawn per child (or per gene with `per_gene=True`)."""

    name = "intermediate"
    sexual = True

    def __init__(self, per_gene: bool = False) -> None:
        self.per_gene = per_gene

    def __repr__(self) -> str:
        return f"IntermediateRecombination(per_gene={self.per_gene})"

    def weights(self, count: int, dim: int, rng: RandomStream, backend: Backend) -> Array:
        return rng.uniform((count, dim if self.per_gene else 1))


class UniformRecombination:
    """Each gene comes from either parent with probability 0.5."""

    name = "uniform"
    sexual = True

    def __repr__(self) -> str:
        return "UniformRecombination()"

    def weights(self, count: int, dim: int, rng: RandomStream, backend: Backend) -> Array:
        return backend.xp.astype(rng.uniform((count, dim)) < 0.5, backend.dtype)


class NoRecombination:
    """Asexual reproduction: a child is a copy of its first parent, to be mutated."""

    name = "none"
    sexual = False

    def __repr__(self) -> str:
        return "NoRecombination()"

    def weights(self, count: int, dim: int, rng: RandomStream, backend: Backend) -> Array:
        return backend.xp.ones((count, 1), dtype=backend.dtype, device=backend.device)
