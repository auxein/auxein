"""Recombination: intermediate, uniform, simulated binary crossover (SBX), and none (asexual)."""

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


class SimulatedBinaryCrossover:
    """Simulated binary crossover (SBX, Deb and Agrawal), the standard real-coded crossover of NSGA-II.

    SBX makes two children that sit symmetrically around their parents' midpoint, spread by a factor `beta` whose
    distribution has the shape of the spread of a single-point crossover on binary strings, controlled by the *distribution
    index* `eta`: the larger it is, the closer the children stay to their parents (default 15, the textbook value).
    Our strategies make one child per pair of parents, so one of the two is made: in each crossed gene a fair coin puts it on
    either side of the midpoint, and in a gene that is not crossed it keeps the gene of the first or of the second parent (one
    coin for the whole child).

    It fits the recombination protocol because a child gene is `w·a + (1 − w)·b`: the child gene `(1 ± beta)/2·a + (1 ∓ beta)/2·b` has
    `w = (1 ± beta)/2`. A weight **outside [0, 1]** is how SBX extrapolates beyond its parents, so the child can leave the box; the
    bounds repair of the strategy brings it back (this is the unbounded form of SBX; the bounded form of the reference code differs
    near the bounds). As in the usual implementation, each gene is crossed with probability `variable_probability` (default 0.5).
    """

    name = "sbx"
    sexual = True

    def __init__(self, eta: float = 15.0, variable_probability: float = 0.5) -> None:
        if not eta >= 0:
            raise ValueError(f"the distribution index eta must not be negative, got {eta}")
        if not 0.0 <= variable_probability <= 1.0:
            raise ValueError(f"variable_probability must be in [0, 1], got {variable_probability}")
        self.eta = eta
        self.variable_probability = variable_probability

    def __repr__(self) -> str:
        return f"SimulatedBinaryCrossover(eta={self.eta}, variable_probability={self.variable_probability})"

    def weights(self, count: int, dim: int, rng: RandomStream, backend: Backend) -> Array:
        xp = backend.xp
        u = rng.uniform((count, dim))
        power = 1.0 / (self.eta + 1.0)
        # the spread factor: (2u)^(1/(eta+1)) below the median, (1 / (2 (1 - u)))^(1/(eta+1)) above it
        below = xp.pow(2.0 * u, power)
        above = xp.pow(1.0 / (2.0 * (1.0 - u)), power)
        beta = xp.where(u <= 0.5, below, above)
        # a crossed gene lands on either side of the midpoint, with a fair coin per gene: w = (1 +/- beta) / 2. This is what makes
        # SBX contract a population (the children of a pair are centred on the parents' midpoint), unlike a child that is always
        # on its first parent's side
        side = xp.where(rng.uniform((count, dim)) < 0.5, 1.0, -1.0)
        crossed_weight = (1.0 + side * beta) / 2.0
        # a gene that is not crossed comes from one parent, the same for the whole child: the first or the second of the pair
        first_child = rng.uniform((count, 1)) < 0.5
        kept_weight = xp.astype(xp.broadcast_to(first_child, (count, dim)), backend.dtype)
        crossed = rng.uniform((count, dim)) < self.variable_probability
        return xp.where(crossed, crossed_weight, kept_weight)


class NoRecombination:
    """Asexual reproduction: a child is a copy of its first parent, to be mutated."""

    name = "none"
    sexual = False

    def __repr__(self) -> str:
        return "NoRecombination()"

    def weights(self, count: int, dim: int, rng: RandomStream, backend: Backend) -> Array:
        return backend.xp.ones((count, 1), dtype=backend.dtype, device=backend.device)
