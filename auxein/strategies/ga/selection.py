"""Parent selection: tournament, and stochastic universal sampling with sigma scaling."""

from auxein.backend import Array
from auxein.random import RandomStream
from auxein.strategies.ga.base import PopulationView


def distinct_partners(first: Array, second: Array, population_size: int, rng: RandomStream, view: PopulationView) -> Array:
    """The second parents, with any that equals its first parent replaced by a uniformly random *other* member.

    This is what keeps the parents of a child distinct (the 0.x engine allowed self-mating). It needs at least two members.
    """
    xp = view.backend.xp
    valid = view.valid
    if valid < population_size:
        # some members failed: the partner is another member that did not, found by moving along the ranking
        offset = rng.integers(1, valid, (int(first.shape[0]),))  # 1 .. valid-1
        moved = xp.take(view.order, xp.remainder(xp.take(view.rank, first, axis=0) + offset, valid), axis=0)
        return xp.where(first == second, moved, second)
    offset = rng.integers(1, population_size, (int(first.shape[0]),))  # 1 .. m-1: never the member itself
    return xp.where(first == second, xp.remainder(first + offset, population_size), second)


class TournamentSelection:
    """Each parent is the best of `size` members drawn uniformly at random (with replacement), by the ranking.

    Tournaments only look at ranks, so the selection pressure doesn't depend on the scale of the objective, and infeasible
    members lose to feasible ones without any special case. Members that failed are not drawn at all while any other
    member exists, so a tournament of failed members can never produce a parent.
    """

    name = "tournament"

    def __init__(self, size: int = 2) -> None:
        if size < 1:
            raise ValueError(f"the tournament size must be at least 1, got {size}")
        self.size = size

    def __repr__(self) -> str:
        return f"TournamentSelection(size={self.size})"

    def select(self, population: PopulationView, count: int, rng: RandomStream) -> tuple[Array, Array]:
        xp = population.backend.xp
        m = population.size
        if population.valid < m:
            # the contestants are drawn from the members that did not fail, which are the first `valid` of the ranking
            ranks = rng.integers(0, population.valid, (2 * count, self.size))
        else:
            draws = rng.integers(0, m, (2 * count, self.size))
            ranks = xp.reshape(xp.take(population.rank, xp.reshape(draws, (-1,)), axis=0), (2 * count, self.size))
        winners = xp.take(population.order, xp.min(ranks, axis=1), axis=0)
        first, second = winners[:count], winners[count:]
        return first, distinct_partners(first, second, m, rng, population)


class SigmaScalingSUS:
    """Stochastic universal sampling with sigma-scaled weights.

    The weight of a feasible member is `max(g - (mean(g) - c * std(g)), 0)` with goodness `g = -value` (the objective in
    minimisation form), over the feasible members, so a few standard deviations below the mean get no chance and the
    best get the most. Infeasible members have weight 0; when no member is feasible the weights come from a lower
    violation in the same way. If the weights are all zero or not finite, the selection falls back to uniform:
    it never divides by zero.

    All the pointers of one call are equally spaced from one random start, which keeps the sample close to the expected
    counts; the sample is then shuffled so that the first and second parents are paired at random.
    """

    name = "sus"

    def __init__(self, scaling: float = 2.0) -> None:
        if scaling < 0:
            raise ValueError(f"the sigma-scaling constant must not be negative, got {scaling}")
        self.scaling = scaling

    def __repr__(self) -> str:
        return f"SigmaScalingSUS(scaling={self.scaling})"

    def weights(self, population: PopulationView) -> Array:
        """The selection weights `(m,)`: sigma-scaled goodness of the feasible members, never NaN, never all zero."""
        xp = population.backend.xp
        violation = population.violation
        feasible = violation == 0
        finite = xp.isfinite(violation)

        def scaled(goodness: Array, mask: Array) -> Array:
            # an objective that is finite in float64 can be infinite in float32 (1e39): such a member has no usable goodness, so it
            # gets no weight, like an infeasible one, instead of turning the scale into infinity and every weight into NaN
            mask = mask & xp.isfinite(goodness)
            goodness = xp.where(mask, goodness, 0.0)
            members = xp.astype(mask, population.backend.dtype)
            n = xp.maximum(xp.sum(members), xp.asarray(1.0, dtype=population.backend.dtype, device=population.backend.device))
            # weights are invariant under a positive rescaling of goodness: normalise it so that squaring never overflows
            scale = xp.max(xp.abs(goodness))
            g = xp.where(mask, goodness / xp.where(scale > 0, scale, 1.0), 0.0)
            mean = xp.sum(g) / n
            std = xp.sqrt(xp.sum(xp.where(mask, (g - mean) ** 2, 0.0)) / n)
            return xp.where(mask, xp.clip(g - (mean - self.scaling * std), 0.0, None), 0.0)

        by_value = scaled(-population.values, feasible)
        by_violation = scaled(-violation, finite & ~feasible)
        weights = xp.where(xp.any(feasible), by_value, by_violation)
        total = xp.sum(weights)
        usable = xp.isfinite(total) & (total > 0)
        # the fallback is uniform over the members that did not fail (all of them, when none failed)
        uniform = xp.where(finite, xp.ones_like(weights), xp.zeros_like(weights))
        uniform = xp.where(xp.any(finite), uniform, xp.ones_like(weights))
        return xp.where(usable, weights, uniform)

    def select(self, population: PopulationView, count: int, rng: RandomStream) -> tuple[Array, Array]:
        backend = population.backend
        xp = backend.xp
        m = population.size
        pointers_count = 2 * count
        cumulative = xp.cumulative_sum(self.weights(population))
        cumulative = cumulative / cumulative[-1]
        start = rng.uniform((1,), 0.0, 1.0 / pointers_count)
        pointers = start + xp.arange(pointers_count, dtype=backend.dtype, device=backend.device) / pointers_count
        # the member of a pointer is the first whose cumulative weight exceeds it: count the members it has passed
        passed = xp.sum(xp.astype(pointers[:, None] >= cumulative[None, :], backend.int_dtype), axis=1)
        chosen = xp.take(xp.clip(passed, 0, m - 1), rng.permutation(pointers_count), axis=0)
        first, second = chosen[:count], chosen[count:]
        return first, distinct_partners(first, second, m, rng, population)
