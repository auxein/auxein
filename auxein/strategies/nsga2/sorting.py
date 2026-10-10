"""Non-dominated sorting and crowding distance, vectorised on the backend (design doc §3.3).

Everything works on the objectives **in minimisation form** (a strategy never handles directions by hand: the
`EvaluationBatch` converts them) and on the total constraint violation, with Deb's constrained-domination rule:

- a feasible member dominates every infeasible one;
- among infeasible members, the one with the lower total violation dominates;
- among feasible members, ordinary Pareto domination.

A **failed** member has NaN objectives and infinite violation. It is dominated by every member that did not fail, and two
failed members do not dominate each other (their violations are equal), so all of them end in the last front, which is where
they must rank. Their NaNs are never compared: they are replaced before any comparison, so nothing warns and nothing is NaN.

The domination relation is a boolean `(n, n)` matrix, built with an `(n, n, k)` broadcast: memory grows with the square of
the pooled population (population plus offspring). That is fine to a few thousand members (4,000 members and 3 objectives
take about 50 MB for the broadcast) and is the documented limit of this implementation. The only Python loop is over the
*fronts*.
"""

from auxein.backend import Array, Backend


def dominance_matrix(values: Array, violation: Array, backend: Backend) -> Array:
    """`dominates[i, j]`: member `i` constrained-dominates member `j`. `values` is `(n, k)`, `violation` is `(n,)`."""
    xp = backend.xp
    clean = xp.where(xp.isnan(values), 0.0, values)  # a failed member's NaNs are never compared, see the module docstring
    feasible = violation == 0
    first, second = clean[:, None, :], clean[None, :, :]
    pareto = xp.all(first <= second, axis=2) & xp.any(first < second, axis=2)
    both_feasible = feasible[:, None] & feasible[None, :]
    feasible_wins = feasible[:, None] & ~feasible[None, :]
    both_infeasible = ~feasible[:, None] & ~feasible[None, :]
    less_violation = violation[:, None] < violation[None, :]
    return (both_feasible & pareto) | feasible_wins | (both_infeasible & less_violation)


def nondominated_ranks(values: Array, violation: Array, backend: Backend) -> Array:
    """The front of each member, `0` for the non-dominated ones, `1` for those dominated only by front 0 and so on: an int
    array `(n,)`. The loop runs once per front."""
    xp = backend.xp
    count = int(values.shape[0])
    if count == 0:
        return xp.zeros((0,), dtype=backend.int_dtype, device=backend.device)
    dominates = dominance_matrix(values, violation, backend)
    ranks = xp.zeros((count,), dtype=backend.int_dtype, device=backend.device)
    remaining = xp.ones((count,), dtype=backend.bool_dtype, device=backend.device)
    front = 0
    while True:
        # members of the remaining set that no other remaining member dominates form the next front
        dominated = xp.any(dominates & remaining[:, None], axis=0)
        current = remaining & ~dominated
        ranks = xp.where(current, front, ranks)
        remaining = remaining & ~current
        front += 1
        if not bool(xp.any(remaining)):
            return ranks


def crowding_distance(values: Array, violation: Array, ranks: Array, backend: Backend) -> Array:
    """The crowding distance of each member within its front: for every objective, the gap between its neighbours in the
    front, normalised by the front's range of that objective, summed over the objectives. The two extreme members of a front
    have an infinite distance. A front whose range is zero in an objective contributes nothing for it (no NaN), and a failed
    member has distance 0 (its front is the last one and its objectives are not numbers)."""
    xp = backend.xp
    count, k = int(values.shape[0]), int(values.shape[1])
    if count == 0:
        return xp.zeros((0,), dtype=backend.dtype, device=backend.device)
    clean = xp.where(xp.isnan(values), 0.0, values)
    fronts = int(xp.max(ranks)) + 1
    front_ids = xp.arange(fronts, dtype=backend.int_dtype, device=backend.device)
    member_of = ranks[:, None] == front_ids[None, :]  # (n, fronts)
    total = xp.zeros((count,), dtype=backend.dtype, device=backend.device)
    infinity = xp.asarray(float("inf"), dtype=backend.dtype, device=backend.device)
    for m in range(k):
        column = clean[:, m]
        # sort by front, then by the objective (two stable sorts: the least significant key first)
        order = xp.argsort(column, stable=True)
        order = xp.take(order, xp.argsort(xp.take(ranks, order, axis=0), stable=True), axis=0)
        sorted_value = xp.take(column, order, axis=0)
        sorted_front = xp.take(ranks, order, axis=0)
        has_previous = sorted_front[1:] == sorted_front[:-1]  # (n-1,): this member and the one before are in the same front
        previous = xp.concat([xp.asarray([False], device=backend.device), has_previous])
        following = xp.concat([has_previous, xp.asarray([False], device=backend.device)])
        before = xp.concat([sorted_value[:1], sorted_value[:-1]])
        after = xp.concat([sorted_value[1:], sorted_value[-1:]])
        # the range of the objective within each member's front
        low = xp.min(xp.where(member_of, column[:, None], infinity), axis=0)
        high = xp.max(xp.where(member_of, column[:, None], -infinity), axis=0)
        spread = xp.take(high - low, sorted_front, axis=0)
        interior = previous & following
        gap = xp.where(spread > 0, (after - before) / xp.where(spread > 0, spread, 1.0), 0.0)
        contribution = xp.where(interior, gap, infinity)
        # back to the original positions: `order[p]` is the member at sorted position p
        inverse = xp.argsort(order)
        total = total + xp.take(contribution, inverse, axis=0)
    failed = xp.isinf(violation)
    return xp.where(failed, 0.0, total)


def crowded_order(values: Array, violation: Array, ids: Array, backend: Backend) -> tuple[Array, Array, Array]:
    """Member indices, best first, by the crowded-comparison order: lower front, then larger crowding distance, then lower
    candidate id (so the order is total). Returns the order, the fronts and the crowding distances."""
    xp = backend.xp
    ranks = nondominated_ranks(values, violation, backend)
    crowding = crowding_distance(values, violation, ranks, backend)
    order = xp.argsort(ids, stable=True)
    order = xp.take(order, xp.argsort(-xp.take(crowding, order, axis=0), stable=True), axis=0)
    order = xp.take(order, xp.argsort(xp.take(ranks, order, axis=0), stable=True), axis=0)
    return order, ranks, crowding
