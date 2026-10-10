"""Non-dominated sorting and crowding distance against brute-force references, on random small cases."""

import math

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.strategies.nsga2 import crowded_order, crowding_distance, dominance_matrix, nondominated_ranks
from tests.support.fixtures import assert_on_backend

INF = math.inf


# --- brute-force references, written for clarity and nothing else ---


def dominates(a: np.ndarray, va: float, b: np.ndarray, vb: float) -> bool:
    """Deb's constrained domination, for two members (failed members have NaN objectives and infinite violation)."""
    feasible_a, feasible_b = va == 0, vb == 0
    if feasible_a and feasible_b:
        return bool((a <= b).all() and (a < b).any())
    if feasible_a:
        return True
    if feasible_b:
        return False
    return va < vb


def reference_ranks(values: np.ndarray, violation: np.ndarray) -> list[int]:
    n = len(values)
    ranks = [-1] * n
    remaining = set(range(n))
    front = 0
    while remaining:
        current = [
            i for i in remaining if not any(dominates(values[j], violation[j], values[i], violation[i]) for j in remaining if j != i)
        ]
        for i in current:
            ranks[i] = front
        remaining -= set(current)
        front += 1
    return ranks


def reference_crowding(values: np.ndarray, violation: np.ndarray, ranks: list[int]) -> list[float]:
    n, k = values.shape
    distance = [0.0] * n
    for front in set(ranks):
        members = [i for i in range(n) if ranks[i] == front]
        if all(math.isinf(violation[i]) for i in members):
            continue  # failed members have distance 0
        for m in range(k):
            ordered = sorted(members, key=lambda i: (values[i, m], i))  # ties by position, as a stable sort does
            low, high = values[ordered[0], m], values[ordered[-1], m]
            distance[ordered[0]] = distance[ordered[-1]] = INF
            if high > low:
                for p in range(1, len(ordered) - 1):
                    if not math.isinf(distance[ordered[p]]):
                        distance[ordered[p]] += (values[ordered[p + 1], m] - values[ordered[p - 1], m]) / (high - low)
    return distance


def make_case(rng: np.random.Generator, n: int, k: int, *, ties: bool, failures: bool, constraints: bool, degenerate: bool):
    values = rng.integers(0, 4, (n, k)).astype(np.float64) if ties else rng.normal(size=(n, k))
    if degenerate:
        values[:, 0] = 1.0  # an objective with no range at all
    violation = np.zeros(n)
    if constraints:
        violation = np.where(rng.random(n) < 0.4, rng.integers(1, 4, n).astype(np.float64) if ties else rng.random(n) + 0.1, 0.0)
    if failures:
        failed = rng.random(n) < 0.2
        values[failed] = np.nan
        violation[failed] = INF
    return values, violation


CASES = [
    {"ties": False, "failures": False, "constraints": False, "degenerate": False},
    {"ties": True, "failures": False, "constraints": False, "degenerate": False},  # ties and duplicate points
    {"ties": False, "failures": False, "constraints": True, "degenerate": False},
    {"ties": False, "failures": True, "constraints": True, "degenerate": False},
    {"ties": True, "failures": True, "constraints": True, "degenerate": False},
    {"ties": False, "failures": False, "constraints": False, "degenerate": True},
]
IDS = ["plain", "ties", "constraints", "failures", "all", "degenerate"]


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_ranks_and_crowding_match_the_brute_force_references(backend: Backend, case: dict[str, bool], k: int):
    rng = np.random.default_rng(100 + k)
    for n in (1, 2, 3, 7, 16, 30):
        values, violation = make_case(rng, n, k, **case)
        v, c = backend.asarray(values), backend.asarray(violation)
        ranks = nondominated_ranks(v, c, backend)
        assert_on_backend(ranks, backend, backend.int_dtype)
        expected = reference_ranks(values, violation)
        assert backend.to_numpy(ranks).tolist() == expected
        got = backend.to_numpy(crowding_distance(v, c, ranks, backend)).astype(np.float64)
        want = np.array(reference_crowding(values, violation, expected))
        assert not np.isnan(got).any()
        assert (np.isinf(got) == np.isinf(want)).all()
        finite = ~np.isinf(want)
        np.testing.assert_allclose(got[finite], want[finite], rtol=1e-4, atol=1e-5)


def test_the_dominance_matrix_follows_the_constrained_domination_rule(backend: Backend):
    values = np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [np.nan, np.nan], [np.nan, np.nan], [1.0, 3.0]])
    violation = np.array([0.0, 0.0, 2.0, INF, INF, 0.0])
    matrix = backend.to_numpy(dominance_matrix(backend.asarray(values), backend.asarray(violation), backend))
    assert matrix[0, 1] and not matrix[1, 0]  # ordinary domination
    assert matrix[0, 2] and matrix[1, 2] and not matrix[2, 0]  # a feasible member beats an infeasible one, even with worse objectives
    assert matrix[2, 3] and matrix[0, 3] and not matrix[3, 2]  # and anything beats a failed member
    assert not matrix[3, 4] and not matrix[4, 3] and not matrix[3, 3]  # failed members do not dominate each other
    assert matrix[0, 5] and not matrix[5, 0]
    assert not matrix.diagonal().any()


def test_among_infeasible_members_a_lower_violation_dominates(backend: Backend):
    values = np.array([[5.0, 5.0], [0.0, 0.0], [1.0, 1.0]])
    violation = np.array([1.0, 3.0, 3.0])
    ranks = backend.to_numpy(nondominated_ranks(backend.asarray(values), backend.asarray(violation), backend)).tolist()
    assert ranks == [0, 1, 1]  # equal violations do not dominate each other, whatever the objectives


def test_failed_members_are_in_the_last_front_and_never_nan(backend: Backend):
    values = np.array([[1.0, 2.0], [np.nan, np.nan], [2.0, 1.0], [np.nan, np.nan]])
    violation = np.array([0.0, INF, 0.0, INF])
    v, c = backend.asarray(values), backend.asarray(violation)
    ids = backend.asarray([0, 1, 2, 3], dtype=backend.int_dtype)
    order, ranks, crowding = crowded_order(v, c, ids, backend)
    assert backend.to_numpy(ranks).tolist() == [0, 1, 0, 1]
    assert backend.to_numpy(order).tolist() == [0, 2, 1, 3]  # the failed ones last, by id
    assert not np.isnan(backend.to_numpy(crowding)).any() and backend.to_numpy(crowding)[[1, 3]].tolist() == [0.0, 0.0]


def test_duplicates_share_a_front_and_a_degenerate_front_has_no_nan(backend: Backend):
    values = np.array([[1.0, 1.0]] * 5)
    zero = np.zeros(5)
    ranks = nondominated_ranks(backend.asarray(values), backend.asarray(zero), backend)
    assert backend.to_numpy(ranks).tolist() == [0] * 5
    crowding = backend.to_numpy(crowding_distance(backend.asarray(values), backend.asarray(zero), ranks, backend)).astype(np.float64)
    assert not np.isnan(crowding).any() and np.isinf(crowding).sum() == 2 and (crowding[~np.isinf(crowding)] == 0).all()


def test_the_crowded_order_is_a_total_order_by_front_then_crowding_then_id(backend: Backend):
    rng = np.random.default_rng(7)
    values = rng.integers(0, 3, (24, 2)).astype(np.float64)
    violation = np.zeros(24)
    ids = rng.permutation(24)
    order, ranks, crowding = crowded_order(
        backend.asarray(values), backend.asarray(violation), backend.asarray(ids.tolist(), dtype=backend.int_dtype), backend
    )
    o, r, d = backend.to_numpy(order), backend.to_numpy(ranks), backend.to_numpy(crowding).astype(np.float64)
    assert sorted(o.tolist()) == list(range(24))
    keys = [(r[i], -d[i], ids[i]) for i in o]
    assert keys == sorted(keys)


def test_the_extremes_of_each_front_are_kept_first(backend: Backend):
    """On a trade-off curve the ends have an infinite distance, so they come first and then the least crowded."""
    t = np.linspace(0, 1, 9)
    values = np.stack([t, 1 - t], axis=1)
    values[4] = [0.5, 0.52]  # a point crowded between its neighbours
    order, _, _ = crowded_order(
        backend.asarray(values), backend.asarray(np.zeros(9)), backend.asarray(list(range(9)), dtype=backend.int_dtype), backend
    )
    first = backend.to_numpy(order).tolist()
    assert set(first[:2]) == {0, 8}


def test_one_objective_ranks_by_value_with_equal_values_sharing_a_front(backend: Backend):
    values = np.array([[3.0], [1.0], [2.0], [1.0]])
    ranks = backend.to_numpy(nondominated_ranks(backend.asarray(values), backend.asarray(np.zeros(4)), backend)).tolist()
    assert ranks == [2, 0, 1, 0]


def test_an_empty_population_is_fine(backend: Backend):
    v = backend.asarray(np.zeros((0, 2)))
    ranks = nondominated_ranks(v, backend.asarray(np.zeros(0)), backend)
    assert ranks.shape == (0,) and crowding_distance(v, backend.asarray(np.zeros(0)), ranks, backend).shape == (0,)
