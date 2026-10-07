import math

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.random import RunSeed
from auxein.spaces import Box
from auxein.strategies.ga import (
    ClipRepair,
    GaussianMutation,
    IntermediateRecombination,
    NoRecombination,
    ReflectRepair,
    SelfAdaptiveMutation,
    SigmaScalingSUS,
    TournamentSelection,
    UniformRecombination,
    mix_genes,
    mix_steps,
    rank_order,
    view_of,
)
from tests.support.fixtures import assert_on_backend


def stream(backend: Backend, name: str = "ga-test", seed: int = 1):
    return RunSeed(seed).stream(name, backend=backend)


def view(backend: Backend, values, violation=None, ids=None):
    n = len(values)
    violation = [0.0] * n if violation is None else violation
    ids = list(range(n)) if ids is None else ids
    return view_of(backend.asarray(values), backend.asarray(violation), backend.asarray(ids, dtype=backend.int_dtype), backend)


def host(backend: Backend, x):
    return backend.to_numpy(x)


# --- ranking ---


def test_ranking_puts_feasible_first_then_lower_violation_then_lower_value_then_lower_id(backend: Backend):
    values = [5.0, 1.0, 3.0, 3.0, 0.0, math.nan]
    violation = [0.0, 2.0, 0.0, 0.0, 0.5, math.inf]
    ids = [10, 11, 12, 13, 14, 15]
    order = host(
        backend, rank_order(backend.asarray(values), backend.asarray(violation), backend.asarray(ids, dtype=backend.int_dtype), backend)
    )
    # feasible by value (3.0 with id 12 before id 13, then 5.0), then violation 0.5, then 2.0, then the failed one
    assert [ids[i] for i in order] == [12, 13, 10, 14, 11, 15]


def test_rank_is_the_inverse_of_order(backend: Backend):
    v = view(backend, [3.0, 1.0, 2.0, 0.5, 4.0])
    order, rank = host(backend, v.order), host(backend, v.rank)
    assert order.tolist() == [3, 1, 2, 0, 4] and rank.tolist() == [3, 1, 2, 0, 4]
    assert rank[order].tolist() == list(range(5))


def test_ranking_matches_a_brute_force_sort_on_random_populations(backend: Backend):
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(2, 40))
        values = rng.integers(0, 6, n).astype(float)  # many ties
        violation = np.where(rng.random(n) < 0.4, rng.integers(0, 4, n).astype(float), 0.0)
        failed = rng.random(n) < 0.1
        values[failed], violation[failed] = math.nan, math.inf
        ids = rng.permutation(n * 3)[:n]
        expected = sorted(range(n), key=lambda i, v=values, w=violation, k=ids: (w[i], math.inf if math.isnan(v[i]) else v[i], k[i]))
        actual = host(
            backend, rank_order(backend.asarray(values), backend.asarray(violation), backend.asarray(ids, dtype=backend.int_dtype), backend)
        )
        assert actual.tolist() == expected


# --- tournament ---


def test_a_large_tournament_always_picks_the_best_member_as_first_parent(backend: Backend):
    v = view(backend, [4.0, 2.0, 0.5, 3.0, 1.0])
    first, second = TournamentSelection(size=200).select(v, 500, stream(backend))
    assert set(host(backend, first).tolist()) == {2}  # the member with the lowest value
    assert 2 not in host(backend, second).tolist()  # the second parent is another member


def test_tournaments_rank_infeasible_and_failed_members_last(backend: Backend):
    # member 0 has the best value but violates a constraint, member 1 failed, members 2-3 are feasible
    v = view(backend, [-100.0, math.nan, 5.0, 3.0], [1.0, math.inf, 0.0, 0.0])
    first, _ = TournamentSelection(size=200).select(v, 200, stream(backend))
    assert set(host(backend, first).tolist()) == {3}


def test_the_parents_of_a_child_are_always_distinct(backend: Backend):
    for m in (2, 3, 10):
        v = view(backend, list(range(m)))
        for selection in (TournamentSelection(size=2), TournamentSelection(size=50), SigmaScalingSUS()):
            first, second = selection.select(v, 3000, stream(backend))
            a, b = host(backend, first), host(backend, second)
            assert (a != b).all() and a.min() >= 0 and a.max() < m and b.min() >= 0 and b.max() < m


def test_the_selection_returns_arrays_on_the_backend_with_the_right_shape(backend: Backend):
    v = view(backend, [3.0, 1.0, 2.0, 4.0])
    for selection in (TournamentSelection(), SigmaScalingSUS()):
        first, second = selection.select(v, 7, stream(backend))
        assert_on_backend(first, backend, backend.int_dtype)
        assert_on_backend(second, backend, backend.int_dtype)
        assert tuple(first.shape) == (7,) and tuple(second.shape) == (7,)


def test_tournament_selection_pressure_matches_the_theory(backend: Backend):
    m, n = 10, 20000
    v = view(backend, list(range(m)))
    first, _ = TournamentSelection(size=2).select(v, n, stream(backend))
    counts = np.bincount(host(backend, first), minlength=m) / n
    # P(member of rank r wins a size-2 tournament) = ((m - r)^2 - (m - r - 1)^2) / m^2
    expected = np.array([((m - r) ** 2 - (m - r - 1) ** 2) / m**2 for r in range(m)])
    np.testing.assert_allclose(counts, expected, atol=0.015)


def test_tournament_size_one_is_uniform_and_validated(backend: Backend):
    first, _ = TournamentSelection(size=1).select(view(backend, list(range(5))), 20000, stream(backend))
    np.testing.assert_allclose(np.bincount(host(backend, first), minlength=5) / 20000, 0.2, atol=0.02)
    with pytest.raises(ValueError, match="at least 1"):
        TournamentSelection(size=0)
    assert "size=3" in repr(TournamentSelection(size=3))


# --- stochastic universal sampling with sigma scaling ---


def reference_weights(values, c=2.0):
    g = -np.asarray(values, dtype=float)
    return np.maximum(g - (g.mean() - c * g.std()), 0.0)


def test_sigma_scaled_weights_follow_the_formula(backend: Backend):
    values = [0.0, 1.0, 2.0, 3.0, 10.0, 4.0, 5.0, 2.5]
    weights = host(backend, SigmaScalingSUS().weights(view(backend, values)))
    expected = reference_weights(values)
    # the weights are only defined up to a positive factor (they are normalised internally): compare the shares
    np.testing.assert_allclose(weights / weights.sum(), expected / expected.sum(), rtol=1e-4, atol=1e-6)
    assert weights[0] == weights.max() and weights[4] == 0.0  # the best has the most, the outlier has none


def test_sus_selection_counts_are_proportional_to_the_weights(backend: Backend):
    values = [0.0, 1.0, 2.0, 3.0, 10.0, 4.0, 5.0, 2.5]
    n = 20000
    first, _ = SigmaScalingSUS().select(view(backend, values), n, stream(backend))
    counts = np.bincount(host(backend, first), minlength=len(values)) / n
    expected = reference_weights(values) / reference_weights(values).sum()
    np.testing.assert_allclose(counts, expected, atol=0.012)
    assert counts[4] == 0.0  # a member with zero weight is never selected


def test_sus_with_the_scaling_constant_of_a_larger_c_keeps_more_members_alive(backend: Backend):
    v = view(backend, [0.0, 1.0, 2.0, 3.0, 10.0])
    assert host(backend, SigmaScalingSUS(scaling=0.0).weights(v)).tolist().count(0.0) > host(
        backend, SigmaScalingSUS(scaling=3.0).weights(v)
    ).tolist().count(0.0)
    with pytest.raises(ValueError, match="must not be negative"):
        SigmaScalingSUS(scaling=-1.0)


@pytest.mark.parametrize(
    "values",
    [[1.0, 1.0, 1.0, 1.0], [math.nan] * 4, [0.0, 0.0, 0.0, 0.0], [1e30, -1e30, 1e30, 0.0]],
    ids=["all-equal", "all-nan", "all-zero", "huge"],
)
def test_sus_falls_back_to_uniform_and_never_hangs_when_the_weights_are_unusable(backend: Backend, values: list[float]):
    violation = [math.inf] * 4 if values[0] != values[0] else None
    v = view(backend, values, violation)
    weights = host(backend, SigmaScalingSUS().weights(v))
    assert np.isfinite(weights).all() and weights.sum() > 0
    if len(set(values)) == 1 or values[0] != values[0]:
        np.testing.assert_array_equal(weights, np.ones(4))  # all equal: no information, so uniform
    first, second = SigmaScalingSUS().select(v, 500, stream(backend))
    assert tuple(host(backend, first).shape) == (500,) and (host(backend, first) != host(backend, second)).all()


def test_sus_never_selects_infeasible_members_while_a_feasible_one_has_weight(backend: Backend):
    # members 0-2 are infeasible but have the best values; 3-6 are feasible
    values = [-50.0, -40.0, -30.0, 1.0, 2.0, 3.0, 0.0]
    violation = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    v = view(backend, values, violation)
    weights = host(backend, SigmaScalingSUS().weights(v))
    assert weights[:3].tolist() == [0.0, 0.0, 0.0] and weights[3:].sum() > 0
    first, _ = SigmaScalingSUS().select(v, 3000, stream(backend))
    assert set(host(backend, first).tolist()) <= {3, 4, 5, 6}


def test_sus_with_no_feasible_member_weights_by_lower_violation(backend: Backend):
    v = view(backend, [0.0, 0.0, 0.0, math.nan], [1.0, 4.0, 2.5, math.inf])
    weights = host(backend, SigmaScalingSUS().weights(v))
    assert weights[0] == weights.max() > 0 and weights[3] == 0.0  # the failed member gets nothing
    assert weights[0] > weights[2] >= weights[1]


def test_sus_weights_are_always_finite_with_a_positive_sum(backend: Backend):
    rng = np.random.default_rng(1)
    for _ in range(40):
        m = int(rng.integers(2, 30))
        values = rng.normal(size=m) * rng.choice([1e-9, 1.0, 1e9])
        violation = np.where(rng.random(m) < 0.5, 0.0, rng.random(m))
        failed = rng.random(m) < 0.2
        values[failed], violation[failed] = math.nan, math.inf
        weights = host(backend, SigmaScalingSUS().weights(view(backend, values, violation)))
        assert np.isfinite(weights).all() and (weights >= 0).all() and weights.sum() > 0


# --- recombination ---


def parents(backend: Backend, n: int = 400, d: int = 5):
    rng = np.random.default_rng(3)
    return backend.asarray(rng.uniform(-5, 5, (n, d))), backend.asarray(rng.uniform(-5, 5, (n, d)))


def test_intermediate_children_lie_between_their_parents(backend: Backend):
    a, b = parents(backend)
    for per_gene in (False, True):
        w = IntermediateRecombination(per_gene=per_gene).weights(400, 5, stream(backend), backend)
        assert tuple(w.shape) == ((400, 5) if per_gene else (400, 1))
        child, ha, hb = host(backend, mix_genes(w, a, b)), host(backend, a), host(backend, b)
        eps = 1e-5 if backend.precision == "float32" else 1e-12
        assert (child >= np.minimum(ha, hb) - eps).all() and (child <= np.maximum(ha, hb) + eps).all()
        wh = host(backend, w)
        assert (wh >= 0).all() and (wh < 1).all()
        if per_gene:
            assert not np.allclose(wh[:, [0]], wh)  # a different weight per gene
        else:
            assert np.allclose(host(backend, mix_genes(w, a, b)), wh * ha + (1 - wh) * hb, atol=1e-5)


def test_uniform_recombination_takes_each_gene_from_one_parent(backend: Backend):
    a, b = parents(backend, 600, 6)
    w = UniformRecombination().weights(600, 6, stream(backend), backend)
    child, ha, hb = host(backend, mix_genes(w, a, b)), host(backend, a), host(backend, b)
    from_a, from_b = child == ha, child == hb
    assert (from_a | from_b).all()  # every gene is exactly one parent's
    np.testing.assert_allclose(from_a.mean(), 0.5, atol=0.04)
    assert set(host(backend, w).ravel().tolist()) <= {0.0, 1.0}


def test_no_recombination_copies_the_first_parent_exactly(backend: Backend):
    a, b = parents(backend)
    w = NoRecombination().weights(400, 5, stream(backend), backend)
    np.testing.assert_array_equal(host(backend, mix_genes(w, a, b)), host(backend, a))


def test_step_sizes_are_mixed_by_a_weighted_geometric_mean_with_the_gene_weights(backend: Backend):
    a, b = backend.asarray([2.0, 8.0, 1.0]), backend.asarray([8.0, 2.0, 100.0])
    half = backend.asarray([[0.5], [0.5], [0.5]])
    np.testing.assert_allclose(host(backend, mix_steps(half, a, b, backend)), [4.0, 4.0, 10.0], rtol=1e-5)  # geometric means
    np.testing.assert_allclose(host(backend, mix_steps(backend.asarray([[1.0], [1.0], [1.0]]), a, b, backend)), host(backend, a), rtol=0)
    np.testing.assert_allclose(host(backend, mix_steps(backend.asarray([[0.0], [0.0], [0.0]]), a, b, backend)), host(backend, b), rtol=0)
    quarter = host(backend, mix_steps(backend.asarray([[0.25], [0.25], [0.25]]), a, b, backend))
    np.testing.assert_allclose(quarter, host(backend, a) ** 0.25 * host(backend, b) ** 0.75, rtol=1e-5)


def test_per_gene_step_sizes_follow_the_gene_mask_exactly(backend: Backend):
    a, b = backend.asarray([[2.0, 3.0, 4.0]]), backend.asarray([[20.0, 30.0, 40.0]])
    mask = backend.asarray([[1.0, 0.0, 1.0]])
    assert host(backend, mix_steps(mask, a, b, backend)).tolist()[0] == pytest.approx([2.0, 30.0, 4.0], rel=0)


def test_a_single_step_with_uniform_weights_uses_the_share_of_the_first_parent(backend: Backend):
    a, b = backend.asarray([4.0]), backend.asarray([16.0])
    mask = backend.asarray([[1.0, 1.0, 0.0, 0.0]])  # half of the genes come from the first parent: the geometric mean
    assert float(host(backend, mix_steps(mask, a, b, backend))[0]) == pytest.approx(8.0, rel=1e-5)


# --- mutation ---


def test_gaussian_mutation_is_relative_to_the_box_width(backend: Backend):
    width = backend.asarray([1.0, 10.0, 100.0])
    zeros = backend.asarray(np.zeros((40000, 3)))
    mutated, steps = GaussianMutation(step=0.1).mutate(zeros, None, width, stream(backend), backend)
    assert (
        steps is None
        and not GaussianMutation().adaptive
        and GaussianMutation().min_step is None
        and GaussianMutation().initial_steps(3, 3, backend) is None
    )
    assert (np.abs(host(backend, mutated).mean(axis=0)) < np.array([0.1, 1.0, 10.0]) * 0.03).all()
    np.testing.assert_allclose(host(backend, mutated).std(axis=0), [0.1, 1.0, 10.0], rtol=0.03)
    assert_on_backend(mutated, backend)
    with pytest.raises(ValueError, match="positive"):
        GaussianMutation(step=0.0)


def test_self_adaptive_default_learning_rates():
    d = 16
    single = SelfAdaptiveMutation()
    assert single.tau(d) == pytest.approx(1 / math.sqrt(d)) and single.name == "self_adaptive"
    per_gene = SelfAdaptiveMutation(per_gene=True)
    assert per_gene.tau_prime(d) == pytest.approx(1 / math.sqrt(2 * d))
    assert per_gene.tau(d) == pytest.approx(1 / math.sqrt(2 * math.sqrt(d))) and per_gene.name == "self_adaptive_per_gene"
    custom = SelfAdaptiveMutation(per_gene=True, tau=0.3, tau_prime=0.2)
    assert custom.tau(d) == 0.3 and custom.tau_prime(d) == 0.2
    assert SelfAdaptiveMutation(tau=0.7).tau(d) == 0.7


def test_self_adaptive_initial_steps(backend: Backend):
    single = SelfAdaptiveMutation(initial_step=0.2).initial_steps(7, 4, backend)
    per_gene = SelfAdaptiveMutation(per_gene=True, initial_step=0.2).initial_steps(7, 4, backend)
    assert tuple(single.shape) == (7,) and tuple(per_gene.shape) == (7, 4)
    assert_on_backend(single, backend) and assert_on_backend(per_gene, backend)
    np.testing.assert_allclose(host(backend, per_gene), 0.2, rtol=1e-6)


def test_single_step_mutation_updates_the_step_log_normally_then_moves_the_genes(backend: Backend):
    mutation, n, d = SelfAdaptiveMutation(initial_step=0.1), 5000, 4
    steps = mutation.initial_steps(n, d, backend)
    width = backend.asarray([2.0] * d)
    genomes = backend.asarray(np.zeros((n, d)))
    mutated, new_steps = mutation.mutate(genomes, steps, width, stream(backend, "m"), backend)
    expected_noise = host(backend, stream(backend, "m").normal((n,)))  # the first draw of the same stream
    np.testing.assert_allclose(host(backend, new_steps), 0.1 * np.exp(mutation.tau(d) * expected_noise), rtol=2e-5)
    assert tuple(new_steps.shape) == (n,)
    moves = host(backend, mutated) / (host(backend, new_steps)[:, None] * 2.0)  # standard normal moves, scaled by step x width
    np.testing.assert_allclose(moves.mean(), 0.0, atol=0.03)
    np.testing.assert_allclose(moves.std(), 1.0, atol=0.03)


def test_per_gene_mutation_gives_every_gene_its_own_step(backend: Backend):
    mutation, n, d = SelfAdaptiveMutation(per_gene=True), 3000, 6
    steps = mutation.initial_steps(n, d, backend)
    mutated, new_steps = mutation.mutate(backend.asarray(np.zeros((n, d))), steps, backend.asarray([1.0] * d), stream(backend), backend)
    assert tuple(new_steps.shape) == (n, d)
    log_ratio = np.log(host(backend, new_steps) / 0.1)
    expected_std = math.sqrt(mutation.tau_prime(d) ** 2 + mutation.tau(d) ** 2)
    np.testing.assert_allclose(log_ratio.std(), expected_std, rtol=0.05)
    assert not np.allclose(log_ratio[:, [0]], log_ratio)  # the genes of an individual get different steps...
    shared = log_ratio - log_ratio.mean(axis=1, keepdims=True)
    assert abs(np.corrcoef(log_ratio[:, 0], log_ratio[:, 1])[0, 1]) > 0.05  # ...that share the global noise term
    assert shared.std() > 0


def test_step_sizes_never_fall_below_the_floor(backend: Backend):
    mutation = SelfAdaptiveMutation(initial_step=1e-3, min_step=1e-3, tau=2.0)  # a big learning rate, steps start at the floor
    steps = mutation.initial_steps(2000, 3, backend)
    for _ in range(5):
        _, steps = mutation.mutate(backend.asarray(np.zeros((2000, 3))), steps, backend.asarray([1.0] * 3), stream(backend), backend)
    assert float(host(backend, steps).min()) >= 1e-3 * (1 - 1e-6)
    per_gene = SelfAdaptiveMutation(per_gene=True, initial_step=1e-3, min_step=1e-3, tau=2.0, tau_prime=2.0)
    steps = per_gene.initial_steps(2000, 3, backend)
    for _ in range(5):
        _, steps = per_gene.mutate(backend.asarray(np.zeros((2000, 3))), steps, backend.asarray([1.0] * 3), stream(backend), backend)
    assert float(host(backend, steps).min()) >= 1e-3 * (1 - 1e-6)


def test_self_adaptive_validation():
    with pytest.raises(ValueError, match="initial step must be positive"):
        SelfAdaptiveMutation(initial_step=0.0)
    with pytest.raises(ValueError, match="min_step must be positive and at most the initial step"):
        SelfAdaptiveMutation(initial_step=0.1, min_step=0.5)
    with pytest.raises(ValueError, match="min_step"):
        SelfAdaptiveMutation(min_step=0.0)
    with pytest.raises(AssertionError, match="needs the step sizes"):
        SelfAdaptiveMutation().mutate(
            Backend().asarray(np.zeros((2, 2))), None, Backend().asarray([1.0, 1.0]), stream(Backend()), Backend()
        )


# --- bounds repair ---

BOX = Box([-1.0, 0.0, 10.0], [1.0, 4.0, 20.0])


def test_clip_brings_everything_inside_the_box(backend: Backend):
    wild = backend.asarray(np.random.default_rng(2).uniform(-100, 100, (500, 3)))
    clipped = ClipRepair().repair(wild, BOX)
    assert all(BOX.contains(clipped[i]) for i in range(500))
    inside = backend.asarray([[0.5, 2.0, 15.0]])
    np.testing.assert_array_equal(host(backend, ClipRepair().repair(inside, BOX)), host(backend, inside))
    np.testing.assert_allclose(host(backend, ClipRepair().repair(backend.asarray([[5.0, -3.0, 99.0]]), BOX)), [[1.0, 0.0, 20.0]])


def test_reflect_is_a_true_reflection(backend: Backend):
    x = backend.asarray([[1.25, 4.5, 21.0], [-1.5, -0.25, 9.0]])
    y = host(backend, ReflectRepair().repair(x, BOX))
    np.testing.assert_allclose(y, [[0.75, 3.5, 19.0], [-0.5, 0.25, 11.0]], rtol=1e-5, atol=1e-5)  # a mirror image across the nearest wall


def test_reflect_folds_far_away_values_back_and_keeps_inside_values(backend: Backend):
    far = backend.asarray([[1.0 + 2 * 2 + 0.25, 4.0 + 2 * 4 + 0.5, 20.0 + 10.0 * 3 + 1.0]])  # several widths beyond the upper bounds
    y = host(backend, ReflectRepair().repair(far, BOX))
    assert BOX.contains(y[0]) and y[0, 0] == pytest.approx(0.75, abs=1e-5)
    wild = backend.asarray(np.random.default_rng(5).uniform(-1e3, 1e3, (500, 3)))
    repaired = ReflectRepair().repair(wild, BOX)
    assert all(BOX.contains(repaired[i]) for i in range(500))
    inside = backend.asarray(np.random.default_rng(6).uniform([-1, 0, 10], [1, 4, 20], (50, 3)))
    np.testing.assert_allclose(host(backend, ReflectRepair().repair(inside, BOX)), host(backend, inside), rtol=1e-6, atol=1e-6)


def test_reflect_and_clip_keep_the_dtype_and_device(backend: Backend):
    x = backend.asarray([[2.0, 5.0, 25.0]])
    for repair in (ClipRepair(), ReflectRepair()):
        assert_on_backend(repair.repair(x, BOX), backend)
    assert ClipRepair().name == "clip" and ReflectRepair().name == "reflect"
