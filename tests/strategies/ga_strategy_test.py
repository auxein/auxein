import json
import math

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import (
    Cost,
    Evaluation,
    EvaluationBatch,
    IdIssuer,
    Objective,
    ProblemSpec,
    Status,
    StrategyContext,
    validate_state_dict,
)
from auxein.random import RunSeed
from auxein.spaces import Box
from auxein.strategies import GeneticAlgorithm
from auxein.strategies.ga import (
    GaussianMutation,
    IntermediateRecombination,
    NoRecombination,
    SelfAdaptiveMutation,
    SigmaScalingSUS,
    TournamentSelection,
    UniformRecombination,
)

DIM = 3
VALUE = (Objective("value"),)


def bind(
    ga: GeneticAlgorithm,
    backend: Backend,
    seed: int = 1,
    dim: int = DIM,
    objectives=VALUE,
    space=None,
    issuer: IdIssuer | None = None,
):
    issuer = issuer or IdIssuer()
    ctx = StrategyContext(RunSeed(seed).stream("strategy", backend=backend), backend, issuer.next)
    ga.bind(ProblemSpec(space or Box(-5.0, 5.0, dim=dim), tuple(objectives)), ctx)
    return ga, issuer


def evaluate(batch, value_of, violation_of=None, status=Status.OK):
    """Evaluations for a batch: `value_of(candidate)` and `violation_of(candidate)` decide the numbers."""
    evaluations = []
    for c in batch.candidates:
        constraints = {} if violation_of is None else {"cpa": violation_of(c)}
        objectives = {"value": value_of(c)} if status is Status.OK else {"value": math.nan}
        evaluations.append(Evaluation(c, status, objectives, constraints, cost=Cost()))
    return evaluations


def sphere_value(backend: Backend):
    return lambda c: float((backend.to_numpy(c.genome).astype(np.float64) ** 2).sum())


def tell_all(ga, batch, value_of, violation_of=None):
    results = EvaluationBatch(evaluate(batch, value_of, violation_of))
    ga.tell(results)
    return results


def started(backend: Backend, **kwargs):
    """A strategy whose initial population has been asked for and told, with the given values."""
    ga, issuer = bind(GeneticAlgorithm(**kwargs), backend)
    batch = ga.ask(1)
    tell_all(ga, batch, sphere_value(backend))
    return ga, issuer


# --- construction and binding ---


def test_defaults_and_repr():
    ga = GeneticAlgorithm()
    assert ga.population_size == 50 and ga.offspring_size == 50
    assert isinstance(ga.selection, TournamentSelection) and ga.selection.size == 2
    assert isinstance(ga.recombination, IntermediateRecombination) and not ga.recombination.per_gene
    assert isinstance(ga.mutation, SelfAdaptiveMutation) and not ga.mutation.per_gene and ga.mutation.min_step == 1e-12
    assert ga.crossover_probability == 1.0 and ga.convergence_tolerance is None
    caps = GeneticAlgorithm.capabilities
    assert caps.max_objectives == 1 and caps.supports_constraints and caps.tell_mode == "both"
    assert "population_size=50" in repr(ga) and "TournamentSelection" in repr(ga)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"population_size": 1}, "at least 2"),
        ({"offspring_size": 0}, "offspring_size must be at least 1"),
        ({"crossover_probability": 1.5}, "crossover_probability"),
        ({"crossover_probability": -0.1}, "crossover_probability"),
    ],
)
def test_constructor_validation(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        GeneticAlgorithm(**kwargs)


def test_it_is_single_objective(backend: Backend):
    with pytest.raises(ValueError, match=r"single-objective.*2 objectives \('a', 'b'\)"):
        bind(GeneticAlgorithm(), backend, objectives=[Objective("a"), Objective("b")])


def test_it_needs_a_box_space(backend: Backend):
    class Words:
        def sample_genomes(self, n, rng, backend):
            return ["w"] * n

        def contains(self, genome):
            return True

    with pytest.raises(TypeError, match="needs a Box search space"):
        bind(GeneticAlgorithm(), backend, space=Words())


def test_it_must_be_bound_before_use():
    ga = GeneticAlgorithm()
    for call in (lambda: ga.ask(1), lambda: ga.state_dict(), lambda: ga.tell(EvaluationBatch([])), lambda: ga.load_state_dict({})):
        with pytest.raises(RuntimeError, match="must be bound"):
            call()
    assert ga.ranked_ids() == [] and ga.size == 0 and not ga.should_stop()


def test_n_must_be_positive(backend: Backend):
    ga, _ = bind(GeneticAlgorithm(), backend)
    with pytest.raises(ValueError, match="n must be at least 1"):
        ga.ask(0)


# --- the initial population and the children ---


def test_the_first_ask_returns_the_initial_population(backend: Backend):
    ga, issuer = bind(GeneticAlgorithm(population_size=12, offspring_size=5), backend)
    batch = ga.ask(1)  # whatever n the driver suggests, the initial batch is the whole population
    assert len(batch.candidates) == 12 and batch.step == 0 and issuer.issued == 12
    assert all(c.origin == "init" and c.parents == () for c in batch.candidates)
    x = backend.to_numpy(batch.as_array())
    assert x.shape == (12, DIM) and (x >= -5).all() and (x <= 5).all()
    assert backend.matches(batch.as_array())


def test_then_exactly_lambda_children_per_ask(backend: Backend):
    ga, _ = started(backend, population_size=12, offspring_size=5)
    for expected_step in (1, 2, 3):
        batch = ga.ask(100)  # n is ignored when offspring_size is given
        assert len(batch.candidates) == 5 and batch.step == expected_step
        tell_all(ga, batch, sphere_value(backend))
    assert ga.size == 12


def test_with_no_offspring_size_the_children_are_exactly_n(backend: Backend):
    ga, _ = started(backend, population_size=12, offspring_size=None)
    for n in (1, 7, 40):
        batch = ga.ask(n)
        assert len(batch.candidates) == n
        tell_all(ga, batch, sphere_value(backend))


def test_children_are_arrays_within_the_box_with_their_parents_and_origins_recorded(backend: Backend):
    ga, _ = started(backend, population_size=10, offspring_size=40)
    members = set(ga.ranked_ids())
    batch = ga.ask(1)
    x = backend.to_numpy(batch.as_array())
    assert x.shape == (40, DIM) and (x >= -5).all() and (x <= 5).all()
    for c in batch.candidates:
        assert len(c.parents) == 2 and set(c.parents) <= members
        assert c.origin == "tournament+intermediate+self_adaptive"


def test_the_two_parents_of_a_child_are_distinct_members(backend: Backend):
    for kwargs in ({}, {"selection": SigmaScalingSUS()}, {"selection": TournamentSelection(size=30)}):
        ga, _ = started(backend, population_size=2, offspring_size=500, **kwargs)  # the hardest case: two members
        for c in ga.ask(1).candidates:
            assert c.parents[0] != c.parents[1]


def test_without_crossover_a_child_copies_one_parent(backend: Backend):
    ga, _ = started(backend, population_size=8, offspring_size=30, crossover_probability=0.0)
    for c in ga.ask(1).candidates:
        assert len(c.parents) == 1 and c.origin == "tournament+copy+self_adaptive"
    asexual, _ = started(backend, population_size=8, offspring_size=30, recombination=NoRecombination())
    assert all(len(c.parents) == 1 and "+copy+" in c.origin for c in asexual.ask(1).candidates)


def test_some_children_cross_over_with_a_probability(backend: Backend):
    ga, _ = started(backend, population_size=8, offspring_size=2000, crossover_probability=0.7)
    batch = ga.ask(1)
    crossed = np.mean([len(c.parents) == 2 for c in batch.candidates])
    assert crossed == pytest.approx(0.7, abs=0.04)
    assert {c.origin for c in batch.candidates} == {"tournament+intermediate+self_adaptive", "tournament+copy+self_adaptive"}


def test_origins_name_the_operators(backend: Backend):
    ga, _ = started(
        backend,
        population_size=6,
        offspring_size=3,
        selection=SigmaScalingSUS(),
        recombination=UniformRecombination(),
        mutation=GaussianMutation(),
    )
    assert {c.origin for c in ga.ask(1).candidates} == {"sus+uniform+gaussian"}


def test_ids_are_issued_by_the_context_in_order(backend: Backend):
    ga, issuer = bind(GeneticAlgorithm(population_size=5, offspring_size=3), backend)
    first = ga.ask(1)
    assert first.ids == (0, 1, 2, 3, 4)
    tell_all(ga, first, sphere_value(backend))
    assert ga.ask(1).ids == (5, 6, 7) and issuer.issued == 8


# --- the population: the best mu of everything told ---


def test_the_population_is_the_best_mu_of_everything_told_against_a_brute_force_reference(backend: Backend):
    rng = np.random.default_rng(7)
    for trial in range(25):
        mu, lam = int(rng.integers(2, 9)), int(rng.integers(1, 10))
        ga, _ = bind(GeneticAlgorithm(population_size=mu, offspring_size=lam), backend, seed=trial)
        reference: list[tuple[float, float, int]] = []

        def tell_random(batch, ga=ga, reference=reference):
            evaluations = []
            for c in batch.candidates:
                if rng.random() < 0.08:
                    evaluations.append(Evaluation(c, Status.FAILED, {"value": math.nan}, error="boom"))
                    key = (math.inf, math.inf, c.id)
                else:
                    value = float(rng.integers(0, 5))  # lots of ties
                    violation = float(rng.integers(0, 3)) if rng.random() < 0.4 else 0.0
                    evaluations.append(Evaluation(c, Status.OK, {"value": value}, {"cpa": violation}))
                    key = (violation, value, c.id)
                reference.append(key)
            order = list(rng.permutation(len(evaluations)))  # told in any order, in one or several tells
            cut = int(rng.integers(1, len(order) + 1))
            ga.tell(EvaluationBatch([evaluations[i] for i in order[:cut]]))
            if cut < len(order):
                ga.tell(EvaluationBatch([evaluations[i] for i in order[cut:]]))

        for _ in range(6):
            tell_random(ga.ask(1))
            assert ga.ranked_ids() == [k[2] for k in sorted(reference)[:mu]]


def test_telling_children_one_at_a_time_gives_the_same_population_as_telling_them_together(backend: Backend):
    def final_population(one_by_one: bool):
        ga, _ = bind(GeneticAlgorithm(population_size=6, offspring_size=10), backend, seed=3)
        values = np.random.default_rng(0)
        table: dict[int, float] = {}
        for _ in range(5):
            batch = ga.ask(1)
            for c in batch.candidates:
                table[c.id] = float(values.integers(0, 50))
            evaluations = evaluate(batch, lambda c: table[c.id])
            if one_by_one:
                for e in evaluations:
                    ga.tell(EvaluationBatch([e]))
            else:
                ga.tell(EvaluationBatch(evaluations))
        return ga.ranked_ids()

    assert final_population(True) == final_population(False)


def test_feasible_members_dominate_infeasible_ones(backend: Backend):
    ga, _ = bind(GeneticAlgorithm(population_size=6, offspring_size=10), backend)
    initial = ga.ask(1)
    # half of the initial members are infeasible with the best objective values
    ga.tell(
        EvaluationBatch(evaluate(initial, lambda c: -100.0 + c.id if c.id < 3 else 5.0 + c.id, lambda c: 1.0 + c.id if c.id < 3 else 0.0))
    )
    assert ga.ranked_ids()[:3] == [3, 4, 5] and ga.ranked_ids()[3:] == [0, 1, 2]  # feasible first; among infeasible, lower violation
    children = ga.ask(1)
    ga.tell(EvaluationBatch(evaluate(children, lambda c: 1000.0, lambda c: 0.0)))  # feasible, but with terrible values
    ranked = ga.ranked_ids()
    assert set(ranked[:3]) == {3, 4, 5}  # still the best of the feasible
    assert all(i >= 6 for i in ranked[3:])  # the new feasible children beat the infeasible ones


def test_the_population_holds_each_candidate_once(backend: Backend):
    ga, _ = bind(GeneticAlgorithm(population_size=5, offspring_size=8), backend)
    told = []
    for _ in range(6):
        batch = ga.ask(1)
        told.extend(batch.ids)
        tell_all(ga, batch, sphere_value(backend))
        assert len(set(ga.ranked_ids())) == ga.size <= 5
    assert len(set(told)) == len(told)  # ids are never reused: every candidate is evaluated once


def test_until_the_population_is_full_all_members_survive(backend: Backend):
    ga, _ = bind(GeneticAlgorithm(population_size=10, offspring_size=5), backend)
    batch = ga.ask(1)
    first_half = batch.take(4)
    ga.tell(EvaluationBatch(evaluate(first_half, sphere_value(backend))))
    assert ga.size == 4
    ga.tell(EvaluationBatch(evaluate(batch.take(10), sphere_value(backend))[4:]))
    assert ga.size == 10


# --- pending children ---


def test_children_wait_as_pending_until_they_are_told_and_asking_again_is_allowed(backend: Backend):
    ga, _ = started(backend, population_size=8, offspring_size=6)
    first, second = ga.ask(1), ga.ask(1)  # a second ask before the first batch has results
    assert len(first.candidates) == len(second.candidates) == 6 and set(first.ids).isdisjoint(second.ids) and second.step == first.step + 1
    members = set(ga.ranked_ids())
    assert all(set(c.parents) <= members for c in second.candidates)  # bred from the population as it is, not from pending children
    tell_all(ga, second, sphere_value(backend))  # in any order
    tell_all(ga, first, sphere_value(backend))
    assert ga.size == 8 and not set(first.ids) & set(ga.ranked_ids()) ^ (set(first.ids) & set(ga.ranked_ids()))


def test_a_tell_can_mix_children_of_different_asks(backend: Backend):
    ga, _ = started(backend, population_size=8, offspring_size=4)
    first, second = ga.ask(1), ga.ask(1)
    value = lambda c: 10.0 + c.id  # noqa: E731
    mixed = [*evaluate(second, value)[:2], *evaluate(first, value)[1:3]]
    ga.tell(EvaluationBatch(mixed))
    ga.tell(EvaluationBatch([*evaluate(first, value)[:1], *evaluate(first, value)[3:], *evaluate(second, value)[2:]]))
    assert ga.size == 8


def test_children_that_are_never_told_are_dropped(backend: Backend):
    ga, _ = started(backend, population_size=8, offspring_size=6)
    before = ga.ranked_ids()
    ga.ask(1)  # asked, but the budget ended before they were told
    assert ga.ranked_ids() == before  # they never joined the population


def test_tell_validates_what_it_is_told(backend: Backend):
    ga, _ = bind(GeneticAlgorithm(population_size=5, offspring_size=3), backend)
    batch = ga.ask(1)
    evaluations = evaluate(batch, sphere_value(backend))
    with pytest.raises(ValueError, match="no results"):
        ga.tell(EvaluationBatch([]))
    with pytest.raises(ValueError, match="more than once"):
        ga.tell(EvaluationBatch([evaluations[0], evaluations[0]]))
    stranger = evaluate(started(backend)[0].ask(1), sphere_value(backend))[0]  # a candidate of another run
    other_ids = Evaluation(type(stranger.candidate)(type(stranger.candidate.id)(999), None, (), "x", 0), Status.OK, {"value": 1.0})
    with pytest.raises(ValueError, match=r"not pending.*\[999\]"):
        ga.tell(EvaluationBatch([other_ids]))
    ga.tell(EvaluationBatch(evaluations))
    with pytest.raises(ValueError, match="not pending"):
        ga.tell(EvaluationBatch(evaluations[:1]))  # already told


# --- breeding is deterministic ---


def test_the_same_seed_gives_the_same_run_and_different_seeds_differ(backend: Backend):
    def run_for(seed: int):
        ga, _ = bind(GeneticAlgorithm(population_size=8, offspring_size=6), backend, seed=seed)
        batches = []
        for _ in range(4):
            batch = ga.ask(1)
            batches.append((batch.ids, backend.to_numpy(batch.as_array()).copy(), batch.parents))
            tell_all(ga, batch, sphere_value(backend))
        return batches, ga.ranked_ids()

    a, b, c = run_for(5), run_for(5), run_for(6)
    for (ids_a, x_a, parents_a), (ids_b, x_b, parents_b) in zip(a[0], b[0]):
        assert ids_a == ids_b and parents_a == parents_b
        np.testing.assert_array_equal(x_a, x_b)
    assert a[1] == b[1]
    assert not np.array_equal(a[0][2][1], c[0][2][1])


# --- state ---


def test_state_dict_is_valid_and_json_serialisable_apart_from_its_arrays(backend: Backend):
    ga, _ = started(backend, population_size=6, offspring_size=4)
    ga.ask(1)  # leaves a pending block in the state
    state = ga.state_dict()
    validate_state_dict(state)
    assert json.loads(json.dumps(state["stream"])) == state["stream"] and json.loads(json.dumps(state["ids"])) == state["ids"]
    assert isinstance(state["pending"], list) and len(state["pending"]) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mutation": SelfAdaptiveMutation(per_gene=True)},
        {"mutation": GaussianMutation(0.1), "selection": SigmaScalingSUS()},
        {"recombination": UniformRecombination(), "crossover_probability": 0.6},
    ],
    ids=["default", "per-gene", "gaussian-sus", "uniform"],
)
def test_state_round_trip_continues_identically(backend: Backend, kwargs: dict):
    def fresh():
        return bind(GeneticAlgorithm(population_size=8, offspring_size=5, **kwargs), backend, seed=11)

    ga, _ = fresh()
    value = sphere_value(backend)
    for _ in range(3):
        tell_all(ga, ga.ask(1), value)
    pending = ga.ask(1)  # a block that has not been told when the state is saved
    state = ga.state_dict()
    state["stream"] = json.loads(json.dumps(state["stream"]))  # the stream part goes through JSON, as in a checkpoint

    def continue_run(strategy):
        out = []
        for _ in range(3):
            batch = strategy.ask(1)
            out.append((batch.ids, backend.to_numpy(batch.as_array()).copy(), batch.parents, batch.origins))
            tell_all(strategy, batch, value)
        return out, strategy.ranked_ids()

    issuer = IdIssuer(start=8 + 5 * 3)  # 8 initial candidates, 2 generations of 5, and the pending block: ids 23 on
    restored, _ = bind(GeneticAlgorithm(population_size=8, offspring_size=5, **kwargs), backend, seed=11, issuer=issuer)
    restored.load_state_dict(state)
    expected = continue_run(ga)
    actual = continue_run(restored)
    for (ids_a, x_a, parents_a, origins_a), (ids_b, x_b, parents_b, origins_b) in zip(expected[0], actual[0]):
        assert ids_a == ids_b and parents_a == parents_b and origins_a == origins_b
        np.testing.assert_array_equal(x_a, x_b)
    assert expected[1] == actual[1]
    # the pending block of the saved state can still be told after the restore
    restored.tell(EvaluationBatch(evaluate(pending, value)))
    assert restored.size == 8


def test_load_state_dict_validates_its_keys(backend: Backend):
    ga, _ = bind(GeneticAlgorithm(), backend)
    with pytest.raises(ValueError, match="invalid GeneticAlgorithm state"):
        ga.load_state_dict({"step": 1})


# --- stopping ---


def test_should_stop_is_false_by_default(backend: Backend):
    ga, _ = started(backend, population_size=6)
    assert not ga.should_stop()


def test_convergence_stops_when_the_population_has_collapsed(backend: Backend):
    ga, _ = bind(
        GeneticAlgorithm(
            population_size=4, offspring_size=4, convergence_tolerance=1e-9, mutation=SelfAdaptiveMutation(min_step=1e-6, initial_step=1e-6)
        ),
        backend,
    )
    batch = ga.ask(1)
    assert not ga.should_stop()  # not full yet
    ga.tell(EvaluationBatch(evaluate(batch, lambda c: 1.0)))  # identical objectives, and the steps start at the floor
    assert ga.should_stop()
    ga2, _ = bind(
        GeneticAlgorithm(population_size=4, offspring_size=4, convergence_tolerance=1e-9, mutation=SelfAdaptiveMutation(initial_step=0.1)),
        backend,
    )
    ga2.tell(EvaluationBatch(evaluate(ga2.ask(1), lambda c: 1.0)))
    assert not ga2.should_stop()  # the steps are still large
    ga3, _ = bind(GeneticAlgorithm(population_size=4, offspring_size=4, convergence_tolerance=1e-9, mutation=GaussianMutation()), backend)
    ga3.tell(EvaluationBatch(evaluate(ga3.ask(1), lambda c: 1.0)))
    assert ga3.should_stop()  # no step sizes: the spread decides
    ga4, _ = bind(GeneticAlgorithm(population_size=4, offspring_size=4, convergence_tolerance=1e-9, mutation=GaussianMutation()), backend)
    ga4.tell(EvaluationBatch(evaluate(ga4.ask(1), lambda c: float(c.id))))
    assert not ga4.should_stop()  # the objectives still differ


# --- step sizes are strategy state ---


def test_step_sizes_follow_the_survivors(backend: Backend):
    for mutation, shape in ((SelfAdaptiveMutation(), (6,)), (SelfAdaptiveMutation(per_gene=True), (6, DIM))):
        ga, _ = started(backend, population_size=6, offspring_size=9, mutation=mutation)
        assert tuple(ga._steps.shape) == shape  # one row per member
        for _ in range(3):
            tell_all(ga, ga.ask(1), sphere_value(backend))
            assert tuple(ga._steps.shape) == shape and tuple(ga._genomes.shape) == (6, DIM)  # the steps of non-survivors are gone
        assert float(backend.to_numpy(ga._steps).min()) >= 1e-12


def test_children_inherit_their_parents_step_sizes(backend: Backend):
    # asexual copies with no mutation noise: a child's step is its parent's step, mutated by exp(tau * N) only
    mutation = SelfAdaptiveMutation(initial_step=0.1, tau=1e-9)
    ga, _ = started(backend, population_size=4, offspring_size=6, recombination=NoRecombination(), mutation=mutation)
    ga._steps = backend.asarray([0.01, 0.02, 0.03, 0.04])  # give the four members distinct steps
    ranked = ga.ranked_ids()
    steps_by_id = dict(zip(ga._ids, backend.to_numpy(ga._steps).tolist()))
    batch = ga.ask(1)
    block = next(iter(ga._blocks.values()))
    for c, step in zip(batch.candidates, backend.to_numpy(block.steps).tolist()):
        assert step == pytest.approx(steps_by_id[c.parents[0]], rel=1e-5)
    assert set(c.parents[0] for c in batch.candidates) <= set(ranked)


def test_non_adaptive_mutations_keep_no_step_sizes(backend: Backend):
    ga, _ = started(backend, population_size=5, offspring_size=5, mutation=GaussianMutation(0.1))
    assert ga._steps is None
    tell_all(ga, ga.ask(1), sphere_value(backend))
    assert ga._steps is None and ga.state_dict()["steps"] is None
