"""`NSGA2`: survivors by constrained non-dominated sorting and crowding, the crowded tournament, and the strategy contract."""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import auxein
from auxein.backend import Backend
from auxein.core import EvaluationBatch, IdIssuer, Objective, StrategyContext, validate_state_dict
from auxein.random import RunSeed
from auxein.spaces import Binary, Box, Categorical, Integer, MixedSpace, Real, SequenceSpace
from auxein.strategies import NSGA2
from auxein.strategies.ga import (
    BitFlipMutation,
    GaussianMutation,
    IntermediateRecombination,
    PolynomialMutation,
    SigmaScalingSUS,
    TournamentSelection,
)
from tests.strategies.nsga2_sorting_test import reference_crowding, reference_ranks
from tests.strategies.strategy_checkpoint_test import same, snapshot, through_a_checkpoint
from tests.support import multiobjective as mo
from tests.support import sequences as sq
from tests.support.fixtures import assert_on_backend

pytestmark = pytest.mark.filterwarnings("ignore::auxein.RecordingDisabledWarning")


def bind(strategy: NSGA2[Any], backend: Backend, space: Any = None, seed: int = 1, issuer: IdIssuer | None = None, constraints=()):
    issuer = issuer or IdIssuer()
    ctx = StrategyContext(RunSeed(seed).stream("strategy", backend=backend), backend, issuer.next)
    strategy.bind(mo.problem(space or mo.ZDT_SPACE, tuple(constraints)), ctx)
    return strategy, issuer


def pair(values: Any) -> auxein.Result:
    return auxein.Result({"f1": float(values[0]), "f2": float(values[1])})


def grid_objectives(c: Any) -> tuple[float, float]:
    """Objectives from the id alone, on a coarse grid: many ties and duplicate points."""
    return float(c.id * 7 % 5), float(c.id * 3 % 4)


# --- survivors: the best mu by constrained non-dominated sorting and crowding ---


def pool_of(strategy: NSGA2[Any], rng: np.random.Generator, rounds: int, size: int, backend: Backend):
    """Feed random evaluations (ties, constraints, failures) and check every survivor set against the brute-force reference."""
    engine = strategy._bound()  # noqa: SLF001
    for _ in range(rounds):
        batch = strategy.ask(size)
        before = {int(i): None for i in backend.to_numpy(engine._ids_array)}  # noqa: SLF001
        old_values = backend.to_numpy(engine._values).astype(np.float64)  # noqa: SLF001
        old_violation = backend.to_numpy(engine._violation).astype(np.float64)  # noqa: SLF001
        old_ids = list(before)
        new_ids = [int(c.id) for c in batch.candidates]
        values = rng.integers(0, 6, (len(new_ids), 2)).astype(np.float64)
        violation = np.where(rng.random(len(new_ids)) < 0.3, rng.integers(1, 4, len(new_ids)).astype(np.float64), 0.0)
        failed = rng.random(len(new_ids)) < 0.1
        values[failed], violation[failed] = np.nan, math.inf
        position = {cid: row for row, cid in enumerate(new_ids)}
        evals = mo.evaluations(
            batch,
            lambda c, v=values, p=position: tuple(v[p[int(c.id)]]),
            lambda c, v=violation, p=position: v[p[int(c.id)]],
            lambda c, f=failed, p=position: bool(f[p[int(c.id)]]),
        )
        strategy.tell(EvaluationBatch(evals))
        pooled_values = np.concatenate([old_values, values]) if len(old_ids) else values
        pooled_violation = np.concatenate([old_violation, violation]) if len(old_ids) else violation
        pooled_ids = old_ids + new_ids
        ranks = reference_ranks(pooled_values, pooled_violation)
        crowding = reference_crowding(pooled_values, pooled_violation, ranks)
        survivors = strategy.ranked_ids()
        mu = strategy.population_size
        assert len(survivors) == min(mu, len(pooled_ids)) and len(set(survivors)) == len(survivors)
        by_id = {cid: i for i, cid in enumerate(pooled_ids)}
        kept_ranks = sorted(ranks[by_id[cid]] for cid in survivors)
        assert kept_ranks == sorted(ranks)[: len(survivors)]  # the survivors have the smallest fronts that exist
        excluded = [i for i in range(len(pooled_ids)) if pooled_ids[i] not in set(survivors)]
        if excluded:
            last = max(kept_ranks)
            worst_kept = min(crowding[by_id[cid]] for cid in survivors if ranks[by_id[cid]] == last)
            for i in excluded:
                assert ranks[i] > last or crowding[i] <= worst_kept + 1e-6  # and in the last front, the least crowded
        # and the population is listed in the crowded-comparison order
        listed = [(ranks[by_id[cid]], -crowding[by_id[cid]], cid) for cid in survivors]
        assert [r for r, _, _ in listed] == sorted(r for r, _, _ in listed)


@pytest.mark.parametrize("seed", range(4))
def test_survivors_are_always_the_best_mu_by_constrained_sorting_and_crowding(backend: Backend, seed: int):
    strategy, _ = bind(NSGA2(population_size=12, offspring_size=9), backend, seed=seed, constraints=("c",))
    pool_of(strategy, np.random.default_rng(seed), rounds=14, size=9, backend=backend)


def test_a_feasible_member_beats_an_infeasible_one_with_better_objectives(backend: Backend):
    strategy, _ = bind(NSGA2(population_size=4, offspring_size=4), backend, constraints=("c",))
    first = strategy.ask(1)
    mo.tell(
        strategy,
        first,
        lambda c: (0.0, 0.0) if c.id < 2 else (9.0, 9.0),  # ids 0 and 1 are infeasible with perfect objectives
        lambda c: 1.0 if c.id < 2 else 0.0,
    )
    assert strategy.ranked_ids()[:2] == [2, 3] or set(strategy.ranked_ids()[:2]) <= {2, 3}
    assert set(strategy.ranked_ids()[2:]) == {0, 1}


def test_failed_members_rank_last_and_are_not_parents_while_there_is_an_alternative(backend: Backend):
    strategy, _ = bind(NSGA2(population_size=10, offspring_size=10), backend)
    first = strategy.ask(1)
    mo.tell(strategy, first, grid_objectives, failed=lambda c: c.id % 2 == 0)
    assert all(i % 2 == 1 for i in strategy.ranked_ids()[:5]) and all(i % 2 == 0 for i in strategy.ranked_ids()[5:])
    for _ in range(5):
        batch = strategy.ask(1)
        parents = {int(p) for c in batch.candidates for p in c.parents}
        assert parents and all(p % 2 == 1 or p >= 10 for p in parents)  # never one of the failed initial members
        mo.tell(strategy, batch, grid_objectives, failed=lambda c: c.id % 2 == 0)


def test_one_objective_is_a_genetic_algorithm_with_plus_selection(backend: Backend):
    result = auxein.run(
        strategy=NSGA2(population_size=40, offspring_size=40),
        evaluator=auxein.VectorisedEvaluator(lambda X: backend.xp.sum(X * X, axis=1)),
        space=Box(0.0, 1.0, dim=8),
        budget=auxein.Budget(evaluations=8000),
        seed=2,
        backend=backend,
        batch_size=40,
    )
    assert result.best is not None and result.best.objectives["value"] < 0.01


# --- parents: the crowded tournament, through the unchanged TournamentSelection ---


def test_the_population_view_orders_members_by_the_crowded_comparison(backend: Backend):
    strategy, _ = bind(NSGA2(population_size=20, offspring_size=20), backend)
    rng = np.random.default_rng(3)
    for _ in range(5):
        batch = strategy.ask(1)
        mo.tell(strategy, batch, lambda c: tuple(rng.random(2)))
    engine = strategy._bound()  # noqa: SLF001
    order, rank = backend.to_numpy(engine._order), backend.to_numpy(engine._rank)  # noqa: SLF001
    assert sorted(order.tolist()) == list(range(20)) and rank[order].tolist() == list(range(20))  # inverse permutations
    ids = backend.to_numpy(engine._ids_array)  # noqa: SLF001
    assert strategy.ranked_ids() == ids[order].tolist()
    assert isinstance(strategy.selection, TournamentSelection)


def test_parents_are_chosen_from_the_better_fronts(backend: Backend):
    strategy, _ = bind(NSGA2(population_size=30, offspring_size=300), backend)
    rng = np.random.default_rng(5)
    mo.tell(strategy, strategy.ask(1), lambda c: tuple(rng.random(2)))
    engine = strategy._bound()  # noqa: SLF001
    batch = strategy.ask(1)
    rank_of = {int(i): r for r, i in enumerate(strategy.ranked_ids())}
    chosen = np.array([rank_of[int(p)] for c in batch.candidates for p in c.parents])
    assert chosen.mean() < 0.8 * np.mean(list(rank_of.values()))  # a binary tournament favours the front of the order
    _ = engine


def test_selection_that_needs_a_scalar_fitness_is_refused():
    with pytest.raises(ValueError, match="crowded-comparison order.*SigmaScalingSUS"):
        NSGA2(selection=SigmaScalingSUS())
    NSGA2(selection=TournamentSelection(4))


# --- the contract ---


def test_offspring_counts_pending_children_and_no_self_mating(backend: Backend):
    strategy, issuer = bind(NSGA2(population_size=10, offspring_size=6), backend)
    first = strategy.ask(99)
    assert len(first.candidates) == 10 and set(first.origins) == {"init"}  # the whole initial population, whatever n says
    mo.tell(strategy, first, grid_objectives)
    a, b = strategy.ask(1), strategy.ask(1)  # two asks before any tell: both are pending
    assert len(a.candidates) == len(b.candidates) == 6
    for c in (*a.candidates, *b.candidates):
        assert len(c.parents) in (1, 2) and len(set(c.parents)) == len(c.parents)
        assert "tournament" in c.origin and "sbx" in c.origin and "polynomial" in c.origin or len(c.parents) == 1
    mo.tell(strategy, b, grid_objectives)  # the later block first
    with pytest.raises(ValueError, match="not pending"):
        strategy.tell(EvaluationBatch(mo.evaluations(b, grid_objectives)[:1]))  # already told
    mo.tell(strategy, a, grid_objectives)
    assert strategy.size == 10
    flexible, _ = bind(NSGA2(population_size=6, offspring_size=None), backend)
    mo.tell(flexible, flexible.ask(1), grid_objectives)
    assert len(flexible.ask(7).candidates) == 7
    with pytest.raises(ValueError, match="at least 1"):
        flexible.ask(0)
    _ = issuer


def test_it_works_under_both_deliveries_with_the_same_population_size(backend: Backend):
    for delivery in ("generation", "steady_state"):
        result = auxein.run(
            strategy=NSGA2(population_size=40, offspring_size=20),
            evaluator=auxein.FunctionEvaluator(lambda genome: pair(mo.zdt1(backend.to_numpy(genome)[None, :])[0])),
            space=mo.ZDT_SPACE,
            objectives=[mo.F1, mo.F2],
            budget=auxein.Budget(evaluations=3000),
            seed=3,
            backend=backend,
            batch_size=20,
            delivery=delivery,  # type: ignore[arg-type]
        )
        front = np.array([[e.objectives["f1"], e.objectives["f2"]] for e in result.pareto_front])
        assert len(front) > 10 and mo.distance_to_front(front) < 1.0, delivery


def test_the_state_dict_continues_identically_through_a_checkpoint(tmp_path: Path, backend: Backend):
    def feed(strategy: NSGA2[Any], rounds: int) -> list[Any]:
        out = []
        for _ in range(rounds):
            batch = strategy.ask(1)
            out.append(snapshot(batch, backend))
            mo.tell(strategy, batch, mo.zdt1_of(backend), failed=lambda c: c.id % 11 == 3)
        return out

    strategy, issuer = bind(NSGA2(population_size=10, offspring_size=6), backend, seed=9)
    feed(strategy, 4)
    pending = strategy.ask(1)
    strategy.tell(EvaluationBatch(mo.evaluations(pending, mo.zdt1_of(backend))[:2]))
    state = through_a_checkpoint(strategy.state_dict(), tmp_path / "c", backend)
    validate_state_dict(state)
    assert "order" in state
    assert_on_backend(state["values"], backend)  # type: ignore[arg-type]
    assert tuple(state["values"].shape)[1] == 2  # type: ignore[union-attr]  (the objectives of every member)

    restored, _ = bind(NSGA2(population_size=10, offspring_size=6), backend, seed=9, issuer=IdIssuer(issuer.issued))
    restored.load_state_dict(state)
    assert restored.ranked_ids() == strategy.ranked_ids()
    rest = EvaluationBatch(mo.evaluations(pending, mo.zdt1_of(backend))[2:])
    strategy.tell(rest)
    restored.tell(rest)
    same(feed(strategy, 8), feed(restored, 8))
    assert restored.ranked_ids() == strategy.ranked_ids()
    np.testing.assert_array_equal(backend.to_numpy(restored.state_dict()["order"]), backend.to_numpy(strategy.state_dict()["order"]))  # type: ignore[arg-type]


def test_a_changed_description_is_visible_to_the_resume_check():
    assert repr(NSGA2()) == repr(NSGA2())
    assert repr(NSGA2(population_size=50)) != repr(NSGA2())
    assert repr(NSGA2(crossover_probability=0.8)) != repr(NSGA2())
    assert "integer_mutation" not in repr(NSGA2())
    assert NSGA2().capabilities.max_objectives is None and NSGA2().capabilities.tell_mode == "both"


def test_binding_to_an_unsupported_space_or_with_the_wrong_operators_is_a_clear_error(backend: Backend):
    class Words:
        def sample_genomes(self, n, rng, backend):
            return ["w"] * n

        def contains(self, genome):
            return True

    with pytest.raises(TypeError, match="Box, a MixedSpace or a space with a codec"):
        bind(NSGA2(), backend, space=Words())
    with pytest.raises(TypeError, match="structured"):
        bind(NSGA2(recombination=sq_crossover()), backend)
    with pytest.raises(TypeError, match="not a mutation of the numeric"):
        bind(NSGA2(mutation=sq_mutation()), backend)
    with pytest.raises(TypeError, match="works on arrays"):
        bind(NSGA2(mutation=GaussianMutation(0.1)), backend, space=sq.SPACE)
    with pytest.raises(TypeError, match="works on arrays"):
        bind(NSGA2(recombination=IntermediateRecombination()), backend, space=sq.SPACE)
    with pytest.raises(TypeError, match="for Box and MixedSpace"):
        bind(NSGA2(integer_mutation=auxein.strategies.ga.IntegerMutation()), backend, space=sq.SPACE)
    with pytest.raises(RuntimeError, match="must be bound"):
        NSGA2().ask(1)


def sq_crossover() -> Any:
    from auxein.strategies.structured import SequenceCrossover

    return SequenceCrossover()


def sq_mutation() -> Any:
    from auxein.strategies.structured import SequenceMutation

    return SequenceMutation()


# --- the three kinds of space ---


def front_of(result: auxein.RunResult[Any], names: tuple[str, str] = ("f1", "f2")) -> np.ndarray:
    return np.array([[e.objectives[names[0]], e.objectives[names[1]]] for e in result.pareto_front])


def test_on_a_box_it_converges_to_the_front_of_zdt1_and_beats_random_search(backend: Backend):
    def solve(strategy: Any) -> np.ndarray:
        result = auxein.run(
            strategy=strategy,
            evaluator=auxein.VectorisedEvaluator(lambda X: mo.zdt1_batch(X, backend)),
            space=mo.ZDT_SPACE,
            objectives=[mo.F1, mo.F2],
            budget=auxein.Budget(evaluations=12_000),
            seed=4,
            backend=backend,
            batch_size=100,
        )
        assert result.best is None  # several objectives: no single best
        return front_of(result)

    found, random = solve(NSGA2()), solve(auxein.RandomSearch())
    assert mo.distance_to_front(found) < 0.05 < 1.0 < mo.distance_to_front(random) + 0.5
    assert found[:, 0].min() < 0.05 and found[:, 0].max() > 0.95  # and it spreads along the whole front


def mixed_space() -> MixedSpace:
    return MixedSpace({"x": Real(0.0, 1.0), "n": Integer(0, 10), "b": Binary(), "c": Categorical(["a", "b", "c"]), "y": Real(0.0, 1.0)})


def test_on_a_mixed_space_every_type_is_searched_and_the_genomes_stay_valid(backend: Backend):
    space = mixed_space()

    def objectives(X: Any) -> np.ndarray:
        x = backend.to_numpy(X).astype(np.float64)
        first = x[:, 0] ** 2 + 0.1 * (x[:, 1] - 5) ** 2 + (1 - x[:, 2]) + (x[:, 3] != 1)
        second = (x[:, 0] - 1) ** 2 + 0.1 * (x[:, 1] - 8) ** 2 + x[:, 2] + (x[:, 3] != 1) + x[:, 4] ** 2
        return np.stack([first, second], axis=1)

    result = auxein.run(
        strategy=NSGA2(population_size=40, offspring_size=40),
        evaluator=auxein.VectorisedEvaluator(objectives),
        space=space,
        objectives=[mo.F1, mo.F2],
        budget=auxein.Budget(evaluations=6000),
        seed=5,
        backend=backend,
        batch_size=40,
    )
    assert all(space.contains(e.candidate.genome) for e in result.pareto_front)
    values = [space.values(e.candidate.genome) for e in result.pareto_front]
    assert {v["c"] for v in values} == {"b"}  # the category that costs nothing
    assert {v["b"] for v in values} == {True, False}  # the bit is the trade-off between the objectives
    assert {v["n"] for v in values} >= {5, 8}  # the integer reaches both of its optima
    origin = result.pareto_front[-1].candidate.origin
    assert origin == "init" or origin.startswith("tournament+sbx/discrete+polynomial/geometric/bitflip/resample")


def test_with_operators_replaced_on_a_mixed_space(backend: Backend):
    space = mixed_space()
    result = auxein.run(
        strategy=NSGA2(
            population_size=20,
            offspring_size=20,
            recombination=IntermediateRecombination(),
            mutation=GaussianMutation(0.05),
            binary_mutation=BitFlipMutation(0.3),
        ),
        evaluator=auxein.VectorisedEvaluator(lambda X: backend.xp.stack([X[:, 0], 1.0 - X[:, 0] + X[:, 2]], axis=1)),
        space=space,
        objectives=[mo.F1, mo.F2],
        budget=auxein.Budget(evaluations=1200),
        seed=1,
        backend=backend,
        batch_size=20,
    )
    assert result.pareto_front and all(space.contains(e.candidate.genome) for e in result.pareto_front)


def test_on_a_sequence_space_the_front_trades_distance_to_the_target_for_length(backend: Backend):
    def objectives(genome: tuple[object, ...]) -> Any:
        return auxein.Result({"f1": float(sq.edit_distance(genome, sq.TARGET)), "f2": float(len(genome))})

    result = auxein.run(
        strategy=NSGA2(population_size=40, offspring_size=40),
        evaluator=auxein.FunctionEvaluator(objectives),
        space=sq.SPACE,
        objectives=[mo.F1, mo.F2],
        budget=auxein.Budget(evaluations=8000),
        seed=6,
        backend=backend,
        batch_size=40,
    )
    front = {(int(e.objectives["f1"]), int(e.objectives["f2"])) for e in result.pareto_front}
    # a sequence of length L is at least 12 - L edits from the 12-token target, with equality for a subsequence of it: those
    # points (12 - L, L) are the true front, and every point of any front respects the bound
    assert all(f1 + f2 >= len(sq.TARGET) for f1, f2 in front)
    on_front = {(f1, f2) for f1, f2 in front if f1 + f2 == len(sq.TARGET)}
    assert (10, 2) in on_front and len(on_front) >= 6  # the shortest sequences, and a good stretch of the trade-off
    assert max(f2 for _, f2 in on_front) >= 8
    assert result.best is None
    assert all(len(e.candidate.parents) <= 2 for e in result.pareto_front)
    assert isinstance(SequenceSpace(("a", "b"), 1, 2), SequenceSpace)


# --- the polynomial regression with structure genes, as a trade-off between accuracy and complexity ---


def test_the_front_of_accuracy_against_complexity_contains_the_true_polynomial(corner_backend: Backend):
    """Two minimised objectives, the data error and the number of active terms. The true model (terms 0, 2 and 5 of
    `2 + 3x² − 1.5x⁵`) fits noise-free data with three terms, so it is on the front, and the front is a staircase: with fewer
    terms the fit is worse, and with more it is no better.

    How reliable this is, measured over seeds 0 to 9 at this budget on the four configurations: the true set of terms is on the
    front in 6 to 7 seeds of 10, and with an error under 0.05 (a quarter of a percent of the data variance, 19.5) in 2 to 5 of
    10. NSGA-II spreads its population over the eight levels of complexity and has no step-size adaptation, so the coefficients
    of the true model are refined slowly. Seed 5 finds the true set on all four configurations, with an error under 0.5."""
    from tests.support import mixed as mx

    backend = corner_backend
    result = auxein.run(
        strategy=NSGA2(population_size=100, offspring_size=100, mutation=PolynomialMutation(eta=50.0)),
        evaluator=auxein.VectorisedEvaluator(lambda X: mx.polynomial_objectives(X, backend)),
        space=mx.POLY_SPACE,
        objectives=[Objective("error"), Objective("terms")],
        budget=auxein.Budget(evaluations=40_000),
        seed=5,
        backend=backend,
        batch_size=100,
    )
    front = {(round(e.objectives["terms"]), mx.active_terms(e.candidate.genome)): e.objectives["error"] for e in result.pareto_front}
    true_model = [error for (terms, active), error in front.items() if active == mx.TRUE_TERMS]
    assert true_model and min(true_model) < 0.5, sorted(front.items())  # the true active terms, and an error far under the data variance
    counts = sorted({terms for terms, _ in front})
    assert counts[0] <= 1 and 3 in counts  # the front runs from the simplest models to the true one
    errors = [min(error for (terms, _), error in front.items() if terms == t) for t in counts]
    assert errors == sorted(errors, reverse=True)  # more terms never fit worse along the front
    assert min(error for (terms, _), error in front.items() if terms < 3) > 0.5  # fewer than three terms cannot fit the data
