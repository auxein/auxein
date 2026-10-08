"""The strategies told FAILED and TIMEOUT evaluations (design doc §6.6): failed candidates rank last and are never parents."""

import warnings

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import Evaluation, EvaluationBatch, ProblemSpec, Status
from auxein.driver import Budget, EvaluationFailureWarning, RecordingDisabledWarning, run
from auxein.evaluators import FunctionEvaluator
from auxein.spaces import Box
from auxein.strategies import GeneticAlgorithm, RandomSearch
from auxein.strategies.ga import GaussianMutation, SigmaScalingSUS, TournamentSelection, view_of
from tests.strategies.ga_strategy_test import bind, evaluate
from tests.support import workers

SELECTIONS = {"tournament": lambda: TournamentSelection(2), "tournament-5": lambda: TournamentSelection(5), "sus": SigmaScalingSUS}


def told(ga, batch, outcome):
    """Tell a batch whose candidates succeed or fail as `outcome(candidate)` says (a status, or a value)."""
    evaluations = []
    for c in batch.candidates:
        what = outcome(c)
        if isinstance(what, Status):
            evaluations.append(Evaluation.failed(c, what, f"{what.value} {c.id}", 0.1))
        else:
            evaluations.append(evaluate(type("B", (), {"candidates": [c]}), lambda _c, v=what: v)[0])
    ga.tell(EvaluationBatch(evaluations))
    return evaluations


@pytest.mark.parametrize("status", [Status.FAILED, Status.TIMEOUT])
def test_failed_children_never_displace_feasible_members(backend: Backend, status: Status):
    ga, _ = bind(GeneticAlgorithm(population_size=6, offspring_size=4), backend)
    told(ga, ga.ask(6), lambda c: float(c.id) + 1.0)
    before = ga.ranked_ids()
    for _ in range(5):
        told(ga, ga.ask(4), lambda c: status)
        assert ga.ranked_ids() == before  # nothing about the population changed
    told(ga, ga.ask(4), lambda c: float(c.id) % 3 + 0.5)  # and good children still get in
    assert ga.ranked_ids() != before


def test_failed_members_rank_last_when_the_population_is_not_full_of_good_ones(backend: Backend):
    ga, _ = bind(GeneticAlgorithm(population_size=6), backend)
    first = ga.ask(6)
    told(ga, first, lambda c: Status.FAILED if c.id % 2 == 0 else 10.0 + c.id)
    ranked = ga.ranked_ids()
    assert [i % 2 for i in ranked] == [1, 1, 1, 0, 0, 0]  # the three that worked, then the three that failed


@pytest.mark.parametrize("selection", list(SELECTIONS))
def test_a_failed_member_is_never_a_parent_while_there_is_an_alternative(backend: Backend, selection: str):
    for good in (2, 3, 5):
        ga, _ = bind(GeneticAlgorithm(population_size=8, offspring_size=300, selection=SELECTIONS[selection]()), backend, seed=good)
        first = ga.ask(8)
        told(ga, first, lambda c, good=good: float(c.id) + 1.0 if c.id < good else Status.FAILED)
        ok = {c.id for c in first.candidates if c.id < good}
        children = ga.ask(300)
        parents = {p for ps in children.parents for p in ps}
        assert parents and parents <= ok, f"{selection} with {good} valid members chose a failed parent: {parents - ok}"
        assert all(len(ps) == len(set(ps)) for ps in children.parents)  # and the two parents are still distinct


def test_with_fewer_than_two_valid_members_the_ga_samples_at_random_instead_of_breeding(backend: Backend):
    ga, _ = bind(GeneticAlgorithm(population_size=6, offspring_size=5), backend)
    first = ga.ask(6)
    told(ga, first, lambda c: 1.0 if c.id == 0 else Status.FAILED)
    children = ga.ask(5)
    assert set(children.origins) == {"init"} and all(ps == () for ps in children.parents)
    told(ga, children, lambda c: 2.0 + c.id)
    assert ga.ranked_ids()[:2] == [0, children.ids[0]]  # and it recovers: the new valid members can be parents again
    assert set(ga.ask(5).origins) != {"init"}


def test_everything_failing_at_the_start_does_not_break_asking(backend: Backend):
    ga, _ = bind(GeneticAlgorithm(population_size=4, offspring_size=4), backend)
    for _ in range(3):
        batch = ga.ask(4)
        told(ga, batch, lambda c: Status.TIMEOUT)
    assert len(ga.ranked_ids()) == 4
    batch = ga.ask(4)
    assert set(batch.origins) == {"init"}
    values = np.asarray(backend.to_numpy(batch.as_array()))
    assert np.isfinite(values).all()


def test_infeasible_members_that_did_not_fail_are_still_valid_parents(backend: Backend):
    problem_ga = GeneticAlgorithm(population_size=6, offspring_size=100)
    problem_ga.bind(
        ProblemSpec(Box(-5.0, 5.0, dim=3), (_value(),), ("cpa",)),
        _context(backend),
    )
    first = problem_ga.ask(6)
    evaluations = []
    for c in first.candidates:
        if c.id == 0:
            evaluations.append(Evaluation.failed(c, Status.FAILED, "x"))
        else:
            evaluations.append(Evaluation(c, Status.OK, {"value": float(c.id)}, {"cpa": float(c.id)}))  # all violate the constraint
    problem_ga.tell(EvaluationBatch(evaluations))
    parents = {p for ps in problem_ga.ask(100).parents for p in ps}
    assert parents <= {1, 2, 3, 4, 5} and len(parents) >= 4  # violating members breed; the failed one does not


def _value():
    from auxein.core import Objective

    return Objective("value")


def _context(backend: Backend):
    from auxein.core import IdIssuer, StrategyContext
    from auxein.random import RunSeed

    return StrategyContext(RunSeed(3).stream("strategy", backend=backend), backend, IdIssuer().next)


def test_sus_weights_are_finite_and_zero_for_failed_members(backend: Backend):
    xp = backend.xp
    values = backend.asarray([1.0, float("nan"), 3.0, float("nan"), 2.0])
    violation = backend.asarray([0.0, float("inf"), 0.0, float("inf"), 0.0])
    ids = backend.asarray([0, 1, 2, 3, 4], dtype=backend.int_dtype)
    weights = backend.to_numpy(SigmaScalingSUS().weights(view_of(values, violation, ids, backend)))
    assert np.isfinite(weights).all() and weights[1] == 0 and weights[3] == 0 and weights.sum() > 0
    # no feasible member: weights come from the violation of the infeasible ones, still never from a failed one
    violation = backend.asarray([2.0, float("inf"), 2.0, float("inf"), 2.0])
    weights = backend.to_numpy(SigmaScalingSUS().weights(view_of(values, violation, ids, backend)))
    assert np.isfinite(weights).all() and weights[1] == 0 and weights[3] == 0 and weights.sum() > 0
    del xp


def test_should_stop_ignores_the_values_of_failed_members(backend: Backend):
    ga, _ = bind(
        GeneticAlgorithm(population_size=4, offspring_size=4, convergence_tolerance=1e-3, mutation=GaussianMutation(step=0.1)), backend
    )
    batch = ga.ask(4)
    told(ga, batch, lambda c: Status.FAILED if c.id == 0 else 5.0)
    assert ga.should_stop() is True  # the three that did not fail agree on 5.0: converged, whatever the failed one holds
    told(ga, ga.ask(4), lambda c: 100.0 + c.id)  # a child with a real (worse) value replaces the failed member
    assert ga.should_stop() is False
    ga2, _ = bind(
        GeneticAlgorithm(population_size=4, offspring_size=4, convergence_tolerance=1e-3, mutation=GaussianMutation(step=0.1)), backend
    )
    told(ga2, ga2.ask(4), lambda c: Status.FAILED)
    assert ga2.should_stop() is False  # nothing is known about an all-failed population


def test_random_search_is_unaffected_by_failed_evaluations(backend: Backend):
    strategy = RandomSearch()
    strategy.bind(ProblemSpec(Box(-1.0, 1.0, dim=2), (_value(),)), _context(backend))
    batch = strategy.ask(5)
    strategy.tell(EvaluationBatch([Evaluation.failed(c, Status.TIMEOUT, "slow") for c in batch.candidates]))
    assert len(strategy.ask(5).candidates) == 5


# --- whole runs ---


def _run(strategy, rate: float, budget: int, backend: Backend, seed: int = 3):
    evaluator = FunctionEvaluator(lambda g, rng: workers.flaky_sphere(g, rng, rate), uses_rng=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RecordingDisabledWarning)
        warnings.simplefilter("ignore", EvaluationFailureWarning)
        return run(
            strategy=strategy, evaluator=evaluator, space=Box(-5.0, 5.0, dim=10), budget=Budget(evaluations=budget), seed=seed,
            backend=backend, batch_size=50,
        )  # fmt: skip


def test_a_ga_with_thirty_percent_random_failures_still_improves_over_random_search(backend: Backend):
    ga = _run(GeneticAlgorithm(), 0.3, 5000, backend)
    random = _run(RandomSearch(), 0.3, 5000, backend)
    assert 0.2 < ga.status_counts["failed"] / 5000 < 0.4 and ga.status_counts["ok"] > 3000
    assert ga.best is not None and random.best is not None
    assert ga.best.objectives["value"] < random.best.objectives["value"] / 10
    clean = _run(GeneticAlgorithm(), 0.0, 5000, backend)
    assert ga.best.objectives["value"] < 50 * clean.best.objectives["value"] + 1.0  # failures cost some progress, not all of it


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_the_ga_makes_progress_with_failures_under_both_deliveries(backend: Backend, delivery: str):
    evaluator = FunctionEvaluator(lambda g, rng: workers.flaky_sphere(g, rng, 0.3), uses_rng=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run(
            strategy=GeneticAlgorithm(population_size=20, offspring_size=10), evaluator=evaluator, space=Box(-5.0, 5.0, dim=5),
            budget=Budget(evaluations=2500), seed=5, backend=backend, batch_size=10, delivery=delivery, concurrency=3,
        )  # fmt: skip
    assert result.best is not None and result.best.objectives["value"] < 1e-2
