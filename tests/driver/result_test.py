import math

from auxein.backend import Backend
from auxein.core import Candidate, CandidateId, Cost, Evaluation, Objective, Result, Status
from auxein.driver import Budget, RecordingDisabledWarning, RunResult, run
from auxein.driver.result import ResultTracker
from auxein.evaluators import FunctionEvaluator, VectorisedEvaluator
from auxein.spaces import Box
from auxein.strategies import RandomSearch
from tests.support.fakes import ScriptedStrategy

LOSS = Objective("loss")
SCORE = Objective("score", "maximise")


def ev(i: int, *values: float, violation: float | None = None, status: Status = Status.OK, names=("loss", "score")) -> Evaluation:
    candidate = Candidate(CandidateId(i), None, (), "x", 0)
    constraints = {} if violation is None else {"cpa": violation}
    objectives = dict(zip(names, values))
    if status is not Status.OK:
        objectives = {n: math.nan for n in names[: len(values)]}
    return Evaluation(candidate, status, objectives, constraints, cost=Cost())


def ids(evaluations) -> list[int]:
    return [e.candidate.id for e in evaluations]


# --- best ---


def test_the_lowest_objective_wins_for_a_minimised_objective():
    tracker = ResultTracker([LOSS])
    tracker.add([ev(0, 5.0), ev(1, 3.0), ev(2, 4.0)], 0)
    assert tracker.best is not None and tracker.best.candidate.id == 1


def test_the_declared_direction_is_respected():
    tracker = ResultTracker([SCORE])
    tracker.add([ev(0, 5.0, names=("score",)), ev(1, 9.0, names=("score",)), ev(2, 7.0, names=("score",))], 0)
    assert tracker.best is not None and tracker.best.candidate.id == 1 and tracker.best.objectives["score"] == 9.0


def test_ties_go_to_the_earliest_id_whatever_the_arrival_order():
    tracker = ResultTracker([LOSS])
    tracker.add([ev(5, 1.0)], 0)
    tracker.add([ev(2, 1.0), ev(9, 1.0)], 1)
    assert tracker.best is not None and tracker.best.candidate.id == 2
    again = ResultTracker([LOSS])
    again.add([ev(2, 1.0), ev(5, 1.0), ev(9, 1.0)], 0)
    assert again.best is not None and again.best.candidate.id == 2


def test_feasible_beats_infeasible_even_with_a_worse_objective():
    tracker = ResultTracker([LOSS])
    tracker.add([ev(0, 0.0, violation=2.0), ev(1, 100.0, violation=0.0), ev(2, -50.0, violation=0.1)], 0)
    assert tracker.best is not None and tracker.best.candidate.id == 1


def test_among_infeasible_candidates_the_smaller_violation_wins_then_the_objective():
    tracker = ResultTracker([LOSS])
    tracker.add([ev(0, 1.0, violation=3.0), ev(1, 99.0, violation=1.0), ev(2, 5.0, violation=1.0)], 0)
    assert tracker.best is not None and tracker.best.candidate.id == 2  # equal violations: the lower objective


def test_the_best_among_feasible_candidates_is_by_objective():
    tracker = ResultTracker([LOSS])
    tracker.add([ev(0, 3.0, violation=0.0), ev(1, 1.0, violation=0.0), ev(2, 0.0, violation=0.5)], 0)
    assert tracker.best is not None and tracker.best.candidate.id == 1


def test_failed_evaluations_are_ignored_and_best_is_none_without_an_ok_one():
    tracker = ResultTracker([LOSS])
    tracker.add([ev(0, 1.0, status=Status.FAILED), ev(1, 1.0, status=Status.TIMEOUT)], 0)
    assert tracker.best is None and tracker.trace == () and tracker.pareto_front == ()
    tracker.add([ev(2, 7.0)], 2)
    assert tracker.best is not None and tracker.best.candidate.id == 2


def test_best_is_only_defined_for_a_single_objective():
    tracker = ResultTracker([LOSS, SCORE])
    tracker.add([ev(0, 1.0, 2.0)], 0)
    assert tracker.best is None and tracker.trace == () and len(tracker.pareto_front) == 1


def test_the_trace_records_each_improvement_at_the_evaluation_that_made_it():
    tracker = ResultTracker([LOSS])
    tracker.add([ev(0, 5.0), ev(1, 6.0), ev(2, 4.0)], 0)  # improvements at evaluations 1 and 3
    tracker.add([ev(3, 4.5), ev(4, 2.0)], 3)  # and 5
    tracker.add([ev(5, 2.0), ev(6, 1.0)], 5)  # a tie is not an improvement; 7 is
    assert tracker.trace == ((1, 5.0), (3, 4.0), (5, 2.0), (7, 1.0))
    values = [v for _, v in tracker.trace]
    assert values == sorted(values, reverse=True)  # monotone for a minimised objective


def test_the_trace_is_in_natural_units_for_a_maximised_objective():
    tracker = ResultTracker([SCORE])
    tracker.add([ev(0, 1.0, names=("score",)), ev(1, 3.0, names=("score",)), ev(2, 2.0, names=("score",))], 0)
    assert tracker.trace == ((1, 1.0), (2, 3.0))


def test_a_single_objective_tracker_holds_constant_memory():
    tracker = ResultTracker([LOSS])
    for batch in range(200):
        tracker.add([ev(batch * 10 + i, float((batch * 10 + i) % 37)) for i in range(10)], batch * 10)
    assert len(tracker._front) <= 1 and len(tracker.pareto_front) == 1
    assert len(tracker.trace) < 10  # improvements only


# --- the Pareto front ---


def test_a_hand_checked_two_objective_front():
    tracker = ResultTracker([LOSS, Objective("time")])
    points = {0: (1.0, 5.0), 1: (2.0, 3.0), 2: (3.0, 4.0), 3: (4.0, 1.0), 4: (2.0, 3.0), 5: (5.0, 5.0)}
    tracker.add([ev(i, *p, names=("loss", "time")) for i, p in points.items()], 0)
    # 2 is dominated by 1, 5 by everything, 4 equals 1 (the earlier candidate keeps the place)
    assert ids(tracker.pareto_front) == [0, 1, 3]


def test_a_new_point_removes_the_points_it_dominates():
    tracker = ResultTracker([LOSS, Objective("time")])
    tracker.add(
        [ev(0, 3.0, 3.0, names=("loss", "time")), ev(1, 1.0, 5.0, names=("loss", "time")), ev(2, 5.0, 1.0, names=("loss", "time"))], 0
    )
    assert ids(tracker.pareto_front) == [0, 1, 2]
    tracker.add([ev(3, 0.5, 0.5, names=("loss", "time"))], 3)
    assert ids(tracker.pareto_front) == [3]
    tracker.add([ev(4, 0.4, 9.0, names=("loss", "time"))], 4)
    assert ids(tracker.pareto_front) == [3, 4]


def test_mixed_directions_are_handled_in_minimisation_form():
    tracker = ResultTracker([LOSS, SCORE])  # minimise loss, maximise score
    points = {0: (1.0, 1.0), 1: (2.0, 5.0), 2: (3.0, 4.0), 3: (0.5, 0.5), 4: (2.0, 4.0)}
    tracker.add([ev(i, *p) for i, p in points.items()], 0)
    # 2 and 4 are dominated by 1 (lower or equal loss, higher or equal score); the rest trade loss for score
    assert ids(tracker.pareto_front) == [0, 1, 3]


def test_infeasible_and_failed_evaluations_never_enter_the_front():
    tracker = ResultTracker([LOSS, Objective("time")])
    tracker.add(
        [
            ev(0, 1.0, 1.0, violation=0.5, names=("loss", "time")),
            ev(1, 9.0, 9.0, violation=0.0, names=("loss", "time")),
            ev(2, 0.0, 0.0, status=Status.FAILED, names=("loss", "time")),
        ],
        0,
    )
    assert ids(tracker.pareto_front) == [1]


def test_for_a_single_objective_the_front_is_the_best_feasible_one():
    tracker = ResultTracker([LOSS])
    tracker.add([ev(0, 3.0), ev(1, 1.0), ev(2, 2.0), ev(3, 1.0)], 0)
    assert ids(tracker.pareto_front) == [1]


# --- through the driver ---


def sphere(g):
    return float((g * g).sum())


def test_the_run_result_of_a_single_objective_run(backend: Backend):
    result = run_quietly(RandomSearch(), FunctionEvaluator(sphere), Budget(evaluations=500), backend=backend)
    assert isinstance(result, RunResult) and result.evaluations_used == 500 and result.run_dir is None and result.wall_time > 0
    assert result.best is not None and result.pareto_front == (result.best,)
    assert result.trace[-1][1] == result.best.objectives["value"] and result.trace[0][0] == 1
    assert [e for e, _ in result.trace] == sorted(e for e, _ in result.trace) and len({e for e, _ in result.trace}) == len(result.trace)
    values = [v for _, v in result.trace]
    assert values == sorted(values, reverse=True) and len(set(values)) == len(values)  # strictly improving


def test_best_is_the_minimum_over_every_evaluation(backend: Backend):
    seen = []

    def fn(genome):
        seen.append(sphere(genome))
        return seen[-1]

    result = run_quietly(RandomSearch(), FunctionEvaluator(fn), Budget(evaluations=300), backend=backend)
    assert result.best is not None and result.best.objectives["value"] == min(seen)


def test_constraints_decide_the_best_candidate_of_a_run():
    evaluator = FunctionEvaluator(lambda g: Result({"loss": g}, {"cpa": max(0.0, 5.0 - g)}))  # feasible only from 5 up
    result = run_quietly(
        ScriptedStrategy(genomes=[1.0, 2.0, 9.0, 6.0, 5.5, 0.0]),
        evaluator,
        Budget(evaluations=6),
        objectives=[LOSS],
        constraints=["cpa"],
        batch_size=3,
    )
    assert result.best is not None and result.best.objectives["loss"] == 5.5  # the lowest loss among the feasible ones
    assert [e.objectives["loss"] for e in result.pareto_front] == [5.5]


def test_a_two_objective_run_has_a_front_and_no_best():
    evaluator = VectorisedEvaluator(lambda X: X[:, :2])
    result = run_quietly(RandomSearch(), evaluator, Budget(evaluations=400), objectives=[Objective("a"), Objective("b", "maximise")])
    assert result.best is None and result.trace == () and len(result.pareto_front) >= 2
    points = [(e.objectives["a"], -e.objectives["b"]) for e in result.pareto_front]
    for p in points:
        assert not any(q != p and q[0] <= p[0] and q[1] <= p[1] for q in points)  # nobody in the front dominates another


VALUE = (Objective("value"),)


def run_quietly(strategy, evaluator, budget, *, objectives=VALUE, constraints=(), backend=None, batch_size=32, seed=3):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RecordingDisabledWarning)
        return run(
            strategy=strategy, evaluator=evaluator, space=Box(-5.0, 5.0, dim=3), objectives=objectives, constraints=constraints,
            budget=budget, seed=seed, backend=backend, batch_size=batch_size,
        )  # fmt: skip
