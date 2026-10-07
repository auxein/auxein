import numpy as np
import pytest

from benchmarks.objective import BudgetExhausted, CountingObjective, log_checkpoints
from benchmarks.problems import Sphere, make_problem


def test_log_checkpoints_are_log_spaced_and_include_first_and_last():
    for budget in (1, 2, 7, 100, 4000, 60000):
        points = log_checkpoints(budget)
        assert points[0] == 1 and points[-1] == budget
        assert points == sorted(set(points))
    points = log_checkpoints(100000)
    assert 90 <= len(points) <= 105  # about 20 per decade over 5 decades
    ratios = np.array(points[40:], dtype=float)
    ratios = ratios[1:] / ratios[:-1]
    assert ratios.max() < 1.15


def test_counts_exactly_and_never_exceeds_the_budget():
    problem = make_problem("sphere", 3, 0)
    objective = CountingObjective(problem, budget=10)
    x = np.zeros(3)
    for expected in range(1, 11):
        objective(x)
        assert objective.evals == expected
    for _ in range(3):
        with pytest.raises(BudgetExhausted):
            objective(x)
    assert objective.evals == 10
    assert objective.remaining == 0


def test_the_call_that_would_exceed_the_budget_is_not_evaluated():
    calls = []

    class Spy(type(make_problem("sphere", 2, 0))):
        def true_error(self, x):
            calls.append(1)
            return super().true_error(x)

    objective = CountingObjective(Spy(2, 0), budget=3)
    for _ in range(3):
        objective(np.zeros(2))
    with pytest.raises(BudgetExhausted):
        objective(np.zeros(2))
    assert len(calls) == 3


def test_trace_is_non_increasing_and_ends_at_the_last_evaluation():
    problem = make_problem("rastrigin", 5, 0)
    objective = CountingObjective(problem, budget=2000)
    rng = np.random.default_rng(0)
    with pytest.raises(BudgetExhausted):
        while True:
            objective(rng.uniform(-5, 5, 5))
    trace = objective.trace
    assert trace[0][0] == 1 and trace[-1][0] == 2000
    errors = [e for _, e in trace]
    assert all(a >= b for a, b in zip(errors, errors[1:]))
    assert trace[-1][1] == objective.best_error


def test_trace_ends_at_the_last_evaluation_when_a_run_stops_early():
    objective = CountingObjective(make_problem("sphere", 2, 0), budget=1000)
    rng = np.random.default_rng(0)
    for _ in range(37):
        objective(rng.uniform(-5, 5, 2))
    assert objective.trace[-1][0] == 37
    assert objective.trace[-1][1] == objective.best_error
    assert CountingObjective(make_problem("sphere", 2, 0), budget=5).trace == []


def test_best_so_far_tracks_the_true_error_at_each_checkpoint():
    problem = make_problem("sphere", 2, 0)
    objective = CountingObjective(problem, budget=50)
    rng = np.random.default_rng(1)
    best = np.inf
    best_at = {}
    for i in range(1, 51):
        x = rng.uniform(-5, 5, 2)
        best = min(best, problem.true_error(x))
        best_at[i] = best
        objective(x)
    for evals, error in objective.trace:
        assert error == best_at[evals]


def test_for_noisy_problems_the_algorithm_sees_noise_but_the_trace_is_noise_free():
    problem = make_problem("noisy_sphere", 3, 0)
    objective = CountingObjective(problem, budget=200)
    rng = np.random.default_rng(0)
    seen, true = [], []
    for _ in range(200):
        x = rng.uniform(-5, 5, 3)
        seen.append(objective(x))
        true.append(problem.true_error(x))
    assert not np.allclose(seen, true)
    assert objective.best_error == min(true)  # not the best noisy value
    assert objective.trace[-1][1] == min(true)


def test_first_hit_of_each_target_is_recorded_exactly():
    problem = Sphere(2, 0)
    objective = CountingObjective(problem, budget=100, targets=[1.0, 1e-3, 1e-12])
    optimum = problem.optimum
    for i in range(1, 101):
        objective(optimum + 2.0 ** (-i))
    assert objective.hits[1.0] == 1
    hit_1e3 = objective.hits[1e-3]
    assert hit_1e3 is not None and hit_1e3 < 10
    hit = objective.hits[1e-12]
    assert hit is not None and problem.true_error(optimum + 2.0 ** (-hit)) <= 1e-12
    assert problem.true_error(optimum + 2.0 ** (-(hit - 1))) > 1e-12
