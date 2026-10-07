import math

import pytest

from auxein.driver import Budget


def test_budget_fields_and_description():
    budget = Budget(evaluations=100, wall_time=2.5, cost={"tokens": 1000.0})
    assert budget.evaluations == 100 and budget.wall_time == 2.5 and dict(budget.cost) == {"tokens": 1000.0}
    assert budget.describe() == {"evaluations": 100, "wall_time": 2.5, "cost": {"tokens": 1000.0}}
    assert Budget(evaluations=5).describe() == {"evaluations": 5, "wall_time": None, "cost": {}}


def test_at_least_one_limit_is_required():
    with pytest.raises(ValueError, match="at least one limit"):
        Budget()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"evaluations": 0}, "at least 1"),
        ({"evaluations": -5}, "at least 1"),
        ({"wall_time": 0.0}, "positive"),
        ({"wall_time": -1.0}, "positive"),
        ({"wall_time": math.inf}, "positive"),
        ({"wall_time": math.nan}, "positive"),
        ({"cost": {"tokens": 0.0}}, "cost unit 'tokens'"),
        ({"cost": {"tokens": math.inf}}, "cost unit 'tokens'"),
    ],
)
def test_limits_must_be_positive_and_finite(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        Budget(**kwargs)


def test_a_budget_is_immutable_and_copies_its_costs():
    costs = {"tokens": 10.0}
    budget = Budget(cost=costs)
    costs["tokens"] = 99.0
    assert budget.cost["tokens"] == 10.0
    with pytest.raises(TypeError):
        budget.cost["tokens"] = 1.0  # type: ignore[index]
    with pytest.raises(AttributeError):
        budget.evaluations = 3  # type: ignore[misc]
