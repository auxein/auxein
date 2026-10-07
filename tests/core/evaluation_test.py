import math
from types import MappingProxyType

import pytest

from auxein.core import ArtifactRef, Candidate, CandidateId, Cost, Evaluation, Objective, RawRef, Status


def candidate(i: int = 0) -> Candidate[list[float]]:
    return Candidate(CandidateId(i), [1.0, 2.0], (), "init", 0)


def test_candidate_fields_and_validation():
    c = Candidate(CandidateId(3), "genome", (CandidateId(1), CandidateId(2)), "crossover:arithmetic", 4)
    assert (c.id, c.genome, c.parents, c.origin, c.step) == (3, "genome", (1, 2), "crossover:arithmetic", 4)
    with pytest.raises(ValueError, match="origin"):
        Candidate(CandidateId(0), 1, (), "", 0)
    with pytest.raises(ValueError, match="step"):
        Candidate(CandidateId(0), 1, (), "init", -1)
    with pytest.raises(AttributeError):
        c.step = 5  # type: ignore[misc]


def test_objective_defaults_to_minimise_and_has_a_sign():
    assert Objective("loss").direction == "minimise"
    assert Objective("loss").sign == 1.0
    assert Objective("score", "maximise").sign == -1.0
    with pytest.raises(ValueError, match="non-empty"):
        Objective("")
    with pytest.raises(ValueError, match="direction"):
        Objective("x", "up")  # type: ignore[arg-type]


def test_status_values():
    assert {s.name for s in Status} == {"OK", "FAILED", "TIMEOUT"}
    assert Status("ok") is Status.OK


def test_cost_validation_and_immutability():
    assert Cost().wall_time == 0.0 and dict(Cost().units) == {}
    cost = Cost(1.5, {"tokens": 120.0, "money": 0.01})
    assert cost.units["tokens"] == 120.0
    with pytest.raises(TypeError):
        cost.units["tokens"] = 1.0  # type: ignore[index]
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="wall_time"):
            Cost(bad)
    with pytest.raises(ValueError, match="cost unit 'tokens'"):
        Cost(1.0, {"tokens": -3.0})
    with pytest.raises(ValueError, match="cost unit"):
        Cost(1.0, {"tokens": math.nan})
    with pytest.raises(ValueError, match="non-empty"):
        Cost(1.0, {"": 1.0})


def test_references_need_a_key():
    assert RawRef("raw/1").key == "raw/1" and ArtifactRef("a/1").key == "a/1"
    with pytest.raises(ValueError):
        RawRef("")
    with pytest.raises(ValueError):
        ArtifactRef("")


def test_a_valid_evaluation():
    e = Evaluation(
        candidate(),
        Status.OK,
        {"loss": -3.5, "time": 0.0},
        constraints={"cpa": 0.0, "speed": 2.5},
        descriptors={"mean_speed": 4.0},
        cost=Cost(0.2),
        raw=RawRef("r"),
    )
    assert e.objectives["loss"] == -3.5  # objective values may be any finite float, negative included
    assert e.error is None and e.raw == RawRef("r")


def test_evaluation_defaults():
    e = Evaluation(candidate(), Status.OK, {"loss": 1.0})
    assert dict(e.constraints) == {} and dict(e.descriptors) == {} and e.cost == Cost() and e.raw is None


def test_mappings_are_copied_and_read_only():
    objectives = {"loss": 1.0}
    e = Evaluation(candidate(), Status.OK, objectives)
    objectives["loss"] = 99.0
    assert e.objectives["loss"] == 1.0
    assert isinstance(e.objectives, MappingProxyType)
    with pytest.raises(TypeError):
        e.constraints["x"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        e.descriptors["x"] = 1.0  # type: ignore[index]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_ok_evaluations_need_finite_objectives(value: float):
    with pytest.raises(ValueError, match="objective 'loss'.*OK"):
        Evaluation(candidate(), Status.OK, {"loss": value})


@pytest.mark.parametrize("status", [Status.FAILED, Status.TIMEOUT])
@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_failed_evaluations_may_carry_non_finite_objectives(status: Status, value: float):
    e = Evaluation(candidate(), status, {"loss": value}, error="boom")
    assert e.status is status and e.error == "boom"


@pytest.mark.parametrize("violation", [-0.1, -1e-12, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("status", list(Status))
def test_constraint_violations_must_be_finite_and_non_negative_whatever_the_status(violation: float, status: Status):
    with pytest.raises(ValueError, match="constraint 'cpa'"):
        Evaluation(candidate(), status, {"loss": 1.0}, constraints={"cpa": violation})


def test_zero_and_positive_violations_are_valid():
    e = Evaluation(candidate(), Status.OK, {"loss": 1.0}, constraints={"a": 0.0, "b": 1e9})
    assert e.constraints["b"] == 1e9


def test_names_must_be_non_empty():
    with pytest.raises(ValueError, match="non-empty"):
        Evaluation(candidate(), Status.OK, {"": 1.0})
    with pytest.raises(ValueError, match="non-empty"):
        Evaluation(candidate(), Status.OK, {"loss": 1.0}, descriptors={"": 1.0})
