import pytest

from auxein.core import Objective, ProblemSpec
from auxein.spaces import Box

SPACE = Box(-1.0, 1.0, dim=2)


def test_a_valid_problem_spec():
    spec = ProblemSpec(SPACE, (Objective("loss"), Objective("time", "maximise")), ("cpa",), ("speed", "side"))
    assert spec.space is SPACE
    assert spec.objective_names == ("loss", "time")
    assert spec.constraints == ("cpa",) and spec.descriptors == ("speed", "side")


def test_constraints_and_descriptors_default_to_none():
    spec = ProblemSpec(SPACE, (Objective("loss"),))
    assert spec.constraints == () and spec.descriptors == ()


def test_sequences_are_stored_as_tuples():
    spec = ProblemSpec(SPACE, [Objective("loss")], ["a"], ["b"])  # type: ignore[arg-type]
    assert isinstance(spec.objectives, tuple) and spec.constraints == ("a",) and spec.descriptors == ("b",)


def test_at_least_one_objective_is_required():
    with pytest.raises(ValueError, match="at least one objective"):
        ProblemSpec(SPACE, ())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"objectives": (Objective("a"), Objective("a", "maximise"))}, r"objective names must be unique.*\['a'\]"),
        ({"objectives": (Objective("a"),), "constraints": ("c", "c")}, r"constraint names must be unique.*\['c'\]"),
        ({"objectives": (Objective("a"),), "descriptors": ("d", "e", "d")}, r"descriptor names must be unique.*\['d'\]"),
        ({"objectives": (Objective("a"),), "constraints": ("",)}, "constraint names must be non-empty"),
        ({"objectives": (Objective("a"),), "descriptors": ("x", "")}, "descriptor names must be non-empty"),
    ],
)
def test_names_must_be_unique_and_non_empty(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        ProblemSpec(SPACE, **kwargs)


def test_the_same_name_may_be_used_in_different_groups():
    spec = ProblemSpec(SPACE, (Objective("cost"),), ("cost",), ("cost",))
    assert spec.objective_names == ("cost",)


def test_problem_spec_is_frozen():
    spec = ProblemSpec(SPACE, (Objective("a"),))
    with pytest.raises(AttributeError):
        spec.constraints = ("x",)  # type: ignore[misc]
