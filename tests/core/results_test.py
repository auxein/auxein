import math
from types import MappingProxyType

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import BatchResult, Result


def test_a_valid_result():
    r = Result({"loss": 1}, {"cpa": 0.0}, {"speed": 4}, {"tokens": 120})
    assert dict(r.objectives) == {"loss": 1.0} and isinstance(r.objectives["loss"], float)
    assert dict(r.constraints) == {"cpa": 0.0} and dict(r.descriptors) == {"speed": 4.0} and dict(r.cost) == {"tokens": 120.0}


def test_result_values_may_be_numpy_scalars_and_zero_dimensional_arrays(backend: Backend):
    r = Result({"a": np.float32(1.5), "b": backend.asarray(2.5), "c": np.int64(3)})
    assert [r.objectives[n] for n in "abc"] == [1.5, 2.5, 3.0]


def test_results_are_copied_and_read_only():
    objectives = {"loss": 1.0}
    r = Result(objectives)
    objectives["loss"] = 9.0
    assert r.objectives["loss"] == 1.0 and isinstance(r.objectives, MappingProxyType)
    with pytest.raises(TypeError):
        r.constraints["x"] = 1.0  # type: ignore[index]
    with pytest.raises(AttributeError):
        r.objectives = {}  # type: ignore[misc]


def test_result_needs_an_objective_and_valid_names():
    with pytest.raises(ValueError, match="at least one objective"):
        Result({})
    with pytest.raises(ValueError, match="non-empty"):
        Result({"": 1.0})
    with pytest.raises(ValueError, match="non-empty"):
        Result({"a": 1.0}, descriptors={"": 1.0})


@pytest.mark.parametrize("violation", [-0.5, math.nan, math.inf])
def test_result_constraints_are_finite_and_non_negative(violation: float):
    with pytest.raises(ValueError, match="constraint 'cpa'"):
        Result({"a": 1.0}, {"cpa": violation})


@pytest.mark.parametrize("amount", [-1.0, math.nan, math.inf])
def test_result_cost_units_are_finite_and_non_negative(amount: float):
    with pytest.raises(ValueError, match="cost unit 'tokens'"):
        Result({"a": 1.0}, cost={"tokens": amount})


def test_result_values_must_be_numbers():
    with pytest.raises(TypeError, match="objective 'a' must be a number, got str"):
        Result({"a": "high"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="constraint 'c' must be a number"):
        Result({"a": 1.0}, {"c": None})  # type: ignore[dict-item]


def test_non_finite_objectives_are_a_normalisation_error_not_a_result_error():
    assert math.isnan(Result({"a": math.nan}).objectives["a"])  # the evaluator names the candidate (see normalise tests)


def test_a_valid_batch_result(backend: Backend):
    r = BatchResult(
        {"a": backend.asarray([1.0, 2.0, 3.0]), "b": [4, 5, 6]},
        {"cpa": backend.asarray([0.0, 1.0, 0.0])},
        {"speed": [1.0, 2.0, 3.0]},
        {"tokens": [10, 20, 30]},
    )
    assert r.size == 3
    for mapping in (r.objectives, r.constraints, r.descriptors, r.cost):
        for column in mapping.values():
            assert isinstance(column, np.ndarray) and column.dtype == np.float64 and column.shape == (3,)
    assert r.objectives["b"].tolist() == [4.0, 5.0, 6.0]


def test_batch_result_columns_are_copied_to_the_host():
    source = np.array([1.0, 2.0])
    r = BatchResult({"a": source})
    source[0] = 99.0
    assert r.objectives["a"][0] == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"objectives": {}}, "at least one objective"),
        ({"objectives": {"": [1.0]}}, "non-empty"),
        ({"objectives": {"a": [[1.0, 2.0]]}}, "1-D"),
        ({"objectives": {"a": 1.0}}, "1-D"),
        ({"objectives": {"a": [1.0, 2.0]}, "constraints": {"c": [0.0]}}, "same length"),
        ({"objectives": {"a": [1.0, 2.0], "b": [1.0]}}, "same length"),
        ({"objectives": {"a": [1.0]}, "cost": {"t": [1.0, 2.0]}}, "same length"),
        ({"objectives": {"a": [1.0, 2.0]}, "constraints": {"c": [0.0, -1.0]}}, "constraint 'c'"),
        ({"objectives": {"a": [1.0, 2.0]}, "constraints": {"c": [0.0, math.nan]}}, "constraint 'c'"),
        ({"objectives": {"a": [1.0, 2.0]}, "cost": {"t": [1.0, -1.0]}}, "cost unit 't'"),
    ],
)
def test_batch_result_validation(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        BatchResult(**kwargs)
