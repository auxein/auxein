import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import StateDictError, validate_state_dict


def test_valid_state_dicts_are_accepted(backend: Backend):
    validate_state_dict({})
    validate_state_dict(
        {
            "none": None,
            "flag": True,
            "count": 3,
            "scale": 0.5,
            "inf": float("inf"),
            "name": "abc",
            "list": [1, 2.0, "x", None, [True]],
            "nested": {"a": {"b": [1, {"c": 2}]}},
            "array": backend.asarray([[1.0, 2.0]]),
            "ints": backend.asarray([1, 2], dtype=backend.int_dtype),
            "numpy": np.zeros(3),
            "list_of_arrays": [backend.asarray([1.0])],
        }
    )


@pytest.mark.parametrize(
    ("state", "path"),
    [
        ({"t": (1, 2)}, r"state\['t'\]: tuple"),
        ({"s": {1, 2}}, r"state\['s'\]: set"),
        ({"f": lambda: 1}, r"state\['f'\]: function"),
        ({"o": object()}, r"state\['o'\]: object"),
        ({"n": np.float64(1.0).item, "x": 1}, r"state\['n'\]"),
        ({"i": np.int64(3)}, r"state\['i'\]: int64"),
        ({"b": np.bool_(True)}, r"state\['b'\]: bool"),
        ({"obj": np.array([object()], dtype=object)}, r"state\['obj'\]: ndarray"),
        ({"deep": {"x": [1, [2, (3,)]]}}, r"state\['deep'\]\['x'\]\[1\]\[1\]: tuple"),
        ({"list": [1, 2, b"bytes"]}, r"state\['list'\]\[2\]: bytes"),
        ({1: "int key"}, "keys must be strings"),
        ({"d": {2: 1}}, r"state\['d'\]: keys must be strings"),
        ({"d": {None: 1}}, "keys must be strings"),
        ({"c": complex(1, 2)}, r"state\['c'\]: complex"),
    ],
)
def test_invalid_values_are_rejected_with_their_path(state: dict, path: str):
    with pytest.raises(StateDictError, match=path):
        validate_state_dict(state)


@pytest.mark.parametrize("state", [None, [1, 2], "x", 3, (1,), np.zeros(2)])
def test_a_state_dict_must_be_a_dict(state: object):
    with pytest.raises(StateDictError, match="must be a dict"):
        validate_state_dict(state)


def test_self_references_are_rejected_without_recursing_forever():
    cyclic: dict = {"a": 1}
    cyclic["self"] = cyclic
    with pytest.raises(StateDictError, match="contains itself"):
        validate_state_dict(cyclic)
    items: list = []
    items.append(items)
    with pytest.raises(StateDictError, match="contains itself"):
        validate_state_dict({"l": items})


def test_a_shared_value_that_is_not_a_cycle_is_fine():
    shared = [1, 2]
    validate_state_dict({"a": shared, "b": shared, "c": [shared, shared]})


def test_errors_are_value_errors():
    assert issubclass(StateDictError, ValueError)


def test_json_round_trip_of_a_valid_state_is_lossless():
    import json

    state = {"a": [1, 2.5, "x", None, True], "b": {"c": {"d": [0]}}}
    validate_state_dict(state)
    assert json.loads(json.dumps(state)) == state
