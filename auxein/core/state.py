"""State dicts: how strategies, the driver and the random streams save their state (design doc §10.4).

A state dict is a dictionary with string keys whose values are JSON scalars, lists, dictionaries, or arrays of a
supported backend. There is no pickle. A checkpoint (design doc §10.4) saves arrays in a standard array format and everything
else as JSON, and loads the arrays with pickling disabled; see `auxein.recording.checkpoints`.
"""

from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from auxein.backend import Array, devices

JsonScalar: TypeAlias = None | bool | int | float | str
StateValue: TypeAlias = "JsonScalar | list[StateValue] | dict[str, StateValue] | Array"
StateDict: TypeAlias = dict[str, StateValue]
"""The state of a component: string keys, and values that are JSON scalars, lists, dicts or arrays."""


class StateDictError(ValueError):
    """A state dict contains a value that cannot be saved without pickle."""


def _is_supported_array(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return cast("npt.NDArray[Any]", value).dtype != np.dtype(object)  # object arrays can only be saved with pickle
    if devices.torch_installed():
        return isinstance(value, devices.import_torch().Tensor)
    return False


def _as_list(value: object) -> list[object] | None:
    return cast("list[object]", value) if isinstance(value, list) else None


def _as_dict(value: object) -> dict[object, object] | None:
    return cast("dict[object, object]", value) if isinstance(value, dict) else None


def validate_state_dict(state: object) -> None:
    """Raise `StateDictError` unless `state` is a valid state dict.

    Accepted values are None, bool, int, float and str; lists; dicts with string keys; and numpy arrays (not of dtype
    object) or torch tensors. Tuples, sets, numpy scalars and every other object are rejected, so that a state
    survives a JSON round trip unchanged. The error message gives the path to the offending value.
    """
    if _as_dict(state) is None:
        raise StateDictError(f"a state dict must be a dict, got {type(state).__name__}")
    _check(state, "state", set())


def _check(value: object, path: str, active: set[int]) -> None:
    if value is None or isinstance(value, bool | int | float | str):
        return
    if _is_supported_array(value):
        return
    items = _as_list(value)
    mapping = _as_dict(value)
    if items is None and mapping is None:
        raise StateDictError(f"{path}: {type(value).__name__} is not a JSON scalar, list, dict or supported array")
    if id(value) in active:
        raise StateDictError(f"{path}: contains itself")
    active.add(id(value))
    try:
        if items is not None:
            for i, item in enumerate(items):
                _check(item, f"{path}[{i}]", active)
        elif mapping is not None:
            for key, item in mapping.items():
                if not isinstance(key, str):
                    raise StateDictError(f"{path}: keys must be strings, got {key!r} ({type(key).__name__})")
                _check(item, f"{path}[{key!r}]", active)
    finally:
        active.discard(id(value))
