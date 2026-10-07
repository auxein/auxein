"""Device strings and availability probes.

The probes are module-level functions that `Backend` looks up at call time, so that tests can replace them and
exercise every validation path without a GPU (or without torch).
"""

import importlib
import importlib.util
import re
from typing import Any

SUPPORTED_DEVICES = "'cpu', 'mps', 'cuda' or 'cuda:<index>'"
_DEVICE = re.compile(r"(?P<kind>cpu|mps|cuda)(?::(?P<index>0|[1-9][0-9]*))?")


def parse_device(device: str) -> tuple[str, int | None] | None:
    """Split a device string into (kind, index), or return None if it is not a supported device string.

    Only `cuda` takes an index: `cpu:0` and `mps:0` are not accepted.
    """
    match = _DEVICE.fullmatch(device)
    if match is None:
        return None
    kind, index = match["kind"], match["index"]
    if index is not None and kind != "cuda":
        return None
    return kind, None if index is None else int(index)


def torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def import_torch() -> Any:
    """Import torch lazily. It is optional, and typed as `Any` so that the new core type-checks without it."""
    return importlib.import_module("torch")


def cuda_available() -> bool:
    return bool(import_torch().cuda.is_available())


def cuda_device_count() -> int:
    return int(import_torch().cuda.device_count())


def mps_available() -> bool:
    return bool(import_torch().backends.mps.is_available())
