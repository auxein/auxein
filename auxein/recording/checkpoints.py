"""Checkpoint files (design doc §10.4): a state dict on disk, without pickle.

One directory per checkpoint, `checkpoints/ckpt-<event seq>/`, holding

    state.json   everything that is not an array: JSON, with a reference where an array was
    arrays.npz   the arrays (numpy, or torch tensors copied through numpy), loaded with pickling disabled

A checkpoint is written to a temporary directory, flushed to disk and renamed into place, so a crash while writing leaves
the previous checkpoints untouched and the half-written one is never mistaken for a real one. The recorder registers it
in the `checkpoints` table afterwards and deletes the oldest beyond the number to keep.
"""

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from auxein.backend import Backend, devices, is_array
from auxein.core import StateDict, validate_state_dict

CHECKPOINT_FORMAT = 1
"""Bumped when the layout of a checkpoint changes; a checkpoint of another format is refused."""

_ARRAY = "__array__"
_TORCH = "__torch__"
STATE_FILE = "state.json"
ARRAYS_FILE = "arrays.npz"


class CheckpointError(RuntimeError):
    """A checkpoint cannot be read: missing files, a different format, or damaged contents."""


def directory_name(event_seq: int) -> str:
    return f"ckpt-{event_seq}"


def _to_host(value: object) -> tuple[npt.NDArray[Any], bool]:
    """A supported array as a numpy array, and whether it was a torch tensor."""
    if isinstance(value, np.ndarray):
        return cast("npt.NDArray[Any]", value), False
    return Backend().to_numpy(value), True  # type: ignore[arg-type]


def _encode(value: object, arrays: dict[str, npt.NDArray[Any]]) -> object:
    """Replace every array in a (validated) state by a reference into `arrays`."""
    if is_array(value):
        host, torch = _to_host(value)
        key = f"a{len(arrays)}"
        arrays[key] = np.ascontiguousarray(host)
        return {_ARRAY: key, _TORCH: torch}
    if isinstance(value, list):
        return [_encode(item, arrays) for item in cast("list[object]", value)]
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        if _ARRAY in mapping:
            raise ValueError(f"{_ARRAY!r} is reserved by the checkpoint format and cannot be a key of a state dict")
        return {key: _encode(item, arrays) for key, item in mapping.items()}
    return value


def _decode(value: object, arrays: Mapping[str, npt.NDArray[Any]], backend: Backend) -> object:
    if isinstance(value, list):
        return [_decode(item, arrays, backend) for item in cast("list[object]", value)]
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        if _ARRAY in mapping:
            array = arrays[cast("str", mapping[_ARRAY])]
            if mapping.get(_TORCH):
                torch = devices.import_torch()
                return torch.as_tensor(array, device=backend.device)
            return array
        return {key: _decode(item, arrays, backend) for key, item in mapping.items()}
    return value


def _flush_directory(path: Path) -> None:
    """Make a rename or file creation in `path` durable (not possible everywhere, e.g. on Windows)."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write(checkpoints_dir: Path, event_seq: int, state: StateDict) -> Path:
    """Write `state` as the checkpoint for `event_seq` and return its directory.

    The state must be a valid state dict (`validate_state_dict`). Writing the same sequence number again leaves the
    existing checkpoint alone: it describes the same moment.
    """
    validate_state_dict(state)
    final = checkpoints_dir / directory_name(event_seq)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if final.exists():
        return final
    arrays: dict[str, npt.NDArray[Any]] = {}
    document = {"format": CHECKPOINT_FORMAT, "event_seq": event_seq, "state": _encode(state, arrays)}
    temporary = checkpoints_dir / f".tmp-{event_seq}-{os.getpid()}"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir()
    try:
        with open(temporary / ARRAYS_FILE, "wb") as handle:
            np.savez(handle, **cast("dict[str, Any]", arrays))  # pyright: ignore[reportUnknownMemberType]
            handle.flush()
            os.fsync(handle.fileno())
        with open(temporary / STATE_FILE, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _flush_directory(temporary)
        os.replace(temporary, final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _flush_directory(checkpoints_dir)
    return final


def read(directory: Path, backend: Backend) -> tuple[int, StateDict]:
    """Read a checkpoint: its event sequence number and its state, with arrays restored (torch tensors on the run's device).

    Arrays are loaded with pickling disabled, so a checkpoint can never run code.
    """
    try:
        document = json.loads((directory / STATE_FILE).read_text(encoding="utf-8"))
        if document.get("format") != CHECKPOINT_FORMAT:
            raise CheckpointError(
                f"{directory} has checkpoint format {document.get('format')!r}; this version of Auxein reads format {CHECKPOINT_FORMAT}"
            )
        with np.load(directory / ARRAYS_FILE, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        state = cast("StateDict", _decode(document["state"], arrays, backend))
        return int(document["event_seq"]), state
    except CheckpointError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CheckpointError(f"the checkpoint {directory} cannot be read: {error}") from error


def remove(directory: Path) -> None:
    shutil.rmtree(directory, ignore_errors=True)


def clear_temporary(checkpoints_dir: Path) -> None:
    """Delete what a crash left half-written."""
    if checkpoints_dir.is_dir():
        for path in checkpoints_dir.glob(".tmp-*"):
            shutil.rmtree(path, ignore_errors=True)
