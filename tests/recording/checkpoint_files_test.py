"""The checkpoint format: JSON plus arrays, no pickle, written atomically, kept newest-first (design doc §10.4)."""

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.backend.devices import torch_installed
from auxein.core import StateDict, StateDictError
from auxein.recording import SQLiteRecorder, checkpoints
from auxein.recording.checkpoints import CheckpointError
from tests.support.reading import peek


def state_with_arrays() -> StateDict:
    return {
        "scalars": {"n": 3, "x": 0.1, "flag": True, "none": None, "text": "t", "big": 2**100, "nan": float("nan"), "inf": float("inf")},
        "list": [1, [2, 3], {"a": np.arange(4, dtype=np.int64)}],
        "genomes": np.arange(12, dtype=np.float32).reshape(3, 4) / 7,
        "empty": np.zeros((0, 4)),
        "bits": np.array([True, False]),
    }


def test_a_state_round_trips_with_its_arrays_exactly(tmp_path: Path):
    state = state_with_arrays()
    directory = checkpoints.write(tmp_path / "checkpoints", 42, state)
    assert directory.name == "ckpt-42" and sorted(p.name for p in directory.iterdir()) == ["arrays.npz", "state.json"]
    seq, back = checkpoints.read(directory, Backend())
    assert seq == 42
    assert back["scalars"]["n"] == 3 and back["scalars"]["big"] == 2**100 and back["scalars"]["inf"] == float("inf")  # type: ignore[index]
    assert np.isnan(back["scalars"]["nan"])  # type: ignore[index]
    assert back["list"][:2] == [1, [2, 3]]  # type: ignore[index]
    for name in ("genomes", "empty", "bits"):
        np.testing.assert_array_equal(back[name], state[name])  # type: ignore[arg-type]
        assert back[name].dtype == state[name].dtype  # type: ignore[union-attr]
    np.testing.assert_array_equal(back["list"][2]["a"], np.arange(4))  # type: ignore[index]
    assert not json.loads((directory / "state.json").read_text())["state"]["genomes"].get("data")  # the arrays are not in the JSON


def test_there_is_no_pickle_anywhere(tmp_path: Path):
    with pytest.raises(StateDictError):
        checkpoints.write(tmp_path / "c", 1, {"a": np.array([object()], dtype=object)})
    with pytest.raises(StateDictError):
        checkpoints.write(tmp_path / "c", 1, {"a": (1, 2)})  # type: ignore[dict-item]
    directory = checkpoints.write(tmp_path / "c", 2, {"a": np.zeros(2)})
    np.savez(
        directory / "arrays.npz", a0=np.array([{"x": 1}], dtype=object), allow_pickle=True
    )  # a file that could only be read with pickle
    with pytest.raises(CheckpointError, match="cannot be read"):
        checkpoints.read(directory, Backend())
    assert "allow_pickle=False" in Path(checkpoints.__file__).read_text()


def test_a_key_the_format_reserves_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="reserved"):
        checkpoints.write(tmp_path / "c", 1, {"a": {"__array__": "a0"}})


@pytest.mark.skipif(not torch_installed(), reason="torch is not installed")
def test_torch_tensors_go_through_numpy_and_come_back_as_tensors_on_the_runs_device(tmp_path: Path):
    import torch

    state: StateDict = {"w": torch.arange(6, dtype=torch.float32).reshape(2, 3) / 3, "i": torch.tensor([1, 2, 3])}
    directory = checkpoints.write(tmp_path / "c", 5, state)
    _, back = checkpoints.read(directory, Backend("torch", "cpu", "float32"))
    assert isinstance(back["w"], torch.Tensor) and back["w"].dtype == torch.float32 and back["w"].device.type == "cpu"
    assert torch.equal(back["w"], state["w"]) and torch.equal(back["i"], state["i"])  # type: ignore[arg-type]
    _, as_numpy = checkpoints.read(directory, Backend())
    assert isinstance(as_numpy["w"], torch.Tensor)  # the checkpoint remembers what it was saved from


def test_a_crash_while_writing_leaves_the_previous_checkpoint_usable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    directory = tmp_path / "checkpoints"
    checkpoints.write(directory, 1, {"a": np.arange(3)})

    def explode(*_: object) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(OSError, match="disk gone"):
        checkpoints.write(directory, 2, {"a": np.arange(5)})
    monkeypatch.undo()
    assert sorted(p.name for p in directory.iterdir()) == ["ckpt-1"]  # nothing of the failed one is left, not even a temporary directory
    seq, state = checkpoints.read(directory / "ckpt-1", Backend())
    assert seq == 1 and list(state["a"]) == [0, 1, 2]  # type: ignore[arg-type]


def test_a_half_written_temporary_directory_is_never_mistaken_for_a_checkpoint_and_is_cleaned(tmp_path: Path):
    directory = tmp_path / "checkpoints"
    checkpoints.write(directory, 1, {"a": np.arange(3)})
    leftover = directory / ".tmp-2-99999"
    leftover.mkdir()
    (leftover / "state.json").write_text('{"format": 1, "event_se')
    checkpoints.clear_temporary(directory)
    assert not leftover.exists() and (directory / "ckpt-1").exists()


def test_writing_the_same_sequence_again_keeps_the_existing_checkpoint(tmp_path: Path):
    first = checkpoints.write(tmp_path / "c", 7, {"a": np.arange(3)})
    again = checkpoints.write(tmp_path / "c", 7, {"a": np.arange(9)})
    assert first == again and checkpoints.read(first, Backend())[1]["a"].shape == (3,)  # type: ignore[union-attr]


def test_a_checkpoint_of_another_format_or_a_damaged_one_is_refused_with_a_clear_error(tmp_path: Path):
    directory = checkpoints.write(tmp_path / "c", 3, {"a": 1})
    document = json.loads((directory / "state.json").read_text())
    (directory / "state.json").write_text(json.dumps({**document, "format": 99}))
    with pytest.raises(CheckpointError, match="format 99"):
        checkpoints.read(directory, Backend())
    (directory / "state.json").write_text("{")
    with pytest.raises(CheckpointError, match="cannot be read"):
        checkpoints.read(directory, Backend())
    with pytest.raises(CheckpointError):
        checkpoints.read(tmp_path / "missing", Backend())


# --- through the recorder ---


def started(run_dir: Path) -> SQLiteRecorder:
    recorder = SQLiteRecorder(run_dir)
    recorder.on_start({"name": "t", "budget": {"evaluations": 10}})
    return recorder


def test_the_recorder_registers_checkpoints_and_keeps_only_the_newest(tmp_path: Path):
    recorder = started(tmp_path / "r")
    for used in (10, 20, 30, 40):
        recorder.on_tell(0, used)  # moves the event sequence on, so that each checkpoint has its own
        recorder.checkpoint({"used": used}, used, keep=2)
    recorder.on_end("completed", "budget:evaluations", {})
    rows = peek(tmp_path / "r").checkpoints()
    assert [c.evaluations_used for c in rows] == [30, 40]
    assert sorted(p.name for p in (tmp_path / "r" / "checkpoints").iterdir()) == sorted(Path(c.path).name for c in rows)
    assert all(c.event_seq > 0 and c.created_at > 0 for c in rows)
    assert checkpoints.read(tmp_path / "r" / rows[-1].path, Backend())[1] == {"used": 40}


def test_two_checkpoints_at_the_same_moment_are_one(tmp_path: Path):
    recorder = started(tmp_path / "r")
    recorder.on_tell(0, 1)
    recorder.checkpoint({"a": 1}, 1, keep=2)
    recorder.checkpoint({"a": 1}, 1, keep=2)
    recorder.on_end("completed", "x", {})
    assert len(peek(tmp_path / "r").checkpoints()) == 1


def test_the_checkpoints_table_records_what_the_prompt_asks(tmp_path: Path):
    recorder = started(tmp_path / "r")
    recorder.on_tell(0, 1)
    recorder.checkpoint({"a": 1}, 7, keep=2)
    recorder.on_end("completed", "x", {})
    db = sqlite3.connect(tmp_path / "r" / "events.sqlite")
    columns = [row[1] for row in db.execute("PRAGMA table_info(checkpoints)")]
    db.close()
    assert columns == ["id", "event_seq", "evaluations_used", "path", "created_at"]
