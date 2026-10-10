"""Mixed-space runs through the driver: recording and replay, extension, and killing and resuming (design doc §4.5, §10.4)."""

import json
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import auxein
from auxein.backend import Backend
from auxein.recording import open_run
from tests.driver.resume_test import comparable, same_result
from tests.support import mixed as mx
from tests.support.fixtures import integration_backend
from tests.support.reading import peek
from tests.support.resumable import backend_config, settings

ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.usefixtures("use_corner_backend"),
    pytest.mark.filterwarnings("ignore::auxein.RecordingDisabledWarning"),
    pytest.mark.filterwarnings("ignore::auxein.driver.errors.SteadyStateVectorisationWarning"),
]


def arguments(run_dir: Path | None, **over: Any) -> dict[str, Any]:
    backend = integration_backend()
    settings_: dict[str, Any] = {
        "strategy": auxein.GeneticAlgorithm(population_size=20, offspring_size=20),
        "evaluator": auxein.VectorisedEvaluator(lambda X: mx.evaluate(X, backend)),
        "space": mx.SPACE,
        "constraints": ["too_big"],
        "seed": 5,
        "batch_size": 20,
        "backend": backend,
        "run_dir": run_dir,
    }
    settings_.update(over)
    return settings_


def test_mixed_genomes_are_recorded_as_arrays_and_decode_with_the_space(tmp_path: Path):
    backend = integration_backend()
    result = auxein.run(budget=auxein.Budget(evaluations=300), **arguments(tmp_path / "r"))
    assert result.best is not None
    with open_run(tmp_path / "r") as run:
        rows = list(run.evaluations())
    assert len(rows) == 300
    genome = rows[0].genome
    assert isinstance(genome, np.ndarray) and genome.dtype == np.dtype(backend.precision) and genome.shape == (mx.SPACE.dim,)
    assert all(mx.SPACE.contains(row.genome) for row in rows)  # type: ignore[arg-type]
    best = min(rows, key=lambda r: (r.constraints["too_big"], r.objectives["value"]))
    assert best.candidate_id == result.best.candidate.id
    decoded = mx.SPACE.values(best.genome)  # type: ignore[arg-type]
    assert list(decoded) == list(mx.SPACE.names) and isinstance(decoded["mode"], str) and isinstance(decoded["k"], int)
    np.testing.assert_array_equal(best.genome, backend.to_numpy(result.best.candidate.genome))
    metadata = peek(tmp_path / "r").metadata
    assert metadata["problem"]["space"]["type"] == "MixedSpace"
    assert [d["name"] for d in metadata["problem"]["space"]["dimensions"]] == list(mx.SPACE.names)


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_extending_a_mixed_run_gives_the_run_a_larger_budget_would_have_made(tmp_path: Path, delivery: str):
    """Replay regenerates the recorded candidates from the seed and compares them byte for byte, so the mixed operators must be
    exactly reproducible, with their integer step sizes in the checkpoint."""
    auxein.run(budget=auxein.Budget(evaluations=130), delivery=delivery, **arguments(tmp_path / "a"))
    extended = auxein.resume(budget=auxein.Budget(evaluations=400), delivery=delivery, **arguments(tmp_path / "a"))
    reference = auxein.run(budget=auxein.Budget(evaluations=400), delivery=delivery, **arguments(tmp_path / "b"))
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")
    same_result(extended, reference)


def test_replay_without_checkpoints_regenerates_a_mixed_run_exactly(tmp_path: Path):
    auxein.run(budget=auxein.Budget(evaluations=160), keep_checkpoints=0, **arguments(tmp_path / "a"))
    auxein.resume(budget=auxein.Budget(evaluations=300), keep_checkpoints=0, **arguments(tmp_path / "a"))
    auxein.run(budget=auxein.Budget(evaluations=300), keep_checkpoints=0, **arguments(tmp_path / "b"))
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")


def test_a_changed_space_is_refused_on_resume(tmp_path: Path):
    from auxein.driver import ConfigurationMismatchError
    from auxein.spaces import Integer, MixedSpace

    auxein.run(budget=auxein.Budget(evaluations=60), **arguments(tmp_path / "r"))
    changed = MixedSpace({**mx.SPACE.dimensions, "k": Integer(0, 25)})
    with pytest.raises(ConfigurationMismatchError, match="space"):
        auxein.resume(budget=auxein.Budget(evaluations=120), **arguments(tmp_path / "r", space=changed))


def recorded(run_dir: Path) -> int:
    try:
        db = sqlite3.connect(f"file:{run_dir / 'events.sqlite'}?mode=ro", uri=True)
        try:
            return int(db.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0])
        finally:
            db.close()
    except sqlite3.Error:
        return -1


@pytest.mark.parametrize(
    ("config", "kill_after"),
    [
        ({"delivery": "generation", "executor": "inline", "concurrency": 1}, 70),
        ({"delivery": "steady_state", "executor": "thread", "concurrency": 4}, 90),
    ],
)
def test_a_killed_mixed_run_resumes_to_the_identical_event_log(tmp_path: Path, config: dict[str, Any], kill_after: int):
    total = 220
    run_dir = tmp_path / "run"
    full = {
        "strategy": "ga",
        "evaluator": "mixed",
        "run_dir": str(run_dir),
        "evaluations": total,
        "backend": backend_config(integration_backend()),
        **config,
    }
    victim = subprocess.Popen(
        [sys.executable, "-m", "tests.support.resume_cli", json.dumps(full)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 120
    try:
        while victim.poll() is None and recorded(run_dir) < kill_after and time.monotonic() < deadline:
            time.sleep(0.003)
        victim.send_signal(signal.SIGKILL)
        victim.wait(timeout=30)
    finally:
        if victim.poll() is None:
            victim.kill()
            victim.wait()
    survivor = subprocess.run(
        [sys.executable, "-m", "tests.support.resume_cli", json.dumps({**full, "resume": True})],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert survivor.returncode == 0, survivor.stderr[-3000:]
    reference = tmp_path / "reference"
    auxein.run(budget=auxein.Budget(evaluations=total), **settings({**full, "run_dir": str(reference)}))
    assert comparable(run_dir) == comparable(reference)
    assert peek(run_dir).metadata["status"] == "completed"
    _ = Backend
