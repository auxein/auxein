"""`NSGA2` through the driver: recording, extension, and killing and resuming (design doc §3.3, §10.4)."""

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
from auxein.recording import open_run
from tests.driver.resume_test import comparable, same_result
from tests.support import multiobjective as mo
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
        "strategy": auxein.NSGA2(population_size=20, offspring_size=20),
        "evaluator": auxein.VectorisedEvaluator(lambda X: mo.zdt1_batch(X, backend)),
        "space": mo.ZDT_SPACE,
        "objectives": [mo.F1, mo.F2],
        "seed": 5,
        "batch_size": 20,
        "backend": backend,
        "run_dir": run_dir,
    }
    settings_.update(over)
    return settings_


def test_all_the_objectives_are_recorded_and_the_front_is_the_pareto_archive(tmp_path: Path):
    result = auxein.run(budget=auxein.Budget(evaluations=400), **arguments(tmp_path / "r"))
    assert result.best is None and len(result.pareto_front) > 3
    with open_run(tmp_path / "r") as run:
        rows = list(run.evaluations())
    assert len(rows) == 400 and all(set(r.objectives) == {"f1", "f2"} for r in rows)
    recorded = np.array([[r.objectives["f1"], r.objectives["f2"]] for r in rows])
    front = np.array([[e.objectives["f1"], e.objectives["f2"]] for e in result.pareto_front])
    # nothing recorded dominates a member of the front
    for point in front:
        assert not ((recorded <= point).all(axis=1) & (recorded < point).any(axis=1)).any()


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_extending_a_run_gives_the_run_a_larger_budget_would_have_made(tmp_path: Path, delivery: str):
    """Replay regenerates the candidates from the seed, so the ranking order of the population must be restored exactly."""
    auxein.run(budget=auxein.Budget(evaluations=130), delivery=delivery, **arguments(tmp_path / "a"))
    extended = auxein.resume(budget=auxein.Budget(evaluations=400), delivery=delivery, **arguments(tmp_path / "a"))
    reference = auxein.run(budget=auxein.Budget(evaluations=400), delivery=delivery, **arguments(tmp_path / "b"))
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")
    same_result(extended, reference)


def test_replay_without_checkpoints_regenerates_the_run_exactly(tmp_path: Path):
    auxein.run(budget=auxein.Budget(evaluations=160), keep_checkpoints=0, **arguments(tmp_path / "a"))
    auxein.resume(budget=auxein.Budget(evaluations=300), keep_checkpoints=0, **arguments(tmp_path / "a"))
    auxein.run(budget=auxein.Budget(evaluations=300), keep_checkpoints=0, **arguments(tmp_path / "b"))
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")


def test_the_event_log_does_not_depend_on_the_executor_in_deterministic_mode(tmp_path: Path):
    """Steady-state results depend on how many children are told at once, but in deterministic mode that is fixed by the seed
    and the batch size, whatever the concurrency."""
    backend = integration_backend()

    def fn(genome: Any) -> auxein.Result:
        f1, f2 = mo.zdt1(backend.to_numpy(genome)[None, :])[0]
        return auxein.Result({"f1": float(f1), "f2": float(f2)})

    budget = auxein.Budget(evaluations=240)
    options = {"evaluator": auxein.FunctionEvaluator(fn), "delivery": "steady_state"}
    auxein.run(budget=budget, concurrency=1, executor="inline", **arguments(tmp_path / "a", **options))
    auxein.run(budget=budget, concurrency=4, executor="thread", **arguments(tmp_path / "b", **options))
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")


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
def test_a_killed_run_resumes_to_the_identical_event_log(tmp_path: Path, config: dict[str, Any], kill_after: int):
    total = 220
    run_dir = tmp_path / "run"
    full = {
        "strategy": "nsga2",
        "evaluator": "biobjective",
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
