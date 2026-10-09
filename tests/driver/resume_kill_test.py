"""Runs killed with SIGKILL at varied points, then resumed: the log must equal an uninterrupted run's (design doc §10.4, §11.4)."""

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

import auxein
from tests.driver.resume_test import comparable
from tests.support.fixtures import integration_backend
from tests.support.resumable import backend_config, settings

pytestmark = pytest.mark.usefixtures("use_corner_backend")

ROOT = Path(__file__).resolve().parents[2]
EVALUATIONS = 240

# (strategy, delivery, executor, concurrency): inline, thread and process executors, both deliveries, both strategies
CONFIGURATIONS = [
    ("random", "generation", "inline", 1),
    ("ga", "generation", "thread", 4),
    ("random", "generation", "process", 4),
    ("ga", "steady_state", "inline", 1),
    ("random", "steady_state", "thread", 4),
    ("ga", "steady_state", "process", 4),
]
# before the first checkpoint, between checkpoints (every 40 evaluations), and close to the end
KILL_AFTER = [3, 100, 210]


def recorded(run_dir: Path) -> int:
    try:
        db = sqlite3.connect(f"file:{run_dir / 'events.sqlite'}?mode=ro", uri=True)
        try:
            return int(db.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0])
        finally:
            db.close()
    except sqlite3.Error:
        return -1


def launch(config: dict[str, object], calls: Path | None) -> subprocess.Popen[str]:
    env = dict(os.environ)
    if calls is not None:
        env["AUXEIN_CALL_LOG"] = str(calls)
    return subprocess.Popen(
        [sys.executable, "-m", "tests.support.resume_cli", json.dumps(config)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


class References:
    """One uninterrupted run per backend, strategy and delivery: determinism makes it the reference whatever the executor."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._runs: dict[tuple[str, str, str, str], Path] = {}

    def __call__(self, strategy: str, delivery: str) -> Path:
        backend = integration_backend()
        key = (backend.name, backend.precision, strategy, delivery)
        if key not in self._runs:
            run_dir = self._directory / "-".join(key)
            config = {"strategy": strategy, "delivery": delivery, "run_dir": str(run_dir), "backend": backend_config(backend)}
            auxein.run(budget=auxein.Budget(evaluations=EVALUATIONS), **settings(config))
            self._runs[key] = run_dir
        return self._runs[key]


@pytest.fixture(scope="module")
def references(tmp_path_factory: pytest.TempPathFactory) -> References:
    return References(tmp_path_factory.mktemp("references"))


@pytest.mark.parametrize("kill_after", KILL_AFTER)
@pytest.mark.parametrize(("strategy", "delivery", "executor", "concurrency"), CONFIGURATIONS)
def test_a_killed_run_resumes_to_the_log_of_an_uninterrupted_run_and_repeats_no_evaluation(
    tmp_path: Path,
    references: References,
    strategy: str,
    delivery: str,
    executor: str,
    concurrency: int,
    kill_after: int,
):
    run_dir = tmp_path / "run"
    config = {"strategy": strategy, "delivery": delivery, "executor": executor, "concurrency": concurrency, "run_dir": str(run_dir), "backend": backend_config(integration_backend())}
    victim = launch({**config, "evaluations": EVALUATIONS}, None)
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
    done = recorded(run_dir)
    assert done >= 0, "the run was killed before it had created its recording"

    calls = tmp_path / "calls"
    survivor = launch({**config, "evaluations": EVALUATIONS, "resume": True}, calls)
    out, err = survivor.communicate(timeout=240)
    assert survivor.returncode == 0, err[-3000:]
    assert json.loads(out.strip().splitlines()[-1]) == {"evaluations": EVALUATIONS, "stop": "budget:evaluations"}

    assert len(calls.read_text().split()) == EVALUATIONS - done if calls.exists() else done == EVALUATIONS  # only what was not recorded
    assert comparable(run_dir) == comparable(references(strategy, delivery))
    assert not (run_dir / "writer.lock").exists()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert [s["status"] for s in metadata["sessions"]] == ["running", "completed"]  # the killed session never ended
    assert metadata["status"] == "completed" and metadata["summary"]["evaluations_used"] == EVALUATIONS


def test_a_run_killed_twice_still_resumes_to_the_same_log(tmp_path: Path, references: References):
    run_dir = tmp_path / "run"
    config = {"strategy": "ga", "delivery": "steady_state", "executor": "thread", "concurrency": 4, "run_dir": str(run_dir), "backend": backend_config(integration_backend())}
    for resume, kill_after in ((False, 30), (True, 120)):
        process = launch({**config, "evaluations": EVALUATIONS, "resume": resume}, None)
        deadline = time.monotonic() + 120
        while process.poll() is None and recorded(run_dir) < kill_after and time.monotonic() < deadline:
            time.sleep(0.003)
        process.send_signal(signal.SIGKILL)
        process.wait(timeout=30)
    final = launch({**config, "evaluations": EVALUATIONS, "resume": True}, None)
    out, err = final.communicate(timeout=240)
    assert final.returncode == 0, err[-3000:]
    assert comparable(run_dir) == comparable(references("ga", "steady_state"))
    assert [s["status"] for s in json.loads((run_dir / "metadata.json").read_text())["sessions"]] == ["running", "running", "completed"]
