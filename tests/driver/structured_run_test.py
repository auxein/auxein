"""Structured genomes through the driver: evolving sequences, every delivery and executor, recording, and resuming."""

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import auxein
from auxein.core import Objective
from auxein.driver import ConfigurationMismatchError
from auxein.environments import ScenarioSet
from auxein.recording import open_run
from auxein.spaces import SequenceSpace
from tests.driver.resume_test import comparable, same_result
from tests.support import pointmass as pm
from tests.support import sequences as sq
from tests.support.eventlog import event_log
from tests.support.reading import peek
from tests.support.resumable import settings

ROOT = Path(__file__).resolve().parents[2]


def arguments(strategy: Any, run_dir: Path | None = None, **over: Any) -> dict[str, Any]:
    settings_: dict[str, Any] = {
        "strategy": strategy,
        "evaluator": auxein.FunctionEvaluator(sq.distance),
        "space": sq.SPACE,
        "objectives": [Objective("distance")],
        "constraints": ["too_long"],
        "seed": 2,
        "batch_size": 20,
        "run_dir": run_dir,
    }
    settings_.update(over)
    if run_dir is None:
        warnings.simplefilter("ignore", auxein.RecordingDisabledWarning)
    return settings_


def structured(population: int = 20) -> auxein.StructuredGeneticAlgorithm[Any]:
    return auxein.StructuredGeneticAlgorithm(population_size=population, offspring_size=population)


def test_it_beats_random_search_at_the_same_budget_and_respects_the_length_constraint():
    ga, rs = [], []
    for seed in range(4):
        found = auxein.run(budget=auxein.Budget(evaluations=800), **arguments(structured(), seed=seed))
        random = auxein.run(budget=auxein.Budget(evaluations=800), **arguments(auxein.RandomSearch(), seed=seed))
        assert found.best is not None and found.best.constraints["too_long"] == 0.0
        ga.append(found.best.objectives["distance"])
        rs.append(random.best.objectives["distance"])  # type: ignore[union-attr]
    assert np.mean(ga) < np.mean(rs) - 1.0 and all(g < r for g, r in zip(ga, rs, strict=True))


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
@pytest.mark.parametrize(("executor", "concurrency"), [("inline", 1), ("thread", 4), ("process", 3)])
def test_the_run_is_the_same_with_every_delivery_and_executor(delivery: str, executor: str, concurrency: int):
    """Structured genomes pickle to worker processes and back, and deterministic mode makes the runs identical."""
    base = auxein.run(budget=auxein.Budget(evaluations=120), delivery=delivery, **arguments(structured(12)))
    other = auxein.run(
        budget=auxein.Budget(evaluations=120), delivery=delivery, concurrency=concurrency, executor=executor, **arguments(structured(12))
    )
    same_result(base, other)
    assert base.best is not None and other.best is not None and base.best.candidate.genome == other.best.candidate.genome


def test_structured_genomes_are_recorded_as_canonical_json_and_decode_with_the_codec(tmp_path: Path):
    result = auxein.run(budget=auxein.Budget(evaluations=60), **arguments(structured(12), tmp_path / "r"))
    db = sqlite3.connect(tmp_path / "r" / "events.sqlite")
    kind, data = db.execute("SELECT genome_kind, genome FROM candidates WHERE id = 5").fetchone()
    db.close()
    assert kind == "json" and data == json.dumps(json.loads(data), separators=(",", ":"), ensure_ascii=False).encode()
    with open_run(tmp_path / "r") as plain, open_run(tmp_path / "r", codec=sq.SPACE.codec) as decoded:
        assert isinstance(next(iter(plain.evaluations())).genome, list)  # without the codec: plain JSON
        genomes = {e.candidate_id: e.genome for e in decoded.evaluations()}
    assert isinstance(genomes[5], tuple) and all(sq.SPACE.contains(g) for g in genomes.values())
    assert result.best is not None and genomes[result.best.candidate.id] == result.best.candidate.genome


def test_the_space_is_described_in_the_metadata_and_a_changed_vocabulary_is_refused_on_resume(tmp_path: Path):
    auxein.run(budget=auxein.Budget(evaluations=40), **arguments(structured(12), tmp_path / "r"))
    assert peek(tmp_path / "r").metadata["problem"]["space"]["items"] == list(sq.VOCABULARY)  # type: ignore[index]
    changed = SequenceSpace((*sq.VOCABULARY, "z"), 2, 18)
    with pytest.raises(ConfigurationMismatchError, match="problem.space: recorded"):
        auxein.resume(budget=auxein.Budget(evaluations=80), **arguments(structured(12), tmp_path / "r", space=changed))


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_extending_a_structured_run_gives_the_run_a_larger_budget_would_have_made(tmp_path: Path, delivery: str):
    auxein.run(budget=auxein.Budget(evaluations=130), delivery=delivery, **arguments(structured(), tmp_path / "a"))
    extended = auxein.resume(budget=auxein.Budget(evaluations=300), delivery=delivery, **arguments(structured(), tmp_path / "a"))
    reference = auxein.run(budget=auxein.Budget(evaluations=300), delivery=delivery, **arguments(structured(), tmp_path / "b"))
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")
    same_result(extended, reference)


# --- structured genomes with the agent layer ---


@pytest.mark.filterwarnings("ignore::auxein.RecordingDisabledWarning")
def test_a_sequence_genome_becomes_an_agent_for_the_point_mass_through_the_episode_evaluator():
    scenarios = ScenarioSet.generate(pm.scenario_params, 4, seed=3)
    evolver = auxein.EpisodeEvaluator(sq.PlanDecoder(), pm.step_environment(), scenarios, pm.aggregator())
    common: dict[str, Any] = {
        "evaluator": evolver,
        "space": sq.PLAN_SPACE,
        "objectives": [Objective("error")],
        "constraints": ["overshoot"],
        "descriptors": ["success_rate"],
        "batch_size": 20,
    }
    found = auxein.run(strategy=structured(), budget=auxein.Budget(evaluations=400), seed=1, **common)  # type: ignore[arg-type]
    random = auxein.run(strategy=auxein.RandomSearch(), budget=auxein.Budget(evaluations=400), seed=1, **common)  # type: ignore[arg-type]
    assert found.best is not None and random.best is not None
    assert found.best.objectives["error"] <= random.best.objectives["error"]
    assert isinstance(found.best.candidate.genome, tuple) and set(found.best.candidate.genome) <= set(sq.FORCES)
    assert found.best.raw is not None  # the episodes behind it were recorded as for any episode run


# --- killing a structured run and resuming it ---


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
def test_a_killed_structured_run_resumes_to_the_identical_event_log(tmp_path: Path, config: dict[str, Any], kill_after: int):
    total = 220
    run_dir = tmp_path / "run"
    full = {"strategy": "structured", "evaluator": "sequence", "run_dir": str(run_dir), "evaluations": total, **config}
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
    _ = event_log, os
