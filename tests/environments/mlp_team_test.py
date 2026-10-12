"""`MLPPolicy` driving a batched world: a team of identical point masses, one shared network per candidate (`tests/support/team.py`).

The pattern of a batched multi-agent world: observations `(n, scenarios, agents, obs)`, actions `(n, scenarios, agents, outputs)`.
"""

import asyncio
import json
import os
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
from auxein.core import ArrayBatch, CandidateId, Objective, Status
from auxein.driver import evaluate_held_out
from auxein.environments import BatchPolicy
from auxein.recording import open_run
from tests.driver.resume_test import comparable, same_result
from tests.support import team
from tests.support.eventlog import event_log
from tests.support.fixtures import assert_on_backend, integration_backend
from tests.support.helpers import eval_context
from tests.support.reading import peek
from tests.support.resumable import backend_config, settings

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.usefixtures("use_corner_backend"), pytest.mark.filterwarnings("ignore::auxein.RecordingDisabledWarning")]


def evaluator(*, batched: bool = True, scenarios: int = 6, environment: Any = None) -> auxein.EpisodeEvaluator[Any]:
    decoder = team.POLICY if batched else team.PerEpisodeDecoder()
    return auxein.EpisodeEvaluator(decoder, environment or team.TeamEnvironment(), team.scenario_set(scenarios), team.aggregator())


def arguments(strategy: Any, evaluations: int, **over: Any) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "evaluator": evaluator(),
        "space": team.POLICY.space(),
        "objectives": [Objective("error")],
        "descriptors": ["success_rate"],
        "budget": auxein.Budget(evaluations=evaluations),
        "seed": 0,
        "batch_size": 30,
        "backend": integration_backend(),
        **over,
    }


def genetic() -> auxein.GeneticAlgorithm:
    return auxein.GeneticAlgorithm(population_size=30, offspring_size=30)


# --- the world: batched and per episode agree ---


GENOMES = np.random.default_rng(0).uniform(-1.0, 1.0, size=(4, team.POLICY.dim)) * np.asarray(team.POLICY.space().upper)


def measure(backend: Backend, batched: bool) -> np.ndarray:
    batch = ArrayBatch(backend.asarray(GENOMES), [CandidateId(i) for i in range(len(GENOMES))], 0, "random")
    candidate_evaluator = evaluator(batched=batched, scenarios=5)
    assert candidate_evaluator.batched is batched
    results = asyncio.run(candidate_evaluator.evaluate(batch, eval_context(team.problem(), backend, 0)))
    assert results.episodes is not None and all(e.status is Status.OK for e in results)
    return np.asarray(results.episodes.values)


def test_the_batched_and_the_per_episode_paths_give_the_same_measurements_for_fixed_genomes():
    backend = integration_backend()
    batched, single = measure(backend, True), measure(backend, False)
    assert batched.shape == (4, 5, 3)
    exact = backend.name == "numpy" and backend.precision == "float64"
    # the agents' observations reach the matrix products in different groupings, so even numpy float64 differs by a few ulp
    np.testing.assert_allclose(batched, single, rtol=1e-9 if exact else 5e-2, atol=1e-9 if exact else 5e-2)
    reference = measure(Backend(), True)
    np.testing.assert_allclose(batched, reference, rtol=1e-9 if exact else 5e-2, atol=1e-9 if exact else 5e-2)


def test_the_fixture_is_not_trivial_different_networks_give_different_outcomes():
    values = measure(integration_backend(), True)
    distance = values[:, :, 0]
    assert np.ptp(distance.mean(axis=1)) > 0.2 and np.ptp(distance, axis=0).min() > 0.0


def test_the_environment_receives_one_agent_batch_that_acts_for_the_whole_team():
    seen: list[Any] = []

    class Spy(team.TeamEnvironment):
        def run_batch(self, agents: Any, scenarios: Any, rng: Any) -> Any:
            seen.append(agents)
            return super().run_batch(agents, scenarios, rng)

    backend = integration_backend()
    batch = ArrayBatch(backend.asarray(GENOMES), [CandidateId(i) for i in range(len(GENOMES))], 0, "random")
    asyncio.run(evaluator(environment=Spy()).evaluate(batch, eval_context(team.problem(), backend, 0)))
    assert len(seen) == 1 and isinstance(seen[0], BatchPolicy)
    for array in (*seen[0].weights, *seen[0].biases):
        assert_on_backend(array, backend)
    observations = backend.asarray(np.zeros((4, 6, team.AGENTS, team.OBSERVATIONS)))
    assert tuple(seen[0].act_batch(observations).shape) == (4, 6, team.AGENTS, 1)


# --- searching ---


def test_a_genetic_algorithm_solves_the_team_task_and_beats_random_search_at_the_same_budget(tmp_path: Path):
    ga = auxein.run(**arguments(genetic(), 3000, run_dir=tmp_path / "ga"))
    random_search = auxein.run(**arguments(auxein.RandomSearch(), 3000))
    assert ga.best is not None and random_search.best is not None
    assert ga.best.objectives["error"] < 0.25 and ga.best.descriptors["success_rate"] >= 0.5  # the unevolved team scores about 2
    assert ga.best.objectives["error"] < 0.75 * random_search.best.objectives["error"]
    assert ga.evaluations_used == random_search.evaluations_used == 3000


def test_the_evolved_team_generalises_to_scenarios_it_never_saw(tmp_path: Path):
    selection, held_out = auxein.ScenarioSet.generate_split(team.scenario_params, 6, 12, seed=4)
    candidate_evaluator = auxein.EpisodeEvaluator(team.POLICY, team.TeamEnvironment(), selection, team.aggregator())
    run_dir = tmp_path / "split"
    auxein.run(**arguments(genetic(), 3000, evaluator=candidate_evaluator, run_dir=run_dir))
    held = auxein.EpisodeEvaluator(team.POLICY, team.TeamEnvironment(), held_out, team.aggregator())
    report = evaluate_held_out(run_dir, held, held_out, write=False)
    assert report.candidates[0].objectives["error"] < 0.4


# --- recording and resume ---


def test_a_recorded_run_with_mlp_genomes_extends_into_the_uninterrupted_run(tmp_path: Path):
    auxein.run(**arguments(genetic(), 150, run_dir=tmp_path / "a"))
    extended = auxein.resume(**arguments(genetic(), 330, run_dir=tmp_path / "a"))
    reference = auxein.run(**arguments(genetic(), 330, run_dir=tmp_path / "b"))
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")
    assert len(event_log(tmp_path / "a")["episodes"]) == 330 * 6
    same_result(extended, reference)
    best = extended.best
    assert best is not None
    with open_run(tmp_path / "a") as run:
        genome = next(np.asarray(e.genome) for e in run.evaluations() if e.candidate_id == best.candidate.id)
    assert genome.shape == (team.POLICY.dim,)  # the recorded genome is the flat layout
    np.testing.assert_array_equal(genome, Backend().to_numpy(best.candidate.genome))


def recorded(run_dir: Path) -> int:
    try:
        db = sqlite3.connect(f"file:{run_dir / 'events.sqlite'}?mode=ro", uri=True)
        try:
            return int(db.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0])
        finally:
            db.close()
    except sqlite3.Error:
        return -1


def test_a_killed_run_of_mlp_genomes_resumes_to_the_same_log_without_duplicates(tmp_path: Path):
    total, run_dir = 300, tmp_path / "run"
    full = {
        "strategy": "ga",
        "evaluator": "team",
        "run_dir": str(run_dir),
        "evaluations": total,
        "scenarios": 4,
        "batch_size": 10,
        "backend": backend_config(integration_backend()),
    }
    command = [sys.executable, "-m", "tests.support.resume_cli"]
    victim = subprocess.Popen([*command, json.dumps(full)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 120
    try:
        while victim.poll() is None and recorded(run_dir) < 90 and time.monotonic() < deadline:
            time.sleep(0.003)
        victim.send_signal(signal.SIGKILL)
        victim.wait(timeout=30)
    finally:
        if victim.poll() is None:
            victim.kill()
            victim.wait()
    survivor = subprocess.run(
        [*command, json.dumps({**full, "resume": True})], cwd=ROOT, env={**os.environ}, capture_output=True, text=True, timeout=240
    )
    assert survivor.returncode == 0, survivor.stderr[-3000:]
    reference = tmp_path / "reference"
    auxein.run(budget=auxein.Budget(evaluations=total), **settings({**full, "run_dir": str(reference)}))
    assert comparable(run_dir) == comparable(reference)
    assert len(event_log(run_dir)["episodes"]) == total * 4
    assert peek(run_dir).metadata["status"] == "completed"
