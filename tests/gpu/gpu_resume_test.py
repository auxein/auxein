"""A run on the device that is recorded, extended and resumed: arrays go back to the device, and the replay is identical there."""

from pathlib import Path
from typing import Any

import numpy as np

import auxein
from auxein.backend import Backend
from auxein.recording import checkpoints
from tests.driver.resume_test import comparable
from tests.support.fixtures import assert_on_backend


def sphere(X: Any) -> Any:
    return (X * X).sum(axis=1)


def arguments(backend: Backend, run_dir: Path, **over: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "strategy": auxein.GeneticAlgorithm(population_size=20, offspring_size=20),
        "evaluator": auxein.VectorisedEvaluator(sphere),
        "space": auxein.Box(-5.0, 5.0, dim=6),
        "seed": 4,
        "batch_size": 20,
        "backend": backend,
        "run_dir": run_dir,
        "checkpoint_every_evaluations": 100,
    }
    settings.update(over)
    return settings


def test_checkpoints_restore_their_arrays_to_the_device(gpu_backend: Backend, tmp_path: Path):
    auxein.run(budget=auxein.Budget(evaluations=300), **arguments(gpu_backend, tmp_path / "r"))
    newest = max((tmp_path / "r" / "checkpoints").iterdir(), key=lambda p: int(p.name.split("-")[1]))
    _, state = checkpoints.read(newest, gpu_backend)
    strategy = state["strategy"]
    for name in ("genomes", "values", "violation"):
        assert_on_backend(strategy[name], gpu_backend)  # type: ignore[index]


def test_an_extended_run_equals_the_run_of_the_larger_budget_on_the_same_device(gpu_backend: Backend, tmp_path: Path):
    auxein.run(budget=auxein.Budget(evaluations=200), **arguments(gpu_backend, tmp_path / "a"))
    extended = auxein.resume(budget=auxein.Budget(evaluations=500), **arguments(gpu_backend, tmp_path / "a"))
    reference = auxein.run(budget=auxein.Budget(evaluations=500), **arguments(gpu_backend, tmp_path / "b"))
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")  # candidates (genome bytes included), evaluations, events
    assert extended.best is not None and reference.best is not None
    assert extended.best.candidate.id == reference.best.candidate.id
    np.testing.assert_array_equal(
        gpu_backend.to_numpy(extended.best.candidate.genome), gpu_backend.to_numpy(reference.best.candidate.genome)
    )


def test_a_run_whose_last_checkpoint_is_lost_replays_identically_on_the_device(float32_backend: Backend, tmp_path: Path):
    """Without a checkpoint the resumed run regenerates every recorded candidate from the seed and checks it byte for byte."""
    auxein.run(budget=auxein.Budget(evaluations=200), keep_checkpoints=0, **arguments(float32_backend, tmp_path / "a"))
    auxein.resume(budget=auxein.Budget(evaluations=400), keep_checkpoints=0, **arguments(float32_backend, tmp_path / "a"))
    auxein.run(budget=auxein.Budget(evaluations=400), keep_checkpoints=0, **arguments(float32_backend, tmp_path / "b"))
    assert comparable(tmp_path / "a") == comparable(tmp_path / "b")
