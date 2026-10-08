"""The run that tests kill and resume (started as `python -m tests.support.resume_cli '<json>'`).

The configuration names a strategy, a delivery, an executor and so on; the run is recorded in `run_dir`. Every call of the
evaluation function appends a line to the file named by `AUXEIN_CALL_LOG`, if it is set, in the worker process or the
driver's, so that a test can count how often the function really ran.
"""

import os
import time
from typing import Any

import numpy as np

import auxein
from auxein.strategies.ga import GeneticAlgorithm


def logged_sphere(genome: np.ndarray) -> float:
    """A sphere that takes about 3 ms, so that a run is long enough to be killed in the middle, and that logs each call."""
    path = os.environ.get("AUXEIN_CALL_LOG")
    if path:
        with open(path, "a") as handle:
            handle.write("x\n")
    time.sleep(0.003)
    return float((genome * genome).sum())


def strategy_for(name: str) -> Any:
    if name == "random":
        return auxein.RandomSearch()
    return GeneticAlgorithm(population_size=12, offspring_size=12)


def settings(config: dict[str, Any]) -> dict[str, Any]:
    """The arguments of `run` and `resume` for a configuration. The evaluator is built by the caller's module."""
    return {
        "strategy": strategy_for(config["strategy"]),
        "evaluator": auxein.FunctionEvaluator(logged_sphere),
        "space": auxein.Box(-5.0, 5.0, dim=4),
        "seed": config.get("seed", 5),
        "batch_size": config.get("batch_size", 10),
        "concurrency": config.get("concurrency", 1),
        "executor": config.get("executor", "auto"),
        "delivery": config.get("delivery"),
        "deterministic": config.get("deterministic", True),
        "run_dir": config["run_dir"],
        "checkpoint_every_evaluations": config.get("checkpoint_every_evaluations", 40),
        "checkpoint_every": config.get("checkpoint_every"),
        "keep_checkpoints": config.get("keep_checkpoints"),
    }
