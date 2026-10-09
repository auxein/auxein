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
from auxein.strategies.structured import ExternalMutation, SequenceMutation
from tests.support import pointmass, sequences
from tests.support.pointmass import PointMassEnvironment
from tests.support.sequences import FakeLLM


def logged_sphere(genome: np.ndarray) -> float:
    """A sphere that takes about 3 ms, so that a run is long enough to be killed in the middle, and that logs each call."""
    path = os.environ.get("AUXEIN_CALL_LOG")
    if path:
        with open(path, "a") as handle:
            handle.write("x\n")
    time.sleep(0.003)
    return float((genome * genome).sum())


class LoggedPointMass(PointMassEnvironment):
    """The point mass, logging each episode it really runs (see `AUXEIN_CALL_LOG`), and taking a moment so that runs can be killed."""

    def run_episode(self, agents: Any, scenario: Any, rng: Any) -> Any:
        path = os.environ.get("AUXEIN_CALL_LOG")
        if path:
            with open(path, "a") as handle:
                handle.write("x\n")
        time.sleep(0.0005)
        return super().run_episode(agents, scenario, rng)

    def __repr__(self) -> str:
        return "LoggedPointMass()"


def strategy_for(name: str) -> Any:
    if name == "random":
        return auxein.RandomSearch()
    if name == "structured":
        return auxein.StructuredGeneticAlgorithm(population_size=12, offspring_size=12)
    if name == "structured-llm":
        mixed = [(SequenceMutation(), 2.0), (ExternalMutation(FakeLLM()), 1.0)]
        return auxein.StructuredGeneticAlgorithm(population_size=12, offspring_size=12, mutation=mixed)
    return GeneticAlgorithm(population_size=12, offspring_size=12)


def backend_config(backend: auxein.Backend) -> list[str]:
    """A backend as the JSON a test passes to the subprocess that runs the configuration (`config["backend"]`)."""
    return [backend.name, backend.precision]


def settings(config: dict[str, Any]) -> dict[str, Any]:
    """The arguments of `run` and `resume` for a configuration. The evaluator is built by the caller's module."""
    name, precision = config.get("backend", ["numpy", "float64"])
    if config.get("evaluator") == "episode":
        evaluator: Any = auxein.EpisodeEvaluator(
            pointmass.GainsDecoder() if config.get("batched") else pointmass.PerEpisodeDecoder(),
            LoggedPointMass(),
            pointmass.scenario_set(config.get("scenarios", 4)),
            pointmass.aggregator(),
        )
        problem: dict[str, Any] = {
            "space": pointmass.SPACE,
            "objectives": [auxein.Objective("error")],
            "constraints": ["overshoot"],
            "descriptors": ["success_rate"],
        }
    elif config.get("evaluator") == "sequence":
        evaluator = auxein.FunctionEvaluator(sequences.distance)
        problem = {"space": sequences.SPACE, "objectives": [auxein.Objective("distance")], "constraints": ["too_long"]}
    else:
        evaluator = auxein.FunctionEvaluator(logged_sphere)
        problem = {"space": auxein.Box(-5.0, 5.0, dim=4)}
    return {
        **problem,
        "strategy": strategy_for(config["strategy"]),
        "evaluator": evaluator,
        "seed": config.get("seed", 5),
        "batch_size": config.get("batch_size", 10),
        "concurrency": config.get("concurrency", 1),
        "executor": config.get("executor", "auto"),
        "delivery": config.get("delivery"),
        "deterministic": config.get("deterministic", True),
        "run_dir": config["run_dir"],
        "backend": auxein.Backend(name, "cpu", precision),
        "checkpoint_every_evaluations": config.get("checkpoint_every_evaluations", 40),
        "checkpoint_every": config.get("checkpoint_every"),
        "keep_checkpoints": config.get("keep_checkpoints"),
        "genome_store_threshold": config.get("genome_store_threshold", 4096),
    }
