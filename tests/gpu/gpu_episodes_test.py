"""The agent layer on the device: the point mass, batched, with the aggregators reducing device arrays."""

import asyncio

import numpy as np

from auxein.backend import Backend
from auxein.core import ArrayBatch, CandidateId, Status
from auxein.evaluators import EpisodeEvaluator
from tests.support import pointmass as pm
from tests.support.fixtures import assert_on_backend
from tests.support.helpers import eval_context

GAINS = np.array([[2.0, 3.0, 0.0], [8.0, 0.5, 0.2], [0.1, 0.1, -0.5], [5.0, 5.0, 0.9], [3.0, 2.0, 0.1]])


def evaluate(backend: Backend, environment: pm.PointMassEnvironment, scenarios: int = 6):
    batch = ArrayBatch(backend.asarray(GAINS), [CandidateId(i) for i in range(len(GAINS))], 0, "random")
    evaluator = EpisodeEvaluator(pm.GainsDecoder(), environment, pm.scenario_set(scenarios), pm.aggregator())
    assert evaluator.batched
    return asyncio.run(evaluator.evaluate(batch, eval_context(pm.problem(), backend, 0)))


def test_the_batched_environment_runs_on_the_device_and_agrees_with_numpy_float64(gpu_backend: Backend):
    seen: list[object] = []

    class Spy(pm.PointMassEnvironment):
        def run_batch(self, agents, scenarios, rng):
            out = super().run_batch(agents, scenarios, rng)
            seen.append((agents, list(out.measurements.values())))
            return out

    results = evaluate(gpu_backend, Spy())
    agents, measured = seen[0]  # type: ignore[misc]
    assert_on_backend(agents, gpu_backend)
    for array in measured:
        assert_on_backend(array, gpu_backend)  # the simulation's outputs never left the device
    assert all(e.status is Status.OK for e in results)
    reference = evaluate(Backend(), pm.PointMassEnvironment())
    atol = 0.05 if gpu_backend.precision == "float32" else 1e-9  # float32 may settle a step earlier or later: the numbers stay close
    for ours, theirs in zip(results, reference, strict=True):
        assert abs(ours.objectives["error"] - theirs.objectives["error"]) <= atol
        assert abs(ours.descriptors["success_rate"] - theirs.descriptors["success_rate"]) <= 1.0 / 6 + 1e-9


def test_the_per_scenario_measurements_come_back_to_the_host_once_for_the_recorder(float32_backend: Backend):
    results = evaluate(float32_backend, pm.PointMassEnvironment())
    assert results.episodes is not None
    assert isinstance(results.episodes.values, np.ndarray) and results.episodes.values.shape == (5, 6, 5)
    assert np.isfinite(results.episodes.values).all()


def test_the_aggregator_reductions_run_on_the_device(gpu_backend: Backend):
    from auxein.aggregators import cvar_upper, mean, quantile

    values = gpu_backend.asarray(np.random.default_rng(0).normal(size=(7, 40)))
    for reduction in (mean("m"), quantile("m", 0.9), cvar_upper("m", 0.25)):
        assert_on_backend(reduction.reduce({"m": values}, gpu_backend.xp), gpu_backend)
