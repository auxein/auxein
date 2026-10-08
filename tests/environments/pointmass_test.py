"""The fixture world, its three implementations, and the step-level adapter."""

import asyncio

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import ArrayBatch, CandidateId, Status
from auxein.environments import StepEnvironment
from auxein.evaluators import EpisodeEvaluator
from tests.support import pointmass as pm
from tests.support.helpers import eval_context

GAINS = np.array([[2.0, 3.0, 0.0], [8.0, 0.5, 0.2], [0.1, 0.1, -0.5], [5.0, 5.0, 0.9]])


def measure(environment, decoder, scenarios, backend: Backend, batched: bool):
    """The measurements of every candidate and scenario, as an `(n, s, names)` array in sorted name order."""
    batch = ArrayBatch(backend.asarray(GAINS), [CandidateId(i) for i in range(len(GAINS))], 0, "random")
    evaluator = EpisodeEvaluator(decoder, environment, scenarios, pm.aggregator())
    assert evaluator.batched is batched
    results = asyncio.run(evaluator.evaluate(batch, eval_context(pm.problem(), backend, 0)))
    assert results.episodes is not None and all(e.status is Status.OK for e in results)
    return results.episodes


def test_the_three_implementations_agree_exactly_in_float64_numpy():
    backend, scenarios = Backend("numpy", "cpu", "float64"), pm.scenario_set(7)
    batched = measure(pm.PointMassEnvironment(), pm.GainsDecoder(), scenarios, backend, True)
    single = measure(pm.PointMassEnvironment(), pm.PerEpisodeDecoder(), scenarios, backend, False)
    stepped = measure(pm.step_environment(), pm.StepGainsDecoder(), scenarios, backend, False)
    common = [name for name in batched.names if name in stepped.names]
    assert set(common) == set(batched.names) == {"final_distance", "steps", "effort", "success", "max_overshoot"}
    np.testing.assert_array_equal(single.values, batched.values)
    for name in common:
        np.testing.assert_array_equal(stepped.values[:, :, stepped.names.index(name)], batched.values[:, :, batched.names.index(name)], err_msg=name)


def test_the_fixture_really_exercises_the_world():
    """Different gains give different outcomes, some settle early, some overshoot, some fail: the comparison above is not trivial."""
    episodes = measure(pm.PointMassEnvironment(), pm.GainsDecoder(), pm.scenario_set(7), Backend(), True)
    steps = episodes.values[:, :, episodes.names.index("steps")]
    overshoot = episodes.values[:, :, episodes.names.index("max_overshoot")]
    success = episodes.values[:, :, episodes.names.index("success")]
    assert steps.min() < pm.STEPS == steps.max() and overshoot.max() > pm.OVERSHOOT_LIMIT and overshoot.min() == 0.0
    assert 0.0 < success.mean() < 1.0


def test_the_three_implementations_agree_on_numpy_float32_and_torch(backend: Backend):
    scenarios = pm.scenario_set(7)
    reference = measure(pm.PointMassEnvironment(), pm.GainsDecoder(), scenarios, Backend("numpy", "cpu", "float64"), True)
    batched = measure(pm.PointMassEnvironment(), pm.GainsDecoder(), scenarios, backend, True)
    single = measure(pm.PointMassEnvironment(), pm.PerEpisodeDecoder(), scenarios, backend, False)
    stepped = measure(pm.step_environment(), pm.StepGainsDecoder(), scenarios, backend, False)
    exact = backend.name == "numpy" and backend.precision == "float64"
    atol = 0.0 if exact else 5e-2  # float32 may settle a step earlier or later; the measurements stay close
    for other in (batched, single, stepped):
        for name in ("final_distance", "effort", "max_overshoot"):
            np.testing.assert_allclose(
                other.values[:, :, other.names.index(name)], reference.values[:, :, reference.names.index(name)], atol=atol, rtol=0.05 if not exact else 0
            )
    # torch float64 is as exact as numpy float64, up to rounding of identical operations
    if backend.precision == "float64":
        np.testing.assert_allclose(batched.values, reference.values, atol=1e-9)


# --- the step adapter ---


class CountingWorld:
    """Counts: each step reports 1 of `effort`, and the running count as `level`; it ends after `limit` steps."""

    limit = 5

    def reset(self, scenario, rng):
        self.t = 0
        return 0

    def step(self, action):
        self.t += 1
        return self.t, {"effort": 1.0, "level": float(self.t * 10)}, self.t >= self.limit


class Doubler:
    def act(self, observation):
        return observation * 2


class Stochastic:
    def reset(self, rng):
        self.value = float(rng.uniform((1,))[0])

    def act(self, observation):
        return self.value


def run(world_factory, agent, **kwargs):
    from auxein.environments import ScenarioSet

    scenario = ScenarioSet.from_params([{}], seed=1)[0]
    env = StepEnvironment(world_factory, role="r", **kwargs)
    from auxein.random import RunSeed

    return env.run_episode({"r": agent}, scenario, RunSeed(1).stream("episode", 0, 0))


def test_updates_are_summed_except_for_the_names_kept_as_last_and_the_adapter_adds_steps_and_done():
    result = run(CountingWorld, Doubler(), max_steps=100, last=("level",))
    assert result.status is Status.OK
    assert dict(result.measurements) == {"effort": 5.0, "level": 50.0, "steps": 5.0, "done": 1.0}


def test_a_world_that_does_not_finish_is_stopped_at_max_steps_and_done_says_so():
    result = run(CountingWorld, Doubler(), max_steps=3)
    assert dict(result.measurements) == {"effort": 3.0, "level": 60.0, "steps": 3.0, "done": 0.0}  # "level" is summed by default


def test_a_world_is_built_for_every_episode_and_an_agents_reset_gets_its_stream():
    worlds = []

    def factory():
        worlds.append(CountingWorld())
        return worlds[-1]

    agent = Stochastic()
    run(factory, agent, max_steps=2)
    run(factory, agent, max_steps=2)
    assert len(worlds) == 2 and worlds[0] is not worlds[1]
    assert 0.0 <= agent.value < 1.0


def test_reserved_names_and_bad_arguments_are_rejected():
    class Bad(CountingWorld):
        def step(self, action):
            return 0, {"steps": 1.0}, True

    with pytest.raises(ValueError, match="measures itself"):
        run(Bad, Doubler(), max_steps=3)
    with pytest.raises(ValueError, match="max_steps"):
        StepEnvironment(CountingWorld, max_steps=0)
    with pytest.raises(ValueError, match="measured by the adapter"):
        StepEnvironment(CountingWorld, max_steps=3, last=("done",))


def test_the_step_environment_describes_itself_stably():
    text = repr(pm.step_environment())
    assert text == "StepEnvironment(world=PointMassWorld, role='controller', max_steps=60, last=['final_distance', 'max_overshoot', 'success'])"
