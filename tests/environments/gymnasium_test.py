"""The Gymnasium adapter and `LinearPolicy`: common random numbers, measurements, action spaces, workers, CartPole end to end."""

import importlib
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import auxein
from auxein.aggregators import Aggregator, mean
from auxein.backend import Backend
from auxein.core import Objective
from auxein.driver import evaluate_held_out
from auxein.environments import GymnasiumEnvironment, LinearAgent, LinearPolicy, ScenarioSet
from auxein.evaluators import EpisodeEvaluator
from auxein.random import RunSeed
from auxein.spaces import Box

gym = pytest.importorskip("gymnasium")

pytestmark = pytest.mark.filterwarnings("ignore::auxein.RecordingDisabledWarning")


class Walk(gym.Env):  # type: ignore[misc]
    """A tiny environment with known numbers: it terminates after `length` steps, pays 1 per step and reports `info` fields."""

    observation_space = gym.spaces.Box(-1e6, 1e6, shape=(2,), dtype=np.float64)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, length: int = 7, limit: int | None = None) -> None:
        self.length = length
        self.t = 0
        self.limit = limit

    def reset(self, *, seed: int | None = None, options: Any = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        self.t = 0
        return np.array([float(self.np_random.uniform()), float(seed or 0)]), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.t += 1
        terminated = self.t >= self.length
        truncated = self.limit is not None and self.t >= self.limit and not terminated
        return np.array([float(self.t), 0.0]), 1.0, terminated, truncated, {"energy": 2.0 * self.t, "level": float(self.t), "text": "x"}


def make_walk(length: int = 7, limit: int | None = None) -> Walk:
    return Walk(length, limit)


class Recorder:
    """An agent that records the first observation of each episode, and plays action 0."""

    def __init__(self) -> None:
        self.first: list[np.ndarray] = []
        self.fresh = True

    def reset(self, rng: Any) -> None:
        self.fresh = True

    def act(self, observation: Any) -> int:
        if self.fresh:
            self.first.append(np.array(observation, dtype=float))
            self.fresh = False
        return 0


SCENARIOS = ScenarioSet.from_params([{}, {}, {}], seed=11)


def episode(environment: GymnasiumEnvironment, agent: Any, scenario_index: int = 0) -> dict[str, float]:
    scenario = SCENARIOS[scenario_index]
    return dict(environment.run_episode({"agent": agent}, scenario, RunSeed(0).stream("episode", 0, scenario_index)).measurements)


# --- common random numbers ---


def test_candidates_in_the_same_scenario_start_from_the_same_state_and_scenarios_differ():
    environment = GymnasiumEnvironment(make_walk, max_steps=50)
    first, second = Recorder(), Recorder()
    episode(environment, first, 0)
    episode(environment, second, 0)  # another candidate, the same scenario
    episode(environment, second, 1)
    episode(environment, second, 2)
    np.testing.assert_array_equal(first.first[0], second.first[0])  # the same initial observation
    assert len({tuple(o) for o in second.first}) == 3  # and a different one in each scenario
    assert second.first[0][1] != second.first[1][1]  # the seed handed to `reset` is derived from the scenario


def test_the_reset_seed_depends_on_the_scenario_alone_not_on_the_agents_stream():
    environment = GymnasiumEnvironment(make_walk, max_steps=50)
    scenario = SCENARIOS[1]
    seeds = []
    for stream_key in (0, 5):
        agent = Recorder()
        environment.run_episode({"agent": agent}, scenario, RunSeed(9).stream("episode", stream_key, 1))
        seeds.append(agent.first[0][1])
    assert seeds[0] == seeds[1]


# --- measurements ---


def test_the_measurements_of_an_episode():
    environment = GymnasiumEnvironment(make_walk, max_steps=100, info_fields=("energy",), last_info_fields=("level",))
    m = episode(environment, Recorder())
    assert m["return"] == 7.0 and m["steps"] == 7.0 and m["done"] == 1.0
    assert m["terminated"] == 1.0 and m["truncated"] == 0.0
    assert m["energy"] == 2.0 * sum(range(1, 8)) and m["level"] == 7.0  # summed, and the last value
    assert set(m) == {"return", "steps", "done", "terminated", "truncated", "energy", "level"}  # the text field was not asked for


def test_gymnasiums_time_limit_and_the_adapters_limit_interact_as_documented():
    # the environment truncates itself first: truncated and done
    inner = episode(GymnasiumEnvironment(lambda: make_walk(100, limit=5), max_steps=50), Recorder())
    assert inner["steps"] == 5.0 and inner["truncated"] == 1.0 and inner["terminated"] == 0.0 and inner["done"] == 1.0
    # the adapter cuts the episode first: neither flag, and done is 0
    outer = episode(GymnasiumEnvironment(lambda: make_walk(100, limit=50), max_steps=5), Recorder())
    assert (
        outer["steps"] == 5.0
        and outer["truncated"] == 0.0
        and outer["terminated"] == 0.0
        and outer["done"] == 0.0
        and outer["return"] == 5.0
    )


def test_max_steps_defaults_to_the_environments_time_limit():
    assert GymnasiumEnvironment("CartPole-v1").max_steps == 500
    assert GymnasiumEnvironment("CartPole-v1", max_steps=50).max_steps == 50
    with pytest.raises(ValueError, match="no max_episode_steps"):
        GymnasiumEnvironment(make_walk)
    with pytest.raises(ValueError, match="at least 1"):
        GymnasiumEnvironment(make_walk, max_steps=0)


def test_reserved_names_and_misplaced_options_are_refused():
    with pytest.raises(ValueError, match="measures those itself"):
        GymnasiumEnvironment(make_walk, max_steps=10, info_fields=("return",))
    with pytest.raises(ValueError, match="make_kwargs only applies to an environment id"):
        GymnasiumEnvironment(make_walk, max_steps=10, make_kwargs={"a": 1})
    assert "CartPole-v1" in repr(GymnasiumEnvironment("CartPole-v1")) and "max_steps=500" in repr(GymnasiumEnvironment("CartPole-v1"))


def test_the_environment_pickles_by_configuration():
    environment = GymnasiumEnvironment("CartPole-v1", max_steps=100, info_fields=("x",))
    copy = pickle.loads(pickle.dumps(environment))
    assert repr(copy) == repr(environment) and copy.max_steps == 100 and copy.roles == ("agent",)
    assert episode(copy, Recorder())["steps"] > 0


# --- the linear policy ---


def test_a_linear_policy_for_a_discrete_action_space_takes_the_argmax():
    policy = LinearPolicy(GymnasiumEnvironment("CartPole-v1"))
    assert policy.dim == 2 * (4 + 1) and policy.discrete and policy.observation_size == 4 and policy.outputs == 2
    genome = np.zeros(policy.dim)
    genome[:4] = [1.0, 0.0, 0.0, 0.0]  # score of action 0: the first observation component
    genome[4:8] = [-1.0, 0.0, 0.0, 0.0]  # score of action 1: its opposite
    agent = policy.decode(genome)
    assert isinstance(agent, LinearAgent)
    assert agent.act(np.array([2.0, 0, 0, 0])) == 0 and agent.act(np.array([-2.0, 0, 0, 0])) == 1
    genome[8:] = [0.0, 5.0]  # a bias towards action 1
    assert policy.decode(genome).act(np.array([2.0, 0, 0, 0])) == 1
    assert isinstance(policy.decode(genome).act(np.zeros(4)), int)
    assert policy.explain(genome)["bias"] == [0.0, 5.0]


def test_a_linear_policy_for_a_continuous_action_space_clips_to_the_action_bounds():
    environment = GymnasiumEnvironment("Pendulum-v1")
    policy = LinearPolicy(environment)
    assert policy.dim == 1 * (3 + 1) and not policy.discrete
    agent = policy.decode(np.array([10.0, 0.0, 0.0, 0.0]))
    assert agent.act(np.array([1.0, 0.0, 0.0])) == pytest.approx([2.0])  # clipped to the torque limit of 2
    assert agent.act(np.array([-1.0, 0.0, 0.0])) == pytest.approx([-2.0])
    assert agent.act(np.array([0.05, 0.0, 0.0])) == pytest.approx([0.5])
    assert policy.space(3.0).dim == 4 and isinstance(policy.space(), Box)
    assert episode(environment, agent)["steps"] == 200.0  # the pendulum is truncated at its limit


def test_the_policy_decodes_genomes_of_every_backend_and_checks_their_shape(backend: Backend):
    policy = LinearPolicy(observation_space=gym.spaces.Box(-1, 1, (3,)), action_space=gym.spaces.Discrete(4))
    genome = backend.asarray(np.arange(policy.dim, dtype=np.float64))
    agent = policy.decode(genome)
    assert agent.weights.shape == (4, 3) and agent.bias.shape == (4,) and agent.bias[0] == 12.0
    with pytest.raises(ValueError, match="has 16 values"):
        policy.decode(backend.asarray(np.zeros(5)))


def test_the_policy_refuses_what_it_cannot_do():
    with pytest.raises(TypeError, match="Box observation space"):
        LinearPolicy(observation_space=gym.spaces.Discrete(3), action_space=gym.spaces.Discrete(2))
    with pytest.raises(TypeError, match="Discrete or a Box action space"):
        LinearPolicy(observation_space=gym.spaces.Box(-1, 1, (2,)), action_space=gym.spaces.MultiBinary(2))
    with pytest.raises(ValueError, match="give an environment"):
        LinearPolicy()
    assert "discrete" in repr(LinearPolicy(GymnasiumEnvironment("CartPole-v1")))


# --- through the evaluator ---


def aggregator() -> Aggregator:
    return Aggregator({"return": mean("return")})


def evaluator(scenarios: ScenarioSet, environment: GymnasiumEnvironment | None = None) -> EpisodeEvaluator[Any]:
    environment = environment or GymnasiumEnvironment("CartPole-v1")
    return EpisodeEvaluator(LinearPolicy(environment), environment, scenarios, aggregator())


def test_episodes_run_in_worker_processes_and_agree_with_the_inline_run(tmp_path: Path):
    scenarios = ScenarioSet.from_params([{}, {}, {}, {}], seed=5)
    results = {}
    for executor, concurrency in (("inline", 1), ("process", 2)):
        results[executor] = auxein.run(
            strategy=auxein.RandomSearch(),
            evaluator=evaluator(scenarios),
            space=LinearPolicy(GymnasiumEnvironment("CartPole-v1")).space(2.0),
            objectives=[Objective("return", "maximise")],
            budget=auxein.Budget(evaluations=12),
            seed=2,
            batch_size=6,
            executor=executor,  # type: ignore[arg-type]
            concurrency=concurrency,
            run_dir=tmp_path / executor,
        )
    assert results["inline"].best is not None and results["process"].best is not None
    assert results["inline"].best.objectives == results["process"].best.objectives
    assert results["inline"].trace == results["process"].trace


def test_importing_without_gymnasium_names_the_extra(monkeypatch: pytest.MonkeyPatch):
    real = importlib.import_module

    def missing(name: str, package: str | None = None) -> Any:
        if name == "gymnasium":
            raise ModuleNotFoundError("No module named 'gymnasium'", name="gymnasium")
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ImportError, match=r"pip install auxein\[gymnasium\]"):
        GymnasiumEnvironment("CartPole-v1")
    with pytest.raises(ImportError, match=r"pip install auxein\[gymnasium\]"):
        LinearPolicy(observation_space=object(), action_space=object())


def test_auxein_imports_without_gymnasium_at_all():
    """Importing auxein (and the environments package) must not import gymnasium: it is an optional extra."""
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['gymnasium'] = None; import auxein, auxein.environments, auxein.strategies; "
        "assert 'gymnasium' not in [m for m in sys.modules if sys.modules[m] is not None]"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr


# --- CartPole, end to end ---


def test_a_genetic_algorithm_evolves_a_linear_policy_that_balances_the_pole_on_held_out_scenarios(tmp_path: Path):
    environment = GymnasiumEnvironment("CartPole-v1")
    policy = LinearPolicy(environment)
    selection, held_out = ScenarioSet.generate_split(lambda index, rng: {}, 5, 20, seed=3)
    run_dir = tmp_path / "cartpole"
    result = auxein.run(
        strategy=auxein.GeneticAlgorithm(population_size=20, offspring_size=20),
        evaluator=evaluator(selection, environment),
        space=policy.space(2.0),
        objectives=[Objective("return", "maximise")],
        budget=auxein.Budget(evaluations=1000),
        seed=0,
        batch_size=20,
        run_dir=run_dir,
    )
    assert result.best is not None and result.best.objectives["return"] >= 475.0  # on the five scenarios it evolved on
    report = evaluate_held_out(run_dir, evaluator(held_out, environment), held_out, write=False)
    assert report.candidates[0].objectives["return"] >= 475.0  # and on 20 it never saw: the mean return CartPole counts as solved
    assert len(report.candidates[0].scenarios) == 20
    shutil.rmtree(run_dir)
