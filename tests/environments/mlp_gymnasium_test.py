"""`MLPPolicy.for_spaces`: Gymnasium spaces, the lazy import, and CartPole end to end."""

import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import auxein
from auxein.aggregators import Aggregator, mean
from auxein.core import Objective
from auxein.driver import evaluate_held_out
from auxein.environments import Clip, GymnasiumEnvironment, MLPPolicy, ScenarioSet, Tanh
from auxein.evaluators import EpisodeEvaluator

gym = pytest.importorskip("gymnasium")

pytestmark = pytest.mark.filterwarnings("ignore::auxein.RecordingDisabledWarning")


def test_a_discrete_action_space_gets_one_output_per_action_and_argmax():
    policy = MLPPolicy.for_spaces(gym.spaces.Box(-1, 1, (4,)), gym.spaces.Discrete(3), hidden=(5,))
    assert (policy.observation_size, policy.outputs, policy.hidden) == (4, 3, (5,))
    assert repr(policy).endswith("output='argmax')")
    action = policy.decode(np.random.default_rng(0).normal(size=policy.dim)).act(np.ones(4))
    assert isinstance(action, int) and 0 <= action < 3  # an int, as Gymnasium's Discrete expects


def test_a_box_action_space_gets_tanh_rescaled_to_its_bounds_or_clip():
    space = gym.spaces.Box(np.array([-2.0, 0.0]), np.array([2.0, 10.0]), dtype=np.float64)
    policy = MLPPolicy.for_spaces(gym.spaces.Box(-1, 1, (3,)), space, hidden=(4,))
    assert policy.outputs == 2 and repr(policy).endswith("output=Tanh(low=[-2.0, 0.0], high=[2.0, 10.0]))")
    huge = policy.decode(np.random.default_rng(1).normal(size=policy.dim) * 50.0)
    actions = np.asarray(huge.act(np.random.default_rng(2).normal(size=(100, 3)) * 5.0))
    assert (actions[:, 0] >= -2.0).all() and (actions[:, 0] <= 2.0).all() and (actions[:, 1] >= 0.0).all() and (actions[:, 1] <= 10.0).all()
    assert space.contains(actions[0].astype(np.float32))
    clipped = MLPPolicy.for_spaces(gym.spaces.Box(-1, 1, (3,)), space, hidden=(4,), continuous="clip")
    assert repr(clipped).endswith("output=Clip(low=[-2.0, 0.0], high=[2.0, 10.0]))")
    assert np.asarray(clipped.decode(np.random.default_rng(1).normal(size=clipped.dim) * 50.0).act(np.ones(3))).max() <= 10.0


def test_an_unbounded_box_needs_clip_not_tanh_and_other_spaces_are_refused():
    unbounded = gym.spaces.Box(-np.inf, np.inf, (2,))
    with pytest.raises(ValueError, match="Tanh needs finite bounds"):
        MLPPolicy.for_spaces(gym.spaces.Box(-1, 1, (3,)), unbounded)
    assert MLPPolicy.for_spaces(gym.spaces.Box(-1, 1, (3,)), unbounded, continuous="clip").outputs == 2
    with pytest.raises(TypeError, match="Box observation space"):
        MLPPolicy.for_spaces(gym.spaces.Discrete(3), gym.spaces.Discrete(2))
    with pytest.raises(TypeError, match="Discrete or a Box action space"):
        MLPPolicy.for_spaces(gym.spaces.Box(-1, 1, (2,)), gym.spaces.MultiBinary(2))
    with pytest.raises(ValueError, match="continuous must be"):
        MLPPolicy.for_spaces(gym.spaces.Box(-1, 1, (3,)), unbounded, continuous="sigmoid")  # type: ignore[arg-type]


def test_a_multi_dimensional_observation_is_flattened():
    policy = MLPPolicy.for_spaces(gym.spaces.Box(-1, 1, (2, 3)), gym.spaces.Discrete(2))
    assert policy.observation_size == 6
    policy.decode(np.zeros(policy.dim)).act(np.zeros((2, 3)).reshape(-1))


def test_the_helper_names_the_extra_when_gymnasium_is_missing(monkeypatch: pytest.MonkeyPatch):
    real = importlib.import_module

    def missing(name: str, package: str | None = None) -> Any:
        if name == "gymnasium":
            raise ModuleNotFoundError("No module named 'gymnasium'", name="gymnasium")
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ImportError, match=r"pip install auxein\[gymnasium\]"):
        MLPPolicy.for_spaces(object(), object())
    assert MLPPolicy(3, (2,), 1).dim == 11  # building from sizes needs no Gymnasium


def test_the_policy_module_imports_and_works_without_gymnasium():
    code = (
        "import sys; sys.modules['gymnasium'] = None; import numpy as np; import auxein; "
        "from auxein.environments import MLPPolicy, BatchPolicy; p = MLPPolicy(3, (4,), 2, output='argmax'); "
        "print(p.decode(np.zeros(p.dim)).act(np.zeros(3)))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "0"


# --- CartPole, end to end ---


def evaluator(policy: MLPPolicy, scenarios: ScenarioSet, environment: GymnasiumEnvironment) -> EpisodeEvaluator[Any]:
    return EpisodeEvaluator(policy, environment, scenarios, Aggregator({"return": mean("return")}))


def test_a_genetic_algorithm_evolves_an_mlp_policy_that_balances_the_pole_on_held_out_scenarios(tmp_path: Path):
    environment = GymnasiumEnvironment("CartPole-v1")
    policy = MLPPolicy.for_spaces(environment.observation_space, environment.action_space, hidden=(4,))
    selection, held_out = ScenarioSet.generate_split(lambda index, rng: {}, 5, 20, seed=3)
    run_dir = tmp_path / "cartpole"
    result = auxein.run(
        strategy=auxein.GeneticAlgorithm(population_size=20, offspring_size=20),
        evaluator=evaluator(policy, selection, environment),
        space=policy.space(2.0),
        objectives=[Objective("return", "maximise")],
        budget=auxein.Budget(evaluations=1000),
        seed=0,
        batch_size=20,
        run_dir=run_dir,
    )
    assert result.best is not None and result.best.objectives["return"] >= 475.0
    report = evaluate_held_out(run_dir, evaluator(policy, held_out, environment), held_out, write=False)
    assert report.candidates[0].objectives["return"] >= 475.0  # the mean return CartPole counts as solved, on 20 unseen scenarios
    assert len(report.candidates[0].scenarios) == 20
    shutil.rmtree(run_dir)


def test_pendulum_runs_with_a_continuous_mlp_policy():
    environment = GymnasiumEnvironment("Pendulum-v1")
    policy = MLPPolicy.for_spaces(environment.observation_space, environment.action_space, hidden=(8,))
    assert isinstance(policy.output, Tanh) and not isinstance(policy.output, Clip)
    scenarios = ScenarioSet.from_params([{}, {}], seed=1)
    result = auxein.run(
        strategy=auxein.RandomSearch(),
        evaluator=evaluator(policy, scenarios, environment),
        space=policy.space(),
        objectives=[Objective("return", "maximise")],
        budget=auxein.Budget(evaluations=6),
        seed=0,
        batch_size=3,
    )
    assert result.best is not None and result.best.objectives["return"] < 0.0  # the pendulum's rewards are negative costs
