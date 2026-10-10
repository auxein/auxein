"""The Gymnasium adapter and a linear policy (design doc §6.2): reinforcement-learning environments as Auxein environments.

Gymnasium (the maintained fork of OpenAI Gym) is an optional dependency, installed with `pip install auxein[gymnasium]`. This
module does not import it until an environment is built, so `import auxein` works without it, and the error that follows a
missing install names the extra.
"""

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from auxein.backend import Array, Backend
from auxein.environments.episode import EpisodeResult
from auxein.environments.protocols import Agent
from auxein.environments.scenario import Scenario
from auxein.environments.step import StepEnvironment
from auxein.random import RandomStream
from auxein.spaces import Box

_SEED_LIMIT = 2**31 - 1


def _import_gymnasium() -> Any:
    try:
        return importlib.import_module("gymnasium")
    except ImportError as error:
        raise ImportError("the Gymnasium adapter needs the gymnasium package: install it with `pip install auxein[gymnasium]`") from error


def _factory_of(env: str | Callable[[], Any], make_kwargs: Mapping[str, object]) -> Callable[[], Any]:
    """A picklable function that builds the environment: an id is made with `gymnasium.make` in the worker that needs it."""
    if isinstance(env, str):
        return _Make(env, dict(make_kwargs))
    if make_kwargs:
        raise ValueError("make_kwargs only applies to an environment id; give the factory its own arguments (functools.partial)")
    return env


@dataclass(frozen=True)
class _Make:
    """`gymnasium.make(id, **kwargs)`, as an object that pickles (a lambda or a closure would not)."""

    id: str
    kwargs: dict[str, object]

    def __call__(self) -> Any:
        return _import_gymnasium().make(self.id, **self.kwargs)

    def __repr__(self) -> str:
        return f"make({self.id!r}{''.join(f', {k}={v!r}' for k, v in sorted(self.kwargs.items()))})"


class _GymWorld:
    """One episode of a Gymnasium environment as a `StepWorld`: built for the episode, so that episodes share nothing."""

    def __init__(self, factory: Callable[[], Any], info_fields: tuple[str, ...], last_info_fields: tuple[str, ...]) -> None:
        self._env = factory()
        self._fields = info_fields + last_info_fields

    def reset(self, scenario: Scenario, rng: RandomStream) -> object:
        # the environment's seed is drawn from the scenario's own stream, so it depends on the scenario alone: every candidate
        # faces the same initial state (common random numbers, design doc §6.4)
        seed = int(np.asarray(rng.integers(0, _SEED_LIMIT, (1,)))[0])
        observation, _ = self._env.reset(seed=seed)
        return observation

    def step(self, action: object) -> tuple[object, Mapping[str, float], bool]:
        observation, reward, terminated, truncated, info = self._env.step(action)
        update: dict[str, float] = {"return": float(reward), "terminated": float(bool(terminated)), "truncated": float(bool(truncated))}
        for name in self._fields:
            if name in info:
                update[name] = float(info[name])
        done = bool(terminated) or bool(truncated)
        if done:
            self._env.close()
        return observation, update, done

    def __del__(self) -> None:
        try:
            self._env.close()
        except Exception:  # an environment that is already closed, or was never made: nothing to clean up
            pass


class GymnasiumEnvironment:
    """A Gymnasium environment as an Auxein `Environment` (through `StepEnvironment`): an agent with `act(observation) -> action`
    is run in it for one episode per scenario.

    `env` is an environment id (`"CartPole-v1"`, made with `gymnasium.make`, with `make_kwargs`) or a **picklable factory**
    (a module-level function, or `functools.partial`) returning an environment, so that episodes can run in worker processes. A
    new environment is made for every episode.

    **Common random numbers.** Every episode calls `env.reset(seed=...)` with a seed drawn from the scenario's own stream, so
    it depends on the scenario alone: all the candidates evaluated in a scenario face the same initial state, whatever the
    worker or the time. A scenario set of `n` scenarios gives `n` different initial states. (Environments whose dynamics are
    stochastic draw that randomness from the generator Gymnasium seeded at reset, so it is the same for every candidate too
    as long as the agent's actions are the same.)

    **Measurements** of an episode: `return`, the sum of the rewards; `steps`, the number of steps taken and `done`, 1.0 if the
    episode ended and 0.0 if `max_steps` cut it (both from the adapter); `terminated` and `truncated`, the final values of
    Gymnasium's two flags, kept apart (a pole that fell is terminated, a time limit is truncated); and the numeric entries of
    `info` that `info_fields` names, **summed** over the episode, and those in `last_info_fields`, whose **last** value is kept.

    **Episode length.** Gymnasium environments registered with a time limit (`max_episode_steps`, 500 for `CartPole-v1`)
    truncate themselves. `max_steps` is the adapter's own limit. By default it is the environment's `max_episode_steps` (an
    environment without one needs `max_steps`). If the adapter's limit is the smaller, the episode is cut by the adapter: `done`
    is 0 and `truncated` is 0. If Gymnasium's is the smaller (or equal), the environment truncates: `truncated` and `done` are 1.
    """

    def __init__(
        self,
        env: str | Callable[[], Any],
        *,
        max_steps: int | None = None,
        info_fields: Sequence[str] = (),
        last_info_fields: Sequence[str] = (),
        make_kwargs: Mapping[str, object] | None = None,
        role: str = "agent",
    ) -> None:
        _import_gymnasium()
        self._factory = _factory_of(env, make_kwargs or {})
        probe = self._factory()
        try:
            limit = getattr(getattr(probe, "spec", None), "max_episode_steps", None)
            self.observation_space = probe.observation_space
            self.action_space = probe.action_space
        finally:
            probe.close()
        if max_steps is None:
            if limit is None:
                raise ValueError("this environment has no max_episode_steps: give max_steps, the length at which episodes are cut")
            max_steps = int(limit)
        elif max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        names = (*info_fields, *last_info_fields)
        reserved = {"return", "terminated", "truncated", "steps", "done"} & set(names)
        if reserved:
            raise ValueError(f"info fields cannot be named {sorted(reserved)}: the adapter measures those itself")
        self.max_steps = max_steps
        self._info = tuple(info_fields)
        self._last_info = tuple(last_info_fields)
        self._inner = StepEnvironment(
            self._world,
            role=role,
            max_steps=max_steps,
            last=("terminated", "truncated", *self._last_info),
        )
        self.roles = (role,)

    def _world(self) -> _GymWorld:
        return _GymWorld(self._factory, self._info, self._last_info)

    def __repr__(self) -> str:
        fields = f"info={list(self._info)}, last_info={list(self._last_info)}"
        return f"GymnasiumEnvironment(env={self._factory!r}, max_steps={self.max_steps}, {fields})"

    def run_episode(self, agents: Mapping[str, Agent], scenario: Scenario, rng: RandomStream) -> EpisodeResult:
        return self._inner.run_episode(agents, scenario, rng)

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle by configuration: the environment's spaces and the world factory are rebuilt in the worker."""
        return (
            _rebuild,
            (self._factory, self.max_steps, self._info, self._last_info, self.roles[0]),
        )


def _rebuild(
    factory: Callable[[], Any], max_steps: int, info: tuple[str, ...], last_info: tuple[str, ...], role: str
) -> GymnasiumEnvironment:
    return GymnasiumEnvironment(factory, max_steps=max_steps, info_fields=info, last_info_fields=last_info, role=role)


@dataclass(frozen=True)
class LinearAgent:
    """What `LinearPolicy` decodes a genome into: `action = argmax(W·o + b)` for discrete actions, `clip(W·o + b)` otherwise."""

    weights: npt.NDArray[np.float64]
    bias: npt.NDArray[np.float64]
    low: npt.NDArray[np.float64] | None
    high: npt.NDArray[np.float64] | None

    def act(self, observation: object) -> object:
        scores: npt.NDArray[np.float64] = self.weights @ np.asarray(observation, dtype=np.float64).reshape(-1) + self.bias
        if self.low is None or self.high is None:
            return int(scores.argmax())
        return np.minimum(np.maximum(scores, self.low), self.high)


class LinearPolicy:
    """A minimal decoder for tests and notebooks: genome to a linear policy of the observation.

    The genome is the weight matrix `W` (`outputs × observation size`, row by row) followed by the bias `b` (`outputs`), so its
    length is `outputs × (observation size + 1)`: `dim`. For a **discrete** action space (`n` actions) there are `n` outputs and
    the action is the one with the largest score, `argmax(W·o + b)`; for a **continuous** (`Box`) action space the outputs are the
    action's components and the action is `W·o + b` clipped to the action space's bounds. Observation spaces must be a `Box`
    (flattened); anything else (dictionaries, tuples) is an error, as is any other action space.

    Build it from an environment (`LinearPolicy(environment)`, anything with `observation_space` and `action_space`, such as
    a `GymnasiumEnvironment` or a Gymnasium environment) or from the two spaces. `space(bound)` is a `Box` of genomes.
    """

    def __init__(
        self, environment: object | None = None, *, observation_space: object | None = None, action_space: object | None = None
    ) -> None:
        observation: Any = getattr(environment, "observation_space", None) if environment is not None else observation_space
        action: Any = getattr(environment, "action_space", None) if environment is not None else action_space
        if observation is None or action is None:
            raise ValueError("give an environment, or both observation_space and action_space")
        spaces: Any = _import_gymnasium().spaces  # Gymnasium's spaces are not typed for strict checking, and it is an optional import
        if not isinstance(observation, spaces.Box):
            raise TypeError(f"LinearPolicy needs a Box observation space (flattened), got {type(observation).__name__}")
        self.observation_size = int(np.prod(tuple(cast("Sequence[int]", observation.shape))))
        self._low: npt.NDArray[np.float64] | None = None
        self._high: npt.NDArray[np.float64] | None = None
        if isinstance(action, spaces.Discrete):
            self.outputs = int(cast("int", action.n))
            self.discrete = True
        elif isinstance(action, spaces.Box):
            self.outputs = int(np.prod(tuple(cast("Sequence[int]", action.shape))))
            self.discrete = False
            self._low = np.asarray(action.low, dtype=np.float64).reshape(-1)
            self._high = np.asarray(action.high, dtype=np.float64).reshape(-1)
        else:
            raise TypeError(f"LinearPolicy needs a Discrete or a Box action space, got {type(action).__name__}")
        self.dim = self.outputs * (self.observation_size + 1)

    def __repr__(self) -> str:
        kind = "discrete" if self.discrete else "continuous"
        return f"LinearPolicy(observation_size={self.observation_size}, outputs={self.outputs}, actions={kind})"

    def space(self, bound: float = 1.0) -> Box:
        """A box of genomes: every weight and bias in `[-bound, bound]`."""
        return Box(-bound, bound, dim=self.dim)

    def decode(self, genome: Array) -> LinearAgent:
        values = np.asarray(Backend().to_numpy(genome), dtype=np.float64)
        if values.shape != (self.dim,):
            raise ValueError(f"a genome of this policy has {self.dim} values, got shape {values.shape}")
        split = self.outputs * self.observation_size
        weights = values[:split].reshape(self.outputs, self.observation_size)
        return LinearAgent(weights, values[split:].copy(), self._low, self._high)

    def explain(self, genome: Array) -> Mapping[str, object]:
        """The weights and the bias of a genome, for reading."""
        agent = self.decode(genome)
        return {"weights": agent.weights.tolist(), "bias": agent.bias.tolist()}
