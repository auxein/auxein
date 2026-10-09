"""An adapter that makes a step-level world an `Environment` (design doc §6.2).

Many simulators and reinforcement-learning environments are written as `reset` and `step`. This adapter runs that loop. It is
deliberately small and generic; the Gymnasium adapter (step 9) builds on it.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

from auxein.environments.episode import EpisodeResult
from auxein.environments.protocols import Agent
from auxein.environments.scenario import Scenario
from auxein.random import RandomStream

RESERVED = ("steps", "done")


class StepWorld(Protocol):
    """A world that is stepped. A new one is built for every episode (see `StepEnvironment`), so it may keep its state."""

    def reset(self, scenario: Scenario, rng: RandomStream) -> object:
        """Start the scenario and return the first observation. `rng` is the world's stream, from the scenario's seed."""
        ...

    def step(self, action: object) -> tuple[object, Mapping[str, float], bool]:
        """Apply an action. Returns the next observation, the **measurement updates** of this step (name to value), and
        whether the episode is over."""
        ...


class StepAgent(Protocol):
    def act(self, observation: object) -> object: ...


class StepEnvironment:
    """Runs a `StepWorld` and an agent as an episode: `observation = reset(...)`, then `act` and `step` until the world says it
    is done or `max_steps` have been taken.

    The agent is anything with `act(observation) -> action`; if it also has `reset(rng)` that is called first with the
    agent's stream for the episode, which is how a stochastic policy gets its randomness.

    **How measurements accumulate.** The updates that `step` returns are **summed** over the episode (effort per step becomes
    total effort), except for the names listed in `last`, whose **last** value is kept (a final distance). The adapter adds two
    measurements itself, so worlds must not use these names: `steps`, the number of steps taken, and `done`, 1.0 if the world
    ended the episode and 0.0 if `max_steps` did. A name that is never updated is simply absent.

    `world_factory` is called once per episode (a class works), so episodes share nothing and can run concurrently, in
    threads or in worker processes if the factory is picklable.
    """

    def __init__(self, world_factory: Callable[[], StepWorld], *, role: str = "agent", max_steps: int, last: Sequence[str] = ()) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        if set(last) & set(RESERVED):
            raise ValueError(f"{sorted(set(last) & set(RESERVED))} are measured by the adapter itself")
        self._factory = world_factory
        self.roles = (role,)
        self._max_steps = max_steps
        self._last = frozenset(last)

    def __repr__(self) -> str:
        name = getattr(self._factory, "__qualname__", type(self._factory).__name__)
        return f"StepEnvironment(world={name}, role={self.roles[0]!r}, max_steps={self._max_steps}, last={sorted(self._last)})"

    def run_episode(self, agents: Mapping[str, Agent], scenario: Scenario, rng: RandomStream) -> EpisodeResult:
        agent = cast("StepAgent", agents[self.roles[0]])
        start = getattr(agent, "reset", None)
        if callable(start):
            start(rng)
        world = self._factory()
        observation = world.reset(scenario, scenario.rng())
        totals: dict[str, float] = {}
        steps = 0
        done = False
        for _ in range(self._max_steps):
            action = agent.act(observation)
            observation, update, done = world.step(action)
            steps += 1
            for name, value in update.items():
                if name in RESERVED:
                    raise ValueError(f"the world reported {name!r}, which the adapter measures itself")
                totals[name] = float(value) if name in self._last else totals.get(name, 0.0) + float(value)
            if done:
                break
        return EpisodeResult({**totals, "steps": float(steps), "done": 1.0 if done else 0.0})
