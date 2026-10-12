"""A test fixture for policy-driven batched worlds, not an example: a team of identical point masses that must reach a target.

It is the pattern a batched multi-agent world follows with `MLPPolicy`. Each scenario holds a team of `AGENTS` point masses on a
line, at different starting positions, all to be brought to one target against a constant drift. Every agent observes only
itself and the team's mean position, `(target - x, v, mean(x) - x)`, and acts with a force; **one genome decodes into one network,
shared by the whole team** (parameter sharing), so the observations are `(n, scenarios, agents, 3)` and the actions
`(n, scenarios, agents, 1)`, for `n` candidates evaluated at once.

Two implementations share the same maths (`simulate`): batched (`TeamEnvironment.run_batch`, which calls `agents.act_batch` at
every step in the array namespace of the run's backend) and per episode (`run_episode`, with the agent `MLPPolicy.decode` made).
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from array_api_compat import numpy as xp_numpy

from auxein.aggregators import Aggregator, mean
from auxein.backend import Array, backend_of
from auxein.core import Objective, ProblemSpec
from auxein.environments import EpisodeBatchResult, EpisodeResult, MLPPolicy, Scenario, ScenarioSet, Tanh
from auxein.random import RandomStream

AGENTS = 5
DT = 0.1
STEPS = 40
FORCE = 2.0
TOLERANCE = 0.15
EFFORT_WEIGHT = 0.02
OBSERVATIONS = 3
HIDDEN = (6,)

POLICY = MLPPolicy(OBSERVATIONS, HIDDEN, 1, output=Tanh(-FORCE, FORCE))
"""The shared network: the output is a force, squashed into the actuator's limits."""


def scenario_params(index: int, rng: RandomStream) -> dict[str, Any]:
    draws = rng.uniform((AGENTS + 2,)).tolist()
    return {"target": 1.0 + 2.0 * draws[0], "drift": 0.4 * (draws[1] - 0.5), "starts": [-2.0 + 4.0 * d for d in draws[2:]]}


def scenario_set(n: int = 6, seed: int = 4) -> ScenarioSet:
    return ScenarioSet.generate(scenario_params, n, seed=seed)


def simulate(
    xp: Any, act: Callable[[Array], Array], n: int, targets: Array, drifts: Array, starts: Array, dtype: Any, device: Any = None
) -> dict[str, Array]:
    """The team's whole simulation for `n` candidates on `s` scenarios at once; the single-episode path is `n = s = 1`.

    `act` maps the observations `(n, s, agents, 3)` to the forces `(n, s, agents)`. `targets` and `drifts` are `(s,)` and
    `starts` is `(s, agents)`. Returns `(n, s)` arrays, averaged over the agents of each team.
    """
    target, drift = targets[None, :, None], drifts[None, :, None]
    x = starts[None, :, :] * xp.ones((n, 1, 1), dtype=dtype, device=device)
    v = xp.zeros_like(x)
    effort = xp.zeros((n, targets.shape[0]), dtype=dtype, device=device)
    for _ in range(STEPS):
        centre = xp.mean(x, axis=-1, keepdims=True)
        observations = xp.stack([target - x, v, centre - x], axis=-1)
        u = act(observations)
        v = v + (u + drift) * DT
        x = x + v * DT
        effort = effort + xp.mean(xp.abs(u), axis=-1) * DT
    distance = xp.abs(target - x)
    return {
        "final_distance": xp.mean(distance, axis=-1),
        "effort": effort,
        "success": xp.mean(xp.where(distance < TOLERANCE, xp.ones_like(distance), xp.zeros_like(distance)), axis=-1),
    }


def _arrays(xp: Any, asarray: Callable[[Any], Array], scenarios: Sequence[Scenario]) -> tuple[Array, Array, Array]:
    targets = asarray(np.array([s.params["target"] for s in scenarios], dtype=np.float64))
    drifts = asarray(np.array([s.params["drift"] for s in scenarios], dtype=np.float64))
    starts = asarray(np.array([s.params["starts"] for s in scenarios], dtype=np.float64))
    return targets, drifts, starts


class TeamEnvironment:
    """The team of point masses, batched over candidates and scenarios (`run_batch`) or one episode at a time (`run_episode`)."""

    roles: tuple[str, ...] = ("brain",)

    def __repr__(self) -> str:
        return "TeamEnvironment()"

    def run_episode(self, agents: Mapping[str, Any], scenario: Scenario, rng: RandomStream) -> EpisodeResult:
        brain = agents["brain"]

        def act(observations: Array) -> Array:
            return np.asarray(brain.act(observations[0, 0]))[None, None, :, 0]  # one team: (agents, 3) -> (agents, 1)

        targets, drifts, starts = _arrays(xp_numpy, lambda a: a, [scenario])
        out = simulate(xp_numpy, act, 1, targets, drifts, starts, np.float64)
        return EpisodeResult({name: float(value[0, 0]) for name, value in out.items()})

    def run_batch(self, agents: Any, scenarios: Sequence[Scenario], rng: RandomStream) -> EpisodeBatchResult:
        backend = backend_of(agents.weights[0])
        xp = backend.xp
        targets, drifts, starts = _arrays(xp, backend.asarray, scenarios)

        def act(observations: Array) -> Array:
            return agents.act_batch(observations)[..., 0]  # (n, s, agents, 1) -> (n, s, agents)

        return EpisodeBatchResult(simulate(xp, act, agents.n, targets, drifts, starts, backend.dtype, backend.device))


class PerEpisodeDecoder:
    """The policy's decoder without `decode_batch`, so that `TeamEnvironment` takes the per-episode path."""

    def decode(self, genome: Any) -> Any:
        return POLICY.decode(genome)

    def __repr__(self) -> str:
        return "PerEpisodeDecoder()"


def aggregator() -> Aggregator:
    return Aggregator(
        objectives={"error": mean(lambda m: m["final_distance"] + EFFORT_WEIGHT * m["effort"])},
        descriptors={"success_rate": mean("success")},
    )


def problem() -> ProblemSpec[Any]:
    return ProblemSpec(POLICY.space(), (Objective("error"),), (), ("success_rate",))
