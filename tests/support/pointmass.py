"""A test fixture for the agent layer, not an example: a point mass that a PD controller must bring to a target.

The world is a point on a line, starting at rest at 0, pushed by the controller's force and by a constant drift, with a little
noise on the force. A scenario sets the target and the drift in its params and the noise through its own seed. The genome is
three gains (`kp`, `kd`, `bias`) of a controller `u = clip(kp * (target - x) - kd * v + bias)`.

It is implemented three ways that share the same maths, to test that the paths of the agent layer agree: per episode
(`PointMassEnvironment.run_episode`), batched over candidates and scenarios (`run_batch`, in the array namespace of the run's
backend), and step by step (`PointMassWorld` and `GainController`, through `StepEnvironment`).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from array_api_compat import numpy as xp_numpy

from auxein.aggregators import Aggregator, maximum, mean
from auxein.backend import Array, Backend, backend_of
from auxein.core import ProblemSpec
from auxein.environments import EpisodeBatchResult, EpisodeResult, Scenario, ScenarioSet, StepEnvironment
from auxein.random import RandomStream
from auxein.spaces import Box

DT = 0.1
STEPS = 60
TOLERANCE = 0.05
FORCE_LIMIT = 2.0
NOISE = 0.05
OVERSHOOT_LIMIT = 0.5
SPACE = Box([0.0, 0.0, -1.0], [10.0, 10.0, 1.0])
"""The genome: `kp`, `kd`, `bias`."""


def scenario_params(index: int, rng: RandomStream) -> dict[str, float]:
    draws = rng.uniform((2,)).tolist()
    return {"target": 1.0 + 2.0 * draws[0], "drift": 0.4 * (draws[1] - 0.5)}


def scenario_set(n: int = 8, seed: int = 3) -> ScenarioSet:
    return ScenarioSet.generate(scenario_params, n, seed=seed)


def world_noise(scenario: Scenario) -> np.ndarray:
    """The noise on the force over the episode: it depends on the scenario's seed alone, so every candidate sees it."""
    return scenario.rng().normal((STEPS,)) * NOISE


def simulate(xp: Any, gains: Array, targets: Array, drifts: Array, noise: Array, dtype: Any, device: Any = None) -> dict[str, Array]:
    """The whole simulation for `n` gain triples on `s` scenarios at once; the single-episode path is `n = s = 1`.

    `gains` is `(n, 3)`, `targets` and `drifts` are `(s,)`, `noise` is `(s, STEPS)`. Returns `(n, s)` arrays. An episode ends
    when the mass has settled on the target; the state then stays frozen, so that nothing more is counted.
    """
    kp, kd, bias = (gains[:, i][:, None] for i in range(3))
    target, drift = targets[None, :], drifts[None, :]
    shape = (gains.shape[0], targets.shape[0])
    zeros = xp.zeros(shape, dtype=dtype, device=device)
    x, v, effort, steps, overshoot = zeros, zeros, zeros, zeros, zeros
    done = xp.zeros(shape, dtype=xp.bool, device=device)
    for t in range(STEPS):
        error = target - x
        u = xp.clip(kp * error - kd * v + bias, -FORCE_LIMIT, FORCE_LIMIT)
        v_next = v + (u + drift + noise[None, :, t]) * DT
        x_next = x + v_next * DT
        active = ~done
        x = xp.where(active, x_next, x)
        v = xp.where(active, v_next, v)
        effort = effort + xp.where(active, xp.abs(u) * DT, zeros)
        steps = steps + xp.where(active, xp.ones(shape, dtype=dtype, device=device), zeros)
        overshoot = xp.maximum(overshoot, xp.where(active, x - target, zeros))
        done = done | ((xp.abs(target - x) < TOLERANCE) & (xp.abs(v) < TOLERANCE))
    distance = xp.abs(target - x)
    return {
        "final_distance": distance,
        "steps": steps,
        "effort": effort,
        "success": xp.where(distance < TOLERANCE, xp.ones(shape, dtype=dtype, device=device), zeros),
        "max_overshoot": overshoot,
    }


class GainsDecoder:
    """A genome of three gains as the controller's gains: a float64 numpy array (picklable, whatever the backend)."""

    def decode(self, genome: Any) -> np.ndarray:
        return np.asarray(Backend().to_numpy(genome), dtype=np.float64).copy()

    def decode_batch(self, genomes: Array) -> Array:
        return genomes

    def __repr__(self) -> str:
        return "GainsDecoder()"


class PerEpisodeDecoder:
    """The same decoder without `decode_batch`, so that an environment with `run_batch` still takes the per-episode path."""

    def decode(self, genome: Any) -> np.ndarray:
        return GainsDecoder().decode(genome)

    def __repr__(self) -> str:
        return "PerEpisodeDecoder()"


class PointMassEnvironment:
    """The point mass, per episode and batched. Raw measurements only: final distance, steps, effort, success, overshoot."""

    roles = ("controller",)

    def __repr__(self) -> str:
        return "PointMassEnvironment()"

    def run_episode(self, agents: Mapping[str, Any], scenario: Scenario, rng: RandomStream) -> EpisodeResult:
        gains = np.asarray(agents["controller"], dtype=np.float64)[None, :]
        out = simulate(
            xp_numpy,  # numpy's array API namespace: plain `numpy` has `device=`, `bool` and `concat` only from 2.0, and 1.26 is supported
            gains,
            np.array([scenario.params["target"]], dtype=np.float64),
            np.array([scenario.params["drift"]], dtype=np.float64),
            world_noise(scenario)[None, :],
            np.float64,
        )
        return EpisodeResult({name: float(value[0, 0]) for name, value in out.items()})

    def run_batch(self, agents: Array, scenarios: Sequence[Scenario], rng: RandomStream) -> EpisodeBatchResult:
        backend = backend_of(agents)
        xp = backend.xp
        targets = backend.asarray(np.array([s.params["target"] for s in scenarios], dtype=np.float64))
        drifts = backend.asarray(np.array([s.params["drift"] for s in scenarios], dtype=np.float64))
        noise = backend.asarray(np.stack([world_noise(s) for s in scenarios]))
        return EpisodeBatchResult(simulate(xp, agents, targets, drifts, noise, backend.dtype, backend.device))


class PointMassWorld:
    """The same point mass, stepped: `reset` and `step`, with plain Python floats."""

    def reset(self, scenario: Scenario, rng: RandomStream) -> tuple[float, float, float]:
        self._target = float(scenario.params["target"])  # type: ignore[arg-type]
        self._drift = float(scenario.params["drift"])  # type: ignore[arg-type]
        self._noise = world_noise(scenario).tolist()
        self._t = 0
        self._x = self._v = 0.0
        self._overshoot = 0.0
        return self._x, self._v, self._target

    def step(self, action: object) -> tuple[object, Mapping[str, float], bool]:
        u = min(max(float(action), -FORCE_LIMIT), FORCE_LIMIT)  # type: ignore[arg-type]
        self._v = self._v + (u + self._drift + self._noise[self._t]) * DT
        self._x = self._x + self._v * DT
        self._t += 1
        self._overshoot = max(self._overshoot, self._x - self._target)
        distance = abs(self._target - self._x)
        settled = distance < TOLERANCE and abs(self._v) < TOLERANCE
        update = {
            "effort": abs(u) * DT,
            "final_distance": distance,
            "success": 1.0 if distance < TOLERANCE else 0.0,
            "max_overshoot": self._overshoot,
        }
        return (self._x, self._v, self._target), update, settled


class GainController:
    """The controller as an agent of a step-level world."""

    def __init__(self, gains: np.ndarray) -> None:
        self._kp, self._kd, self._bias = (float(g) for g in gains)

    def act(self, observation: object) -> float:
        x, v, target = observation  # type: ignore[misc]
        return self._kp * (target - x) - self._kd * v + self._bias


class StepGainsDecoder:
    def decode(self, genome: Any) -> GainController:
        return GainController(GainsDecoder().decode(genome))

    def __repr__(self) -> str:
        return "StepGainsDecoder()"


def step_environment() -> StepEnvironment:
    return StepEnvironment(PointMassWorld, role="controller", max_steps=STEPS, last=("final_distance", "success", "max_overshoot"))


def relu(array: Array) -> Array:
    return (array + abs(array)) / 2.0  # works on numpy arrays and torch tensors alike


def aggregator() -> Aggregator:
    return Aggregator(
        objectives={"error": mean(lambda m: m["final_distance"] + 0.02 * m["effort"])},
        constraints={"overshoot": maximum(lambda m: relu(m["max_overshoot"] - OVERSHOOT_LIMIT))},
        descriptors={"success_rate": mean("success")},
    )


def problem() -> ProblemSpec[Any]:
    from auxein.core import Objective

    return ProblemSpec(SPACE, (Objective("error"),), ("overshoot",), ("success_rate",))
