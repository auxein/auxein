"""Small environments for testing the episode evaluator. Importable by name, so that worker processes can use them."""

import asyncio
import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from auxein.environments import EpisodeResult, Scenario
from auxein.random import RandomStream

GAUGE_LOCK = threading.Lock()


class Gauge:
    """Counts the episodes in progress, and the most there ever were (within one process)."""

    def __init__(self) -> None:
        self.current = self.peak = self.calls = 0

    def enter(self) -> None:
        with GAUGE_LOCK:
            self.current += 1
            self.calls += 1
            self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        with GAUGE_LOCK:
            self.current -= 1


class Probe:
    """Reports the first draw of the world's stream and of the agent's stream, and the agent's first gene."""

    roles = ("agent",)

    def __repr__(self) -> str:
        return "Probe()"

    def run_episode(self, agents: Mapping[str, Any], scenario: Scenario, rng: RandomStream) -> EpisodeResult:
        return EpisodeResult(
            {
                "world": float(scenario.rng().normal((1,))[0]),
                "agent": float(rng.normal((1,))[0]),
                "gene": float(np.asarray(agents["agent"]).ravel()[0]),
                "scenario": float(scenario.index),
            }
        )


class AsyncProbe(Probe):
    """The same, as an `async def`, with a gauge and a pause so that episodes overlap."""

    def __init__(self, gauge: Gauge | None = None, pause: float = 0.005) -> None:
        self.gauge, self.pause = gauge or Gauge(), pause

    def __repr__(self) -> str:
        return "AsyncProbe()"

    async def run_episode(self, agents: Mapping[str, Any], scenario: Scenario, rng: RandomStream) -> EpisodeResult:  # type: ignore[override]
        self.gauge.enter()
        try:
            await asyncio.sleep(self.pause)
            return Probe.run_episode(self, agents, scenario, rng)
        finally:
            self.gauge.leave()


class SlowProbe(Probe):
    """A synchronous probe that takes a moment and counts how many run at once (threads of one process)."""

    def __init__(self, gauge: Gauge | None = None, pause: float = 0.01) -> None:
        self.gauge, self.pause = gauge or Gauge(), pause

    def run_episode(self, agents: Mapping[str, Any], scenario: Scenario, rng: RandomStream) -> EpisodeResult:
        self.gauge.enter()
        try:
            time.sleep(self.pause)
            return super().run_episode(agents, scenario, rng)
        finally:
            self.gauge.leave()


class Troubled(Probe):
    """Fails in the scenarios it is told to: by raising, by returning a failure, by taking too long, or by killing its process."""

    def __init__(
        self, raises: tuple[str, ...] = (), returns: tuple[str, ...] = (), slow: tuple[str, ...] = (), seconds: float = 30.0, kills: tuple[str, ...] = ()
    ) -> None:
        self.raises, self.returns, self.slow, self.seconds, self.kills = raises, returns, slow, seconds, kills

    def __repr__(self) -> str:
        return f"Troubled({self.raises}, {self.returns}, {self.slow}, {self.kills})"

    def run_episode(self, agents: Mapping[str, Any], scenario: Scenario, rng: RandomStream) -> EpisodeResult:
        if scenario.id in self.raises:
            raise ValueError(f"the simulator diverged in {scenario.id}")
        if scenario.id in self.returns:
            return EpisodeResult.failed(f"the agent crashed in {scenario.id}")
        if scenario.id in self.slow:
            time.sleep(self.seconds)
        if scenario.id in self.kills:
            import os
            import signal

            os.kill(os.getpid(), signal.SIGKILL)
        return super().run_episode(agents, scenario, rng)


class AsyncTroubled(Troubled):
    async def run_episode(self, agents: Mapping[str, Any], scenario: Scenario, rng: RandomStream) -> EpisodeResult:  # type: ignore[override]
        if scenario.id in self.raises:
            raise ValueError(f"the simulator diverged in {scenario.id}")
        if scenario.id in self.returns:
            return EpisodeResult.failed(f"the agent crashed in {scenario.id}")
        if scenario.id in self.slow:
            await asyncio.sleep(self.seconds)
        return Probe.run_episode(self, agents, scenario, rng)


class Wrong:
    """Returns something that is not an `EpisodeResult`."""

    roles = ("agent",)

    def run_episode(self, agents: Mapping[str, Any], scenario: Scenario, rng: RandomStream) -> object:
        return {"not": "a result"}


class Shifty(Probe):
    """Reports different measurement names for different scenarios."""

    def run_episode(self, agents: Mapping[str, Any], scenario: Scenario, rng: RandomStream) -> EpisodeResult:
        return EpisodeResult({"a": 1.0} if scenario.index == 0 else {"b": 1.0})


class Diverging(Probe):
    """Reports a NaN for one scenario, as a diverged simulation does."""

    def run_episode(self, agents: Mapping[str, Any], scenario: Scenario, rng: RandomStream) -> EpisodeResult:
        value = float("nan") if scenario.index == 1 else 1.0
        return EpisodeResult({"world": value, "agent": 0.0, "gene": 0.0, "scenario": float(scenario.index)})


class TwoRoles(Probe):
    roles = ("evolved", "opponent")


class ListDecoder:
    """Decodes a genome into a plain list of floats (picklable)."""

    def decode(self, genome: Any) -> list[float]:
        from auxein.backend import Backend

        return [float(x) for x in Backend().to_numpy(genome).ravel()]

    def __repr__(self) -> str:
        return "ListDecoder()"


class BrokenDecoder:
    def decode(self, genome: Any) -> Any:
        if float(np.asarray(genome).ravel()[0]) > 100.0:
            raise ValueError("cannot decode this genome")
        return [0.0]

    def __repr__(self) -> str:
        return "BrokenDecoder()"

