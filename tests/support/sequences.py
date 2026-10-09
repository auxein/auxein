"""A structured-genome fixture, not an example: evolve a sequence of tokens to match a hidden target.

The objective is the edit distance to the hidden target; a constraint caps the length. A deliberately non-deterministic
"language model" operator rewrites a sequence, and counts its calls in a file so that tests can see how often it really ran.
"""

import os
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from auxein.core import Objective, ProblemSpec, Proposal, Result
from auxein.random import RandomStream
from auxein.spaces import SequenceSpace

VOCABULARY = tuple("abcdefghijkl")
TARGET = tuple("badcfehgjilk")  # twelve tokens, none repeated
MAX_ALLOWED = 14
SPACE = SequenceSpace(VOCABULARY, min_length=2, max_length=18)


def edit_distance(a: Sequence[object], b: Sequence[object]) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def distance(genome: Sequence[object]) -> Result:
    """Edit distance to the hidden target, with the length over `MAX_ALLOWED` as the constraint violation."""
    return Result(
        objectives={"distance": float(edit_distance(genome, TARGET))},
        constraints={"too_long": float(max(0, len(genome) - MAX_ALLOWED))},
    )


def problem() -> ProblemSpec[tuple[object, ...]]:
    return ProblemSpec(SPACE, (Objective("distance"),), ("too_long",))


class FakeLLM:
    """Rewrites a sequence by replacing one item, using *its own unseeded* randomness and a call counter: two calls with the same
    input differ, so only recording can make a run that uses it reproducible. Each real call appends a line to the file named by
    `AUXEIN_CALL_LOG` (in whatever process it runs), if that variable is set.
    """

    def __init__(self, name: str = "fake-llm") -> None:
        self.name = name
        self.calls = 0
        self._random = random.Random()  # seeded from the operating system, on purpose

    def __repr__(self) -> str:
        return f"FakeLLM({self.name!r})"

    def propose(self, parents: Sequence[tuple[object, ...]], rng: RandomStream) -> Proposal[tuple[object, ...]]:
        self.calls += 1
        path = os.environ.get("AUXEIN_LLM_LOG")
        if path:
            with open(path, "a") as handle:
                handle.write("call\n")
        genome = list(parents[0])
        if genome:
            genome[self._random.randrange(len(genome))] = self._random.choice(VOCABULARY)
        return Proposal(tuple(genome), {"tokens": 12.0 + len(genome)}, {"model": "fake", "call": self.calls})


@dataclass
class ActionPlan:
    """An open-loop agent for the step-level point mass: it plays its sequence of forces, then keeps the last one."""

    actions: tuple[float, ...]

    def act(self, observation: object) -> float:
        self.step = getattr(self, "step", -1) + 1
        return float(self.actions[min(self.step, len(self.actions) - 1)]) if self.actions else 0.0


class PlanDecoder:
    def decode(self, genome: tuple[object, ...]) -> ActionPlan:
        return ActionPlan(tuple(float(x) for x in genome))  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return "PlanDecoder()"


FORCES = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
PLAN_SPACE = SequenceSpace(FORCES, min_length=1, max_length=24)

_ = np
