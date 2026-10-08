"""Scenarios and scenario sets (design doc §6.4): the fixed, seeded instances every candidate is evaluated on."""

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast, overload

from auxein.backend import Backend
from auxein.random import RandomStream, RunSeed

SET_FORMAT = 1


class Params(Mapping[str, object]):
    """The parameters of a scenario: a read-only mapping of JSON values.

    It is a class of its own, not a `MappingProxyType`, so that a scenario can be pickled to a worker process. Nested lists
    and dicts are copies owned by the scenario; do not modify them.
    """

    def __init__(self, values: Mapping[str, object]) -> None:
        try:
            self._values = cast("dict[str, object]", json.loads(json.dumps(dict(values), sort_keys=True)))
        except (TypeError, ValueError) as error:
            raise TypeError(f"scenario params must be JSON-serialisable (numbers, strings, booleans, lists, dicts): {error}") from error

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"Params({self._values!r})"

    def to_json(self) -> dict[str, object]:
        return dict(self._values)


@dataclass(frozen=True)
class Scenario:
    """One instance an agent is evaluated on: an id, its place in its set, a seed for the world's randomness, and parameters.

    `seed` is what makes every candidate face the same world (common random numbers, design doc §6.4): an environment draws
    its disturbances and noise from `scenario.rng()`, never from the agent's stream. `params` (initial geometry, sea state,
    a task instance...) must be JSON-serialisable, because scenarios are fingerprinted, saved and recorded. Two scenarios
    are the same when all their fields are; they hash by id.
    """

    id: str
    index: int
    seed: int
    params: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("a scenario needs a non-empty id")
        if self.index < 0:
            raise ValueError(f"the index of a scenario must not be negative, got {self.index}")
        RunSeed(self.seed)  # validates: a non-negative int
        object.__setattr__(self, "params", Params(self.params))

    def __hash__(self) -> int:
        return hash(self.id)

    def rng(self, *keys: int, backend: Backend | None = None) -> RandomStream:
        """A stream for the world's randomness, derived from this scenario's seed alone (numpy on the CPU by default).

        Every candidate calling it gets the same numbers, which is the point. Independent streams within a scenario (wind and
        waves, say) take different `keys`.
        """
        return RunSeed(self.seed).stream("world", *keys, backend=backend)


def _fingerprint(scenarios: Sequence[Scenario]) -> str:
    document = [[s.id, s.seed, cast("Params", s.params).to_json()] for s in scenarios]
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class ScenarioSet:
    """An ordered, immutable collection of scenarios, identified by a fingerprint.

    The scenarios' indices are their positions, and ids are unique. The `fingerprint` is a stable hash of the ids, seeds and
    parameters in order: it goes into the episode evaluator's description, so resuming a run with a different scenario set is
    refused (design doc §10.4). Build one from a list of parameters (`from_params`), from a generating function with a seed
    (`generate`), or as a selection and a held-out set that cannot overlap (`generate_split`, or `split` on a set).
    """

    def __init__(self, scenarios: Sequence[Scenario]) -> None:
        items = tuple(scenarios)
        if not items:
            raise ValueError("a scenario set needs at least one scenario")
        for position, scenario in enumerate(items):
            if scenario.index != position:
                raise ValueError(f"scenario {scenario.id!r} has index {scenario.index} but is at position {position} of the set")
        ids = [s.id for s in items]
        if len(set(ids)) != len(ids):
            raise ValueError(f"scenario ids must be unique, but {sorted({i for i in ids if ids.count(i) > 1})} appear more than once")
        self._scenarios = items
        self.fingerprint = _fingerprint(items)

    @classmethod
    def from_params(cls, params: Sequence[Mapping[str, object]], *, seed: int = 0, ids: Sequence[str] | None = None) -> "ScenarioSet":
        """A set from a list of parameter mappings. Each scenario's seed is derived from `seed` and its index."""
        if ids is not None and len(ids) != len(params):
            raise ValueError(f"{len(ids)} ids for {len(params)} scenarios")
        run = RunSeed(seed)
        return cls([Scenario(ids[i] if ids is not None else f"s{i:04d}", i, _scenario_seed(run, i), p) for i, p in enumerate(params)])

    @classmethod
    def generate(cls, fn: Callable[[int, RandomStream], Mapping[str, object]], n: int, *, seed: int) -> "ScenarioSet":
        """`n` scenarios whose parameters are `fn(index, rng)`, with `rng` the stream of that index derived from `seed`."""
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        return cls.from_params(_draw(fn, 0, n, seed), seed=seed)

    @classmethod
    def generate_split(
        cls, fn: Callable[[int, RandomStream], Mapping[str, object]], n_selection: int, n_held_out: int, *, seed: int
    ) -> tuple["ScenarioSet", "ScenarioSet"]:
        """A selection set and a held-out set generated from one seed, which cannot overlap: they are consecutive indices of
        one stream of scenarios, so their parameters, seeds and ids all differ."""
        if n_selection < 1 or n_held_out < 1:
            raise ValueError(f"both sets need at least one scenario, got {n_selection} and {n_held_out}")
        return cls.generate(fn, n_selection + n_held_out, seed=seed).split(n_selection, n_held_out)

    def split(self, n_selection: int, n_held_out: int) -> tuple["ScenarioSet", "ScenarioSet"]:
        """The first `n_selection` scenarios and the next `n_held_out`, as two sets (disjoint by construction).

        The scenarios keep their ids and seeds, and are re-indexed from 0 within their new set.
        """
        if n_selection < 1 or n_held_out < 1 or n_selection + n_held_out > len(self):
            raise ValueError(f"cannot split {len(self)} scenarios into {n_selection} for selection and {n_held_out} held out")
        first = self._scenarios[:n_selection]
        rest = self._scenarios[n_selection : n_selection + n_held_out]
        return ScenarioSet(first), ScenarioSet([replace(s, index=i) for i, s in enumerate(rest)])

    def __len__(self) -> int:
        return len(self._scenarios)

    def __iter__(self) -> Iterator[Scenario]:
        return iter(self._scenarios)

    @overload
    def __getitem__(self, index: int) -> Scenario: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[Scenario, ...]: ...
    def __getitem__(self, index: int | slice) -> Scenario | tuple[Scenario, ...]:
        return self._scenarios[index]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in self._scenarios)

    def __repr__(self) -> str:
        return f"ScenarioSet({len(self)} scenarios, fingerprint={self.fingerprint[:12]})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ScenarioSet) and self.fingerprint == other.fingerprint

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    # --- JSON ---

    def to_json(self) -> dict[str, object]:
        return {
            "format": SET_FORMAT,
            "fingerprint": self.fingerprint,
            "scenarios": [{"id": s.id, "seed": s.seed, "params": cast("Params", s.params).to_json()} for s in self._scenarios],
        }

    @classmethod
    def from_json(cls, document: Mapping[str, object]) -> "ScenarioSet":
        if document.get("format") != SET_FORMAT:
            raise ValueError(f"unknown scenario set format {document.get('format')!r}; this version reads format {SET_FORMAT}")
        items = cast("list[dict[str, object]]", document["scenarios"])
        loaded = cls(
            [
                Scenario(cast("str", d["id"]), i, cast("int", d["seed"]), cast("Mapping[str, object]", d["params"]))
                for i, d in enumerate(items)
            ]
        )
        if loaded.fingerprint != document.get("fingerprint"):
            raise ValueError("the scenario set was modified after it was saved: its fingerprint does not match its contents")
        return loaded

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioSet":
        return cls.from_json(cast("Mapping[str, object]", json.loads(Path(path).read_text())))


def _scenario_seed(run: RunSeed, index: int) -> int:
    """The seed of the world of scenario `index`: 32 bits derived from the set's seed."""
    return int(run.sequence("scenario-seed", index).generate_state(1)[0])


def _draw(fn: Callable[[int, RandomStream], Mapping[str, object]], first: int, n: int, seed: int) -> list[Mapping[str, object]]:
    run = RunSeed(seed)
    return [fn(i, run.stream("scenarios", i)) for i in range(first, first + n)]
