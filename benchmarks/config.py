"""Benchmark configurations (TOML, see benchmarks/configs/)."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str
    adapter: str  # a module in benchmarks.adapters, or a dotted module path
    params: dict[str, Any]


@dataclass(frozen=True)
class OverheadConfig:
    budget: int
    repeats: int
    dims: tuple[int, ...]
    population_sizes: tuple[int, ...]  # for the algorithms that have a population size parameter
    algorithms: tuple[str, ...] | None  # names; all of the algorithms when None


@dataclass(frozen=True)
class Config:
    name: str
    base_seed: int
    runs: int
    budget_per_dim: int
    dims: tuple[int, ...]
    problems: tuple[str, ...]
    targets: tuple[float, ...]
    algorithms: tuple[AlgorithmConfig, ...]
    overhead: OverheadConfig | None
    raw: dict[str, Any]

    def budget(self, dim: int) -> int:
        return self.budget_per_dim * dim

    def algorithm(self, name: str) -> AlgorithmConfig:
        for algorithm in self.algorithms:
            if algorithm.name == name:
                return algorithm
        raise KeyError(name)


def parse_config(raw: dict[str, Any]) -> Config:
    algorithms = tuple(AlgorithmConfig(a["name"], a["adapter"], dict(a.get("params", {}))) for a in raw["algorithms"])
    names = [a.name for a in algorithms]
    if len(set(names)) != len(names):
        raise ValueError(f"algorithm names must be unique: {names}")

    overhead = None
    if "overhead" in raw:
        o = raw["overhead"]
        selected = tuple(o["algorithms"]) if "algorithms" in o else None
        unknown = set(selected or ()) - set(names)
        if unknown:
            raise ValueError(f"overhead.algorithms refers to unknown algorithms: {sorted(unknown)}")
        overhead = OverheadConfig(o["budget"], o["repeats"], tuple(o["dims"]), tuple(o["population_sizes"]), selected)

    return Config(
        name=raw["name"],
        base_seed=raw.get("base_seed", 0),
        runs=raw["runs"],
        budget_per_dim=raw["budget_per_dim"],
        dims=tuple(raw["dims"]),
        problems=tuple(raw["problems"]),
        targets=tuple(float(t) for t in raw["targets"]),
        algorithms=algorithms,
        overhead=overhead,
        raw=raw,
    )


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        return parse_config(tomllib.load(f))
