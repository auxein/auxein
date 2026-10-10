"""Multi-objective benchmark configurations (TOML with `kind = "multi-objective"`, see benchmarks/configs/)."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.config import AlgorithmConfig
from benchmarks.mo_problems import MO_PROBLEMS

KIND = "multi-objective"


@dataclass(frozen=True)
class MOProblemConfig:
    name: str
    budget: int  # fitness evaluations per run


@dataclass(frozen=True)
class MOConfig:
    name: str
    base_seed: int
    runs: int
    problems: tuple[MOProblemConfig, ...]
    algorithms: tuple[AlgorithmConfig, ...]
    raw: dict[str, Any]

    def algorithm(self, name: str) -> AlgorithmConfig:
        for algorithm in self.algorithms:
            if algorithm.name == name:
                return algorithm
        raise KeyError(name)


def parse_mo_config(raw: dict[str, Any]) -> MOConfig:
    if raw.get("kind") != KIND:
        raise ValueError(f"not a multi-objective config: kind = {raw.get('kind')!r}, expected {KIND!r}")
    algorithms = tuple(AlgorithmConfig(a["name"], a["adapter"], dict(a.get("params", {}))) for a in raw["algorithms"])
    names = [a.name for a in algorithms]
    if len(set(names)) != len(names):
        raise ValueError(f"algorithm names must be unique: {names}")
    problems = tuple(MOProblemConfig(p["name"], int(p["budget"])) for p in raw["problems"])
    unknown = [p.name for p in problems if p.name not in MO_PROBLEMS]
    if unknown:
        raise ValueError(f"unknown multi-objective problems {unknown}, available: {sorted(MO_PROBLEMS)}")
    return MOConfig(raw["name"], raw.get("base_seed", 0), raw["runs"], problems, algorithms, raw)


def load_mo_config(path: str | Path) -> MOConfig:
    with open(path, "rb") as f:
        return parse_mo_config(tomllib.load(f))


def config_kind(path: str | Path) -> str:
    """`"multi-objective"` for a multi-objective config, `"single-objective"` for the others."""
    with open(path, "rb") as f:
        return str(tomllib.load(f).get("kind", "single-objective"))
