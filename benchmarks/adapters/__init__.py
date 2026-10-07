"""Algorithm adapters: one module per algorithm, each exposing `run(objective, dim, seed, params) -> RunInfo`."""

import importlib

from benchmarks.adapters.base import Adapter, RunInfo

__all__ = ["Adapter", "RunInfo", "load_adapter"]


def load_adapter(name: str) -> Adapter:
    """Find an adapter by name: a module in benchmarks.adapters, or a dotted module path for an external one."""
    module = importlib.import_module(name if "." in name else f"benchmarks.adapters.{name}")
    return module.run
