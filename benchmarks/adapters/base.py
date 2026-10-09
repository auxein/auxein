"""What an algorithm adapter looks like."""

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from benchmarks.objective import CountingObjective


@dataclass
class RunInfo:
    """Algorithm-specific extras of one run. They are stored with the run but never used to judge quality."""

    generations: int | None = None  # completed generations, for generational algorithms
    stop_reason: str = "budget"
    evals_per_generation: float | None = None  # average cost of a completed generation, in fitness evaluations
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def backend_from(params: dict[str, Any]) -> Any:
    """The Auxein `Backend` named by an adapter's params, or None when they name none (the driver's numpy-float64 default).

    The core adapters take `backend` (`"numpy"` or `"torch"`), `precision` (`"float64"` or `"float32"`) and `device` (`"cpu"`,
    `"cuda"`, `"mps"`), so that the same benchmark can be run on each backend: `backend = "torch"`, `precision = "float32"`.
    Without any of them nothing changes, so the existing entries and their committed reports keep meaning what they meant.
    """
    if not any(key in params for key in ("backend", "precision", "device")):
        return None
    from auxein.backend import Backend

    return Backend(params.get("backend", "numpy"), params.get("device", "cpu"), params.get("precision", "float64"))


class Adapter(Protocol):
    """An adapter is a module with a `run` function of this shape.

    It must evaluate points only through `objective(x)` and finish cleanly when that raises `BudgetExhausted`
    (a partial generation is fine). All of its randomness must come from `seed`.
    """

    def __call__(self, objective: CountingObjective, dim: int, seed: int, params: dict[str, Any]) -> RunInfo: ...
