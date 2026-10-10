"""Fixtures for the multi-objective strategy, not examples: ZDT1 with its known front, and a generic evaluation builder."""

from typing import Any

import numpy as np

from auxein.backend import Array, Backend
from auxein.core import Candidate, Cost, Evaluation, EvaluationBatch, Objective, ProblemSpec, Status
from auxein.spaces import Box

F1, F2 = Objective("f1"), Objective("f2")
ZDT_DIM = 30
ZDT_SPACE = Box(0.0, 1.0, dim=ZDT_DIM)


def zdt1(X: np.ndarray) -> np.ndarray:
    """ZDT1 on `(n, d)` genomes in [0, 1]: two objectives to minimise, with the front `f2 = 1 - sqrt(f1)` at `x[1:] = 0`."""
    x = np.asarray(X, dtype=np.float64)
    f1 = x[:, 0]
    g = 1.0 + 9.0 * x[:, 1:].mean(axis=1)
    return np.stack([f1, g * (1.0 - np.sqrt(f1 / g))], axis=1)


def zdt1_batch(X: Array, backend: Backend) -> np.ndarray:
    return zdt1(backend.to_numpy(X))


def distance_to_front(front: np.ndarray) -> float:
    """The mean vertical distance of points `(f1, f2)` to ZDT1's true front: 0 on it, about 3 for random points."""
    return float(np.mean(np.abs(front[:, 1] - (1.0 - np.sqrt(front[:, 0])))))


def problem(space: Any = ZDT_SPACE, constraints: tuple[str, ...] = ()) -> ProblemSpec[Any]:
    return ProblemSpec(space, (F1, F2), constraints)


def evaluations(batch: Any, objectives: Any, constraints: Any = None, failed: Any = None) -> list[Evaluation[Any]]:
    """Evaluations of a batch: `objectives(candidate)` gives `(f1, f2)`, `constraints(candidate)` a violation (default none),
    and `failed(candidate)` makes a candidate fail."""
    out: list[Evaluation[Any]] = []
    for c in batch.candidates:
        if failed is not None and failed(c):
            out.append(Evaluation.failed(c, Status.FAILED, "boom"))
            continue
        f1, f2 = objectives(c)
        violation = {} if constraints is None else {"c": float(constraints(c))}
        out.append(Evaluation(c, Status.OK, {"f1": float(f1), "f2": float(f2)}, violation, cost=Cost()))
    return out


def tell(strategy: Any, batch: Any, objectives: Any, constraints: Any = None, failed: Any = None) -> EvaluationBatch[Any]:
    results = EvaluationBatch(evaluations(batch, objectives, constraints, failed))
    strategy.tell(results)
    return results


def zdt1_of(backend: Backend) -> Any:
    return lambda c: tuple(zdt1(backend.to_numpy(c.genome)[None, :])[0])


_ = (Candidate,)
