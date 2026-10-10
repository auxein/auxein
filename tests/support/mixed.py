"""Fixtures for the mixed-variable spaces, not examples.

`MixedProblem` has a known optimum that needs all four dimension types to be right: three real genes whose targets *depend on
the discrete choices* (so the real part cannot be solved before the rest), two integers, three bits and a category, each
with a penalty for being wrong, and a constraint. `PolynomialProblem` previews the polynomial notebook: real coefficients
plus binary switches for which terms are active, a data error plus a complexity penalty as the single objective.
"""

from typing import Any

import numpy as np

from auxein.backend import Array, Backend
from auxein.core import BatchResult, Objective, ProblemSpec
from auxein.spaces import Binary, Categorical, Integer, MixedSpace, Real

MODES = ["a", "b", "c", "d"]
BEST = {"k": 13, "m": -4, "bits": (1, 0, 1), "mode": 2}
"""The discrete part of the optimum: the integers, the bits and the index of the mode ("c")."""

SPACE = MixedSpace(
    {
        "x0": Real(-5.0, 5.0),
        "x1": Real(-5.0, 5.0),
        "x2": Real(0.01, 100.0, log=True),
        "k": Integer(0, 20),
        "m": Integer(-10, 10),
        "b0": Binary(),
        "b1": Binary(),
        "b2": Binary(),
        "mode": Categorical(MODES),
    }
)
"""Nine dimensions of four types; the log-scale real has its target at 4."""


def targets(xp: Any, mode: Array, bits: Array, dtype: Any) -> Array:
    """The real targets `(n, 3)` for the discrete choices of each candidate: they move with the mode and with the bits, so
    that the real genes cannot be tuned before the discrete ones are right."""
    mode_f = xp.astype(mode, dtype)
    first = 0.5 * mode_f - 1.0 + 0.25 * bits[:, 0]
    second = 1.0 - 0.5 * mode_f + 0.5 * (bits[:, 1] + bits[:, 2])
    third = 2.0 + mode_f
    return xp.stack([first, second, third], axis=1)


def evaluate(X: Array, backend: Backend) -> BatchResult:
    """The objective (to minimise, 0 at the optimum) and the constraint of a batch of genomes, on the backend's namespace."""
    xp = backend.xp
    columns = SPACE.columns(X)
    bits = xp.astype(xp.stack([columns["b0"], columns["b1"], columns["b2"]], axis=1), backend.dtype)
    t = targets(xp, columns["mode"], bits, backend.dtype)
    real = xp.stack([columns["x0"], columns["x1"], columns["x2"]], axis=1)
    real_error = (real[:, 0] - t[:, 0]) ** 2 + (real[:, 1] - t[:, 1]) ** 2 + (xp.log(real[:, 2]) - xp.log(t[:, 2])) ** 2
    wrong_bits = xp.sum(xp.abs(bits - xp.asarray(BEST["bits"], dtype=backend.dtype, device=backend.device)), axis=1)
    penalty = (
        0.5 * xp.abs(xp.astype(columns["k"], backend.dtype) - BEST["k"])
        + 0.3 * xp.abs(xp.astype(columns["m"], backend.dtype) + 4.0)
        + 1.0 * wrong_bits
        + 2.0 * xp.astype(columns["mode"] != BEST["mode"], backend.dtype)
    )
    violation = xp.clip(
        xp.astype(columns["k"], backend.dtype) - 16.0, 0.0, None
    )  # the integer must not exceed 16 (the optimum, 13, does not)
    return BatchResult({"value": real_error + penalty}, {"too_big": violation})


def problem() -> ProblemSpec[Any]:
    return ProblemSpec(SPACE, (Objective("value"),), ("too_big",))


def optimum_genome(backend: Backend) -> Array:
    """The genome with value 0, as a `(d,)` array on the backend."""
    mode, bits = BEST["mode"], np.array(BEST["bits"], dtype=np.float64)
    t = targets(np, np.array([mode]), bits[None, :], np.float64)[0]
    return backend.asarray(np.array([t[0], t[1], t[2], BEST["k"], BEST["m"], *BEST["bits"], mode], dtype=np.float64))


# --- the polynomial regression with structure genes ---

DEGREE = 6
TRUE_TERMS = (0, 2, 5)
TRUE_COEFFICIENTS = {0: 2.0, 2: 3.0, 5: -1.5}
COMPLEXITY = 0.02

POLY_SPACE = MixedSpace({**{f"c{j}": Real(-5.0, 5.0) for j in range(DEGREE + 1)}, **{f"on{j}": Binary() for j in range(DEGREE + 1)}})
"""Seven real coefficients and seven binary switches: the genome length is constant, so the array path applies."""


def polynomial_data() -> tuple[np.ndarray, np.ndarray]:
    """Noise-free data of the true polynomial `2 + 3x² − 1.5x⁵`, on 40 points of `[-1.5, 1.5]`."""
    x = np.linspace(-1.5, 1.5, 40)
    return x, sum(c * x**j for j, c in TRUE_COEFFICIENTS.items())  # type: ignore[return-value]


def polynomial_evaluate(X: Array, backend: Backend) -> Array:
    """The mean squared error of the polynomial each genome describes (its active terms only) plus `COMPLEXITY` per active term."""
    xp = backend.xp
    x, y = polynomial_data()
    powers = backend.asarray(np.stack([x**j for j in range(DEGREE + 1)], axis=1))  # (points, terms)
    coefficients, switches = X[:, : DEGREE + 1], X[:, DEGREE + 1 :]
    predicted = xp.matmul(coefficients * switches, xp.permute_dims(powers, (1, 0)))  # (n, points)
    error = xp.mean((predicted - backend.asarray(y)[None, :]) ** 2, axis=1)
    return error + COMPLEXITY * xp.sum(switches, axis=1)


def polynomial_problem() -> ProblemSpec[Any]:
    return ProblemSpec(POLY_SPACE, (Objective("loss"),))


def active_terms(genome: Array) -> tuple[int, ...]:
    values = POLY_SPACE.values(genome)
    return tuple(j for j in range(DEGREE + 1) if values[f"on{j}"])


def evaluate_one(genome: Array, backend: Backend) -> Any:
    """The same objective for one genome, as a `Result` (for `FunctionEvaluator`, and so for steady-state delivery)."""
    from auxein.core import Result

    batch = evaluate(genome[None, :], backend)
    value = float(backend.to_numpy(batch.objectives["value"])[0])
    return Result({"value": value}, {"too_big": float(backend.to_numpy(batch.constraints["too_big"])[0])})
