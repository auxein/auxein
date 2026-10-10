"""Scalarisation (design doc §5.4): running a single-objective strategy on a multi-objective problem.

A scalarisation turns the objectives of an evaluation into one number to minimise. `Scalarised(strategy, scalarisation)`
wraps any single-objective strategy so that it sees exactly one minimised objective, while the driver, the recorder and the
`RunResult` keep **all** the original objectives (and the Pareto archive): the wrapper only changes what the inner strategy is
told. Scalarisations work on the objectives **in minimisation form**, which the `EvaluationBatch` produces from the declared
directions; a user never negates anything.
"""

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Protocol, cast

import numpy as np

from auxein.backend import Array, Backend
from auxein.core import (
    Batch,
    Evaluation,
    EvaluationBatch,
    Objective,
    ProblemSpec,
    StateDict,
    Status,
    StrategyCapabilities,
    StrategyContext,
)
from auxein.core._typing import G

SCALARISED = "scalarised"
"""The name of the one objective the inner strategy sees."""


class Scalarisation(Protocol):
    """Turns objective vectors, in minimisation form, into numbers to minimise."""

    def validate(self, objectives: tuple[Objective, ...]) -> None:
        """Raise `ValueError` if the scalarisation does not fit these objectives (names, weights)."""
        ...

    def apply(self, minimised: Array, objectives: tuple[Objective, ...], backend: Backend) -> Array:
        """`(n, k)` objectives in minimisation form (columns in the order of `objectives`) to `(n,)` values."""
        ...


def _check_weights(weights: Mapping[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("a scalarisation needs a weight for every objective")
    checked: dict[str, float] = {}
    for name, weight in weights.items():
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight) or weight < 0:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"the weight of {name!r} must be a finite number that is not negative, got {weight!r}")
        checked[name] = float(weight)
    if not any(w > 0 for w in checked.values()):
        raise ValueError("the weights are all zero: at least one objective needs a positive weight")
    return checked


def _check_names(kind: str, given: Mapping[str, float], objectives: tuple[Objective, ...]) -> None:
    names = [o.name for o in objectives]
    unknown = sorted(set(given) - set(names))
    missing = [n for n in names if n not in given]
    if unknown or missing:
        raise ValueError(
            f"the {kind} do not match the problem's objectives {names}: "
            + (f"unknown {unknown}" if unknown else "")
            + ("; " if unknown and missing else "")
            + (f"missing {missing} (give 0 to ignore an objective)" if missing else "")
        )


def _weight_vector(weights: Mapping[str, float], objectives: tuple[Objective, ...], backend: Backend) -> Array:
    return backend.asarray([weights[o.name] for o in objectives])


@dataclass(frozen=True)
class WeightedSum:
    """`Σ wᵢ·fᵢ` over the objectives in minimisation form, with a non-negative weight for every objective.

    A maximised objective enters with the sign of its direction (so a positive weight always means "more of this is worse"
    for a minimised objective and "less of this is worse" for a maximised one). The weights are named, not positional. A
    weighted sum with positive weights can only find points on the *convex* part of a Pareto front.
    """

    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", _check_weights(self.weights))

    def validate(self, objectives: tuple[Objective, ...]) -> None:
        _check_names("weights", self.weights, objectives)

    def apply(self, minimised: Array, objectives: tuple[Objective, ...], backend: Backend) -> Array:
        return backend.xp.sum(minimised * _weight_vector(self.weights, objectives, backend)[None, :], axis=1)

    def __repr__(self) -> str:
        return f"WeightedSum(weights={dict(self.weights)})"


@dataclass(frozen=True)
class Chebyshev:
    """The weighted Chebyshev (Tchebycheff) scalarisation `maxᵢ wᵢ·(fᵢ − zᵢ)` over the objectives in minimisation form.

    `z` is the **reference point**, given in the objectives' natural units (the value you would like to reach; 0 for every
    objective when omitted) and converted to minimisation form with the objective's direction. It must be **explicit**: the
    obvious alternative, the best value seen so far for each objective, would change as the run goes on, so the same candidate
    would score differently at different times, and a strategy that compares scores across generations (a plus-selection GA
    does) would be comparing numbers on different scales. A fixed reference point makes the scalarisation stationary. Unlike
    the weighted sum, minimising it can reach every point of a Pareto front, convex or not, by choosing the weights and the
    reference point; a reference point that dominates the front gives the textbook form.
    """

    weights: Mapping[str, float]
    reference: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", _check_weights(self.weights))
        if self.reference is not None:
            for name, value in self.reference.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):  # pyright: ignore[reportUnnecessaryIsInstance]
                    raise ValueError(f"the reference value of {name!r} must be a finite number, got {value!r}")
            object.__setattr__(self, "reference", {k: float(v) for k, v in self.reference.items()})

    def validate(self, objectives: tuple[Objective, ...]) -> None:
        _check_names("weights", self.weights, objectives)
        if self.reference is not None:
            _check_names("reference values", self.reference, objectives)

    def apply(self, minimised: Array, objectives: tuple[Objective, ...], backend: Backend) -> Array:
        reference = self.reference or {}
        z = backend.asarray([o.sign * reference.get(o.name, 0.0) for o in objectives])
        weights = _weight_vector(self.weights, objectives, backend)
        return backend.xp.max(weights[None, :] * (minimised - z[None, :]), axis=1)

    def __repr__(self) -> str:
        return f"Chebyshev(weights={dict(self.weights)}, reference={None if self.reference is None else dict(self.reference)})"


class Scalarised(Generic[G]):
    """Runs a single-objective strategy on a multi-objective problem through a scalarisation.

    The inner strategy is bound to a problem with **one** objective, `"scalarised"` (minimised), and the same space,
    constraints and descriptors; each time it is told, every evaluation carries that one value instead of the original
    objectives. The driver keeps asking the wrapper and telling it the full evaluations, so it records them all, and
    `RunResult.pareto_front` is the Pareto archive of the original objectives (`RunResult.best` is None for several objectives:
    use `best_by_scalarisation` for the winner by the scalarisation). Failed evaluations pass through as failures.

    `repr` includes the inner strategy and the scalarisation, so a resumed run whose weights changed is refused, and
    `state_dict` is the inner strategy's.
    """

    def __init__(self, strategy: object, scalarisation: Scalarisation) -> None:
        for method in ("bind", "ask", "tell", "state_dict", "load_state_dict"):
            if not hasattr(strategy, method):
                raise TypeError(f"Scalarised needs a strategy, got {strategy!r}")
        capabilities = getattr(strategy, "capabilities", None)
        if capabilities is None:
            raise TypeError(f"Scalarised needs a strategy with capabilities, got {strategy!r}")
        self._inner = cast("_Inner[G]", strategy)
        self.scalarisation = scalarisation
        inner = cast("StrategyCapabilities", capabilities)
        self.capabilities = StrategyCapabilities(None, inner.supports_constraints, inner.tell_mode)
        self._problem: ProblemSpec[G] | None = None
        self._backend: Backend | None = None

    @property
    def strategy(self) -> object:
        """The wrapped single-objective strategy."""
        return self._inner

    def __repr__(self) -> str:
        return f"Scalarised({self._inner!r}, {self.scalarisation!r})"

    def bind(self, problem: ProblemSpec[G], ctx: StrategyContext) -> None:
        self.scalarisation.validate(problem.objectives)
        inner_problem = ProblemSpec(problem.space, (Objective(SCALARISED),), problem.constraints, problem.descriptors)
        self._inner.bind(inner_problem, ctx)
        self._problem, self._backend = problem, ctx.backend

    def ask(self, n: int) -> Batch[G]:
        return self._inner.ask(n)

    def tell(self, results: EvaluationBatch[G]) -> None:
        if self._problem is None or self._backend is None:
            raise RuntimeError("Scalarised must be bound to a problem before use: the driver calls bind() first")
        objectives, backend = self._problem.objectives, self._backend
        minimised = results.minimisation_matrix(objectives, backend)
        values = np.asarray(backend.to_numpy(self.scalarisation.apply(minimised, objectives, backend)), dtype=np.float64).tolist()
        told: list[Evaluation[G]] = []
        for evaluation, value in zip(results, values, strict=True):
            if evaluation.status is Status.OK:
                if not math.isfinite(value):
                    told.append(Evaluation.failed(evaluation.candidate, Status.FAILED, f"the scalarisation is not finite: {value}"))
                    continue
                told.append(dataclasses.replace(evaluation, objectives={SCALARISED: value}))
            else:
                told.append(evaluation)
        self._inner.tell(EvaluationBatch(told, results.episodes))

    def should_stop(self) -> bool:
        return bool(getattr(self._inner, "should_stop", lambda: False)())

    def state_dict(self) -> StateDict:
        return self._inner.state_dict()

    def load_state_dict(self, state: StateDict) -> None:
        self._inner.load_state_dict(state)


class _Inner(Protocol[G]):
    def bind(self, problem: ProblemSpec[G], ctx: StrategyContext) -> None: ...
    def ask(self, n: int) -> Batch[G]: ...
    def tell(self, results: EvaluationBatch[G]) -> None: ...
    def state_dict(self) -> StateDict: ...
    def load_state_dict(self, state: StateDict) -> None: ...


@dataclass(frozen=True)
class ScalarisedBest:
    """The best evaluation of a run by a scalarisation."""

    candidate_id: int
    value: float
    objectives: Mapping[str, float]
    constraints: Mapping[str, float]
    genome: object


def best_by_scalarisation(run: object, scalarisation: Scalarisation, *, objectives: Sequence[Objective] | None = None) -> ScalarisedBest:
    """The best evaluation of a run by `scalarisation`: a `RunResult` (searched in its Pareto archive, which holds the
    minimiser of any scalarisation that is non-decreasing in every objective) or a run directory (searched in everything it
    recorded; the objectives' directions are read from its metadata). Feasible evaluations come first, then a lower total
    violation, then the lower value, then the lower id; failed evaluations never win.
    """
    from auxein.driver import RunResult

    candidates: list[tuple[int, Mapping[str, float], Mapping[str, float], object]] = []
    if isinstance(run, RunResult):
        if objectives is None:
            raise ValueError("pass objectives= (the problem's objectives, with their directions) to search a RunResult")
        for evaluation in cast("RunResult[object]", run).pareto_front:
            candidates.append((int(evaluation.candidate.id), evaluation.objectives, evaluation.constraints, evaluation.candidate.genome))
    elif isinstance(run, (str, Path)):
        import json

        from auxein.recording import open_run

        metadata = json.loads((Path(run) / "metadata.json").read_text())
        objectives = tuple(Objective(o["name"], o["direction"]) for o in metadata["problem"]["objectives"])
        with open_run(run) as recorded:
            for row in recorded.evaluations():
                if row.status is Status.OK:
                    candidates.append((int(row.candidate_id), row.objectives, row.constraints, row.genome))
    else:
        raise TypeError(f"best_by_scalarisation needs a RunResult or a run directory, got {type(run).__name__}")
    if not candidates:
        raise ValueError("the run has no successful evaluation")
    assert objectives is not None
    objectives = tuple(objectives)
    scalarisation.validate(objectives)
    backend = Backend()
    matrix = backend.asarray([[o.sign * c[1][o.name] for o in objectives] for c in candidates])
    values = np.asarray(scalarisation.apply(matrix, objectives, backend), dtype=np.float64).tolist()
    ranked = sorted(
        zip(candidates, values, strict=True),
        key=lambda item: (sum(item[0][2].values()) > 0, sum(item[0][2].values()), item[1], item[0][0]),
    )
    (cid, objective_values, constraint_values, genome), value = ranked[0]
    return ScalarisedBest(cid, value, dict(objective_values), dict(constraint_values), genome)
