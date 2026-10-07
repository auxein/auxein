"""One place where what user code returned becomes an `Evaluation` (design doc §5.3).

Every evaluator uses it, so the rules are the same whatever the evaluator:

- a bare number is the single objective of a problem with exactly one objective, and nothing else;
- a `Result` (or `BatchResult`) must match the problem exactly: every declared objective, constraint and descriptor
  present, and no unknown name;
- a plain dict is a `TypeError` that points to `Result`;
- a non-finite objective in what would be an `OK` evaluation is an error that names the candidate.
"""

import math
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import numpy.typing as npt

from auxein.backend import Array, Backend, is_array
from auxein.core._typing import G
from auxein.core.candidate import Candidate
from auxein.core.evaluation import Cost, Evaluation, Status
from auxein.core.problem import ProblemSpec
from auxein.core.results import BatchResult, Result

_HOST = Backend()


def _dict_error(origin: str, kind: str) -> TypeError:
    return TypeError(
        f"{origin} returned a dict. Plain dicts are not accepted: a dict cannot say which keys are objectives, constraints or "
        f"descriptors. Return a {kind} instead, e.g. {kind}(objectives={{'loss': 1.0}}, constraints={{...}}), or a bare number "
        f"for a problem with a single objective."
    )


def _as_number(raw: Array) -> float | None:
    """The value of a bare number (int, float, numpy scalar or 0-d array of any supported backend), else None."""
    if isinstance(raw, bool | np.bool_):
        return None
    if isinstance(raw, int | float | np.integer | np.floating):
        return float(cast("float", raw))
    if is_array(raw):
        host = _HOST.to_numpy(raw)
        return float(host) if host.ndim == 0 and host.dtype.kind in "iuf" else None
    return None


def _names(values: Sequence[str]) -> str:
    return ", ".join(repr(v) for v in values)


def _check_bare_allowed(problem: ProblemSpec[G], origin: str, kind: str) -> str:
    """The name of the single objective a bare number stands for, or an error saying why there is none."""
    if len(problem.objectives) != 1:
        raise TypeError(
            f"{origin} returned a bare number but the problem has {len(problem.objectives)} objectives ({_names(problem.objective_names)}): "
            f"return a {kind} with a value for each of them"
        )
    if problem.constraints or problem.descriptors:
        declared = [f"constraints {_names(problem.constraints)}"] * bool(problem.constraints) + [
            f"descriptors {_names(problem.descriptors)}"
        ] * bool(problem.descriptors)
        raise TypeError(
            f"{origin} returned a bare number but the problem declares {' and '.join(declared)}: return a {kind} that reports them"
        )
    return problem.objectives[0].name


def _mismatch(declared: Sequence[str], given: Mapping[str, object], what: str) -> list[str]:
    missing = [n for n in declared if n not in given]
    unknown = [n for n in given if n not in declared]
    details: list[str] = []
    if missing:
        details.append(f"missing {what} {_names(missing)}")
    if unknown:
        details.append(f"unknown {what} {_names(unknown)} (declared: {_names(declared) or 'none'})")
    return details


def _check_finite_objective(candidate_id: int, name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"candidate {candidate_id}: objective {name!r} is {value}, but an OK evaluation needs finite objective values")


def _check_result_matches(
    problem: ProblemSpec[G],
    objectives: Mapping[str, object],
    constraints: Mapping[str, object],
    descriptors: Mapping[str, object],
    origin: str,
    kind: str,
) -> None:
    """Every declared name present and no unknown name, in all three groups, reported together."""
    details = (
        _mismatch(problem.objective_names, objectives, "objectives")
        + _mismatch(problem.constraints, constraints, "constraints")
        + _mismatch(problem.descriptors, descriptors, "descriptors")
    )
    if details:
        raise ValueError(f"the {kind} returned by {origin} does not match the problem: {'; '.join(details)}")


def evaluation_from_return(raw: object, candidate: Candidate[G], problem: ProblemSpec[G], wall_time: float) -> Evaluation[G]:
    """The `Evaluation` of one candidate from what a per-candidate function returned."""
    origin = f"the function for candidate {candidate.id}"
    if isinstance(raw, Result):
        _check_result_matches(problem, raw.objectives, raw.constraints, raw.descriptors, origin, "Result")
        for name, value in raw.objectives.items():
            _check_finite_objective(candidate.id, name, value)
        return Evaluation(candidate, Status.OK, raw.objectives, raw.constraints, raw.descriptors, Cost(wall_time, raw.cost))
    if isinstance(raw, Mapping):
        raise _dict_error(origin, "Result")
    number = _as_number(raw)
    if number is None:
        raise TypeError(f"{origin} must return a number or a Result, got {type(raw).__name__}")
    name = _check_bare_allowed(problem, origin, "Result")
    _check_finite_objective(candidate.id, name, number)
    return Evaluation(candidate, Status.OK, {name: number}, cost=Cost(wall_time))


def evaluations_from_batch_return(
    raw: object, candidates: Sequence[Candidate[G]], problem: ProblemSpec[G], wall_time: float
) -> list[Evaluation[G]]:
    """The `Evaluation`s of a batch from what a vectorised function returned, in batch order.

    `raw` is an array of shape `(n,)` (a single objective) or `(n, k)` (k declared objectives, in declared order), on any
    supported backend, or a `BatchResult`. The batch's wall time is split equally across its candidates.
    """
    n = len(candidates)
    origin = "the vectorised function"
    objectives: dict[str, npt.NDArray[np.float64]]
    constraints: dict[str, npt.NDArray[np.float64]] = {}
    descriptors: dict[str, npt.NDArray[np.float64]] = {}
    costs: dict[str, npt.NDArray[np.float64]] = {}

    if isinstance(raw, BatchResult):
        if raw.size != n:
            raise ValueError(f"the BatchResult covers {raw.size} candidates but the batch has {n}")
        _check_result_matches(problem, raw.objectives, raw.constraints, raw.descriptors, origin, "BatchResult")
        objectives, constraints, descriptors, costs = (
            raw.host_columns(m) for m in (raw.objectives, raw.constraints, raw.descriptors, raw.cost)
        )
    else:
        if isinstance(raw, Mapping):
            raise _dict_error(origin, "BatchResult")
        array = _to_values(raw)
        if array.ndim == 0:
            raise TypeError(f"{origin} must return an array or a BatchResult with one value per candidate, got {type(raw).__name__}")
        if array.ndim == 1:
            name = _check_bare_allowed(problem, origin, "BatchResult")
            objectives = {name: array}
        elif array.ndim == 2:
            if array.shape[1] != len(problem.objectives):
                raise ValueError(
                    f"{origin} returned {array.shape[1]} columns but the problem has {len(problem.objectives)} objectives ({_names(problem.objective_names)})"
                )
            if problem.constraints or problem.descriptors:
                _check_bare_allowed(
                    ProblemSpec(problem.space, problem.objectives[:1], problem.constraints, problem.descriptors), origin, "BatchResult"
                )
            objectives = {name: array[:, i] for i, name in enumerate(problem.objective_names)}
        else:
            raise ValueError(f"{origin} must return an array of shape (n,) or (n, k), got shape {array.shape}")
        if array.shape[0] != n:
            raise ValueError(f"{origin} returned {array.shape[0]} values for a batch of {n} candidates")

    for name, column in objectives.items():
        finite: npt.NDArray[np.bool_] = np.isfinite(column)
        if not finite.all():
            bad = int((~finite).argmax())
            _check_finite_objective(candidates[bad].id, name, float(column[bad]))

    each = wall_time / n if n else 0.0
    objective_rows = {name: column.tolist() for name, column in objectives.items()}
    constraint_rows = {name: column.tolist() for name, column in constraints.items()}
    descriptor_rows = {name: column.tolist() for name, column in descriptors.items()}
    cost_rows = {name: column.tolist() for name, column in costs.items()}
    return [
        Evaluation(
            candidate,
            Status.OK,
            {name: rows[i] for name, rows in objective_rows.items()},
            {name: rows[i] for name, rows in constraint_rows.items()},
            {name: rows[i] for name, rows in descriptor_rows.items()},
            Cost(each, {name: rows[i] for name, rows in cost_rows.items()}),
        )
        for i, candidate in enumerate(candidates)
    ]


def _to_values(raw: Array) -> npt.NDArray[np.float64]:
    try:
        return np.asarray(_HOST.to_numpy(raw), dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(f"the vectorised function must return an array or a BatchResult, got {type(raw).__name__}") from None
