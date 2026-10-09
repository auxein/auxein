import math

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import (
    BatchResult,
    Candidate,
    CandidateId,
    Objective,
    ProblemSpec,
    Result,
    Status,
    evaluation_from_return,
    evaluations_from_batch_return,
)
from auxein.spaces import Box

SPACE = Box(-1.0, 1.0, dim=2)
ONE = ProblemSpec(SPACE, (Objective("loss"),))
TWO = ProblemSpec(SPACE, (Objective("loss"), Objective("score", "maximise")))
FULL = ProblemSpec(SPACE, (Objective("loss"),), ("cpa",), ("speed",))


def cand(i: int = 7) -> Candidate[None]:
    return Candidate(CandidateId(i), None, (), "init", 0)


def one(raw: object, problem: ProblemSpec = ONE, wall: float = 0.25):
    return evaluation_from_return(raw, cand(), problem, wall)  # type: ignore[arg-type]


# --- per-candidate returns ---


@pytest.mark.parametrize("raw", [3, 3.0, np.float32(3.0), np.float64(3.0), np.int64(3), np.array(3.0), np.array(3), np.float16(3.0)])
def test_a_bare_number_is_the_single_objective(raw: object):
    e = one(raw)
    assert e.status is Status.OK and dict(e.objectives) == {"loss": 3.0}
    assert dict(e.constraints) == {} and dict(e.descriptors) == {}
    assert e.cost.wall_time == 0.25 and dict(e.cost.units) == {}


def test_a_bare_number_may_be_a_zero_dimensional_array_of_any_backend(backend: Backend):
    assert dict(one(backend.asarray(2.5)).objectives) == {"loss": 2.5}


def test_a_bare_number_uses_the_name_of_the_declared_objective():
    problem = ProblemSpec(SPACE, (Objective("score", "maximise"),))
    assert dict(one(1.5, problem).objectives) == {"score": 1.5}


def test_a_bare_number_with_several_objectives_is_an_error_naming_them():
    with pytest.raises(TypeError, match=r"bare number.*2 objectives \('loss', 'score'\).*Result"):
        one(1.0, TWO)


def test_a_bare_number_with_declared_constraints_or_descriptors_is_an_error():
    with pytest.raises(TypeError, match=r"constraints 'cpa' and descriptors 'speed'.*Result"):
        one(1.0, FULL)
    with pytest.raises(TypeError, match="constraints 'cpa'"):
        one(1.0, ProblemSpec(SPACE, (Objective("a"),), ("cpa",)))


def test_a_complete_result():
    e = one(Result({"loss": 2.0}, {"cpa": 0.5}, {"speed": 3.0}, {"tokens": 10.0}), FULL)
    assert dict(e.objectives) == {"loss": 2.0} and dict(e.constraints) == {"cpa": 0.5}
    assert dict(e.descriptors) == {"speed": 3.0}
    assert dict(e.cost.units) == {"tokens": 10.0} and e.cost.wall_time == 0.25
    assert e.status is Status.OK and e.candidate.id == 7


def test_a_result_with_several_objectives_in_any_order():
    e = one(Result({"score": 1.0, "loss": 2.0}), TWO)
    assert dict(e.objectives) == {"loss": 2.0, "score": 1.0}


@pytest.mark.parametrize(
    ("result", "problem", "message"),
    [
        (Result({"los": 1.0}), ONE, r"missing objectives 'loss'; unknown objectives 'los' \(declared: 'loss'\)"),
        (Result({"loss": 1.0, "extra": 2.0}), ONE, r"unknown objectives 'extra'"),
        (Result({"loss": 1.0}), TWO, r"missing objectives 'score'"),
        (Result({"loss": 1.0}), FULL, r"missing constraints 'cpa'; missing descriptors 'speed'"),
        (Result({"loss": 1.0}, {"cpa": 0.0}, {"speed": 1.0, "typo": 2.0}), FULL, r"unknown descriptors 'typo'"),
        (Result({"loss": 1.0}, {"cpa": 0.0, "cpa2": 1.0}, {"speed": 1.0}), FULL, r"unknown constraints 'cpa2'"),
        (Result({"loss": 1.0}, {"cpa": 0.0}), ONE, r"unknown constraints 'cpa' \(declared: none\)"),
    ],
)
def test_a_result_must_match_the_problem_exactly(result: Result, problem: ProblemSpec, message: str):
    with pytest.raises(ValueError, match=message):
        one(result, problem)


def test_a_plain_dict_is_rejected_with_a_pointer_to_result():
    with pytest.raises(TypeError, match=r"returned a dict.*Plain dicts are not accepted.*Result\(objectives="):
        one({"loss": 1.0})
    with pytest.raises(TypeError, match="Result"):
        one({})


@pytest.mark.parametrize(
    "raw", [None, "1.0", [1.0], (1.0,), True, np.bool_(True), np.array([1.0]), np.array([1.0, 2.0]), object(), 1 + 2j, np.array("x")]
)
def test_other_returns_are_rejected(raw: object):
    with pytest.raises(TypeError, match="must return a number or a Result"):
        one(raw)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("wrap", [lambda v: v, lambda v: Result({"loss": v})])
def test_a_non_finite_objective_is_a_failed_evaluation_naming_the_objective(value: float, wrap):
    evaluation = one(wrap(value))
    assert evaluation.status is Status.FAILED and evaluation.candidate.id == 7
    assert evaluation.error is not None and f"'loss' is {value}" in evaluation.error and "non-finite" in evaluation.error
    assert dict(evaluation.objectives) == {}  # a failed evaluation has no values: the batch views fill in NaN


# --- vectorised returns ---

CANDIDATES = [cand(i) for i in (10, 11, 12)]


def many(raw: object, problem: ProblemSpec = ONE, wall: float = 0.3):
    return evaluations_from_batch_return(raw, CANDIDATES, problem, wall)  # type: ignore[arg-type]


def test_a_vector_is_the_single_objective(backend: Backend):
    evaluations = many(backend.asarray([3.0, 1.0, 2.0]))
    assert [e.candidate.id for e in evaluations] == [10, 11, 12]
    assert [e.objectives["loss"] for e in evaluations] == pytest.approx([3.0, 1.0, 2.0], rel=1e-6)
    assert all(e.status is Status.OK for e in evaluations)


def test_the_batch_wall_time_is_split_equally(backend: Backend):
    evaluations = many(backend.asarray([1.0, 2.0, 3.0]), wall=0.3)
    assert [e.cost.wall_time for e in evaluations] == pytest.approx([0.1, 0.1, 0.1])


def test_a_matrix_has_one_column_per_declared_objective_in_declared_order(backend: Backend):
    evaluations = many(backend.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]), TWO)
    assert [(e.objectives["loss"], e.objectives["score"]) for e in evaluations] == [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]


def test_a_single_column_matrix_is_fine_for_one_objective(backend: Backend):
    assert [e.objectives["loss"] for e in many(backend.asarray([[1.0], [2.0], [3.0]]))] == [1.0, 2.0, 3.0]


def test_a_batch_result_returns_everything(backend: Backend):
    raw = BatchResult(
        {"loss": backend.asarray([1.0, 2.0, 3.0])},
        {"cpa": backend.asarray([0.0, 0.5, 0.0])},
        {"speed": backend.asarray([4.0, 5.0, 6.0])},
        {"tokens": backend.asarray([10.0, 20.0, 30.0])},
    )
    evaluations = many(raw, FULL)
    assert [e.constraints["cpa"] for e in evaluations] == [0.0, 0.5, 0.0]
    assert [e.descriptors["speed"] for e in evaluations] == [4.0, 5.0, 6.0]
    assert [e.cost.units["tokens"] for e in evaluations] == [10.0, 20.0, 30.0]
    assert [e.cost.wall_time for e in evaluations] == pytest.approx([0.1] * 3)


def test_a_batch_result_must_match_the_problem_exactly():
    with pytest.raises(ValueError, match=r"BatchResult.*missing objectives 'score'"):
        many(BatchResult({"loss": [1.0, 2.0, 3.0]}), TWO)
    with pytest.raises(ValueError, match=r"unknown descriptors 'x'"):
        many(BatchResult({"loss": [1.0, 2.0, 3.0]}, descriptors={"x": [1.0, 2.0, 3.0]}), ONE)
    with pytest.raises(ValueError, match="covers 2 candidates but the batch has 3"):
        many(BatchResult({"loss": [1.0, 2.0]}))


def test_shape_errors(backend: Backend):
    with pytest.raises(ValueError, match="returned 2 values for a batch of 3"):
        many(backend.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="returned 4 values for a batch of 3"):
        many(backend.asarray([[1.0, 2.0]] * 4), TWO)
    with pytest.raises(ValueError, match=r"returned 3 columns but the problem has 2 objectives \('loss', 'score'\)"):
        many(backend.asarray(np.zeros((3, 3))), TWO)
    with pytest.raises(ValueError, match=r"shape \(n,\) or \(n, k\)"):
        many(backend.asarray(np.zeros((3, 1, 1))))
    with pytest.raises(ValueError, match="returned 2 columns but the problem has 1 objectives"):
        many(backend.asarray(np.zeros((3, 2))))


def test_a_vector_with_several_objectives_or_declared_constraints_is_an_error(backend: Backend):
    with pytest.raises(TypeError, match=r"2 objectives \('loss', 'score'\).*BatchResult"):
        many(backend.asarray([1.0, 2.0, 3.0]), TWO)
    with pytest.raises(TypeError, match="constraints 'cpa'"):
        many(backend.asarray([1.0, 2.0, 3.0]), FULL)
    with pytest.raises(TypeError, match="constraints 'cpa'"):
        many(backend.asarray([[1.0], [2.0], [3.0]]), FULL)


def test_a_dict_is_rejected_here_too():
    with pytest.raises(TypeError, match=r"returned a dict.*BatchResult"):
        many({"loss": [1.0, 2.0, 3.0]})


def test_non_array_returns_are_rejected():
    with pytest.raises(TypeError, match="must return an array or a BatchResult"):
        many("text")
    with pytest.raises(TypeError):
        many(None)


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_a_non_finite_value_fails_only_its_own_candidate(backend: Backend, value: float):
    def statuses(evaluations):
        return [e.status for e in evaluations]

    evaluations = many(backend.asarray([1.0, value, 3.0]))
    assert statuses(evaluations) == [Status.OK, Status.FAILED, Status.OK]
    assert "'loss' is" in (evaluations[1].error or "") and evaluations[1].candidate.id == 11
    evaluations = many(backend.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, value]]), TWO)
    assert statuses(evaluations) == [Status.OK, Status.OK, Status.FAILED] and "'score' is" in (evaluations[2].error or "")
    evaluations = many(BatchResult({"loss": [value, 1.0, 2.0]}))
    assert statuses(evaluations) == [Status.FAILED, Status.OK, Status.OK] and evaluations[0].candidate.id == 10
    both = many(backend.asarray([[value, value], [2.0, 2.0], [3.0, 3.0]]), TWO)
    assert "'loss'" in (both[0].error or "") and "'score'" in (both[0].error or "")  # every bad objective is named


def test_an_empty_batch():
    assert evaluations_from_batch_return(np.zeros(0), [], ONE, 0.0) == []


def test_the_evaluations_of_a_batch_equal_the_validated_ones(backend: Backend):
    from auxein.core import Cost, Evaluation

    fast = many(backend.asarray([3.0, 1.0, 2.0]), wall=0.3)
    expected = [Evaluation(c, Status.OK, {"loss": v}, cost=Cost(0.1)) for c, v in zip(CANDIDATES, [3.0, 1.0, 2.0], strict=True)]
    assert [
        (
            e.candidate,
            e.status,
            dict(e.objectives),
            dict(e.constraints),
            dict(e.descriptors),
            e.cost.wall_time,
            dict(e.cost.units),
            e.raw,
            e.error,
        )
        for e in fast
    ] == [
        (
            e.candidate,
            e.status,
            dict(e.objectives),
            dict(e.constraints),
            dict(e.descriptors),
            pytest.approx(e.cost.wall_time),
            dict(e.cost.units),
            e.raw,
            e.error,
        )
        for e in expected
    ]


def test_the_evaluations_of_a_batch_are_immutable(backend: Backend):
    evaluation = many(backend.asarray([3.0, 1.0, 2.0]))[0]
    with pytest.raises(TypeError):
        evaluation.objectives["loss"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        evaluation.constraints["x"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        evaluation.descriptors["x"] = 0.0  # type: ignore[index]
    with pytest.raises(AttributeError):
        evaluation.status = Status.FAILED  # type: ignore[misc]
    with pytest.raises(AttributeError):
        evaluation.cost = None  # type: ignore[misc]


def test_a_batch_shares_one_immutable_cost_unless_cost_units_are_per_candidate(backend: Backend):
    shared = many(backend.asarray([3.0, 1.0, 2.0]))
    assert shared[0].cost is shared[1].cost and shared[0].cost.wall_time == pytest.approx(0.1)
    own = many(BatchResult({"loss": [1.0, 2.0, 3.0]}, cost={"tokens": [1.0, 2.0, 3.0]}))
    assert [e.cost.units["tokens"] for e in own] == [1.0, 2.0, 3.0] and own[0].cost is not own[1].cost


def test_results_pickle_so_that_functions_in_worker_processes_can_return_them():
    import pickle

    import numpy as np

    from auxein.core import BatchResult, Result

    result = Result({"a": 1.0}, {"c": 0.5}, {"d": 2.0}, {"tokens": 3.0})
    back = pickle.loads(pickle.dumps(result))
    assert back == result and dict(back.objectives) == {"a": 1.0} and dict(back.cost) == {"tokens": 3.0}
    batch = BatchResult({"a": np.array([1.0, 2.0])}, {"c": np.array([0.0, 1.0])})
    again = pickle.loads(pickle.dumps(batch))
    assert list(again.objectives["a"]) == [1.0, 2.0] and list(again.constraints["c"]) == [0.0, 1.0]
