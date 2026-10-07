import math

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import Candidate, CandidateId, Cost, Evaluation, EvaluationBatch, Objective, Status, to_minimisation
from tests.support.fixtures import assert_on_backend

LOSS = Objective("loss")
SCORE = Objective("score", "maximise")
TIME = Objective("time")


def evaluation(i: int, status: Status = Status.OK, objectives=None, constraints=None, descriptors=None) -> Evaluation[None]:
    candidate = Candidate(CandidateId(i), None, (), "init", 0)
    if objectives is None:
        objectives = {"loss": float(i), "score": 10.0 * i, "time": 100.0 + i}
    return Evaluation(candidate, status, objectives, constraints or {}, descriptors or {}, Cost(0.1))


def mixed_batch() -> EvaluationBatch[None]:
    return EvaluationBatch(
        [
            evaluation(0, constraints={"cpa": 0.0, "speed": 0.0}, descriptors={"side": 1.0}),
            evaluation(1, constraints={"cpa": 2.0, "speed": 0.5}, descriptors={"side": -1.0}),
            evaluation(2, Status.FAILED, {"loss": math.nan, "score": math.nan}),
            evaluation(3, constraints={"cpa": 0.0, "speed": 0.0}, descriptors={"side": 1.0}),
            evaluation(4, Status.TIMEOUT, {}),
        ]
    )


def host(backend: Backend, x):
    return backend.to_numpy(x)


def test_it_reads_as_a_sequence_of_evaluations():
    batch = mixed_batch()
    assert len(batch) == 5
    assert [e.candidate.id for e in batch] == [0, 1, 2, 3, 4]
    assert batch[1].candidate.id == 1
    assert [e.candidate.id for e in batch[1:3]] == [1, 2]
    assert [c.id for c in batch.candidates] == [0, 1, 2, 3, 4]


def test_it_copies_its_sequence():
    evaluations = [evaluation(0)]
    batch = EvaluationBatch(evaluations)
    evaluations.append(evaluation(1))
    assert len(batch) == 1


def test_objectives_matrix_follows_the_requested_order(backend: Backend):
    batch = EvaluationBatch([evaluation(1), evaluation(2), evaluation(3)])
    m = batch.objectives_matrix([TIME, LOSS], backend)
    assert_on_backend(m, backend)
    np.testing.assert_allclose(host(backend, m), [[101.0, 1.0], [102.0, 2.0], [103.0, 3.0]])
    m = batch.objectives_matrix([LOSS, SCORE, TIME], backend)
    assert tuple(m.shape) == (3, 3)
    np.testing.assert_allclose(host(backend, m)[:, 0], [1.0, 2.0, 3.0])


def test_objectives_matrix_is_in_natural_units(backend: Backend):
    batch = EvaluationBatch([evaluation(1), evaluation(2)])
    np.testing.assert_allclose(host(backend, batch.objectives_matrix([SCORE], backend)), [[10.0], [20.0]])


def test_failed_evaluations_have_nan_objectives(backend: Backend):
    m = host(backend, mixed_batch().objectives_matrix([LOSS, SCORE], backend))
    assert m.shape == (5, 2)
    assert np.isnan(m[2]).all() and np.isnan(m[4]).all()
    assert not np.isnan(m[[0, 1, 3]]).any()


def test_a_missing_objective_in_an_ok_evaluation_is_an_error(backend: Backend):
    batch = EvaluationBatch([evaluation(0, objectives={"loss": 1.0})])
    with pytest.raises(KeyError, match="candidate 0 has status OK but reported no objective 'score'"):
        batch.objectives_matrix([LOSS, SCORE], backend)


def test_minimisation_form_negates_maximised_objectives_only(backend: Backend):
    batch = EvaluationBatch([evaluation(1), evaluation(2), evaluation(3)])
    objectives = [LOSS, SCORE, TIME]
    natural = host(backend, batch.objectives_matrix(objectives, backend))
    minimised = host(backend, batch.minimisation_matrix(objectives, backend))
    np.testing.assert_allclose(minimised[:, 0], natural[:, 0])
    np.testing.assert_allclose(minimised[:, 1], -natural[:, 1])
    np.testing.assert_allclose(minimised[:, 2], natural[:, 2])
    assert minimised[:, 1].tolist() == [-10.0, -20.0, -30.0]  # lower is better in every column


def test_to_minimisation_converts_any_natural_matrix(backend: Backend):
    values = backend.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = to_minimisation(values, [Objective("a", "maximise"), Objective("b"), Objective("c", "maximise")], backend)
    assert_on_backend(out, backend)
    np.testing.assert_allclose(host(backend, out), [[-1.0, 2.0, -3.0], [-4.0, 5.0, -6.0]])
    np.testing.assert_allclose(host(backend, values)[0], [1.0, 2.0, 3.0])  # the input is not modified


def test_failed_rows_stay_nan_in_minimisation_form(backend: Backend):
    m = host(backend, mixed_batch().minimisation_matrix([LOSS, SCORE], backend))
    assert np.isnan(m[2]).all()
    np.testing.assert_allclose(m[1], [1.0, -10.0])


def test_constraints_matrix(backend: Backend):
    m = host(backend, mixed_batch().constraints_matrix(["speed", "cpa"], backend))
    assert m.shape == (5, 2)
    np.testing.assert_allclose(m[0], [0.0, 0.0])
    np.testing.assert_allclose(m[1], [0.5, 2.0])
    assert np.isinf(m[2]).all() and np.isinf(m[4]).all()  # failed evaluations violate everything


def test_a_missing_constraint_in_an_ok_evaluation_is_an_error(backend: Backend):
    batch = EvaluationBatch([evaluation(0, constraints={"cpa": 0.0})])
    with pytest.raises(KeyError, match="no constraint 'speed'"):
        batch.constraints_matrix(["cpa", "speed"], backend)


def test_descriptors_matrix(backend: Backend):
    m = host(backend, mixed_batch().descriptors_matrix(["side"], backend))
    np.testing.assert_allclose(m[[0, 1, 3], 0], [1.0, -1.0, 1.0])
    assert np.isnan(m[2, 0]) and np.isnan(m[4, 0])


def test_total_violation(backend: Backend):
    total = mixed_batch().total_violation(backend)
    assert_on_backend(total, backend)
    t = host(backend, total)
    assert t.shape == (5,)
    np.testing.assert_allclose(t[[0, 1, 3]], [0.0, 2.5, 0.0])
    assert np.isinf(t[2]) and np.isinf(t[4])


def test_total_violation_of_selected_constraints(backend: Backend):
    t = host(backend, mixed_batch().total_violation(backend, ["speed"]))
    np.testing.assert_allclose(t[[0, 1, 3]], [0.0, 0.5, 0.0])


def test_total_violation_of_unconstrained_evaluations_is_zero(backend: Backend):
    t = host(backend, EvaluationBatch([evaluation(0), evaluation(1)]).total_violation(backend))
    np.testing.assert_array_equal(t, [0.0, 0.0])


def test_feasible_mask(backend: Backend):
    mask = mixed_batch().feasible_mask(backend)
    assert_on_backend(mask, backend, backend.bool_dtype)
    assert host(backend, mask).tolist() == [True, False, False, True, False]  # violated, failed and timed out are infeasible


def test_unconstrained_ok_evaluations_are_feasible(backend: Backend):
    assert host(backend, EvaluationBatch([evaluation(0)]).feasible_mask(backend)).tolist() == [True]


def test_status_masks():
    batch = mixed_batch()
    assert batch.status_mask(Status.OK).tolist() == [True, True, False, True, False]
    assert batch.status_mask(Status.FAILED).tolist() == [False, False, True, False, False]
    assert batch.status_mask(Status.TIMEOUT).tolist() == [False, False, False, False, True]
    total = sum(batch.status_mask(s).astype(int) for s in Status)
    assert total.tolist() == [1] * 5  # every evaluation has exactly one status


def test_status_masks_on_a_backend(backend: Backend):
    mask = mixed_batch().status_mask(Status.OK, backend)
    assert_on_backend(mask, backend, backend.bool_dtype)
    assert host(backend, mask).tolist() == [True, True, False, True, False]


def test_an_empty_batch_gives_empty_columns(backend: Backend):
    batch = EvaluationBatch([])
    assert tuple(batch.objectives_matrix([LOSS, SCORE], backend).shape) == (0, 2)
    assert tuple(batch.constraints_matrix(["a"], backend).shape) == (0, 1)
    assert tuple(batch.minimisation_matrix([LOSS], backend).shape) == (0, 1)
    assert tuple(batch.total_violation(backend).shape) == (0,)
    assert tuple(batch.feasible_mask(backend).shape) == (0,)
    assert batch.status_mask(Status.OK).shape == (0,)


def test_columns_with_no_names(backend: Backend):
    batch = EvaluationBatch([evaluation(0), evaluation(1)])
    assert tuple(batch.constraints_matrix([], backend).shape) == (2, 0)
