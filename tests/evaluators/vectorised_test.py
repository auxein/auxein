import asyncio

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import BatchResult, Candidate, CandidateId, ListBatch, Objective, Status
from auxein.evaluators import EvaluationError, VectorisedEvaluator
from auxein.random import RunSeed
from tests.support.helpers import array_batch, eval_context, problem


def evaluate(evaluator, batch, spec=None, backend=None, seed=0):
    spec = spec or problem()
    return asyncio.run(evaluator.evaluate(batch, eval_context(spec, backend or Backend(), seed)))


def test_a_vector_per_batch_for_a_single_objective(backend: Backend):
    results = evaluate(VectorisedEvaluator(lambda X: (X * X).sum(axis=1)), array_batch(backend, 4, 3), backend=backend)
    assert [e.candidate.id for e in results] == [0, 1, 2, 3]
    assert [e.objectives["value"] for e in results] == pytest.approx([14.0, 77.0, 194.0, 365.0], rel=1e-5)


def test_the_function_is_called_once_per_batch_with_the_batch_array(backend: Backend):
    calls = []
    batch = array_batch(backend, 5, 2)
    evaluate(VectorisedEvaluator(lambda X: calls.append(X) or X.sum(axis=1)), batch, backend=backend)
    assert len(calls) == 1 and calls[0] is batch.as_array()


def test_a_matrix_with_a_column_per_declared_objective(backend: Backend):
    spec = problem((Objective("a"), Objective("b", "maximise")))
    results = evaluate(
        VectorisedEvaluator(lambda X: backend.xp.stack([X[:, 0], -X[:, 1]], axis=1)), array_batch(backend, 3, 2), spec, backend
    )
    assert [(e.objectives["a"], e.objectives["b"]) for e in results] == [(1.0, -2.0), (3.0, -4.0), (5.0, -6.0)]


def test_a_batch_result(backend: Backend):
    spec = problem((Objective("loss"),), ("cpa",), ("speed",))

    def fn(X):
        return BatchResult({"loss": X[:, 0]}, {"cpa": X[:, 1] * 0.0}, {"speed": X[:, 0] + X[:, 1]}, {"tokens": X[:, 0] * 10})

    results = evaluate(VectorisedEvaluator(fn), array_batch(backend, 3, 2), spec, backend)
    assert [e.descriptors["speed"] for e in results] == [3.0, 7.0, 11.0]
    assert [e.cost.units["tokens"] for e in results] == [10.0, 30.0, 50.0]
    assert [e.constraints["cpa"] for e in results] == [0.0, 0.0, 0.0]


def test_the_batch_wall_time_is_split_equally():
    import time

    def slow(X):
        time.sleep(0.04)
        return X.sum(axis=1)

    results = evaluate(VectorisedEvaluator(slow), array_batch(Backend(), 4, 2))
    times = [e.cost.wall_time for e in results]
    assert len(set(times)) == 1 and 0.009 <= times[0] < 0.2  # about 0.04 / 4 each


def test_the_function_may_run_on_another_backend_and_return_its_arrays():
    pytest.importorskip("torch")
    torch_backend = Backend("torch", "cpu", "float64")
    numpy_out = evaluate(VectorisedEvaluator(lambda X: np.asarray(X).sum(axis=1)), array_batch(Backend(), 3, 2))
    torch_out = evaluate(VectorisedEvaluator(lambda X: X.sum(dim=1)), array_batch(torch_backend, 3, 2), backend=torch_backend)
    assert [e.objectives["value"] for e in numpy_out] == [e.objectives["value"] for e in torch_out]
    # a numpy array returned for torch genomes, and the other way round, is converted
    mixed = evaluate(VectorisedEvaluator(lambda X: np.array([1.0, 2.0, 3.0])), array_batch(torch_backend, 3, 2), backend=torch_backend)
    assert [e.objectives["value"] for e in mixed] == [1.0, 2.0, 3.0]


def test_an_async_function():
    async def fn(X):
        await asyncio.sleep(0)
        return X.sum(axis=1)

    assert len(evaluate(VectorisedEvaluator(fn), array_batch(Backend(), 3, 2))) == 3


def test_shape_errors(backend: Backend):
    with pytest.raises(ValueError, match="returned 3 values for a batch of 4"):
        evaluate(VectorisedEvaluator(lambda X: X.sum(axis=1)[:3]), array_batch(backend, 4, 2), backend=backend)
    with pytest.raises(ValueError, match="returned 2 columns but the problem has 1 objectives"):
        evaluate(VectorisedEvaluator(lambda X: X), array_batch(backend, 4, 2), backend=backend)
    with pytest.raises(TypeError, match="2 objectives"):
        evaluate(
            VectorisedEvaluator(lambda X: X.sum(axis=1)), array_batch(backend, 4, 2), problem((Objective("a"), Objective("b"))), backend
        )


def test_non_finite_values_fail_only_their_own_candidates(backend: Backend):
    def fn(X):
        out = X.sum(axis=1)
        return backend.xp.where(X[:, 0] > 5.0, float("nan"), out)

    results = evaluate(VectorisedEvaluator(fn), array_batch(backend, 4, 2), backend=backend)
    assert [e.status for e in results] == [Status.OK, Status.OK, Status.OK, Status.FAILED]
    assert "'value' is nan" in (results[3].error or "")


def test_it_needs_an_array_backed_batch():
    candidates = [Candidate(CandidateId(i), [1.0, 2.0], (), "init", 0) for i in range(2)]
    with pytest.raises(TypeError, match="array-backed batch.*FunctionEvaluator"):
        evaluate(VectorisedEvaluator(lambda X: X), ListBatch(candidates))


def test_the_random_stream_is_one_per_batch_from_the_first_candidate_id(backend: Backend):
    streams = []

    def fn(X, rng):
        streams.append(backend.to_numpy(rng.uniform(4)))
        return X.sum(axis=1)

    evaluator = VectorisedEvaluator(fn, uses_rng=True)
    evaluate(evaluator, array_batch(backend, 3, 2, first_id=20), backend=backend, seed=5)
    expected = backend.to_numpy(RunSeed(5).stream("evaluation-batch", 20, backend=backend).uniform(4))
    np.testing.assert_array_equal(streams[0], expected)

    evaluate(evaluator, array_batch(backend, 3, 2, first_id=20), backend=backend, seed=5)  # same batch composition: same draws
    np.testing.assert_array_equal(streams[1], streams[0])
    evaluate(evaluator, array_batch(backend, 5, 2, first_id=20), backend=backend, seed=5)  # the first id decides, not the size
    np.testing.assert_array_equal(streams[2], streams[0])
    evaluate(evaluator, array_batch(backend, 3, 2, first_id=21), backend=backend, seed=5)
    assert not np.array_equal(streams[3], streams[0])
    assert not np.array_equal(
        streams[0], backend.to_numpy(RunSeed(5).stream("evaluation", 20, backend=backend).uniform(4))
    )  # not a per-candidate stream


def test_without_uses_rng_the_function_gets_only_the_array():
    seen = []
    evaluate(VectorisedEvaluator(lambda X: seen.append(1) or X.sum(axis=1)), array_batch(Backend(), 2, 2))
    assert seen == [1]


def test_exceptions_in_user_code_name_all_the_candidates():
    def fn(X):
        raise RuntimeError("gpu on fire")

    with pytest.raises(EvaluationError, match=r"evaluating candidates 5, 6, 7, 8 failed: RuntimeError: gpu on fire") as info:
        evaluate(VectorisedEvaluator(fn), array_batch(Backend(), 4, 2, first_id=5))
    assert isinstance(info.value.__cause__, RuntimeError) and info.value.candidate_ids == (5, 6, 7, 8)


def test_long_candidate_lists_are_abbreviated():
    def fn(X):
        raise RuntimeError("x")

    with pytest.raises(EvaluationError, match=r"candidates 0, 1, 2, 3, 4, \.\.\. \(12 candidates\)"):
        evaluate(VectorisedEvaluator(fn), array_batch(Backend(), 12, 2))


def test_an_empty_batch_does_not_call_the_function():
    calls = []
    empty = array_batch(Backend(), 0, 3)
    assert len(evaluate(VectorisedEvaluator(lambda X: calls.append(1) or X), empty)) == 0 and calls == []


def test_repr_names_the_function():
    def my_batch_fitness(X):
        return X

    assert "my_batch_fitness" in repr(VectorisedEvaluator(my_batch_fitness))
