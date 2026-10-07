import asyncio
import time

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import Candidate, CandidateId, ListBatch, Objective, Result, Status
from auxein.evaluators import EvaluationError, FunctionEvaluator
from auxein.random import RunSeed
from tests.support.helpers import array_batch, eval_context, problem


def evaluate(evaluator, batch, spec=None, backend=None, seed=0):
    spec = spec or problem()
    return asyncio.run(evaluator.evaluate(batch, eval_context(spec, backend or Backend(), seed)))


def test_a_sync_function_returning_bare_numbers(backend: Backend):
    results = evaluate(FunctionEvaluator(lambda g: float(g.sum())), array_batch(backend, 4, 3), backend=backend)
    assert [e.candidate.id for e in results] == [0, 1, 2, 3]
    assert [e.objectives["value"] for e in results] == pytest.approx([6.0, 15.0, 24.0, 33.0])
    assert all(e.status is Status.OK for e in results)


def test_the_function_receives_each_row_as_given(backend: Backend):
    seen = []
    evaluate(FunctionEvaluator(lambda g: seen.append(g) or 0.0), array_batch(backend, 3, 2), backend=backend)
    assert len(seen) == 3 and all(tuple(g.shape) == (2,) for g in seen)
    assert backend.matches(seen[0])  # a row view on the backend: not converted or copied to the host


def test_an_async_function(backend: Backend):
    async def slow_sphere(genome):
        await asyncio.sleep(0)
        return float((genome * genome).sum())

    results = evaluate(FunctionEvaluator(slow_sphere), array_batch(backend, 3, 2), backend=backend)
    assert [e.objectives["value"] for e in results] == pytest.approx([5.0, 25.0, 61.0])


def test_candidates_are_evaluated_sequentially_in_batch_order():
    order = []
    evaluate(FunctionEvaluator(lambda g: order.append(float(g[0])) or 0.0), array_batch(Backend(), 5, 2))
    assert order == [1.0, 3.0, 5.0, 7.0, 9.0]


def test_list_batches_with_any_genome():
    candidates = [Candidate(CandidateId(i), {"depth": i}, (), "init", 0) for i in range(3)]
    results = evaluate(FunctionEvaluator(lambda g: float(g["depth"]) * 2), ListBatch(candidates))
    assert [e.objectives["value"] for e in results] == [0.0, 2.0, 4.0]
    assert [e.candidate for e in results] == candidates


def test_results_with_constraints_descriptors_and_cost_units():
    spec = problem((Objective("loss"),), ("cpa",), ("speed",))
    fn = lambda g: Result({"loss": float(g[0])}, {"cpa": 0.5}, {"speed": 2.0}, {"tokens": 7})  # noqa: E731
    results = evaluate(FunctionEvaluator(fn), array_batch(Backend(), 2, 2), spec)
    assert dict(results[1].objectives) == {"loss": 3.0} and dict(results[1].constraints) == {"cpa": 0.5}
    assert dict(results[1].descriptors) == {"speed": 2.0} and dict(results[1].cost.units) == {"tokens": 7.0}


def test_wall_time_is_recorded_per_candidate():
    def slow(genome):
        time.sleep(0.01)
        return 0.0

    results = evaluate(FunctionEvaluator(slow), array_batch(Backend(), 3, 2))
    assert all(0.009 <= e.cost.wall_time < 0.5 for e in results)


def test_uses_rng_passes_the_candidates_own_stream(backend: Backend):
    draws = {}

    def fn(genome, rng):
        value = rng.uniform(3)
        draws[len(draws)] = backend.to_numpy(value)
        return float(backend.to_numpy(value).sum())

    results = evaluate(FunctionEvaluator(fn, uses_rng=True), array_batch(backend, 3, 2, first_id=40), backend=backend, seed=9)
    for i, cid in enumerate((40, 41, 42)):
        expected = backend.to_numpy(RunSeed(9).stream("evaluation", cid, backend=backend).uniform(3))
        np.testing.assert_array_equal(draws[i], expected)
        assert results[i].objectives["value"] == pytest.approx(float(expected.sum()))


def test_evaluation_randomness_follows_the_candidate_not_the_batch(backend: Backend):
    def fn(genome, rng):
        return float(backend.to_numpy(rng.uniform(1))[0])

    evaluator = FunctionEvaluator(fn, uses_rng=True)
    together = evaluate(evaluator, array_batch(backend, 4, 2, first_id=10), backend=backend)
    alone = evaluate(evaluator, array_batch(backend, 1, 2, first_id=12), backend=backend)  # candidate 12 on its own
    assert together[2].objectives["value"] == alone[0].objectives["value"]
    again = evaluate(evaluator, array_batch(backend, 4, 2, first_id=10), backend=backend)
    assert [e.objectives["value"] for e in again] == [e.objectives["value"] for e in together]
    different = evaluate(evaluator, array_batch(backend, 4, 2, first_id=10), backend=backend, seed=1)
    assert [e.objectives["value"] for e in different] != [e.objectives["value"] for e in together]


def test_without_uses_rng_the_function_gets_only_the_genome():
    calls = []
    evaluate(FunctionEvaluator(lambda genome: calls.append(1) or 0.0), array_batch(Backend(), 2, 2))
    assert len(calls) == 2
    with pytest.raises(EvaluationError, match="positional argument"):
        evaluate(FunctionEvaluator(lambda genome, rng: 0.0), array_batch(Backend(), 1, 2))  # type: ignore[arg-type]


def test_exceptions_in_user_code_are_wrapped_naming_the_candidate_and_chained():
    def fn(genome):
        if float(genome[0]) > 3.5:
            raise ZeroDivisionError("boom")
        return 1.0

    with pytest.raises(EvaluationError, match=r"evaluating candidate 2 failed: ZeroDivisionError: boom") as info:
        evaluate(FunctionEvaluator(fn), array_batch(Backend(), 4, 2))
    assert isinstance(info.value.__cause__, ZeroDivisionError)
    assert info.value.candidate_ids == (2,)


def test_async_exceptions_are_wrapped_too():
    async def fn(genome):
        raise KeyError("missing")

    with pytest.raises(EvaluationError, match="candidate 0 failed: KeyError") as info:
        evaluate(FunctionEvaluator(fn), array_batch(Backend(), 2, 2))
    assert isinstance(info.value.__cause__, KeyError)


def test_interrupts_are_not_wrapped():
    def fn(genome):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        evaluate(FunctionEvaluator(fn), array_batch(Backend(), 2, 2))


def test_invalid_returns_are_not_wrapped_as_evaluation_errors():
    with pytest.raises(TypeError, match="returned a dict"):
        evaluate(FunctionEvaluator(lambda g: {"value": 1.0}), array_batch(Backend(), 2, 2))
    with pytest.raises(ValueError, match="candidate 1: objective 'value' is nan"):
        evaluate(FunctionEvaluator(lambda g: float("nan") if float(g[0]) > 2 else 1.0), array_batch(Backend(), 3, 2))


def test_an_empty_batch():
    assert len(evaluate(FunctionEvaluator(lambda g: 0.0), ListBatch([]))) == 0


def test_repr_names_the_function():
    def my_fitness(genome):
        return 0.0

    assert "my_fitness" in repr(FunctionEvaluator(my_fitness)) and "uses_rng=False" in repr(FunctionEvaluator(my_fitness))
