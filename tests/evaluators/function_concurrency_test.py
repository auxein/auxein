import asyncio
import threading
import time

import pytest

from auxein.backend import Backend
from auxein.core import Candidate, CandidateId, ListBatch
from auxein.evaluators import EvaluationError, FunctionEvaluator
from auxein.execution import make_executor
from tests.support import workers
from tests.support.helpers import array_batch, eval_context, problem

KINDS = ["inline", "thread"]


def evaluate(evaluator, batch, *, concurrency=1, kind="inline", backend=None, seed=0):
    backend = backend or Backend()
    executor = make_executor(kind, concurrency)
    try:
        ctx = eval_context(problem(), backend, seed, concurrency=concurrency, executor=executor)
        return asyncio.run(evaluator.evaluate(batch, ctx))
    finally:
        executor.shutdown()


def listing(n: int) -> ListBatch:
    return ListBatch(tuple(Candidate(CandidateId(i), float(i), (), "t", 0) for i in range(n)))


@pytest.mark.parametrize(("concurrency", "kind"), [(1, "inline"), (1, "thread"), (4, "inline"), (4, "thread"), (4, "process")])
def test_sync_functions_give_the_same_results_with_every_executor_and_concurrency(concurrency: int, kind: str):
    results = evaluate(FunctionEvaluator(workers.double), listing(9), concurrency=concurrency, kind=kind)
    assert [e.candidate.id for e in results] == list(range(9))
    assert [e.objectives["value"] for e in results] == [2.0 * i for i in range(9)]


@pytest.mark.parametrize("concurrency", [1, 4])
def test_async_functions_run_natively_with_any_concurrency(concurrency: int):
    async def slow_double(x):
        await asyncio.sleep(0.001)
        return 2 * x

    results = evaluate(FunctionEvaluator(slow_double), listing(9), concurrency=concurrency)
    assert [e.objectives["value"] for e in results] == [2.0 * i for i in range(9)]


@pytest.mark.parametrize("kind", KINDS)
def test_results_come_back_in_ask_order_even_when_later_candidates_finish_first(kind: str):
    finished: list[float] = []

    def sleepy(x):
        time.sleep(0.03 - 0.01 * x)  # candidate 0 is the slowest, candidate 2 the fastest
        finished.append(x)
        return x

    results = evaluate(FunctionEvaluator(sleepy), listing(3), concurrency=3, kind="thread")
    assert finished == [2.0, 1.0, 0.0]  # completion order really was reversed
    assert [e.candidate.id for e in results] == [0, 1, 2]
    assert [e.objectives["value"] for e in results] == [0.0, 1.0, 2.0]


def test_async_results_come_back_in_ask_order_too():
    finished: list[float] = []

    async def sleepy(x):
        await asyncio.sleep(0.03 - 0.01 * x)
        finished.append(x)
        return x

    results = evaluate(FunctionEvaluator(sleepy), listing(3), concurrency=3)
    assert finished == [2.0, 1.0, 0.0]
    assert [e.candidate.id for e in results] == [0, 1, 2]


@pytest.mark.parametrize("concurrency", [1, 2, 3])
def test_a_threaded_function_never_has_more_evaluations_in_progress_than_the_concurrency(concurrency: int):
    gauge = workers.Gauge()

    def probe(x):
        gauge.enter()
        time.sleep(0.005)
        gauge.leave()
        return x

    evaluate(FunctionEvaluator(probe), listing(12), concurrency=concurrency, kind="thread")
    assert gauge.calls == 12 and gauge.peak <= concurrency
    if concurrency > 1:
        assert gauge.peak > 1  # and it really does overlap evaluations


@pytest.mark.parametrize("concurrency", [1, 2, 5])
def test_an_async_function_never_has_more_evaluations_in_progress_than_the_concurrency(concurrency: int):
    gauge = workers.Gauge()

    async def probe(x):
        gauge.enter()
        await asyncio.sleep(0.003)
        gauge.leave()
        return x

    evaluate(FunctionEvaluator(probe), listing(12), concurrency=concurrency)
    assert gauge.calls == 12 and gauge.peak == min(concurrency, 12)


@pytest.mark.parametrize("kind", KINDS)
def test_an_exception_cancels_the_rest_of_the_batch_and_names_the_candidate(kind: str):
    started: list[float] = []

    async def fn(x):
        started.append(x)
        if x == 1:
            raise RuntimeError("boom")
        await asyncio.sleep(10)  # would hang the test if it were not cancelled
        return x

    t0 = time.perf_counter()
    with pytest.raises(EvaluationError, match="candidate 1 failed: RuntimeError: boom") as raised:
        evaluate(FunctionEvaluator(fn), listing(4), concurrency=4)
    assert raised.value.candidate_ids == (1,) and isinstance(raised.value.__cause__, RuntimeError)
    assert time.perf_counter() - t0 < 5 and sorted(started) == [0.0, 1.0, 2.0, 3.0]


def test_candidates_waiting_for_a_slot_are_not_started_after_a_failure():
    started: list[float] = []

    def fn(x):
        started.append(x)
        if x == 0:
            time.sleep(0.02)
            raise RuntimeError("boom")
        time.sleep(0.02)
        return x

    with pytest.raises(EvaluationError):
        evaluate(FunctionEvaluator(fn), listing(10), concurrency=2, kind="thread")
    assert len(started) < 10


def test_a_sync_exception_in_a_thread_is_an_evaluation_error():
    def fn(x):
        raise ValueError("bad genome")

    with pytest.raises(EvaluationError, match="ValueError: bad genome"):
        evaluate(FunctionEvaluator(fn), listing(3), concurrency=2, kind="thread")


def test_validation_errors_of_what_the_function_returns_look_the_same_with_every_executor():
    messages = []
    for concurrency, kind in [(1, "inline"), (2, "thread"), (2, "process")]:
        with pytest.raises(TypeError) as raised:
            evaluate(FunctionEvaluator(workers.not_a_number), listing(2), concurrency=concurrency, kind=kind)
        messages.append(str(raised.value))
    assert len(set(messages)) == 1 and "candidate 0 must return a number or a Result" in messages[0]  # the first bad one, in the parent


def test_a_process_executor_with_an_async_function_is_an_error():
    async def fn(x):
        return x

    with pytest.raises(ValueError, match="async function.*cannot be sent to worker processes"):
        evaluate(FunctionEvaluator(fn), listing(2), concurrency=2, kind="process")


def test_a_lambda_with_a_process_executor_explains_itself():
    from auxein.execution import ExecutorError

    with pytest.raises(EvaluationError) as raised:
        evaluate(FunctionEvaluator(lambda x: x), listing(2), concurrency=2, kind="process")
    assert isinstance(raised.value.__cause__, ExecutorError) and "executor='thread'" in str(raised.value)


def test_the_candidates_stream_reaches_a_worker_process_and_gives_the_same_numbers(backend: Backend):
    batch = array_batch(backend, 5, 2, first_id=7)
    reference = evaluate(FunctionEvaluator(workers.jittery_sphere, uses_rng=True), batch, backend=backend, seed=4)
    in_processes = evaluate(
        FunctionEvaluator(workers.jittery_sphere, uses_rng=True), batch, concurrency=3, kind="process", backend=backend, seed=4
    )
    assert [e.objectives["value"] for e in in_processes] == [e.objectives["value"] for e in reference]


def test_sequential_evaluation_keeps_its_fast_path():
    names = []

    def fn(x):
        names.append(threading.current_thread())
        return x

    evaluate(FunctionEvaluator(fn), listing(3))
    assert names == [threading.main_thread()] * 3  # called directly: no hand-off to another thread
