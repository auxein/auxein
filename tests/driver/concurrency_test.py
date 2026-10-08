"""Concurrent evaluation, steady-state delivery and deterministic mode (design doc §8.1, §9.2)."""

import asyncio
import json
import multiprocessing
import threading
import time
import warnings
from pathlib import Path

import pytest

from auxein.backend import Backend
from auxein.driver import Budget, RecordingDisabledWarning, SteadyStateVectorisationWarning, arun, run
from auxein.evaluators import EvaluationError, FunctionEvaluator, VectorisedEvaluator
from auxein.recording import open_run
from auxein.spaces import Box
from auxein.strategies import GeneticAlgorithm, RandomSearch
from tests.support import workers
from tests.support.eventlog import event_log, told_ids
from tests.support.fakes import ManualClock, ScriptedStrategy

SPACE = Box(-5.0, 5.0, dim=3)
IDENTITY = FunctionEvaluator(lambda genome: genome)


def go(strategy=None, evaluator=None, budget=None, batch_size=8, seed=1, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RecordingDisabledWarning)
        return run(
            strategy=strategy or ScriptedStrategy(tell_mode="both"),
            evaluator=evaluator or IDENTITY,
            space=kwargs.pop("space", SPACE),
            budget=budget or Budget(evaluations=20),
            seed=seed,
            batch_size=batch_size,
            **kwargs,
        )


class Outstanding(ScriptedStrategy):
    """Remembers the most candidates it ever had asked but not told, and the sizes of what it was told."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, tell_mode="both", **kwargs)
        self.outstanding = 0
        self.peak = 0
        self.before_ask: list[int] = []
        self.told_sizes: list[int] = []

    def ask(self, n):
        self.before_ask.append(self.outstanding)
        batch = super().ask(n)
        self.outstanding += len(batch.candidates)
        self.peak = max(self.peak, self.outstanding)
        return batch

    def tell(self, results):
        super().tell(results)
        self.outstanding -= len(results)
        self.told_sizes.append(len(results))


# --- the shape of steady-state delivery ---


def test_steady_state_tells_one_result_at_a_time_and_refills_after_each_tell():
    strategy = Outstanding()
    result = go(strategy, budget=Budget(evaluations=20), batch_size=4, delivery="steady_state", concurrency=2)
    assert result.evaluations_used == 20 and result.stop_reason == "budget:evaluations"
    assert strategy.told_sizes == [1] * 20
    assert strategy.asked_n[0] == 4 and set(strategy.asked_n[1:]) <= {1} and strategy.peak <= 4


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("concurrency", [1, 3, 10])
def test_the_window_never_exceeds_batch_size_and_the_budget_is_exact(concurrency: int, deterministic: bool):
    strategy = Outstanding()
    result = go(
        strategy, FunctionEvaluator(jitter), Budget(evaluations=37), batch_size=5, delivery="steady_state",
        concurrency=concurrency, deterministic=deterministic,
    )  # fmt: skip
    assert result.evaluations_used == 37 and strategy.peak <= 5
    assert sum(strategy.told_sizes) == 37 and strategy.outstanding == 0


async def jitter(genome):
    await asyncio.sleep((hash(float(genome)) % 7) * 0.0005)
    return genome


def test_never_more_evaluations_in_progress_than_concurrency_in_steady_state():
    gauge = workers.Gauge()

    async def probe(genome):
        gauge.enter()
        await asyncio.sleep(0.002)
        gauge.leave()
        return genome

    go(evaluator=FunctionEvaluator(probe), budget=Budget(evaluations=40), batch_size=16, delivery="steady_state", concurrency=3)
    assert gauge.calls == 40 and gauge.peak == 3  # the window is bigger than the concurrency: the rest wait in the queue


def test_a_surplus_from_a_generation_sized_strategy_is_queued_and_evaluated_in_ask_order():
    evaluated: list[int] = []

    def spy(genome):
        evaluated.append(int(genome))
        return genome

    strategy = Outstanding(sizes=[10, 10, 10])  # always more than the window of 4
    go(strategy, FunctionEvaluator(spy), Budget(evaluations=30), batch_size=4, delivery="steady_state")
    assert evaluated == list(range(30))  # ask order
    assert strategy.asked_n[0] == 4 and strategy.peak > 4  # the surplus counts towards the window...
    assert max(strategy.before_ask) < 4  # ...so nothing is asked while 4 or more are outstanding
    assert strategy.outstanding == 0 and sum(strategy.told_sizes) == 30


def test_the_evaluation_budget_is_a_hard_limit_and_a_queued_surplus_is_dropped_for_good(tmp_path: Path):
    evaluated: list[int] = []

    def spy(genome):
        evaluated.append(int(genome))
        return genome

    strategy = Outstanding(sizes=[10])  # 10 candidates against a budget of 4
    result = go(strategy, FunctionEvaluator(spy), Budget(evaluations=4), batch_size=8, delivery="steady_state", run_dir=tmp_path / "r")
    assert result.evaluations_used == 4 and evaluated == [0, 1, 2, 3]
    assert sum(strategy.told_sizes) == 4  # the other six were never told...
    assert told_ids(tmp_path / "r") == [0, 1, 2, 3]  # ...nor recorded
    log = event_log(tmp_path / "r")
    assert len(log["candidates"]) == 4 and len(log["evaluations"]) == 4


def test_in_throughput_mode_the_budget_holds_and_every_result_is_told_once(tmp_path: Path):
    strategy = Outstanding()
    result = go(
        strategy, FunctionEvaluator(jitter), Budget(evaluations=60), batch_size=8, delivery="steady_state",
        concurrency=4, deterministic=False, run_dir=tmp_path / "r",
    )  # fmt: skip
    assert result.evaluations_used == 60 and strategy.peak <= 8
    ids = told_ids(tmp_path / "r")
    assert sorted(ids) == list(range(60)) and len(set(ids)) == 60


def test_in_deterministic_mode_results_are_told_in_ask_order_even_when_they_finish_out_of_order(tmp_path: Path):
    async def later_candidates_first(genome):
        await asyncio.sleep(0.02 - 0.0025 * (genome % 8))
        return genome

    result = go(
        evaluator=FunctionEvaluator(later_candidates_first), budget=Budget(evaluations=24), batch_size=8,
        delivery="steady_state", concurrency=8, run_dir=tmp_path / "r",
    )  # fmt: skip
    assert result.evaluations_used == 24
    assert told_ids(tmp_path / "r") == list(range(24))


def test_in_throughput_mode_results_are_told_as_they_finish(tmp_path: Path):
    async def later_candidates_first(genome):
        await asyncio.sleep(0.02 - 0.0025 * genome)
        return genome

    go(
        evaluator=FunctionEvaluator(later_candidates_first), budget=Budget(evaluations=8), batch_size=8,
        delivery="steady_state", concurrency=8, deterministic=False, run_dir=tmp_path / "r",
    )  # fmt: skip
    ids = told_ids(tmp_path / "r")
    assert sorted(ids) == list(range(8)) and ids == list(range(7, -1, -1))  # the slowest was asked first and told last


# --- wall-time budgets ---


def test_with_a_wall_time_budget_evaluations_in_progress_finish_and_queued_ones_are_dropped(tmp_path: Path):
    clock = ManualClock()
    started: list[int] = []

    async def fn(genome):
        started.append(int(genome))
        clock.advance(1.0)
        await asyncio.sleep(0.005)
        return genome

    strategy = Outstanding()
    result = go(
        strategy, FunctionEvaluator(fn), Budget(wall_time=3.0), batch_size=6, delivery="steady_state",
        concurrency=2, clock=clock, run_dir=tmp_path / "r",
    )  # fmt: skip
    assert result.stop_reason == "budget:wall_time"
    assert started == [0, 1, 2, 3]  # two in progress when time ran out (t = 2 < 3 when the second pair started)
    assert result.evaluations_used == 4 and sum(strategy.told_sizes) == 4
    assert told_ids(tmp_path / "r") == [0, 1, 2, 3]  # the queued ones left no trace
    assert len(event_log(tmp_path / "r")["candidates"]) == 4


# --- warnings and metadata ---


def test_a_vectorised_evaluator_in_steady_state_warns_once_per_run():
    vectorised = VectorisedEvaluator(lambda X: (X * X).sum(axis=1))
    with pytest.warns(SteadyStateVectorisationWarning, match="one candidate at a time") as caught:
        go(RandomSearch(), vectorised, Budget(evaluations=20), batch_size=4, delivery="steady_state")
    assert len([w for w in caught if issubclass(w.category, SteadyStateVectorisationWarning)]) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", SteadyStateVectorisationWarning)  # generation delivery does not warn
        go(RandomSearch(), vectorised, Budget(evaluations=20), batch_size=4, delivery="generation")


def test_the_metadata_records_how_the_run_was_scheduled(tmp_path: Path):
    go(RandomSearch(), FunctionEvaluator(workers.sphere), Budget(evaluations=12), batch_size=4, concurrency=3, run_dir=tmp_path / "a")
    meta = json.loads((tmp_path / "a" / "metadata.json").read_text())
    assert (meta["concurrency"], meta["executor"], meta["delivery"], meta["deterministic"], meta["in_flight_window"]) == (
        3, "thread", "generation", True, 4,
    )  # fmt: skip
    go(
        RandomSearch(), FunctionEvaluator(workers.sphere), Budget(evaluations=12), batch_size=4, delivery="steady_state",
        deterministic=False, executor="inline", concurrency=2, run_dir=tmp_path / "b",
    )  # fmt: skip
    meta = json.loads((tmp_path / "b" / "metadata.json").read_text())
    assert (meta["executor"], meta["delivery"], meta["deterministic"], meta["concurrency"]) == ("inline", "steady_state", False, 2)


def test_defaults_are_sequential_generation_delivery_in_deterministic_mode(tmp_path: Path):
    go(RandomSearch(), FunctionEvaluator(workers.sphere), Budget(evaluations=12), batch_size=4, run_dir=tmp_path / "a")
    meta = json.loads((tmp_path / "a" / "metadata.json").read_text())
    assert (meta["concurrency"], meta["executor"], meta["delivery"], meta["deterministic"]) == (1, "inline", "generation", True)


def test_bad_arguments_are_rejected_before_the_run_starts():
    with pytest.raises(ValueError, match="concurrency must be at least 1"):
        go(concurrency=0)
    with pytest.raises(ValueError, match="unknown executor"):
        go(executor="fork")


# --- the same event log, whatever the workers ---


def _record(path: Path, backend: Backend, strategy, evaluator, *, delivery, concurrency, executor, deterministic=True):
    run(
        strategy=strategy, evaluator=evaluator, space=Box(-5.0, 5.0, dim=3), budget=Budget(evaluations=45), seed=11,
        backend=backend, batch_size=8, run_dir=path, delivery=delivery, concurrency=concurrency, executor=executor,
        deterministic=deterministic,
    )  # fmt: skip
    return event_log(path)


STRATEGIES = {
    "random": lambda: RandomSearch(),
    "ga": lambda: GeneticAlgorithm(population_size=6, offspring_size=4),
    "ga-default-offspring": lambda: GeneticAlgorithm(population_size=6),
}


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_the_event_log_is_identical_across_concurrency_executors_and_sync_or_async(
    tmp_path: Path, backend: Backend, strategy: str, delivery: str
):
    sync = FunctionEvaluator(workers.jittery_sphere, uses_rng=True)
    async_ = FunctionEvaluator(workers.async_jittery_sphere, uses_rng=True)
    configurations = [
        (sync, 1, "inline"), (sync, 2, "thread"), (sync, 8, "thread"), (async_, 1, "auto"), (async_, 2, "auto"),
        (async_, 8, "auto"), (async_, 8, "thread"),
    ]  # fmt: skip
    if backend.name == "numpy" and backend.precision == "float64":  # spawning workers is slow enough to do once
        configurations += [(sync, 1, "process"), (sync, 3, "process")]
    reference = _record(tmp_path / "ref", backend, STRATEGIES[strategy](), sync, delivery=delivery, concurrency=1, executor="inline")
    assert len(reference["evaluations"]) == 45
    for i, (evaluator, concurrency, executor) in enumerate(configurations):
        log = _record(
            tmp_path / f"run{i}", backend, STRATEGIES[strategy](), evaluator, delivery=delivery, concurrency=concurrency, executor=executor
        )
        assert log == reference, f"{delivery} delivery, {executor} executor, concurrency {concurrency}"
    assert multiprocessing.active_children() == []


def test_generation_delivery_is_the_same_in_throughput_mode(tmp_path: Path):
    backend = Backend()
    sync = FunctionEvaluator(workers.jittery_sphere, uses_rng=True)
    a = _record(tmp_path / "a", backend, STRATEGIES["ga"](), sync, delivery="generation", concurrency=4, executor="thread")
    b = _record(
        tmp_path / "b", backend, STRATEGIES["ga"](), sync, delivery="generation", concurrency=4, executor="thread", deterministic=False
    )
    assert a == b  # nothing to reorder


def test_the_event_log_changes_with_the_seed_and_with_the_window_but_not_with_workers(tmp_path: Path):
    backend = Backend()
    sync = FunctionEvaluator(workers.jittery_sphere, uses_rng=True)

    def record(name, batch_size, seed):
        run(
            strategy=GeneticAlgorithm(population_size=6, offspring_size=4), evaluator=sync, space=SPACE, budget=Budget(evaluations=45),
            seed=seed, backend=backend, batch_size=batch_size, run_dir=tmp_path / name, delivery="steady_state", concurrency=2,
        )  # fmt: skip
        return event_log(tmp_path / name)

    assert record("a", 8, 1) != record("b", 8, 2) and record("c", 8, 1) != record("d", 12, 1)


# --- the strategies under steady-state delivery ---


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize(("executor", "concurrency"), [("inline", 1), ("thread", 4)])
def test_the_genetic_algorithm_keeps_the_best_mu_of_everything_told(
    tmp_path: Path, backend: Backend, executor: str, concurrency: int, deterministic: bool
):
    ga = GeneticAlgorithm(population_size=6, offspring_size=3)
    function = workers.async_jittery_sphere if executor == "inline" else workers.jittery_sphere
    run(
        strategy=ga, evaluator=FunctionEvaluator(function, uses_rng=True),
        space=SPACE, budget=Budget(evaluations=60), seed=3, backend=backend, batch_size=7, run_dir=tmp_path / "r",
        delivery="steady_state", executor=executor, concurrency=concurrency, deterministic=deterministic,
    )  # fmt: skip
    with open_run(tmp_path / "r") as recorded:
        told = sorted((e.objectives["value"], e.candidate_id) for e in recorded.evaluations())
    assert len(told) == 60
    assert ga.ranked_ids() == [cid for _, cid in told[:6]]


def test_the_genetic_algorithm_improves_under_every_delivery(backend: Backend):
    warnings.simplefilter("ignore", RecordingDisabledWarning)
    results = {}
    for delivery, deterministic in [("generation", True), ("steady_state", True), ("steady_state", False)]:
        results[delivery, deterministic] = run(
            strategy=GeneticAlgorithm(population_size=20, offspring_size=10), evaluator=FunctionEvaluator(workers.sphere),
            space=Box(-5.0, 5.0, dim=3), budget=Budget(evaluations=1500), seed=2, backend=backend, batch_size=10,
            delivery=delivery, concurrency=4, deterministic=deterministic,
        ).best  # fmt: skip
    for best in results.values():
        assert best is not None and best.objectives["value"] < 1e-3


# --- throughput ---


def _time_a_run(delivery: str, deterministic: bool) -> float:
    start = time.perf_counter()
    go(
        RandomSearch(), FunctionEvaluator(workers.async_uneven, uses_rng=True), Budget(evaluations=240), batch_size=8,
        delivery=delivery, concurrency=8, deterministic=deterministic,
    )  # fmt: skip
    return time.perf_counter() - start


def test_steady_state_throughput_mode_is_faster_than_generation_delivery_when_evaluation_times_vary(capsys: pytest.CaptureFixture[str]):
    """Evaluations of 1 to 20 ms, 8 at a time: a generation waits for the slowest of its 8 (about 18 ms), whereas the
    steady-state window refills as soon as any one finishes (about 10 ms per slot). The margin is generous."""
    generation = min(_time_a_run("generation", True) for _ in range(3))
    steady = min(_time_a_run("steady_state", False) for _ in range(3))
    ratio = generation / steady
    with capsys.disabled():
        print(f"\nthroughput: generation {generation * 1000:.0f} ms, steady-state {steady * 1000:.0f} ms, ratio {ratio:.2f}x")
    assert ratio >= 1.3


# --- cancellation and leaks ---


def _eval_threads() -> set[threading.Thread]:
    return {t for t in threading.enumerate() if t.name.startswith("auxein-eval")}


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_pools_are_shut_down_when_a_run_ends_normally_or_fails(delivery: str):
    def sometimes_fails(genome):
        if genome == 7:
            raise RuntimeError("boom")
        time.sleep(0.001)
        return genome

    before = threading.active_count()
    go(
        evaluator=FunctionEvaluator(lambda g: g),
        budget=Budget(evaluations=20),
        batch_size=4,
        delivery=delivery,
        executor="thread",
        concurrency=4,
    )
    assert _eval_threads() == set() and threading.active_count() == before
    with pytest.raises(EvaluationError, match="candidate 7"):
        go(
            evaluator=FunctionEvaluator(sometimes_fails),
            budget=Budget(evaluations=20),
            batch_size=4,
            delivery=delivery,
            executor="thread",
            concurrency=4,
        )
    assert _eval_threads() == set() and threading.active_count() == before


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_processes_are_shut_down_when_a_run_ends_normally_or_fails(delivery: str):
    go(
        evaluator=FunctionEvaluator(workers.double),
        budget=Budget(evaluations=12),
        batch_size=4,
        delivery=delivery,
        executor="process",
        concurrency=2,
    )
    assert multiprocessing.active_children() == []
    with pytest.raises(EvaluationError, match="ZeroDivisionError"):
        go(
            evaluator=FunctionEvaluator(workers.divide_by_zero_1), budget=Budget(evaluations=12), batch_size=4,
            delivery=delivery, executor="process", concurrency=2,
        )  # fmt: skip
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
@pytest.mark.parametrize("executor", ["inline", "thread"])
def test_a_keyboard_interrupt_finalises_the_recording_and_leaves_nothing_running(tmp_path: Path, delivery: str, executor: str):
    def interrupted(genome):
        if genome == 5:
            raise KeyboardInterrupt
        time.sleep(0.002)
        return genome

    before = threading.active_count()
    with pytest.raises(KeyboardInterrupt):
        go(
            evaluator=FunctionEvaluator(interrupted), budget=Budget(evaluations=40), batch_size=4, delivery=delivery,
            executor=executor, concurrency=1 if executor == "inline" else 3, run_dir=tmp_path / "r",
        )  # fmt: skip
    assert json.loads((tmp_path / "r" / "metadata.json").read_text())["status"] == "interrupted"
    assert _eval_threads() == set() and threading.active_count() == before


def test_a_cancelled_run_cancels_its_evaluations_and_leaves_no_task_behind():
    state = {"running": 0}

    async def never_ends(genome):
        state["running"] += 1
        await asyncio.sleep(30)
        return genome

    async def main():
        task = asyncio.ensure_future(
            arun(strategy=ScriptedStrategy(tell_mode="both"), evaluator=FunctionEvaluator(never_ends), space=SPACE,
                 budget=Budget(evaluations=100), seed=1, batch_size=8, concurrency=4, delivery="steady_state")
        )  # fmt: skip
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RecordingDisabledWarning)
        leaked = asyncio.run(main())
    assert state["running"] == 4 and leaked == set()


# --- randomness ---


def test_evaluation_streams_are_numpy_and_vectorised_batch_streams_are_backend_native(backend: Backend):
    seen = {"function": set(), "batch": set()}

    def per_candidate(genome, rng):
        seen["function"].add(type(rng.uniform(2)).__module__.split(".")[0])
        return 0.0

    def per_batch(X, rng):
        seen["batch"].add(type(rng.uniform(2)).__module__.split(".")[0])
        return backend.xp.sum(X, axis=1)

    for evaluator in (FunctionEvaluator(per_candidate, uses_rng=True), VectorisedEvaluator(per_batch, uses_rng=True)):
        go(RandomSearch(), evaluator, Budget(evaluations=8), batch_size=4, backend=backend)
    assert seen["function"] == {"numpy"}
    assert seen["batch"] == {backend.name}
