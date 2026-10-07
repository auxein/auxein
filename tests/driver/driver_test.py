import asyncio
import warnings

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import ArrayBatch, Candidate, CandidateId, EvaluationBatch, ListBatch, Objective, Result
from auxein.driver import Budget, EvaluatorError, RecordingDisabledWarning, StrategyError, arun, run
from auxein.evaluators import EvaluationError, FunctionEvaluator, VectorisedEvaluator
from auxein.spaces import Box
from auxein.strategies import RandomSearch
from tests.support.fakes import ManualClock, ScriptedStrategy

SPACE = Box(-5.0, 5.0, dim=3)
IDENTITY = FunctionEvaluator(lambda genome: genome)  # the genome of a scripted candidate is its objective value


def go(strategy=None, evaluator=None, budget: Budget | None = None, batch_size: int = 8, seed: int = 1, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RecordingDisabledWarning)
        return run(
            strategy=strategy or ScriptedStrategy(),
            evaluator=evaluator or IDENTITY,
            space=kwargs.pop("space", SPACE),
            budget=budget or Budget(evaluations=20),
            seed=seed,
            batch_size=batch_size,
            **kwargs,
        )


# --- budgets ---


@pytest.mark.parametrize(("budget", "batch_size"), [(20, 8), (16, 8), (7, 8), (1, 8), (100, 1), (64, 64), (65, 64)])
def test_the_evaluation_budget_is_exact(budget: int, batch_size: int):
    strategy = ScriptedStrategy()
    result = go(strategy, budget=Budget(evaluations=budget), batch_size=batch_size)
    assert result.evaluations_used == budget and result.stop_reason == "budget:evaluations"
    assert sum(len(t) for t in strategy.told) == budget  # no truncation here: the driver asks for what remains


def test_the_driver_asks_for_the_batch_size_or_what_remains():
    strategy = ScriptedStrategy()
    go(strategy, budget=Budget(evaluations=20), batch_size=8)
    assert strategy.asked_n == [8, 8, 4]


def test_an_oversized_final_batch_is_truncated_recorded_and_not_told(tmp_path):
    from auxein.recording import open_run

    strategy = ScriptedStrategy(sizes=[4, 10])  # the second batch is bigger than the 6 evaluations that remain
    evaluated: list[list[int]] = []

    def spy(genome):
        evaluated.append(int(genome))
        return genome

    result = go(strategy, FunctionEvaluator(spy), Budget(evaluations=10), batch_size=4, run_dir=tmp_path / "r")
    assert result.evaluations_used == 10 and result.stop_reason == "budget:evaluations"
    assert evaluated == list(range(10))  # the first 6 of the 10 proposed, in ask order
    assert len(strategy.told) == 1 and len(strategy.told[0]) == 4  # the incomplete batch was not told
    assert strategy.proposed == 14
    with open_run(tmp_path / "r") as recorded:
        assert [r.candidate_id for r in recorded.evaluations()] == list(range(10))
        events = [e for e in recorded._db.execute("SELECT kind, step FROM events ORDER BY seq")]
    assert events == [("ask", 0), ("tell", 0), ("ask", 1), ("stop", None)]  # asked and recorded, but never told


def test_every_evaluated_candidate_counts_towards_the_result():
    result = go(ScriptedStrategy(sizes=[4, 10], genomes=lambda i: 100.0 - i), budget=Budget(evaluations=10), batch_size=4)
    assert result.best is not None and result.best.objectives["value"] == 91.0  # candidate 9, from the truncated batch


def test_a_wall_time_budget_is_checked_between_batches():
    clock = ManualClock()

    def tick(genome):
        clock.advance(1.0)  # every evaluation takes one second
        return genome

    strategy = ScriptedStrategy()
    result = go(strategy, FunctionEvaluator(tick), Budget(wall_time=10.0), batch_size=4, clock=clock)
    assert result.stop_reason == "budget:wall_time"
    assert result.evaluations_used == 12  # the batch in progress completes: 4 + 4 = 8 seconds, then 4 more reach 12 >= 10
    assert result.wall_time == 12.0


def test_the_wall_time_budget_may_be_exceeded_by_one_batch_but_no_more():
    clock = ManualClock()
    result = go(
        ScriptedStrategy(), FunctionEvaluator(lambda g: clock.advance(100.0) or g), Budget(wall_time=1.0), batch_size=5, clock=clock
    )
    assert result.evaluations_used == 5 and result.wall_time == 500.0


def test_an_exhausted_wall_time_stops_before_the_first_ask():
    readings = iter([0.0] + [10.0] * 50)  # the start of the run, then ten seconds later
    strategy = ScriptedStrategy()
    result = go(strategy, budget=Budget(wall_time=5.0), clock=lambda: next(readings))
    assert result.stop_reason == "budget:wall_time" and result.evaluations_used == 0 and strategy.asks == 0


def test_a_cost_budget_is_checked_between_batches():
    spec = {"tokens": 10.0}
    strategy = ScriptedStrategy()
    result = go(strategy, FunctionEvaluator(lambda g: Result({"value": g}, cost=spec)), Budget(cost={"tokens": 100.0}), batch_size=4)
    assert result.stop_reason == "budget:cost:tokens"
    assert result.evaluations_used == 12  # 3 batches of 4 evaluations at 10 tokens: 120 >= 100, checked after each batch


def test_cost_units_without_a_limit_are_ignored_and_several_limits_race():
    evaluator = FunctionEvaluator(lambda g: Result({"value": g}, cost={"tokens": 1.0, "money": 5.0}))
    result = go(ScriptedStrategy(), evaluator, Budget(evaluations=1000, cost={"money": 50.0, "tokens": 1e9}), batch_size=2)
    assert result.stop_reason == "budget:cost:money" and result.evaluations_used == 10


def test_the_evaluation_budget_wins_when_several_are_exhausted_together():
    result = go(ScriptedStrategy(), budget=Budget(evaluations=8, cost={"tokens": 1.0}), batch_size=8)
    assert result.stop_reason == "budget:evaluations"  # there is no cost to speak of: only the evaluations ran out


def test_should_stop_ends_the_run():
    strategy = ScriptedStrategy(stop_after=3)
    result = go(strategy, budget=Budget(evaluations=1000), batch_size=5)
    assert result.stop_reason == "strategy" and result.evaluations_used == 15 and strategy.asks == 3


def test_a_strategy_that_stops_at_once_gets_no_evaluations():
    result = go(ScriptedStrategy(stop_after=0))
    assert (
        result.stop_reason == "strategy"
        and result.evaluations_used == 0
        and result.best is None
        and result.trace == ()
        and result.pareto_front == ()
    )


# --- what the driver checks on the strategy and the evaluator ---


def test_bind_rejection_propagates():
    with pytest.raises(ValueError, match="only one objective please"):
        go(ScriptedStrategy(reject="only one objective please"))


def test_the_strategy_is_bound_before_the_first_ask():
    strategy = ScriptedStrategy()
    go(strategy)
    assert strategy.bound is not None and strategy.bound[0].objective_names == ("value",)


def test_a_strategy_that_needs_steady_state_delivery_is_rejected():
    with pytest.raises(ValueError, match="generation only"):
        go(ScriptedStrategy(tell_mode="steady_state"))
    go(ScriptedStrategy(tell_mode="both"))


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        go(batch_size=0)


class Replaying(ScriptedStrategy):
    """Asks for candidates whose ids it chooses itself."""

    def __init__(self, make_ids, steps=None) -> None:
        super().__init__()
        self._make_ids, self._steps = make_ids, steps

    def ask(self, n: int):
        _, ctx = self.bound
        issued = [ctx.new_id() for _ in range(n)]
        self.asks += 1
        chosen = self._make_ids(self.asks, issued)
        steps = self._steps(self.asks) if self._steps else [self.asks - 1] * len(chosen)
        return ListBatch([Candidate(c, float(i), (), "x", s) for i, (c, s) in enumerate(zip(chosen, steps))])


def test_an_empty_batch_is_rejected():
    with pytest.raises(StrategyError, match="empty batch"):
        go(Replaying(lambda k, issued: []))


def test_reusing_an_id_is_rejected_even_across_batches():
    with pytest.raises(StrategyError, match="candidate id 0 again"):
        go(Replaying(lambda k, issued: [CandidateId(0)] * 0 + [CandidateId(0)] + issued[1:]), batch_size=4, budget=Budget(evaluations=20))


def test_a_duplicate_id_inside_a_batch_is_rejected():
    with pytest.raises(StrategyError, match="candidate id 1 again"):
        go(Replaying(lambda k, issued: [issued[0], issued[1], issued[1]]), batch_size=3)


def test_an_id_that_this_run_never_issued_is_rejected():
    with pytest.raises(StrategyError, match="id 12345, which this run never issued"):
        go(Replaying(lambda k, issued: [CandidateId(12345)]), batch_size=1)
    with pytest.raises(StrategyError, match="id -1, which this run never issued"):
        go(Replaying(lambda k, issued: [CandidateId(-1)]), batch_size=1)


def test_a_batch_must_have_one_step_and_steps_never_go_back():
    with pytest.raises(StrategyError, match="different steps"):
        go(Replaying(lambda k, issued: issued, steps=lambda k: [0, 1, 0, 1]), batch_size=4)
    with pytest.raises(StrategyError, match="step 0 after step 5"):
        go(Replaying(lambda k, issued: issued, steps=lambda k: [5 if k == 1 else 0] * 4), batch_size=4)


def test_array_batches_with_foreign_ids_are_rejected_too():
    class Foreign(ScriptedStrategy):
        def ask(self, n):
            return ArrayBatch(np.zeros((n, 3)), [CandidateId(900 + i) for i in range(n)], 0, "x")

    with pytest.raises(StrategyError, match="never issued"):
        go(Foreign(), batch_size=2)


def test_evaluators_must_return_one_evaluation_per_candidate_in_ask_order():
    class Dropping:
        async def evaluate(self, batch, ctx):
            return await IDENTITY.evaluate(ListBatch(batch.candidates[:-1]), ctx)

    class Reversing:
        async def evaluate(self, batch, ctx):
            return EvaluationBatch(list((await IDENTITY.evaluate(batch, ctx)).evaluations)[::-1])

    class NotABatch:
        async def evaluate(self, batch, ctx):
            return [1.0]

    with pytest.raises(EvaluatorError, match=r"one evaluation per candidate, in ask order.*3 of 4"):
        go(evaluator=Dropping(), batch_size=4)
    with pytest.raises(EvaluatorError, match="in ask order"):
        go(evaluator=Reversing(), batch_size=4)
    with pytest.raises(EvaluatorError, match="must return an EvaluationBatch, got list"):
        go(evaluator=NotABatch(), batch_size=4)


def test_exceptions_in_user_code_fail_the_run_naming_the_candidate():
    def fn(genome):
        if genome >= 5:
            raise RuntimeError("boom")
        return genome

    with pytest.raises(EvaluationError, match="candidate 5 failed: RuntimeError: boom") as info:
        go(evaluator=FunctionEvaluator(fn), batch_size=4)
    assert isinstance(info.value.__cause__, RuntimeError)


# --- the event loop ---


def test_run_works_in_a_plain_script():
    assert go(budget=Budget(evaluations=10)).evaluations_used == 10


def test_run_works_inside_a_running_event_loop():
    async def main():
        loop = asyncio.get_running_loop()
        result = go(budget=Budget(evaluations=10))  # a synchronous call from inside a coroutine, as in Jupyter
        assert asyncio.get_running_loop() is loop
        return result

    assert asyncio.run(main()).evaluations_used == 10


def test_run_inside_a_running_loop_uses_a_dedicated_thread_and_propagates_errors():
    import threading

    names = []

    async def main():
        go(evaluator=FunctionEvaluator(lambda g: names.append(threading.current_thread().name) or g), budget=Budget(evaluations=2))
        with pytest.raises(ValueError, match="nope"):
            go(ScriptedStrategy(reject="nope"))

    asyncio.run(main())
    assert names and all(n.startswith("auxein-driver") for n in names)


def test_arun_is_the_asynchronous_entry_point():
    async def main():
        return await arun(
            strategy=ScriptedStrategy(), evaluator=IDENTITY, space=SPACE, budget=Budget(evaluations=9), seed=1, batch_size=4, run_dir=None
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RecordingDisabledWarning)
        assert asyncio.run(main()).evaluations_used == 9


def test_async_evaluators_run_natively_in_the_driver():
    async def fn(genome):
        await asyncio.sleep(0)
        return genome

    assert go(evaluator=FunctionEvaluator(fn), budget=Budget(evaluations=6)).evaluations_used == 6


# --- the recording warning ---


def run_collecting_warnings(**kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(strategy=ScriptedStrategy(), evaluator=IDENTITY, space=SPACE, budget=Budget(evaluations=10), seed=1, batch_size=4, **kwargs)
    return [w for w in caught if issubclass(w.category, RecordingDisabledWarning)]


def test_the_recording_warning_is_emitted_exactly_once_per_run_without_a_run_dir():
    caught = run_collecting_warnings()
    assert len(caught) == 1  # not once per batch
    assert "not recorded" in str(caught[0].message) and "run_dir" in str(caught[0].message) and "filterwarnings" in str(caught[0].message)
    assert caught[0].filename == __file__  # the stack level points at the user's call
    assert len(run_collecting_warnings()) == 1  # and again for the next run


def test_no_warning_with_a_run_dir(tmp_path):
    assert run_collecting_warnings(run_dir=tmp_path / "r") == []


def test_the_warning_can_be_silenced_with_one_line():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=RecordingDisabledWarning)
        run(strategy=ScriptedStrategy(), evaluator=IDENTITY, space=SPACE, budget=Budget(evaluations=4), seed=1)
    assert caught == []


def test_the_warning_is_a_user_warning_and_arun_warns_at_the_awaiting_call():
    assert issubclass(RecordingDisabledWarning, UserWarning)

    async def main():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await arun(strategy=ScriptedStrategy(), evaluator=IDENTITY, space=SPACE, budget=Budget(evaluations=4), seed=1)
        return caught

    caught = [w for w in asyncio.run(main()) if issubclass(w.category, RecordingDisabledWarning)]
    assert len(caught) == 1 and caught[0].filename == __file__


# --- objectives ---


def test_the_default_problem_has_one_minimised_objective_called_value():
    strategy = ScriptedStrategy()
    go(strategy)
    assert strategy.bound is not None
    assert strategy.bound[0].objectives == (Objective("value"),)


def test_custom_objectives_constraints_and_descriptors_reach_the_strategy_and_the_evaluator():
    evaluator = FunctionEvaluator(lambda g: Result({"loss": g, "score": -g}, {"cpa": 0.0}, {"speed": 1.0}))
    strategy = ScriptedStrategy()
    result = go(
        strategy, evaluator, objectives=[Objective("loss"), Objective("score", "maximise")], constraints=["cpa"], descriptors=["speed"]
    )
    assert strategy.bound is not None and strategy.bound[0].constraints == ("cpa",) and strategy.bound[0].descriptors == ("speed",)
    assert result.best is None  # a single best is only defined for one objective


def test_a_backend_reaches_the_strategy_and_the_arrays(backend: Backend):
    result = go(RandomSearch(), VectorisedEvaluator(lambda X: (X * X).sum(axis=1)), Budget(evaluations=30), backend=backend)
    assert result.evaluations_used == 30 and result.best is not None
    assert backend.matches(result.best.candidate.genome)


def test_invalid_returns_surface_with_their_own_error_types():
    with pytest.raises(TypeError, match="returned a dict"):
        go(evaluator=FunctionEvaluator(lambda g: {"value": g}))
    with pytest.raises(ValueError, match="candidate 0: objective 'value' is nan"):
        go(evaluator=FunctionEvaluator(lambda g: float("nan")))


def test_the_dim_of_a_space_is_independent_of_the_scripted_genomes():
    assert np.isfinite(
        go(RandomSearch(), VectorisedEvaluator(lambda X: X.sum(axis=1)), space=Box(-1.0, 1.0, dim=7)).best.objectives["value"]
    )
