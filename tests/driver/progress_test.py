"""Progress callbacks (design doc §9.6): when they fire, what they see, how they stop a run, replay, errors, and that they cost nothing."""

import asyncio
import io
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import auxein
from auxein.callbacks import stop_when
from auxein.core import Objective, Result, Status
from auxein.driver import Progress, StopRun
from auxein.recording import open_run
from tests.driver.resume_test import comparable
from tests.support import multiobjective
from tests.support.fixtures import integration_backend
from tests.support.reading import peek

pytestmark = [
    pytest.mark.filterwarnings("ignore::auxein.driver.errors.EvaluationFailureWarning"),
    pytest.mark.filterwarnings("ignore::auxein.RecordingDisabledWarning"),
    pytest.mark.usefixtures("use_corner_backend"),
]


def sphere(genome: np.ndarray) -> float:
    return float((genome * genome).sum())


class Spy:
    """A callback that keeps every `Progress` it is given."""

    def __init__(self) -> None:
        self.calls: list[Progress[Any]] = []

    def __call__(self, progress: Progress[Any]) -> None:
        self.calls.append(progress)

    @property
    def told(self) -> list[Progress[Any]]:
        return [p for p in self.calls if not p.final]


def settings(run_dir: Path | None = None, evaluations: int = 55, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "strategy": auxein.GeneticAlgorithm(population_size=10, offspring_size=10),
        "evaluator": auxein.FunctionEvaluator(sphere),
        "space": auxein.Box(-5.0, 5.0, dim=3),
        "budget": auxein.Budget(evaluations=evaluations),
        "seed": 3,
        "batch_size": 10,
        "run_dir": run_dir,
        "backend": integration_backend(),
    }
    base.update(over)
    return base


# --- when callbacks fire ---


def test_a_genetic_algorithm_fires_once_per_generation_then_the_final_event(tmp_path: Path):
    spy = Spy()
    result = auxein.run(**settings(tmp_path / "ga", callbacks=[spy]))
    assert [p.generation for p in spy.told] == [1, 2, 3, 4, 5]
    assert [p.evaluations_used for p in spy.told] == [10, 20, 30, 40, 50]
    assert [p.step for p in spy.told] == sorted({p.step for p in spy.told})  # one ask round each
    assert len(spy.calls) == 6 and spy.calls[-1].final and not any(p.final for p in spy.told)
    assert result.stop_reason == "budget:evaluations" == spy.calls[-1].stop_reason
    assert all(p.stop_reason is None and not p.replaying and p.delivery == "generation" for p in spy.told)


def test_nsga2_fires_once_per_generation_too(tmp_path: Path):
    def zdt(genome: np.ndarray) -> Result:
        f1, f2 = multiobjective.zdt1(genome[None, :])[0]
        return Result({"f1": float(f1), "f2": float(f2)})

    spy = Spy()
    auxein.run(
        strategy=auxein.NSGA2(population_size=12, offspring_size=12),
        evaluator=auxein.FunctionEvaluator(zdt),
        space=multiobjective.ZDT_SPACE,
        objectives=[multiobjective.F1, multiobjective.F2],
        budget=auxein.Budget(evaluations=60),
        seed=1,
        batch_size=12,
        backend=integration_backend(),
        callbacks=[spy],
    )
    assert [p.generation for p in spy.told] == [1, 2, 3, 4, 5] and spy.calls[-1].final
    assert spy.calls[-1].best is None and len(spy.calls[-1].pareto_front) >= 1  # several objectives: a front, no single best


def test_a_truncated_final_batch_fires_no_call_but_the_final_event_counts_it(tmp_path: Path):
    spy = Spy()
    result = auxein.run(**settings(tmp_path / "t", callbacks=[spy]))  # 55 evaluations in batches of 10: the last is cut to 5
    assert result.evaluations_used == 55 and len(spy.told) == 5 and spy.told[-1].evaluations_used == 50
    final = spy.calls[-1]
    assert final.evaluations_used == 55 and final.remaining == 0 and final.generation == 5
    assert final.batch is not None and final.batch.size == 10  # the last batch that was told: the truncated one never was
    assert final.status_counts == {"ok": 55, "failed": 0, "timeout": 0}


def test_callbacks_see_everything_they_count_already_recorded_and_checkpointed(tmp_path: Path):
    run_dir, seen = tmp_path / "disk", []

    def look(progress: Progress[Any]) -> None:
        with open_run(run_dir) as run:
            recorded = sum(1 for _ in run.evaluations())
            checkpoints = run.checkpoints()
        newest = max((c.evaluations_used for c in checkpoints), default=0)
        seen.append((progress.evaluations_used, recorded, newest, progress.final))

    auxein.run(**settings(run_dir, evaluations=60, checkpoint_every_evaluations=20, callbacks=[look]))
    assert all(used == recorded for used, recorded, _, _ in seen)  # the snapshot counts only what is on disk
    by_used = {used: newest for used, _, newest, final in seen if not final}
    assert by_used[20] == 20 and by_used[40] == 40  # a due checkpoint is taken before the callbacks run


def test_steady_state_fires_after_each_tell_and_the_throttle_counts_evaluations():
    from auxein.callbacks import every_evaluations

    every_tell, throttled = Spy(), Spy()
    auxein.run(
        **settings(
            evaluations=20,
            strategy=auxein.RandomSearch(),
            delivery="steady_state",
            batch_size=4,
            callbacks=[every_tell, every_evaluations(5, throttled)],
        )
    )
    assert [p.evaluations_used for p in every_tell.told] == list(range(1, 21))
    assert all(p.delivery == "steady_state" for p in every_tell.calls) and every_tell.calls[-1].final
    assert [(p.evaluations_used, p.final) for p in throttled.calls] == [(5, False), (10, False), (15, False), (20, False), (20, True)]


# --- the snapshot ---


def mixed_evaluator() -> auxein.FunctionEvaluator:
    def judge(genome: np.ndarray) -> Result:
        if genome[0] > 3.0:
            raise ValueError("a bad region")
        return Result(
            {"cost": float(genome[1] ** 2 + genome[2]), "gain": float(genome[0] + genome[2])}, constraints={"c": max(0.0, genome[1] - 2.0)}
        )

    return auxein.FunctionEvaluator(judge)


def mixed_settings(run_dir: Path, **over: Any) -> dict[str, Any]:
    return settings(
        run_dir,
        evaluations=60,
        strategy=auxein.RandomSearch(),
        evaluator=mixed_evaluator(),
        objectives=[Objective("cost", "minimise"), Objective("gain", "maximise")],
        constraints=["c"],
        **over,
    )


def test_the_snapshot_agrees_with_values_computed_from_the_recording(tmp_path: Path):
    spy = Spy()
    result = auxein.run(**mixed_settings(tmp_path / "m", callbacks=[spy]))
    records = peek(tmp_path / "m").evaluations()
    assert any(r.status is Status.FAILED for r in records) and any(any(v > 0 for v in r.constraints.values()) for r in records)
    for progress in spy.told:
        upto = progress.evaluations_used
        batch = records[upto - 10 : upto]
        so_far = records[:upto]
        assert progress.remaining == 60 - upto and progress.budget.evaluations == 60
        counts = {s: sum(1 for r in so_far if r.status is s) for s in (Status.OK, Status.FAILED, Status.TIMEOUT)}
        assert dict(progress.status_counts) == {"ok": counts[Status.OK], "failed": counts[Status.FAILED], "timeout": 0}
        stats = progress.batch
        assert stats is not None and stats.size == 10
        ok = [r for r in batch if r.status is Status.OK]
        assert (stats.ok, stats.failed, stats.timeout) == (len(ok), 10 - len(ok), 0)
        assert stats.feasible_fraction == pytest.approx(sum(1 for r in ok if r.constraints["c"] == 0) / 10)
        for name, best in (("cost", min), ("gain", max)):  # natural units; the best follows the direction
            values = [r.objectives[name] for r in ok]
            worst = max if best is min else min
            expected = (best(values), float(np.mean(values)), float(np.median(values)), worst(values))
            got = stats.objectives[name]
            assert (got.best, got.mean, got.median, got.worst) == pytest.approx(expected)
        feasible = [r for r in so_far if r.status is Status.OK and r.constraints["c"] == 0]
        points = {r.candidate_id: (r.objectives["cost"], -r.objectives["gain"]) for r in feasible}  # minimisation form
        front = {i for i, p in points.items() if not any(all(q[k] <= p[k] for k in (0, 1)) and q != p for j, q in points.items() if j != i)}
        assert {int(e.candidate.id) for e in progress.pareto_front} == front
        assert progress.best is None
    assert [int(e.candidate.id) for e in spy.calls[-1].pareto_front] == [int(e.candidate.id) for e in result.pareto_front]
    assert spy.calls[-1].best is None and result.best is None


def test_a_single_objective_snapshot_reports_the_best_as_the_run_result_does(tmp_path: Path):
    spy = Spy()
    result = auxein.run(**settings(tmp_path / "s", callbacks=[spy]))
    final = spy.calls[-1]
    assert final.best is not None and result.best is not None
    assert final.best.candidate.id == result.best.candidate.id and final.pareto_front[0].candidate.id == result.best.candidate.id
    values = [p.best.objectives["value"] for p in spy.told if p.best is not None]
    assert values == sorted(values, reverse=True)  # the best only improves
    assert final.evaluations_per_second is not None and final.evaluations_per_second > 0
    assert final.elapsed > 0 and final.eta == 0.0 and final.run_dir == tmp_path / "s"
    assert tuple(o.name for o in final.objectives) == ("value",)


def test_a_snapshot_is_immutable():
    spy = Spy()
    auxein.run(**settings(callbacks=[spy]))
    progress = spy.calls[0]
    with pytest.raises(FrozenInstanceError):
        progress.generation = 99  # type: ignore[misc]
    with pytest.raises(TypeError):
        progress.status_counts["ok"] = 0  # type: ignore[index]
    assert progress.batch is not None
    with pytest.raises(TypeError):
        progress.batch.objectives["value"] = None  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        progress.batch.ok = 0  # type: ignore[misc]


def test_it_holds_references_not_copies_and_a_run_without_an_evaluation_budget_has_no_remaining():
    spy = Spy()
    result = auxein.run(**settings(budget=auxein.Budget(wall_time=0.3), strategy=auxein.RandomSearch(), callbacks=[spy]))
    assert spy.calls[0].remaining is None and spy.calls[0].eta is None
    assert spy.calls[-1].best is result.best  # the run's own Evaluation object
    assert result.stop_reason == "budget:wall_time" == spy.calls[-1].stop_reason


# --- callbacks never change what the run computes ---


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_the_event_log_is_identical_with_no_callbacks_a_noop_and_print_progress(tmp_path: Path, delivery: str):
    from auxein.callbacks import print_progress

    def noop(progress: Progress[Any]) -> None:
        return None

    outputs: dict[str, Any] = {"none": [], "noop": [noop], "print": [print_progress(file=io.StringIO())]}
    results = {}
    for name, callbacks in outputs.items():
        results[name] = auxein.run(**settings(tmp_path / name, evaluations=75, delivery=delivery, callbacks=callbacks))
    for name in ("noop", "print"):
        assert comparable(tmp_path / name) == comparable(tmp_path / "none")
        assert results[name].trace == results["none"].trace and results[name].stop_reason == results["none"].stop_reason


# --- StopRun ---


def solved(progress: Progress[Any]) -> StopRun | None:
    return StopRun("solved") if progress.best is not None and progress.best.objectives["value"] < 1.0 else None


def test_stoprun_ends_the_run_with_a_callback_stop_reason_everywhere_it_is_recorded(tmp_path: Path):
    spy = Spy()
    result = auxein.run(**settings(tmp_path / "stop", evaluations=2000, callbacks=[solved, spy]))
    assert result.stop_reason == "callback:solved" and result.evaluations_used < 2000
    assert peek(tmp_path / "stop").metadata["stop_reason"] == "callback:solved"
    assert spy.calls[-1].final and spy.calls[-1].stop_reason == "callback:solved"
    assert spy.told[-1].best is not None and spy.told[-1].best.objectives["value"] < 1.0
    assert all(
        p.best is None or p.best.objectives["value"] >= 1.0 for p in spy.told[:-1]
    )  # it stopped at the first generation that qualified
    assert spy.calls[-1].evaluations_used == result.evaluations_used


def test_a_budget_that_is_also_exhausted_keeps_the_budget_as_the_reason(tmp_path: Path):
    spy = Spy()
    result = auxein.run(**settings(tmp_path / "both", evaluations=30, callbacks=[lambda p: StopRun("always"), spy]))
    assert result.stop_reason == "callback:always" and result.evaluations_used == 10  # the first generation asked to stop
    other = auxein.run(**settings(tmp_path / "same", evaluations=10, callbacks=[lambda p: StopRun("always")]))
    assert other.stop_reason == "budget:evaluations"


def test_a_stopped_run_resumes_and_extends_into_the_run_that_was_never_stopped(tmp_path: Path):
    reference = auxein.run(**settings(tmp_path / "ref", evaluations=300))
    for name, callbacks in (("without", []), ("with", [solved])):
        auxein.run(**settings(tmp_path / name, evaluations=300, callbacks=[solved]))
        assert peek(tmp_path / name).metadata["stop_reason"] == "callback:solved"
        resumed = auxein.resume(**settings(tmp_path / name, evaluations=300, callbacks=callbacks))  # the same budget
        if name == "without":
            assert comparable(tmp_path / name) == comparable(tmp_path / "ref")
            assert resumed.stop_reason == "budget:evaluations" and resumed.trace == reference.trace
        else:
            assert resumed.stop_reason == "callback:solved"  # the callbacks decided again
    extended = auxein.run(**settings(tmp_path / "ext", evaluations=300, callbacks=[solved]))
    longer = auxein.resume(**settings(tmp_path / "ext", evaluations=450))  # a larger budget: the run that never stopped
    auxein.run(**settings(tmp_path / "ref450", evaluations=450))
    assert extended.stop_reason == "callback:solved"
    assert longer.stop_reason == "budget:evaluations" and longer.evaluations_used == 450
    assert comparable(tmp_path / "ext") == comparable(tmp_path / "ref450")


def test_a_stop_in_steady_state_lets_the_window_finish_and_resumes_like_a_budget_stop(tmp_path: Path):
    config = {"strategy": auxein.RandomSearch(), "delivery": "steady_state", "batch_size": 4}
    easy = stop_when(lambda p: p.best is not None and p.best.objectives["value"] < 6.0, "solved")  # random search needs an easy target
    result = auxein.run(**settings(tmp_path / "s", evaluations=200, callbacks=[easy], **config))
    assert result.stop_reason == "callback:solved" and result.evaluations_used < 200
    resumed = auxein.resume(**settings(tmp_path / "s", evaluations=200, **{**config, "strategy": auxein.RandomSearch()}))
    auxein.run(**settings(tmp_path / "r", evaluations=200, **{**config, "strategy": auxein.RandomSearch()}))
    assert resumed.evaluations_used == 200 and comparable(tmp_path / "s") == comparable(tmp_path / "r")


def test_a_stoprun_needs_a_reason_and_a_callback_must_return_none_or_a_stoprun():
    with pytest.raises(ValueError, match="non-empty reason"):
        StopRun("")
    with pytest.raises(TypeError, match="None or a StopRun"):
        auxein.run(**settings(callbacks=[lambda p: "stop"]))
    with pytest.raises(TypeError, match="callables"):
        auxein.run(**settings(callbacks=[42]))  # type: ignore[list-item]


# --- replay ---


def test_callbacks_fire_while_a_resumed_run_replays_and_say_so(tmp_path: Path):
    auxein.run(**settings(tmp_path / "r", evaluations=100, keep_checkpoints=0))  # no checkpoints: a resume replays from the start
    spy = Spy()
    out = io.StringIO()
    from auxein.callbacks import print_progress

    auxein.resume(**settings(tmp_path / "r", evaluations=160, keep_checkpoints=0, callbacks=[spy, print_progress(file=out)]))
    flags = [(p.generation, p.replaying) for p in spy.told]
    assert flags == [(g, g <= 10) for g in range(1, 17)]  # ten replayed generations, then six live ones, counted from the start
    assert not spy.calls[-1].replaying and spy.calls[-1].final and spy.calls[-1].evaluations_used == 160
    text = out.getvalue().splitlines()
    assert sum("replaying" in line for line in text) == 1 and text[0].startswith("replaying")
    assert len(text) == 1 + 6 + 1  # the replay line, six live generations, the final line


@pytest.mark.filterwarnings("ignore::auxein.driver.errors.ResumeWarning")
def test_a_stop_asked_for_while_replaying_is_ignored(tmp_path: Path):
    auxein.run(**settings(tmp_path / "r", evaluations=100, keep_checkpoints=0))
    spy = Spy()
    result = auxein.resume(**settings(tmp_path / "r", evaluations=100, keep_checkpoints=0, callbacks=[lambda p: StopRun("now"), spy]))
    assert result.stop_reason == "budget:evaluations"  # complete with its budget: it returns its result and nothing is replayed
    longer = auxein.resume(**settings(tmp_path / "r", evaluations=130, keep_checkpoints=0, callbacks=[lambda p: StopRun("now"), spy]))
    assert (
        longer.evaluations_used == 110 and longer.stop_reason == "callback:now"
    )  # ignored for ten replayed tells, honoured at the first live one
    assert [p.replaying for p in spy.told] == [True] * 10 + [False]


def test_a_resumed_run_continues_the_generation_count_from_its_checkpoint(tmp_path: Path):
    auxein.run(**settings(tmp_path / "c", evaluations=100))
    spy = Spy()
    auxein.resume(**settings(tmp_path / "c", evaluations=130, callbacks=[spy]))
    assert [(p.generation, p.replaying) for p in spy.told] == [(11, False), (12, False), (13, False)]


# --- errors ---


def test_an_error_in_a_callback_fails_the_run_and_the_run_resumes_once_it_is_fixed(tmp_path: Path):
    def broken(progress: Progress[Any]) -> None:
        if progress.generation == 3:
            raise RuntimeError("the video renderer crashed")

    with pytest.raises(RuntimeError, match="renderer crashed"):
        auxein.run(**settings(tmp_path / "e", evaluations=100, callbacks=[broken]))
    assert peek(tmp_path / "e").metadata["status"] == "failed"
    assert len(peek(tmp_path / "e").evaluations()) == 30  # everything up to the last tell is recorded
    resumed = auxein.resume(**settings(tmp_path / "e", evaluations=100))
    reference = auxein.run(**settings(tmp_path / "ref", evaluations=100))
    assert comparable(tmp_path / "e") == comparable(tmp_path / "ref") and resumed.trace == reference.trace


def test_an_error_in_the_final_event_fails_the_run_too(tmp_path: Path):
    def at_the_end(progress: Progress[Any]) -> None:
        if progress.final:
            raise RuntimeError("could not write the summary")

    with pytest.raises(RuntimeError, match="summary"):
        auxein.run(**settings(tmp_path / "f", evaluations=30, callbacks=[at_the_end]))
    assert peek(tmp_path / "f").metadata["status"] == "failed"
    assert auxein.resume(**settings(tmp_path / "f", evaluations=30)).evaluations_used == 30


def test_a_run_that_raises_gets_no_final_event(tmp_path: Path):
    spy = Spy()

    def explode(genome: np.ndarray) -> float:
        raise ValueError("boom")

    with pytest.raises(Exception, match="boom|failed"):
        auxein.run(**settings(evaluator=auxein.FunctionEvaluator(explode), failure_policy="fail_fast", callbacks=[spy]))
    assert not any(p.final for p in spy.calls)


# --- async callbacks ---


def test_async_callbacks_are_awaited_in_order():
    order: list[tuple[str, int]] = []

    async def slow(progress: Progress[Any]) -> None:
        await asyncio.sleep(0.01)
        order.append(("slow", progress.generation))

    async def quick(progress: Progress[Any]) -> StopRun | None:
        await asyncio.sleep(0)
        order.append(("quick", progress.generation))
        return StopRun("enough") if progress.generation == 3 and not progress.final else None

    def plain(progress: Progress[Any]) -> None:
        order.append(("plain", progress.generation))

    result = auxein.run(**settings(evaluations=100, callbacks=[slow, quick, plain]))
    assert order == [(name, g) for g in (1, 2, 3, 3) for name in ("slow", "quick", "plain")]
    assert result.stop_reason == "callback:enough"


def test_async_callbacks_work_inside_arun():
    seen: list[int] = []

    async def record(progress: Progress[Any]) -> None:
        await asyncio.sleep(0)
        seen.append(progress.generation)

    asyncio.run(auxein.arun(**settings(evaluations=30, callbacks=[record])))
    assert seen == [1, 2, 3, 3]


# --- no callbacks, no cost ---


def test_without_callbacks_no_snapshot_and_no_batch_statistics_are_ever_built(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from auxein.driver import driver as driver_module

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a snapshot was built without callbacks")

    monkeypatch.setattr(driver_module, "Progress", forbidden)
    monkeypatch.setattr(driver_module, "batch_stats", forbidden)
    for delivery in ("generation", "steady_state"):
        result = auxein.run(**settings(tmp_path / delivery, evaluations=40, delivery=delivery))
        assert result.evaluations_used == 40
    resumed = auxein.resume(**settings(tmp_path / "generation", evaluations=60))
    assert resumed.evaluations_used == 60


def test_math_is_not_needed_for_the_import_check():
    assert math.isfinite(1.0)
