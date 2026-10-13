"""The helpers of `auxein.callbacks`: throttles, the progress line, stopping rules and the history."""

import csv
import io
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import auxein
from auxein.callbacks import ProgressHistory, every, every_evaluations, print_progress, stop_when
from auxein.core import Objective, Result
from auxein.driver import Progress, StopRun

pytestmark = pytest.mark.filterwarnings("ignore::auxein.RecordingDisabledWarning")


def sphere(genome: np.ndarray) -> float:
    return float((genome * genome).sum())


def go(callbacks: list[Any], evaluations: int = 100, **over: Any) -> auxein.RunResult[Any]:
    settings: dict[str, Any] = {
        "strategy": auxein.GeneticAlgorithm(population_size=10, offspring_size=10),
        "evaluator": auxein.FunctionEvaluator(sphere),
        "space": auxein.Box(-5.0, 5.0, dim=3),
        "budget": auxein.Budget(evaluations=evaluations),
        "seed": 4,
        "batch_size": 10,
        "callbacks": callbacks,
    }
    settings.update(over)
    return auxein.run(**settings)


def test_every_calls_on_every_nth_generation_and_always_at_the_end():
    seen: list[tuple[int, bool]] = []
    go([every(3, lambda p: seen.append((p.generation, p.final)))], evaluations=100)
    assert seen == [(3, False), (6, False), (9, False), (10, True)]


def test_every_forwards_what_the_callback_returns_so_it_can_stop_the_run():
    result = go([every(2, lambda p: StopRun("enough"))], evaluations=1000)
    assert result.stop_reason == "callback:enough" and result.evaluations_used == 20


def test_every_forwards_async_callbacks_to_be_awaited():
    seen: list[int] = []

    async def record(progress: Progress[Any]) -> None:
        seen.append(progress.generation)

    go([every(5, record)], evaluations=100)
    assert seen == [5, 10, 10]


def test_every_evaluations_throttles_by_evaluations_and_remembers_between_calls():
    seen: list[tuple[int, bool]] = []
    go([every_evaluations(25, lambda p: seen.append((p.evaluations_used, p.final)))], evaluations=100)
    assert seen == [(30, False), (60, False), (90, False), (100, True)]  # at least 25 more since the last call, at tell boundaries


def test_the_throttles_skip_replayed_generations_unless_asked_to(tmp_path: Path):
    run_dir = tmp_path / "r"
    go([], evaluations=50, run_dir=run_dir, keep_checkpoints=0)
    plain, replayed = [], []
    auxein.resume(
        strategy=auxein.GeneticAlgorithm(population_size=10, offspring_size=10),
        evaluator=auxein.FunctionEvaluator(sphere),
        space=auxein.Box(-5.0, 5.0, dim=3),
        budget=auxein.Budget(evaluations=80),
        seed=4,
        batch_size=10,
        run_dir=run_dir,
        keep_checkpoints=0,
        callbacks=[
            every(1, lambda p: plain.append(p.replaying)),
            every(1, lambda p: replayed.append(p.replaying), during_replay=True),
        ],
    )
    assert plain == [False, False, False, False]  # three live generations and the final event
    assert replayed == [True] * 5 + [False] * 4


def test_helpers_validate_their_arguments():
    for bad in (lambda: every(0, print), lambda: every_evaluations(0, print), lambda: print_progress(every=0), lambda: stop_when(bool, "")):
        with pytest.raises(ValueError):
            bad()


def test_stop_when_stops_on_the_predicate_and_not_at_the_final_event():
    calls: list[bool] = []

    def below(progress: Progress[Any]) -> bool:
        calls.append(progress.final)
        return progress.best is not None and progress.best.objectives["value"] < 0.5

    result = go([stop_when(below, "tight")], evaluations=3000)
    assert result.stop_reason == "callback:tight" and result.best is not None and result.best.objectives["value"] < 0.5
    assert calls and not any(calls)
    assert go([stop_when(lambda p: False, "never")], evaluations=30).stop_reason == "budget:evaluations"


def test_stop_when_is_reproducible_for_a_fitness_predicate():
    first = go([stop_when(lambda p: p.best is not None and p.best.objectives["value"] < 0.5, "tight")], evaluations=3000)
    second = go([stop_when(lambda p: p.best is not None and p.best.objectives["value"] < 0.5, "tight")], evaluations=3000)
    assert first.evaluations_used == second.evaluations_used and first.trace == second.trace


# --- print_progress ---

LINE = re.compile(
    r"^gen +\d+ \| evals +[\d,]+/[\d,]+ +\( *\d+%\) \| best +\S+ \| mean +\S+ \| +\S+ ev/s \| elapsed +\d+:\d\d:\d\d \| ETA +[\d:-]+$"
)


def test_print_progress_prints_one_aligned_line_per_call_and_a_final_line():
    out = io.StringIO()
    go([print_progress(file=out)], evaluations=60)
    lines = out.getvalue().splitlines()
    assert len(lines) == 7  # six generations and the final line
    assert all(LINE.match(line) for line in lines[:-1]), lines
    assert len({len(line) for line in lines[:-1]}) <= 2  # aligned (a number may take one more column)
    assert lines[-1].endswith("| done: budget:evaluations") and "100%" in lines[-1]
    assert lines[0].startswith("gen      1 | evals         10/60        ( 17%)")


def test_print_progress_every_n_and_failures_and_the_stdout_default(capsys: pytest.CaptureFixture[str]):
    def sometimes(genome: np.ndarray) -> float:
        if genome[0] > 4.0:
            raise ValueError("no")
        return sphere(genome)

    with pytest.warns(auxein.driver.EvaluationFailureWarning):
        go([print_progress(every=4)], evaluations=100, evaluator=auxein.FunctionEvaluator(sometimes))
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and lines[0].startswith("gen      4") and lines[1].startswith("gen      8")  # 4, 8 and the final line
    assert "failed" in lines[0] and re.search(r"failed +\d+", lines[0])


def test_print_progress_shows_the_front_for_several_objectives_and_works_without_an_evaluation_budget():
    out = io.StringIO()

    def two(genome: np.ndarray) -> Result:
        return Result({"a": float(genome[0] ** 2), "b": float((genome[0] - 1) ** 2)})

    auxein.run(
        strategy=auxein.NSGA2(population_size=10, offspring_size=10),
        evaluator=auxein.FunctionEvaluator(two),
        space=auxein.Box(-2.0, 2.0, dim=2),
        objectives=[Objective("a"), Objective("b")],
        budget=auxein.Budget(wall_time=0.2),
        seed=1,
        batch_size=10,
        callbacks=[print_progress(every=1000, file=out)],
    )
    (line,) = out.getvalue().splitlines()  # only the final line: the budget has no evaluations, so no percentage and no ETA
    assert "front" in line and "%" not in line and "ETA     -:--" in line and "done: budget:wall_time" in line


# --- ProgressHistory ---


def test_progress_history_keeps_one_row_per_call_with_the_batch_statistics():
    history = ProgressHistory()
    go([history], evaluations=50)
    rows = history.rows
    assert len(rows) == len(history) == 6 and [r["generation"] for r in rows] == [1, 2, 3, 4, 5, 5]
    assert [r["final"] for r in rows] == [False] * 5 + [True] and [r["evaluations"] for r in rows] == [10, 20, 30, 40, 50, 50]
    assert all(r["batch_ok"] == 10 and r["batch_failed"] == 0 and r["batch_feasible"] == 1.0 for r in rows)
    best = [r["best"] for r in rows]
    assert best == sorted(best, reverse=True) and rows[0]["value_best"] <= rows[0]["value_median"] <= rows[0]["value_worst"]  # type: ignore[operator]
    assert rows[0]["value_mean"] >= rows[0]["value_best"]  # type: ignore[operator]
    assert rows[-1]["elapsed"] >= rows[0]["elapsed"] and rows[0]["failed"] == 0 and rows[0]["replaying"] is False
    rows[0]["generation"] = 99  # a copy: the history is not changed through it
    assert history.rows[0]["generation"] == 1


def test_progress_history_writes_csv_text_and_a_file(tmp_path: Path):
    history = ProgressHistory()
    assert history.to_csv() == ""
    go([history], evaluations=30)
    text = history.to_csv(tmp_path / "history.csv")
    assert (tmp_path / "history.csv").read_text() == text
    table = list(csv.DictReader(io.StringIO(text)))
    assert len(table) == 4 and table[0]["generation"] == "1" and table[-1]["final"] == "True"
    assert {"evaluations", "elapsed", "best", "value_best", "value_mean", "value_median", "value_worst", "batch_failed"} <= set(table[0])
    assert float(table[1]["best"]) <= float(table[0]["best"]) and math.isfinite(float(table[0]["value_mean"]))


def test_progress_history_covers_replayed_generations_of_a_resumed_run(tmp_path: Path):
    run_dir = tmp_path / "h"
    go([], evaluations=50, run_dir=run_dir, keep_checkpoints=0)
    history = ProgressHistory()
    auxein.resume(
        strategy=auxein.GeneticAlgorithm(population_size=10, offspring_size=10),
        evaluator=auxein.FunctionEvaluator(sphere),
        space=auxein.Box(-5.0, 5.0, dim=3),
        budget=auxein.Budget(evaluations=70),
        seed=4,
        batch_size=10,
        run_dir=run_dir,
        keep_checkpoints=0,
        callbacks=[history],
    )
    assert [r["generation"] for r in history.rows] == [1, 2, 3, 4, 5, 6, 7, 7]
    assert [r["replaying"] for r in history.rows] == [True] * 5 + [False] * 3
