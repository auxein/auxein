"""External proposal operators (an LLM-driven mutation): recorded, replayed, and never paid for twice (design doc §3.5)."""

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import pytest

import auxein
from auxein.core import Objective, Proposal
from auxein.driver import ConfigurationMismatchError, ReplayMismatchError
from auxein.recording import open_run
from auxein.strategies.structured import (
    ExternalMutation,
    ExternalRecombination,
    OperatorError,
    OperatorNotRecordedWarning,
    SequenceMutation,
    call_operator,
    operator_key,
)
from tests.driver.resume_test import comparable
from tests.support import sequences as sq
from tests.support.fixtures import integration_backend
from tests.support.reading import peek
from tests.support.resumable import backend_config

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.usefixtures("use_corner_backend")


def strategy(llm: sq.FakeLLM, **options: Any) -> auxein.StructuredGeneticAlgorithm[Any]:
    mixed = [(SequenceMutation(), 1.0), (ExternalMutation(llm), 1.0)]
    return auxein.StructuredGeneticAlgorithm(population_size=12, offspring_size=12, mutation=mixed, **options)


def arguments(strategy_: Any, run_dir: Path | None, **over: Any) -> dict[str, Any]:
    settings_: dict[str, Any] = {
        "strategy": strategy_,
        "evaluator": auxein.FunctionEvaluator(sq.distance),
        "space": sq.SPACE,
        "objectives": [Objective("distance")],
        "constraints": ["too_long"],
        "seed": 3,
        "batch_size": 12,
        "run_dir": run_dir,
        "backend": integration_backend(),
    }
    settings_.update(over)
    return settings_


def calls_recorded(run_dir: Path) -> int:
    with open_run(run_dir) as run:
        return len(run.operator_calls())


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_the_calls_are_recorded_with_their_output_cost_time_and_metadata(tmp_path: Path, delivery: str):
    llm = sq.FakeLLM()
    result = auxein.run(budget=auxein.Budget(evaluations=100), delivery=delivery, **arguments(strategy(llm), tmp_path / "r"))
    assert result.evaluations_used == 100 and llm.calls > 10
    with open_run(tmp_path / "r") as run:
        recorded = run.operator_calls()
        children = {e.candidate_id: e for e in run.evaluations()}
    assert len(recorded) >= llm.calls - 12 and {c.operator for c in recorded} == {"fake-llm"}  # some calls made dropped candidates
    first = recorded[0]
    assert first.cost["tokens"] > 12 and first.metadata["model"] == "fake" and first.wall_time >= 0.0
    assert (
        isinstance(first.output, list) and all(x in sq.VOCABULARY for x in first.output) and len({c.key for c in recorded}) == len(recorded)
    )
    assert any(e.origin.endswith("external:fake-llm") for e in children.values())  # and the origin names the operator


def test_a_resumed_or_extended_run_regenerates_the_same_candidates_and_does_not_call_the_operator_again(tmp_path: Path):
    first = sq.FakeLLM()
    auxein.run(budget=auxein.Budget(evaluations=96), **arguments(strategy(first), tmp_path / "r"))
    recorded_before = calls_recorded(tmp_path / "r")
    snapshot = comparable(tmp_path / "r")

    again = sq.FakeLLM()
    with pytest.warns(auxein.driver.ResumeWarning):
        auxein.resume(budget=auxein.Budget(evaluations=96), **arguments(strategy(again), tmp_path / "r"))  # nothing to do
    assert again.calls == 0

    extended = sq.FakeLLM()
    auxein.resume(budget=auxein.Budget(evaluations=240), **arguments(strategy(extended), tmp_path / "r"))
    assert calls_recorded(tmp_path / "r") == recorded_before + extended.calls  # only the new candidates were paid for
    assert extended.calls > 0
    after = comparable(tmp_path / "r")
    for table in ("candidates", "evaluations"):  # the first 96 candidates are exactly what was recorded: replay regenerated them
        assert after[table][: len(snapshot[table]) - 0][:96] == snapshot[table][:96]


def test_a_replay_without_checkpoints_still_never_calls_the_operator_for_recorded_candidates(tmp_path: Path):
    llm = sq.FakeLLM()
    auxein.run(budget=auxein.Budget(evaluations=96), keep_checkpoints=0, **arguments(strategy(llm), tmp_path / "r"))
    before = calls_recorded(tmp_path / "r")
    replayer = sq.FakeLLM()
    auxein.resume(budget=auxein.Budget(evaluations=144), keep_checkpoints=0, **arguments(strategy(replayer), tmp_path / "r"))
    assert calls_recorded(tmp_path / "r") == before + replayer.calls and replayer.calls < before  # 96 candidates replayed for free


def test_calls_made_for_dropped_candidates_are_reused_when_the_run_is_extended(tmp_path: Path):
    """The last batch is cut at the budget: its dropped children were asked for, so their calls were made and paid for."""
    llm = sq.FakeLLM()
    auxein.run(budget=auxein.Budget(evaluations=100), **arguments(strategy(llm), tmp_path / "r"))  # 100 is not a multiple of 12
    paid = calls_recorded(tmp_path / "r")
    assert paid == llm.calls
    extended = sq.FakeLLM()
    auxein.resume(budget=auxein.Budget(evaluations=120), **arguments(strategy(extended), tmp_path / "r"))
    reference_calls = calls_recorded(tmp_path / "r")
    assert reference_calls == paid + extended.calls  # no key is ever called twice: the dropped children's calls were served


def test_a_changed_operator_name_is_refused_by_the_configuration_check(tmp_path: Path):
    auxein.run(budget=auxein.Budget(evaluations=60), **arguments(strategy(sq.FakeLLM()), tmp_path / "r"))
    with pytest.raises(ConfigurationMismatchError, match="strategy: recorded"):
        auxein.resume(budget=auxein.Budget(evaluations=120), **arguments(strategy(sq.FakeLLM("other-llm")), tmp_path / "r"))


def test_changed_inputs_are_detected_as_a_divergence_and_the_operator_is_not_called(tmp_path: Path):
    """Same name, same description, but the operator is now given a different parent: no recorded call matches."""
    auxein.run(budget=auxein.Budget(evaluations=96), keep_checkpoints=0, **arguments(strategy(sq.FakeLLM()), tmp_path / "r"))
    drifted = sq.FakeLLM()

    class Drifting(ExternalMutation):  # type: ignore[type-arg]
        def mutate(self, genome, rng, ctx):  # the same operator, called on a changed input
            return super().mutate(genome[:-1] + (genome[0],), rng, ctx) if len(genome) > 2 else super().mutate(genome, rng, ctx)

    Drifting.__name__ = Drifting.__qualname__ = "ExternalMutation"
    changed = auxein.StructuredGeneticAlgorithm(
        population_size=12, offspring_size=12, mutation=[(SequenceMutation(), 1.0), (Drifting(drifted), 1.0)]
    )
    with pytest.raises(ReplayMismatchError, match="fake-llm.*recorded.*diverged"):
        auxein.resume(budget=auxein.Budget(evaluations=144), keep_checkpoints=0, **arguments(changed, tmp_path / "r"))
    assert drifted.calls == 0  # it was not called for a candidate that is recorded


@pytest.mark.filterwarnings("ignore::auxein.RecordingDisabledWarning")
def test_without_recording_the_calls_are_live_and_the_run_says_it_cannot_be_reproduced():
    def run_once() -> list[tuple[object, ...]]:
        seen: list[tuple[object, ...]] = []

        def distance(genome: tuple[object, ...]) -> auxein.Result:
            seen.append(genome)
            return sq.distance(genome)

        with pytest.warns(OperatorNotRecordedWarning, match="cannot be reproduced or resumed"):
            auxein.run(
                budget=auxein.Budget(evaluations=60),
                **arguments(strategy(sq.FakeLLM()), None, evaluator=auxein.FunctionEvaluator(distance)),
            )
        return seen

    # the same seed, but a model that answers differently each time: the two runs are not the same run
    assert run_once() != run_once()


def test_a_recorded_run_does_not_warn(tmp_path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", OperatorNotRecordedWarning)
        auxein.run(budget=auxein.Budget(evaluations=40), **arguments(strategy(sq.FakeLLM()), tmp_path / "r"))


def test_an_exception_in_an_operator_fails_the_run_naming_the_operator(tmp_path: Path):
    class Broken:
        name = "flaky-model"

        def propose(self, parents, rng):
            raise ConnectionError("the model is down")

    broken = auxein.StructuredGeneticAlgorithm(population_size=12, offspring_size=12, mutation=[(ExternalMutation(Broken()), 1.0)])
    with pytest.raises(OperatorError, match="operator 'flaky-model' failed: ConnectionError: the model is down") as raised:
        auxein.run(budget=auxein.Budget(evaluations=60), **arguments(broken, tmp_path / "r"))
    assert isinstance(raised.value.__cause__, ConnectionError)
    assert peek(tmp_path / "r").metadata["status"] == "failed"


def test_an_operator_that_returns_the_wrong_thing_or_a_genome_outside_the_space_fails_the_run():
    class Wrong:
        name = "wrong"

        def propose(self, parents, rng):
            return ("a", "b")

    class Outside:
        name = "outside"

        def propose(self, parents, rng):
            return Proposal(("a",) * 99)

    for operator, message in ((Wrong(), "must return a Proposal"), (Outside(), "not in the search space")):
        ga = auxein.StructuredGeneticAlgorithm(population_size=12, offspring_size=12, mutation=[(ExternalMutation(operator), 1.0)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(OperatorError, match=message):
                auxein.run(budget=auxein.Budget(evaluations=40), **arguments(ga, None))


def test_an_external_recombination_gets_both_parents():
    seen: list[int] = []

    class Splice:
        name = "splice"

        def propose(self, parents, rng):
            seen.append(len(parents))
            return Proposal(tuple(parents[0][:2]) + tuple(parents[1][:2]))

    ga = auxein.StructuredGeneticAlgorithm(population_size=12, offspring_size=12, recombination=ExternalRecombination(Splice()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auxein.run(budget=auxein.Budget(evaluations=48), **arguments(ga, None))
    assert seen and set(seen) == {2}


def test_the_key_of_a_call_depends_on_the_name_the_inputs_and_the_draw():
    base = operator_key("op", [["a", "b"]], 7)
    assert base == operator_key("op", [["a", "b"]], 7)
    assert (
        len({base, operator_key("other", [["a", "b"]], 7), operator_key("op", [["a", "c"]], 7), operator_key("op", [["a", "b"]], 8)}) == 4
    )


def test_the_strategy_stream_advances_by_one_draw_per_call_whether_it_is_replayed_or_live(tmp_path: Path):
    """That is what keeps a resumed run in step with the recorded one."""
    from auxein.random import RunSeed
    from auxein.recording import SQLiteRecorder
    from auxein.strategies.structured import VariationContext

    space = sq.SPACE
    recorder = SQLiteRecorder(tmp_path / "r")
    recorder.on_start({"name": "t"})
    llm = sq.FakeLLM()

    def one(log: Any, seed: int) -> list[float]:
        stream = RunSeed(seed).stream("strategy")
        ctx = VariationContext(space, space.codec, log, 1000)
        call_operator(llm, [("a", "b", "c")], stream, ctx)
        return [float(x) for x in stream.uniform((3,))]

    live = one(recorder, 5)
    before = llm.calls
    replayed = one(recorder, 5)  # same seed, same input, same draw: found in the log
    assert replayed == live and llm.calls == before
    recorder.on_end("completed", "x", {})


# --- killing a run that uses an operator ---


def recorded(run_dir: Path) -> int:
    try:
        db = sqlite3.connect(f"file:{run_dir / 'events.sqlite'}?mode=ro", uri=True)
        try:
            return int(db.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0])
        finally:
            db.close()
    except sqlite3.Error:
        return -1


def lines(path: Path) -> int:
    return len(path.read_text().split()) if path.exists() else 0


@pytest.mark.parametrize("delivery", ["generation", "steady_state"])
def test_a_killed_run_resumes_without_paying_twice_and_with_the_recorded_candidates(tmp_path: Path, delivery: str):
    total = 180
    run_dir = tmp_path / "run"
    full = {
        "strategy": "structured-llm",
        "evaluator": "sequence",
        "run_dir": str(run_dir),
        "evaluations": total,
        "delivery": delivery,
        "backend": backend_config(integration_backend()),
    }
    first_log, second_log = tmp_path / "first.log", tmp_path / "second.log"
    victim = subprocess.Popen(
        [sys.executable, "-m", "tests.support.resume_cli", json.dumps(full)],
        cwd=ROOT,
        env={**os.environ, "AUXEIN_LLM_LOG": str(first_log)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 120
    try:
        while victim.poll() is None and recorded(run_dir) < 70 and time.monotonic() < deadline:
            time.sleep(0.003)
        victim.send_signal(signal.SIGKILL)
        victim.wait(timeout=30)
    finally:
        if victim.poll() is None:
            victim.kill()
            victim.wait()
    stored = calls_recorded(run_dir)
    candidates_before = [row for row in comparable(run_dir)["candidates"]]  # type: ignore[union-attr]
    survivor = subprocess.run(
        [sys.executable, "-m", "tests.support.resume_cli", json.dumps({**full, "resume": True})],
        cwd=ROOT,
        env={**os.environ, "AUXEIN_LLM_LOG": str(second_log)},
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert survivor.returncode == 0, survivor.stderr[-3000:]
    assert lines(second_log) == calls_recorded(run_dir) - stored  # the resumed session paid only for keys nobody had paid for
    assert lines(first_log) in (stored, stored + 1)  # (a call killed after it ran and before it was stored is paid again)
    after = comparable(run_dir)["candidates"]
    assert after[: len(candidates_before)] == candidates_before  # the recorded candidates are untouched
    assert peek(run_dir).metadata["status"] == "completed" and len(after) == total  # type: ignore[arg-type]
