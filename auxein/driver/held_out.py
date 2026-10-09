"""Held-out evaluation (design doc §6.4): judging chosen candidates on scenarios the run never saw.

A run evolves on a selection set. To detect overfitting and reward hacking, its results are judged afterwards on a held-out set
by the same machinery, but *outside* the run: no strategy, no budget, and nothing added to the run's event log.
"""

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from auxein.backend import Backend, Precision
from auxein.core import (
    ArrayBatch,
    Batch,
    Candidate,
    EvalContext,
    Evaluation,
    Evaluator,
    FailurePolicy,
    ListBatch,
    Objective,
    ProblemSpec,
    Status,
)
from auxein.driver.result import ResultTracker, RunResult
from auxein.environments import ScenarioSet
from auxein.evaluators.episode import describe_component
from auxein.execution import ExecutorName, make_executor, resolve_executor
from auxein.random import RandomStream, RunSeed
from auxein.recording.replay import iter_records

REPORT_FILE = "held_out.json"


@dataclass(frozen=True)
class HeldOutScenario:
    """One candidate in one held-out scenario."""

    scenario_id: str
    status: Status
    measurements: dict[str, float]
    error: str | None


@dataclass(frozen=True)
class HeldOutCandidate:
    """A candidate judged on the held-out set: per-scenario measurements, and the aggregated result."""

    candidate_id: int
    status: Status
    objectives: dict[str, float]
    constraints: dict[str, float]
    descriptors: dict[str, float]
    error: str | None
    scenarios: tuple[HeldOutScenario, ...]


@dataclass(frozen=True)
class HeldOutReport:
    """The result of `evaluate_held_out`: what the chosen candidates did on scenarios the run never saw."""

    scenarios_fingerprint: str
    evaluator: str
    candidates: tuple[HeldOutCandidate, ...]
    path: Path | None = None
    """Where `held_out.json` was written, if it was."""

    def to_json(self) -> dict[str, object]:
        return {
            "format": 1,
            "scenarios_fingerprint": self.scenarios_fingerprint,
            "evaluator": self.evaluator,
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "status": c.status.value,
                    "objectives": c.objectives,
                    "constraints": c.constraints,
                    "descriptors": c.descriptors,
                    "error": c.error,
                    "scenarios": [
                        {"scenario_id": s.scenario_id, "status": s.status.value, "measurements": s.measurements, "error": s.error}
                        for s in c.scenarios
                    ],
                }
                for c in self.candidates
            ],
        }


class _RecordedSpace:
    """A stand-in for the search space of a recorded run: held-out evaluation never samples or checks genomes."""

    def sample_genomes(self, n: int, rng: RandomStream, backend: Backend) -> Sequence[Any]:
        raise NotImplementedError("a held-out evaluation does not sample genomes")

    def contains(self, genome: Any) -> bool:
        return True


def _problem_of(metadata: Mapping[str, Any]) -> ProblemSpec[Any]:
    problem = metadata["problem"]
    objectives = tuple(Objective(o["name"], o["direction"]) for o in problem["objectives"])
    return ProblemSpec(_RecordedSpace(), objectives, tuple(problem["constraints"]), tuple(problem["descriptors"]))


def _backend_of(metadata: Mapping[str, Any]) -> Backend:
    spec = metadata["backend"]
    return Backend(spec["name"], spec["device"], cast("Precision", metadata["precision"]))


def evaluate_held_out(
    run: RunResult[Any] | str | Path,
    evaluator: Evaluator[Any],
    scenarios: ScenarioSet,
    *,
    candidates: Literal["best", "pareto"] | Sequence[int] = "best",
    problem: ProblemSpec[Any] | None = None,
    seed: int | None = None,
    backend: Backend | None = None,
    concurrency: int = 1,
    executor: ExecutorName = "auto",
    timeout: float | None = None,
    failure_policy: FailurePolicy = "infeasible",
    write: bool = True,
) -> HeldOutReport:
    """Evaluate candidates of a run on a held-out scenario set, outside the run.

    `run` is a run directory or a `RunResult`; `evaluator` is an `EpisodeEvaluator` built with the *held-out* scenarios
    (typically the selection evaluator's decoder, environment and aggregator with another `ScenarioSet`), and `scenarios` is
    that set, which is recorded in the report. `candidates` is `"best"` (the run's best; single objective), `"pareto"` (its
    front) or a list of candidate ids (those must be in a run directory). Nothing is added to the run's event log, and no
    strategy or budget is involved. The agents' random streams come from a seed derived from the run's, so they are
    independent of the ones used during evolution.

    The problem (objective directions, constraint and descriptor names), the seed and the backend are read from the run
    directory; for a `RunResult` with no `run_dir`, pass `problem=` (and optionally `seed=` and `backend=`). With `write=True`
    (the default) and a run directory, the report is also written to `held_out.json` there, replacing an earlier report.
    """
    return _run_sync(
        aevaluate_held_out(
            run,
            evaluator,
            scenarios,
            candidates=candidates,
            problem=problem,
            seed=seed,
            backend=backend,
            concurrency=concurrency,
            executor=executor,
            timeout=timeout,
            failure_policy=failure_policy,
            write=write,
        )
    )


async def aevaluate_held_out(
    run: RunResult[Any] | str | Path,
    evaluator: Evaluator[Any],
    scenarios: ScenarioSet,
    *,
    candidates: Literal["best", "pareto"] | Sequence[int] = "best",
    problem: ProblemSpec[Any] | None = None,
    seed: int | None = None,
    backend: Backend | None = None,
    concurrency: int = 1,
    executor: ExecutorName = "auto",
    timeout: float | None = None,
    failure_policy: FailurePolicy = "infeasible",
    write: bool = True,
) -> HeldOutReport:
    """The asynchronous variant of `evaluate_held_out`, for use inside a running event loop."""
    run_dir = run if isinstance(run, (str, Path)) else run.run_dir
    metadata: dict[str, Any] | None = None
    if run_dir is not None:
        run_dir = Path(run_dir)
        metadata = cast("dict[str, Any]", json.loads((run_dir / "metadata.json").read_text()))
    if problem is None:
        if metadata is None:
            raise ValueError("a RunResult without a run directory does not say what the problem was: pass problem=")
        problem = _problem_of(metadata)
    if backend is None:
        backend = _backend_of(metadata) if metadata is not None else Backend()
    base_seed = seed if seed is not None else (int(metadata["seed"]) if metadata is not None else 0)
    held_out_seed = int(RunSeed(base_seed).sequence("held-out").generate_state(1)[0])

    chosen = _choose(run, run_dir, candidates, problem, backend)
    batch = _batch_of(chosen, backend)
    seeds = RunSeed(held_out_seed)
    host = Backend("numpy", "cpu", backend.precision)
    pool = make_executor(resolve_executor(executor, concurrency, timeout), concurrency)
    context: EvalContext[Any] = EvalContext(
        problem,
        backend,
        lambda cid: seeds.stream("evaluation", cid, backend=host),
        lambda cid: seeds.stream("evaluation-batch", cid, backend=backend),
        lambda cid, scenario: seeds.stream("episode", cid, scenario, backend=host),
        lambda cid: seeds.stream("episode-batch", cid, backend=backend),
        executor=pool,
        concurrency=concurrency,
        timeout=timeout,
        failure_policy=failure_policy,
    )
    try:
        results = await evaluator.evaluate(batch, context)
    finally:
        pool.shutdown()

    episodes = results.episodes
    per_candidate: dict[int, list[HeldOutScenario]] = {}
    if episodes is not None:
        for row, candidate_id in enumerate(episodes.candidate_ids):
            rows: list[HeldOutScenario] = []
            for index, scenario_id in enumerate(episodes.scenario_ids):
                failure = episodes.failures.get((row, index))
                if failure is None:
                    values = dict(zip(episodes.names, episodes.values[row, index].tolist(), strict=True))
                    rows.append(HeldOutScenario(scenario_id, Status.OK, values, None))
                else:
                    rows.append(HeldOutScenario(scenario_id, failure[0], {}, failure[1]))
            per_candidate[int(candidate_id)] = rows
    report = HeldOutReport(
        scenarios.fingerprint,
        describe_component(evaluator),
        tuple(
            HeldOutCandidate(
                int(e.candidate.id),
                e.status,
                dict(e.objectives),
                dict(e.constraints),
                dict(e.descriptors),
                e.error,
                tuple(per_candidate.get(int(e.candidate.id), ())),
            )
            for e in results
        ),
    )
    if write and run_dir is not None:
        path = run_dir / REPORT_FILE
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report.to_json(), indent=2) + "\n")
        os.replace(temporary, path)
        report = HeldOutReport(report.scenarios_fingerprint, report.evaluator, report.candidates, path)
    return report


def _choose(
    run: RunResult[Any] | str | Path,
    run_dir: Path | None,
    candidates: Literal["best", "pareto"] | Sequence[int],
    problem: ProblemSpec[Any],
    backend: Backend,
) -> list[Candidate[Any]]:
    """The candidates to evaluate, with their genomes."""
    if isinstance(run, RunResult):
        if isinstance(candidates, str):
            if candidates == "best":
                if run.best is None:
                    raise ValueError("the run has no best candidate (several objectives, or nothing succeeded): use candidates='pareto'")
                return [run.best.candidate]
            return [e.candidate for e in run.pareto_front]
        if run_dir is None:
            raise ValueError("candidates given by id need the run directory: pass the run's directory instead of its result")
    assert run_dir is not None
    wanted = None if isinstance(candidates, str) else set(int(c) for c in candidates)
    tracker: ResultTracker[Any] = ResultTracker(problem.objectives, constrained=bool(problem.constraints))
    found: dict[int, Candidate[Any]] = {}
    seen = 0
    for record in iter_records(run_dir / "events.sqlite"):
        evaluation: Evaluation[Any] = record.evaluation(record.decoded_candidate(backend))
        if wanted is not None:
            if int(record.candidate_id) in wanted:
                found[int(record.candidate_id)] = evaluation.candidate
        else:
            tracker.add([evaluation], seen)
        seen += 1
    if wanted is not None:
        missing = sorted(wanted - set(found))
        if missing:
            raise ValueError(f"the run did not record candidates {missing}")
        return [found[int(c)] for c in candidates]  # type: ignore[union-attr]
    if candidates == "best":
        if tracker.best is None:
            raise ValueError("the run has no best candidate (several objectives, or nothing succeeded): use candidates='pareto'")
        return [tracker.best.candidate]
    return [e.candidate for e in tracker.pareto_front]


def _batch_of(chosen: Sequence[Candidate[Any]], backend: Backend) -> Batch[Any]:
    """The candidates as one batch: array-backed if their genomes are arrays of one shape, so that batched evaluators apply."""
    if not chosen:
        raise ValueError("there are no candidates to evaluate")
    genomes = [c.genome for c in chosen]
    try:
        stacked = np.stack([np.asarray(backend.to_numpy(g)) for g in genomes])  # pyright: ignore[reportUnknownMemberType]
    except (TypeError, ValueError):
        return ListBatch(tuple(chosen))
    if stacked.ndim != 2:
        return ListBatch(tuple(chosen))
    return ArrayBatch(backend.asarray(stacked), [c.id for c in chosen], 0, [c.origin for c in chosen], [c.parents for c in chosen])


def _run_sync(coroutine: Any) -> HeldOutReport:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="auxein-held-out") as pool:
        return pool.submit(asyncio.run, coroutine).result()
