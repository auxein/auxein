"""Run the benchmark: independent runs in parallel, then the engine overhead benchmark one at a time."""

import json
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.adapters import RunInfo, load_adapter
from benchmarks.config import AlgorithmConfig, Config
from benchmarks.metadata import collect_metadata, git_dirty, git_sha
from benchmarks.objective import CountingObjective
from benchmarks.problems import make_problem

OVERHEAD_PROBLEM = "sphere"
OVERHEAD_WARMUP_BUDGET = 200


def target_key(target: float) -> str:
    return f"{target:g}"


@dataclass(frozen=True)
class Task:
    """One run: everything a worker needs, so that the result does not depend on which worker runs it."""

    index: int
    algorithm: str
    adapter: str
    params: dict[str, Any]
    problem: str
    dim: int
    instance: int
    seed: int
    budget: int
    targets: tuple[float, ...]


def build_tasks(config: Config) -> list[Task]:
    tasks = []
    for problem in config.problems:
        for dim in config.dims:
            for algorithm in config.algorithms:
                for k in range(config.runs):
                    tasks.append(
                        Task(
                            index=len(tasks),
                            algorithm=algorithm.name,
                            adapter=algorithm.adapter,
                            params=algorithm.params,
                            problem=problem,
                            dim=dim,
                            instance=config.instance_offset + k,
                            seed=config.base_seed + k,
                            budget=config.budget(dim),
                            targets=config.targets,
                        )
                    )
    return tasks


def run_one(
    adapter: str, params: dict[str, Any], problem: str, dim: int, instance: int, seed: int, budget: int, targets: tuple[float, ...] = ()
) -> tuple[CountingObjective, RunInfo, float]:
    """Run one algorithm on one problem instance. Returns the objective (with its trace), the run info and the wall time."""
    run = load_adapter(adapter)
    objective = CountingObjective(make_problem(problem, dim, instance), budget, targets)
    start = time.perf_counter()
    info = run(objective, dim, seed, params)
    return objective, info, time.perf_counter() - start


def execute_task(task: Task) -> dict[str, Any]:
    objective, info, wall = run_one(task.adapter, task.params, task.problem, task.dim, task.instance, task.seed, task.budget, task.targets)
    return {
        "task": task.index,
        "algorithm": task.algorithm,
        "adapter": task.adapter,
        "params": task.params,
        "problem": task.problem,
        "dim": task.dim,
        "instance": task.instance,
        "seed": task.seed,
        "budget": task.budget,
        "evals": objective.evals,
        "wall_time": wall,
        "final_error": objective.best_error,
        "trace": [[evals, error] for evals, error in objective.trace],
        "hits": {target_key(t): hit for t, hit in objective.hits.items()},
        "info": info.to_dict(),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def new_results_dir(root: Path) -> Path:
    sha = (git_sha() or "unknown")[:7] + ("-dirty" if git_dirty() else "")
    path = root / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{sha}"
    path.mkdir(parents=True)
    return path


def run_quality(tasks: list[Task], out_dir: Path, workers: int, progress: Callable[[str], None]) -> list[dict[str, Any]]:
    """Run all tasks in parallel. Records are appended as they complete, then rewritten in task order."""
    path = out_dir / "runs.jsonl"
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool, open(path, "w") as partial:
        futures = [pool.submit(execute_task, task) for task in tasks]
        step = max(1, len(tasks) // 50)
        for done, future in enumerate(as_completed(futures), start=1):
            record = future.result()  # a crash in a run is a crash of the benchmark
            records.append(record)
            partial.write(json.dumps(record) + "\n")
            partial.flush()
            if done % step == 0 or done == len(tasks):
                elapsed = time.perf_counter() - start
                progress(
                    f"[{done}/{len(tasks)}] runs finished, {elapsed:.0f}s elapsed, about {elapsed / done * (len(tasks) - done):.0f}s left"
                )
    records.sort(key=lambda r: r["task"])
    write_jsonl(path, records)
    return records


def overhead_cases(config: Config) -> Iterator[tuple[AlgorithmConfig, int, int | None]]:
    """(algorithm, dimension, population size) for the overhead benchmark. The size is None without a population parameter."""
    assert config.overhead is not None
    selected = config.overhead.algorithms
    for algorithm in config.algorithms:
        if selected is not None and algorithm.name not in selected:
            continue
        for dim in config.overhead.dims:
            if "population_size" in algorithm.params:
                for population_size in config.overhead.population_sizes:
                    yield algorithm, dim, population_size
            else:
                yield algorithm, dim, None


def run_overhead(config: Config, progress: Callable[[str], None]) -> list[dict[str, Any]]:
    """Wall-clock time per evaluation on a negligible-cost objective. Serial, so that runs do not disturb each other."""
    assert config.overhead is not None
    overhead = config.overhead
    cases = list(overhead_cases(config))
    for algorithm in config.algorithms:  # untimed warm-up: imports and first-call costs stay out of the measurements
        run_one(algorithm.adapter, algorithm.params, OVERHEAD_PROBLEM, 2, 0, config.base_seed, OVERHEAD_WARMUP_BUDGET)

    records = []
    for number, (algorithm, dim, population_size) in enumerate(cases, start=1):
        params = dict(algorithm.params)
        if population_size is not None:
            params["population_size"] = population_size
        for repeat in range(overhead.repeats):
            objective, info, wall = run_one(
                algorithm.adapter, params, OVERHEAD_PROBLEM, dim, repeat, config.base_seed + repeat, overhead.budget
            )
            records.append(
                {
                    "algorithm": algorithm.name,
                    "dim": dim,
                    "population_size": population_size,
                    "repeat": repeat,
                    "budget": overhead.budget,
                    "evals": objective.evals,
                    "wall_time": wall,
                    "us_per_eval": wall / objective.evals * 1e6,
                    "generations": info.generations,
                    "evals_per_generation": info.evals_per_generation,
                    "stop_reason": info.stop_reason,
                }
            )
        times = [r["us_per_eval"] for r in records[-overhead.repeats :]]
        progress(f"[{number}/{len(cases)}] overhead {algorithm.name} d={dim} pop={population_size}: {statistics.median(times):.1f} us/eval")
    return records


def run_benchmark(
    config: Config, results_root: Path, workers: int, skip_overhead: bool = False, progress: Callable[[str], None] | None = None
) -> Path:
    """Run a whole config into a new results directory and return its path."""
    log = progress or (lambda message: print(message, file=sys.stderr, flush=True))
    out_dir = new_results_dir(results_root)
    (out_dir / "metadata.json").write_text(json.dumps(collect_metadata(config, workers), indent=2) + "\n")

    tasks = build_tasks(config)
    log(f"config {config.name!r}: {len(tasks)} runs on {workers} workers -> {out_dir}")
    start = time.perf_counter()
    run_quality(tasks, out_dir, workers, log)
    log(f"runs took {time.perf_counter() - start:.0f}s")

    if config.overhead is not None and not skip_overhead:
        start = time.perf_counter()
        write_jsonl(out_dir / "overhead.jsonl", run_overhead(config, log))
        log(f"overhead benchmark took {time.perf_counter() - start:.0f}s")
    return out_dir
