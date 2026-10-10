"""Run a multi-objective benchmark: independent runs in parallel, each measured on the non-dominated set it has found."""

import json
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.adapters import load_adapter
from benchmarks.metadata import collect_metadata
from benchmarks.mo_config import MOConfig
from benchmarks.mo_objective import MOCountingObjective
from benchmarks.mo_problems import make_mo_problem
from benchmarks.runner import new_results_dir, write_jsonl

FRONT_DECIMALS = 6


@dataclass(frozen=True)
class MOTask:
    """One run: everything a worker needs, so that the result does not depend on which worker runs it."""

    index: int
    algorithm: str
    adapter: str
    params: dict[str, Any]
    problem: str
    seed: int
    budget: int


def build_mo_tasks(config: MOConfig) -> list[MOTask]:
    tasks = []
    for problem in config.problems:
        for algorithm in config.algorithms:
            for k in range(config.runs):
                tasks.append(
                    MOTask(
                        len(tasks), algorithm.name, algorithm.adapter, algorithm.params, problem.name, config.base_seed + k, problem.budget
                    )
                )
    return tasks


def execute_mo_task(task: MOTask) -> dict[str, Any]:
    problem = make_mo_problem(task.problem)
    objective = MOCountingObjective(problem, task.budget)
    run = load_adapter(task.adapter)
    start = time.perf_counter()
    info = run(objective, problem.dim, task.seed, task.params)  # type: ignore[arg-type]
    wall = time.perf_counter() - start
    trace = objective.trace
    front = np.round(objective.front, FRONT_DECIMALS)
    front = front[np.lexsort(front.T[::-1])]
    return {
        "task": task.index,
        "algorithm": task.algorithm,
        "adapter": task.adapter,
        "params": task.params,
        "problem": task.problem,
        "dim": problem.dim,
        "n_obj": problem.n_obj,
        "seed": task.seed,
        "budget": task.budget,
        "evals": objective.evals,
        "wall_time": wall,
        "final_hv": objective.final_hypervolume,
        "final_igd_plus": objective.final_igd_plus,
        "trace_hv": [[evals, hv] for evals, hv, _ in trace],
        "trace_igd_plus": [[evals, igd] for evals, _, igd in trace],
        "front": front.tolist(),
        "info": info.to_dict(),
    }


def run_mo_quality(tasks: list[MOTask], out_dir: Path, workers: int, progress: Callable[[str], None]) -> list[dict[str, Any]]:
    path = out_dir / "runs.jsonl"
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool, open(path, "w") as partial:
        futures = [pool.submit(execute_mo_task, task) for task in tasks]
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


def run_mo_benchmark(config: MOConfig, results_root: Path, workers: int, progress: Callable[[str], None] | None = None) -> Path:
    """Run a whole multi-objective config into a new results directory and return its path."""
    log = progress or (lambda message: print(message, file=sys.stderr, flush=True))
    out_dir = new_results_dir(results_root)
    (out_dir / "metadata.json").write_text(json.dumps(collect_metadata(config, workers), indent=2) + "\n")
    tasks = build_mo_tasks(config)
    log(f"config {config.name!r}: {len(tasks)} runs on {workers} workers -> {out_dir}")
    start = time.perf_counter()
    run_mo_quality(tasks, out_dir, workers, log)
    log(f"runs took {time.perf_counter() - start:.0f}s")
    return out_dir
