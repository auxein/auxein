"""How long the generation queries of the recording's reader take on a long run (design doc §10.2).

    uv run python -m benchmarks.reader_queries                     # 300 generations x 200 candidates x 2,000 floats
    uv run python -m benchmarks.reader_queries --generations 30 --population 20 --dim 50 --keep runs/reader

The recording is built through `SQLiteRecorder`, as a run of a generational strategy writes it (one batch and one tell per
generation, genomes in the genome store), without running a strategy: what is measured is the reader. Each query is timed on
its own fresh reader, after the first query, which builds the reader's index once (`generations()`), is timed separately.
Recording 60,000 distinct genomes of 2,000 float64 numbers writes about 1 GB.
"""

import argparse
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from auxein.core import ArrayBatch, CandidateId, Evaluation, EvaluationBatch, Result, Status
from auxein.recording import SQLiteRecorder, open_run


def build(run_dir: Path, generations: int, population: int, dim: int, seed: int = 0) -> None:
    """A recording of `generations` tells of `population` candidates each, with one minimised objective and a constraint."""
    rng = np.random.default_rng(seed)
    recorder = SQLiteRecorder(run_dir)
    recorder.on_start(
        {"name": "reader-queries", "problem": {"objectives": [{"name": "value", "direction": "minimise"}], "constraints": ["c"]}}
    )
    first = 0
    for step in range(generations):
        genomes = rng.standard_normal((population, dim))
        ids = [CandidateId(first + i) for i in range(population)]
        parents = None if step == 0 else [(CandidateId(int(p)),) for p in rng.integers(max(0, first - population), first, population)]
        batch = ArrayBatch(genomes, ids, step, "init" if step == 0 else "tournament+intermediate+self_adaptive", parents)
        values = (genomes * genomes).sum(axis=1) / (1 + step)  # improves over the generations, as a search would
        violations = np.maximum(0.0, genomes[:, 0] - 2.0)
        evaluations: list[Evaluation[object]] = []
        for candidate, value, violation in zip(batch.candidates, values.tolist(), violations.tolist(), strict=True):
            if candidate.id % 97 == 0:
                evaluations.append(Evaluation.failed(candidate, Status.FAILED, "diverged"))
            else:
                result = Result({"value": value}, {"c": violation})
                evaluations.append(Evaluation(candidate, Status.OK, result.objectives, result.constraints))
        recorder.on_batch(step, batch, EvaluationBatch(evaluations))
        recorder.on_tell(step, population)
        first += population
    recorder.on_end("completed", "budget:evaluations", {})


def timed(run_dir: Path, query: Callable[..., object], warm: bool = True) -> float:
    """Seconds for one call of `query(reader)` on a fresh reader (whose index is built first, untimed, when `warm`)."""
    with open_run(run_dir) as run:
        if warm:
            run.generations()
        start = time.perf_counter()
        query(run)
        return time.perf_counter() - start


def measure(run_dir: Path) -> dict[str, float]:
    with open_run(run_dir) as run:
        middle = max(1, run.generations() // 2)
    return {
        "first query (builds the index): generations()": timed(run_dir, lambda r: r.generations(), warm=False),
        f"best_up_to({middle})": timed(run_dir, lambda r: r.best_up_to(middle)),
        "best_up_to() (overall)": timed(run_dir, lambda r: r.best_up_to()),
        f"best_in({middle})": timed(run_dir, lambda r: r.best_in(middle)),
        f"pareto_front_up_to({middle})": timed(run_dir, lambda r: r.pareto_front_up_to(middle)),
        f"evaluations_in({middle})": timed(run_dir, lambda r: r.evaluations_in(middle)),
        "champions(every=25), all of them": timed(run_dir, lambda r: list(r.champions(every=25))),
        "generation_history()": timed(run_dir, lambda r: r.generation_history()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population", type=int, default=200)
    parser.add_argument("--dim", type=int, default=2000)
    parser.add_argument("--keep", type=Path, default=None, help="build the recording here and keep it (default: a temporary directory)")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = args.keep if args.keep is not None else Path(temporary) / "run"
        start = time.perf_counter()
        build(run_dir, args.generations, args.population, args.dim)
        size = (run_dir / "events.sqlite").stat().st_size
        print(
            f"recording: {args.generations} generations x {args.population} candidates x {args.dim} floats, "
            f"{size / 1e6:,.0f} MB, built in {time.perf_counter() - start:.1f} s"
        )
        for name, seconds in measure(run_dir).items():
            print(f"  {name:<45} {seconds * 1000:>9.1f} ms")


if __name__ == "__main__":
    main()
