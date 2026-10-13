"""The reader-queries benchmark at a CI size: its recording is a valid run with the generations it says."""

from pathlib import Path

import numpy as np

from auxein.recording import open_run
from benchmarks.reader_queries import build, measure


def test_the_benchmark_recording_has_its_generations_and_every_query_is_timed(tmp_path: Path):
    build(tmp_path / "run", generations=6, population=8, dim=600)  # 4.8 KB genomes: they go to the genome store
    with open_run(tmp_path / "run") as run:
        assert run.generations() == 6 and run.generation_of(47) == 6
        assert run.genome_store().blobs == 48
        champion = run.best_up_to()
        assert champion is not None and isinstance(champion.genome, np.ndarray) and champion.genome.shape == (600,)
        assert run.generation_history()[0]["failed"] == 1  # candidate 0 fails (every 97th does)
    times = measure(tmp_path / "run")
    assert len(times) == 8 and all(seconds >= 0 for seconds in times.values())
