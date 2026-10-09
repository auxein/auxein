"""A performance probe: does the device pay off, and from what size? Informational: nothing is asserted about the numbers.

A `GeneticAlgorithm` evolves two objectives, a cheap one (a sum of squares per candidate) and a heavy one (a product of the
whole population with a dimension × dimension matrix, then a sum of squares), at the reference size of the design (population
1,000, dimension 1,000) and at a small and a wide one. The time
of a generation is the time between two calls of the evaluator (so it contains `ask`, the evaluation and `tell`), measured
after the device has finished its queue. The first generations are discarded as a warm-up (kernel compilation on Metal,
cuBLAS initialisation on CUDA), the median of the next five is reported.
"""

import statistics
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

import auxein
from auxein.backend import Backend
from tests.gpu import report
from tests.gpu.conftest import device_name, synchronize

SIZES = [(100, 100), (1000, 1000), (1000, 4000)]
"""(population, dimension): the reference size of the design (1,000 × 1,000), a small one and a wide one."""
WARMUP, MEASURED = 2, 5


def objectives(backend: Backend, dimension: int) -> dict[str, Callable[[Any], Any]]:
    matrix = backend.asarray(np.random.default_rng(0).normal(size=(dimension, dimension)) / np.sqrt(dimension))
    xp = backend.xp
    return {
        "cheap": lambda X: xp.sum(X * X, axis=1),
        "heavy": lambda X: xp.sum(xp.matmul(X, matrix) ** 2, axis=1),
    }


def milliseconds_per_generation(backend: Backend, objective_name: str, population: int, dimension: int) -> float:
    function = objectives(backend, dimension)[objective_name]
    stamps: list[float] = []

    def timed(X: Any) -> Any:
        values = function(X)
        synchronize(backend.device)
        stamps.append(time.perf_counter())
        return values

    generations = WARMUP + MEASURED + 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", auxein.RecordingDisabledWarning)
        auxein.run(
            strategy=auxein.GeneticAlgorithm(population_size=population, offspring_size=population),
            evaluator=auxein.VectorisedEvaluator(timed),
            space=auxein.Box(-5.0, 5.0, dim=dimension),
            budget=auxein.Budget(evaluations=population * generations),
            seed=1,
            backend=backend,
            batch_size=population,
        )
    gaps = [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:], strict=False)]  # the first call has no predecessor
    return statistics.median(gaps[WARMUP:])


def test_time_per_generation_on_the_cpu_and_on_the_device(device: str):
    import torch

    configurations = [
        ("numpy float64, CPU", Backend()),
        (f"torch float32, CPU ({torch.get_num_threads()} threads)", Backend("torch", "cpu", "float32")),
    ]
    configurations.append((f"torch float32, {device} ({device_name(device)})", Backend("torch", device, "float32")))
    for population, dimension in SIZES:
        for name in ("cheap", "heavy"):
            for label, backend in configurations:
                row = report.ProbeRow(
                    f"{population:,} × {dimension:,}", name, label, milliseconds_per_generation(backend, name, population, dimension)
                )
                report.PROBE_ROWS.append(row)
    assert len(report.PROBE_ROWS) == len(SIZES) * 2 * len(configurations)  # it ran; what it measured is for the reader
