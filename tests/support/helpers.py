"""Builders shared by the tests of the evaluators, the driver and the recorder."""

import numpy as np

from auxein.backend import Backend
from auxein.core import ArrayBatch, CandidateId, EvalContext, Objective, ProblemSpec
from auxein.execution import Executor, InlineExecutor
from auxein.random import RunSeed
from auxein.spaces import Box


def problem(
    objectives: tuple[Objective, ...] = (Objective("value"),),
    constraints: tuple[str, ...] = (),
    descriptors: tuple[str, ...] = (),
    dim: int = 3,
) -> ProblemSpec[object]:
    return ProblemSpec(Box(-5.0, 5.0, dim=dim), objectives, constraints, descriptors)  # type: ignore[arg-type]


def eval_context(
    spec: ProblemSpec, backend: Backend, seed: int = 0, *, concurrency: int = 1, executor: Executor | None = None
) -> EvalContext:
    run_seed = RunSeed(seed)
    return EvalContext(
        spec,
        backend,
        lambda cid: run_seed.stream("evaluation", cid, backend=Backend("numpy", "cpu", backend.precision)),  # as the driver does
        lambda cid: run_seed.stream("evaluation-batch", cid, backend=backend),
        executor=InlineExecutor() if executor is None else executor,
        concurrency=concurrency,
    )


def array_batch(backend: Backend, n: int = 4, d: int = 3, first_id: int = 0, step: int = 0, origin: str = "random") -> ArrayBatch:
    genomes = backend.asarray(np.arange(n * d, dtype=np.float64).reshape(n, d) + 1.0)
    return ArrayBatch(genomes, [CandidateId(first_id + i) for i in range(n)], step, origin)
