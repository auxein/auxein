# auxein [![license](https://img.shields.io/hexpm/l/plug.svg?maxAge=2592000)](https://github.com/auxein/auxein/blob/master/LICENSE) [![CI](https://github.com/auxein/auxein/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/auxein/auxein/actions/workflows/ci.yml)

**Auxein is a Python framework for evolving agents that act in environments.** It searches over agent designs (genomes), which can be numeric parameters, structures or text. It does this by proposing candidates, evaluating them in an environment across a set of scenarios, and selecting on the outcomes. It works with any type of genome and any environment, and treats evaluation as the expensive step: evaluations are budgeted and batched, and may be asynchronous or noisy. Function optimisation, regression and model fitting are supported directly.

## Status

Auxein is a **0.x development version**: the API may change, and it isn't published to PyPI yet. Install it from the repository with [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/auxein/auxein && cd auxein && uv sync
```

The design is in [`docs/design/core.md`](docs/design/core.md), together with what is built so far and what comes next. Today that is the core (candidates, batches, evaluations, search spaces, a numpy and PyTorch backend with reproducible random streams), a driver with budgets and recording, function and vectorised evaluators, and two strategies: `RandomSearch` and a composable `GeneticAlgorithm`.

This is a rewrite: **there is no backward compatibility with 0.x**. The earlier engine remains at the git tag [`v0.2.0`](https://github.com/auxein/auxein/tree/v0.2.0).

## Quickstart

Minimise the 10-dimensional Rastrigin function with a genetic algorithm:

```python
import numpy as np

import auxein


def rastrigin(X):
    """Rastrigin of a batch of points: X has shape (n, d) and the result shape (n,). Its minimum, 0, is at the origin."""
    return 10 * X.shape[1] + (X**2 - 10 * np.cos(2 * np.pi * X)).sum(axis=1)


result = auxein.run(
    strategy=auxein.GeneticAlgorithm(),
    evaluator=auxein.VectorisedEvaluator(rastrigin),
    space=auxein.Box(-5.12, 5.12, dim=10),
    budget=auxein.Budget(evaluations=20_000),
    seed=42,
    # run_dir="runs/rastrigin",  # records the candidates, their lineage and evaluations; without it a warning says so
)

print(result.best.objectives["value"], result.best.candidate.genome)
```

`result.best` is the best evaluation found. The same seed gives the same run. Pass `run_dir` to record the run, and read it back with `auxein.open_run`. A recorded run keeps checkpoints, so if it is interrupted, killed or finishes with too little budget, run the same script with `auxein.resume` (and a larger `Budget` to extend it) and it carries on as if it had never stopped.

## Links

- [Design document](docs/design/core.md)
- [Benchmark harness](benchmarks/README.md), its [0.2.0 baseline](benchmarks/reports/baseline-0.2.0/report.md) and the [comparison of the new genetic algorithm with it](benchmarks/reports/core-ga-0.3.0-dev/report.md)
- [Changelog](CHANGELOG.md)
- [`v0.2.0`](https://github.com/auxein/auxein/tree/v0.2.0): the previous engine, its notebooks and documentation

## Development

```
uv sync                                     # install the project with its development dependencies
uv run pytest                               # tests
uv run pytest -m gpu --device mps           # the GPU smoke suite, by hand: mps, cuda or cpu (tests/gpu/README.md)
uv run ruff check                           # lint
uv run ruff format --check                  # formatting
uv run pyright                              # type check
uv run python -m benchmarks run --config benchmarks/configs/quick.toml    # a smoke-sized benchmark
```

------------------

## Why this name, Auxein?

[Auxein](https://en.wikipedia.org/wiki/Auxin) (αυξειν) means _to grow_ in Greek.
