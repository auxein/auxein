# auxein [![license](https://img.shields.io/hexpm/l/plug.svg?maxAge=2592000)](https://github.com/auxein/auxein/blob/master/LICENSE) [![CI](https://github.com/auxein/auxein/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/auxein/auxein/actions/workflows/ci.yml)

**Auxein is a Python framework for evolving agents that act in environments.** It searches over agent designs (genomes), which can be numeric parameters, structures or text. It does this by proposing candidates, evaluating them in an environment across a set of scenarios, and selecting on the outcomes. It works with any type of genome and any environment, and treats evaluation as the expensive step: evaluations are budgeted and batched, and may be asynchronous or noisy. Function optimisation, regression and model fitting are supported directly.

## Status

Auxein is a **0.x version**: the API may change before 1.0. Install it with `pip install auxein` (add the extras you need, for example `pip install "auxein[torch,cma]"`), or work on it from the repository with [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/auxein/auxein && cd auxein && uv sync
```

The design is in [`docs/design/core.md`](https://github.com/auxein/auxein/blob/master/docs/design/core.md), together with what is built, how each part was validated and what is left open. The optional dependencies are extras: `auxein[torch]` (PyTorch, for GPUs), `auxein[cma]` (pycma, for `PycmaStrategy`) and `auxein[gymnasium]` (Gymnasium, for the reinforcement-learning adapter); `import auxein` needs none of them.

This is a rewrite: **there is no backward compatibility with 0.x**. The earlier engine remains at the git tag [`v0.2.0`](https://github.com/auxein/auxein/tree/v0.2.0).

## What Auxein offers

- **Strategies** that propose candidates and learn from the results: `RandomSearch`; a composable `GeneticAlgorithm` (tournament and SUS selection, intermediate, uniform and simulated binary crossover, Gaussian, self-adaptive and polynomial mutation) with type-aware operators for integer, binary and categorical genes; `StructuredGeneticAlgorithm` for genomes that are not arrays, with recorded and replayed LLM-style operators; `NSGA2` for several objectives, with constraints; `PycmaStrategy`, CMA-ES from pycma; and `Scalarised`, which runs any single-objective strategy on a multi-objective problem.
- **Search spaces:** `Box` (with log scale), `MixedSpace` of real, integer, binary and categorical dimensions (also `IntegerSpace` and `BinarySpace`), and `SequenceSpace` for variable-length sequences; your own through a small codec protocol.
- **Evaluators and the agent layer:** plain functions (`FunctionEvaluator`), vectorised ones on whole populations (`VectorisedEvaluator`), and `EpisodeEvaluator`, which runs agents in every scenario of a seeded `ScenarioSet` and turns the measurements into objectives, constraints and descriptors with an `Aggregator` (mean, worst case, quantile, CVaR). Step-level worlds, a Gymnasium adapter (`GymnasiumEnvironment`, `LinearPolicy`) and held-out evaluation are included. Directions are explicit per objective, and constraints are violation amounts.
- **Concurrency, failures and timeouts:** concurrent evaluation in threads or worker processes, steady-state or generation delivery, deterministic mode (the same event log whatever the number of workers), `async def` evaluators, per-evaluation timeouts, and failures that are recorded and ranked last instead of stopping the run.
- **Recording, checkpoints and resume:** with a `run_dir`, every candidate, its lineage, its evaluations and its per-scenario measurements go into an SQLite file, read back with `open_run`. Checkpoints are written without pickle, and `resume` continues an interrupted, killed or finished run as if it had never stopped, or extends it with a larger budget.
- **Backends and GPUs:** numpy and PyTorch, float64 and float32, on the CPU, on CUDA and on Apple Metal, with reproducible random streams derived from one seed. The GPU smoke suite is run by hand (`tests/gpu/README.md`).

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

## Documentation

Four short notebooks, executed, with their outputs, introduce the API ([view them on nbviewer](https://nbviewer.org/github/auxein/auxein/tree/master/notebooks/)):

- [Rastrigin](https://github.com/auxein/auxein/blob/master/notebooks/rastrigin.ipynb): three strategies at equal budgets, a recorded run and the ancestry of its best candidate
- [Linear regression](https://github.com/auxein/auxein/blob/master/notebooks/linear_regression.ipynb): evolved coefficients against the closed-form solution
- [Logistic regression](https://github.com/auxein/auxein/blob/master/notebooks/logistic_regression.ipynb): a maximised log-likelihood, against the known true coefficients
- [Polynomial regression](https://github.com/auxein/auxein/blob/master/notebooks/polynomial_regression.ipynb): structure genes in a mixed space, the Pareto front of error against complexity with `NSGA2`, and one trade-off chosen with `Scalarised`

The [design document](https://github.com/auxein/auxein/blob/master/docs/design/core.md) is the reference, and the [benchmark reports](https://github.com/auxein/auxein/blob/master/benchmarks/README.md) show how the algorithms compare.

## Links

- [Design document](https://github.com/auxein/auxein/blob/master/docs/design/core.md)
- [Benchmark harness](https://github.com/auxein/auxein/blob/master/benchmarks/README.md), its [0.2.0 baseline](https://github.com/auxein/auxein/blob/master/benchmarks/reports/baseline-0.2.0/report.md) and the [comparison of the new genetic algorithm with it](https://github.com/auxein/auxein/blob/master/benchmarks/reports/core-ga-0.3.0-dev/report.md)
- [Changelog](https://github.com/auxein/auxein/blob/master/CHANGELOG.md)
- [`v0.2.0`](https://github.com/auxein/auxein/tree/v0.2.0): the previous engine, its notebooks and documentation

## Development

```
uv sync                                     # install the project with its development dependencies
uv run pytest                               # tests
uv run --group notebooks pytest --nbmake notebooks/                      # the notebooks, executed
uv run pytest -m gpu --device mps           # the GPU smoke suite, by hand: mps, cuda or cpu (tests/gpu/README.md)
uv run ruff check                           # lint
uv run ruff format --check                  # formatting
uv run pyright                              # type check
uv run python -m benchmarks run --config benchmarks/configs/quick.toml    # a smoke-sized benchmark
```

------------------

## Why this name, Auxein?

[Auxein](https://en.wikipedia.org/wiki/Auxin) (αυξειν) means _to grow_ in Greek.
