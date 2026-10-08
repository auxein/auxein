# Auxein benchmark harness

A harness to measure, repeatably and fairly, how good the solutions found by Auxein (and by other algorithms) are, and how much time the framework itself spends per fitness evaluation. It is not part of the installed package: it lives in `benchmarks/` and its dependencies are in the `bench` dependency group (installed by a plain `uv sync`).

It exists so that a later redesign of Auxein's core can be judged against a baseline. The committed baseline for Auxein 0.2.0, with its findings, is in [`reports/baseline-0.2.0/`](reports/baseline-0.2.0/report.md).

It measures two things:

1. **Quality**: the best (true) error each algorithm finds for a fixed budget of fitness evaluations, on five problems, in 2, 10 and 30 dimensions, over many paired runs.
2. **Engine overhead**: wall-clock time per evaluation on a negligible-cost objective, for different dimensions and population sizes.

## Why the budget is in evaluations

Generations differ in size between algorithms. An Auxein generation with population 200 costs 204 evaluations, because `Population.update()` re-scores everyone, while a CMA-ES generation in 10-D costs about 10. Comparing algorithms after the same number of generations would compare very different amounts of work. So every algorithm gets the same number of fitness evaluations, and progress is always plotted against evaluations.

The budget is enforced outside the algorithms. `CountingObjective` (`objective.py`) wraps a problem, counts every call, records the best *true* error so far at log-spaced evaluation counts, and raises `BudgetExhausted` on the call that would exceed the budget, so no algorithm can overspend. The initial population counts against the budget, and a partial generation is fine. On noisy problems the algorithm sees the noisy value, but the trace records the noise-free error of the evaluated point, so what is measured is quality, not luck.

## Running it

```
# the full benchmark (about 2 minutes on 12 cores)
uv run --group bench python -m benchmarks run --config benchmarks/configs/full.toml [--workers N]

# a smoke-sized one, as run in CI
uv run --group bench python -m benchmarks run --config benchmarks/configs/quick.toml

# report.md and PNG plots, written next to the results
uv run --group bench python -m benchmarks report benchmarks/results/<timestamp>-<sha>
```

`run` prints its progress to stderr and the results directory, `benchmarks/results/<timestamp>-<short-sha>/`, to stdout. Other options: `--results-root` and `--skip-overhead`. The directory contains `runs.jsonl` (one record per run: algorithm, config, parameters, problem, dimension, instance, seed, budget, evaluations used, wall time, trace, first hit of each target, algorithm-specific run info), `overhead.jsonl` and `metadata.json` (git SHA and dirty flag, library and Python versions, CPU, the full config). `benchmarks/results/` is gitignored; to keep a baseline, copy a results directory with its report to `benchmarks/reports/`. If a `findings.md` sits next to the results, `report` puts it at the top of `report.md`.

The same config, commit and seeds reproduce identical `runs.jsonl` contents (apart from the wall-clock fields). Every run seeds itself, so the result does not depend on which worker process executed it.

### Configs

TOML files in `configs/`: the problems, dimensions, number of runs, the budget per dimension (`budget_per_dim`), the precision targets, the overhead benchmark, and a list of named algorithm entries (`name`, `adapter`, `params`). Run *k* of every algorithm uses problem instance `instance_offset + k` (the offset is 0 unless the config says otherwise, e.g. to tune on other instances than the ones a comparison is judged on) and seed `base_seed + k`, so comparisons are paired. An optional `[report] references = [...]` names the algorithms the statistical comparison is made from (default: `auxein-default`).

## Layout

| Path | What it is |
|---|---|
| `problems.py` | the `Problem` interface and the five problems (sphere, ellipsoid, Rosenbrock, Rastrigin, noisy sphere) |
| `objective.py` | `CountingObjective`, `BudgetExhausted` and the trace checkpoints |
| `adapters/` | one module per algorithm: Auxein's 0.x `Static` playground, random search, CMA-ES, the new core's driver with `RandomSearch` (`auxein_core_random`, a cross-check of the driver) and with `GeneticAlgorithm` (`auxein_core_ga`, configured by operator tables) |
| `runner.py`, `config.py`, `metadata.py` | parallel runs, the overhead benchmark, configs and result metadata |
| `report.py`, `stats.py` | the report: plots, tables and statistics |
| `tests/` | tests of the harness, run by the main `uv run pytest` |

## Adding a problem

Subclass `Problem` in `problems.py` (or anywhere you can import it from) and register it:

```python
class MyProblem(Problem):
    name = "my_problem"
    noisy = False  # True if evaluate() differs from true_error()

    def true_error(self, x):  # f(x) - f*, >= 0, and 0 at the optimum
        ...

    def evaluate(self, x):  # what the algorithm sees; only override it for noisy problems
        ...


register(MyProblem)  # the factory is called as factory(dim, instance)
```

A problem has a `name`, a dimension `dim`, a domain (`lower`, `upper`), an `evaluate(x)` that may be noisy and a `true_error(x)`. Draw anything random about an instance (its optimum, a rotation, noise) from `instance_rng(instance, dim, stream)`, so that it depends only on the instance id and never on an algorithm's randomness. Then add the name to `problems` in a config. Problems that are not analytic functions, such as agent domains, fit the same interface.

## Adding an algorithm

Create a module in `adapters/` with a `run` function (or point `adapter` at any importable module, with a dotted path):

```python
from benchmarks.adapters.base import RunInfo
from benchmarks.objective import BudgetExhausted, CountingObjective


def run(objective: CountingObjective, dim: int, seed: int, params: dict) -> RunInfo:
    try:
        while True:
            ...  # evaluate points only through objective(x)
    except BudgetExhausted:
        pass
    return RunInfo(generations=..., stop_reason="budget", evals_per_generation=...)
```

Rules: evaluate only through `objective(x)` and stop cleanly when it raises `BudgetExhausted`; draw all randomness from `seed`; take the search domain from `objective.problem.lower` and `.upper`. `RunInfo` carries algorithm-specific extras (generations completed, the stop reason, evaluations per generation, anything else in `extra`). Then add an entry to a config:

```toml
[[algorithms]]
name = "my-algorithm"
adapter = "my_algorithm"   # benchmarks/adapters/my_algorithm.py, or a dotted module path
[algorithms.params]
whatever = 1
```

To benchmark another version of Auxein, install it in the environment and run the same config: the Auxein adapter imports whatever `auxein` is installed, and `metadata.json` records its version.

## Checks of the harness itself

`benchmarks/tests/sanity_test.py` (and the "Sanity checks" section of the report) confirm that the harness is right, not just the algorithms: CMA-ES must reach 10⁻⁶ on the 10-D sphere and ellipsoid within the full budget in at least 90% of runs, and random search must not reach 10⁻³ on the 10-D sphere. If one fails, suspect the harness first.
