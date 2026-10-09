# Changelog

## 0.3.0 (unreleased)

Auxein is rewritten around a new core, designed as a framework for evolving agents that act in environments (see the
[design document](docs/design/core.md)). **There is no backward compatibility with 0.x**: the old engine, its API and its
notebooks are gone from the repository, and remain available at the git tag [`v0.2.0`](https://github.com/auxein/auxein/tree/v0.2.0).
Nothing is published to PyPI yet, and the release workflow was removed.

What exists now:

- **Strategies ask, evaluators evaluate, the driver runs the loop.** `auxein.run` / `auxein.arun`, budgets in evaluations (and wall time and cost units), results with `best`, a Pareto front and a trace.
- **Evaluators:** `FunctionEvaluator` and `VectorisedEvaluator`; fitness functions return a number or an explicit `Result` / `BatchResult`.
- **Failures, timeouts and crashes:** `failure_policy="infeasible"` (the default) records an exception, a timeout or a worker crash as a `FAILED` or `TIMEOUT` evaluation that ranks last, and `"fail_fast"` stops at the first one. The first failure is shown at once with its traceback (`EvaluationFailureWarning`), and a run whose first 10 evaluations all fail is stopped (`initial_failure_guard`). A non-finite objective is a failure. `timeout=` is hard for `async def` and worker processes and soft for threads (abandoned daemon threads that never block exit). Process execution uses a pool of Auxein's own that kills a timed-out worker, survives a crashed one and leaves no orphan. `RunResult.status_counts` counts the outcomes. In the genetic algorithm a failed member is never a parent while there is another.
- **Concurrent evaluation:** `concurrency=N` and `executor="auto" | "inline" | "thread" | "process"` (processes are never automatic and always use `spawn`); `async def` functions run natively. `delivery="steady_state"` keeps a window of `batch_size` candidates in flight, and `deterministic=True` (the default) makes the event log identical whatever the number of workers; `deterministic=False` tells results as they finish. Evaluation streams (`rng_for`) are now numpy-backed on every backend.
- **Strategies:** `RandomSearch`, and a composable `GeneticAlgorithm` (tournament and SUS selection, intermediate and uniform recombination, Gaussian and self-adaptive mutation, plus selection of survivors, constraints supported). It beats the 0.2.0 engine on every problem of the benchmark suite at a lower cost per evaluation: see the [comparison report](benchmarks/reports/core-ga-0.3.0-dev/report.md).
- **Backends:** numpy and PyTorch, float64 and float32, reproducible random streams derived from one seed. CPU runs are tested in CI; CUDA and Metal devices are covered by a GPU smoke suite that is run by hand (see below).
- **Search spaces:** `Box`, a bounded real vector with optional log scale.
- **Recording:** an opt-in run directory (`run_dir`) with metadata and an SQLite event log of candidates, lineage and evaluations, and a reader (`auxein.open_run`).
- **Checkpoints and resume:** with `run_dir`, the strategy's and the driver's state is checkpointed (JSON and arrays, no pickle, written atomically; `checkpoint_every`, `checkpoint_every_evaluations`, `keep_checkpoints`), also at the end and on interrupt, and `auxein.resume` / `auxein.aresume` continue a run that was interrupted, killed or finished. Resuming in deterministic mode replays the recording (recorded evaluations are never repeated, and a candidate that differs is an error naming it), so a killed run, or a finished run extended with a larger budget, has exactly the event log of the run that was never stopped. In throughput mode it truncates to the latest checkpoint and redoes. Only the budget may change; a lock file keeps two processes from writing to one run. The run directory has schema version 2 (older runs cannot be resumed), with `checkpoints`, a `resume` event and the sessions in `metadata.json`.
- **The agent layer:** `EpisodeEvaluator(decoder, environment, scenarios, aggregator)` evaluates agents by running them in every scenario of a `ScenarioSet` (seeded instances, fingerprinted, with selection and held-out splits) and turning the raw measurements of the episodes into objectives, constraints and descriptors with a declarative `Aggregator` (mean, worst case, quantile, CVaR). Environments run whole episodes (plain or `async def`), optionally batched on the array backend, and step-level `reset`/`step` worlds plug in through `StepEnvironment`. Every candidate faces the same world (common random numbers); a candidate fails if any of its episodes does; timeouts apply per episode. The per-scenario measurements are recorded (schema version 3, an `episodes` table), `RunReader.reaggregate` re-judges a recorded run with another aggregator without re-simulating, and `auxein.driver.evaluate_held_out` judges chosen candidates on held-out scenarios after a run. Runs of episode evaluators resume like any other.
- **Structured genomes:** a search space can have a codec (genome to canonical JSON and back), so that genomes that are not arrays are recorded, replayed byte for byte and checkpointed. `SequenceSpace` is a built-in variable-length space (tuples of items from a vocabulary, optionally unique), and `StructuredGeneticAlgorithm` evolves such genomes with the numeric GA's ranking, survivor selection and parent selection and with structure-aware operators (insert, delete, replace and swap mutations, one- and two-point cut-and-splice crossover). **External proposal operators** (the slot for LLM-driven mutation) are called through the run's log: each call is recorded with its output, cost and time, and replayed on resume or extension, so a resumed run never pays twice. The recording gets a **content-addressed genome store** in `events.sqlite` (schema version 4): genomes over 4 KiB are stored once by their SHA-256, which shrinks a run with large, repetitive genomes by 6 to 56 times. Functions in worker processes can now return a `Result`.
- **PyTorch everywhere:** every numeric code path (spaces, both strategies' ranking and selection, operators, the three evaluators, aggregators, checkpoints and the genome store for arrays) is tested on numpy and PyTorch, in float64 and float32: unit tests on all four configurations, integration tests (runs, kill and resume, failures, timeouts, episodes) on numpy-float64 and torch-float32. A statistical cross-check requires the torch GA to search as well as numpy's (Vargha–Delaney A₁₂ in [0.35, 0.65] on the sphere and Rastrigin), `quick.toml` runs a torch entry in CI, and the benchmark adapters take `backend`, `precision` and `device`. Fixed on the way: the structured GA read a random draw with `np.asarray`, which raises on CUDA and Metal tensors; CVaR put one scenario too many in the tail for some `alpha` (`cvar_upper(.., 0.07)` over 100 scenarios); sigma-scaled selection lost all selection pressure when one objective overflowed float32. **The GPU smoke suite** (`tests/gpu/`, `uv run pytest -m gpu --device mps`) checks backends, streams, spaces, runs, the batched point mass and checkpoint/resume on a device, times a CPU-against-device probe, and writes a report to `docs/gpu-smoke/`. It has been run on Apple Metal; a CUDA run is pending. The design document now states the reproducibility scope across CPU architectures (float32 results are not portable between arm64 and x86-64).
- **A benchmark harness** (`benchmarks/`), with the frozen 0.2.0 baseline.
- Version `0.3.0.dev0`, strict typing across the package, and a global-random-state ban enforced by a test.

## 0.2.0

### Behaviour changes

- `Fps` now raises `ValueError` when any fitness is negative or when the total fitness is zero. Previously it silently favoured the *worst* individual for negative fitness. Use `FpsWithWindowing` or `SigmaScaling` for negative fitness (e.g. with `GlobalMinimum` or the regression fitness functions).
- `MaximumLikelihood` now returns the log-likelihood `Σ y·log(p) + (1−y)·log(1−p)`, with `p` clipped to `[1e-12, 1 − 1e-12]`, instead of a sum of probabilities. Fitness values are now `<= 0`.
- `build_individual(dna)` now defaults the mask to ones (it used to be empty), also when an empty mask is passed. `Genotype` raises `ValueError` if the mask and dna lengths differ.
- `Mutation` extension now appends the existing step size (`mask[-1]`) to the mask, instead of a fresh random value, so `SelfAdaptiveSingleStep` keeps a single shared step size.
- `MatrixRecombination` now really crosses over: the matrices are flattened to 1-D before they are handed to the inner recombination. It also calls `Recombination.__init__`, so it has an `allow_uneven` attribute.
- `StochasticUniversalSampling.select` validates its input (same lengths, finite, non-negative, positive sum) and normalises the probabilities, raising `ValueError` otherwise.
- `polynomial_fit` requires `x` to be a `np.ndarray` of size 1; the coefficients may be a list.
- Importing `auxein` no longer configures global logging (`logging.basicConfig` was removed). `Static` logs through `logging.getLogger("auxein.playgrounds.static")`.
- `FixedVariance` mutation draws its noise with a single vectorised call, so results for a given seed differ from previous versions.

### Fixes

- `StochasticUniversalSampling` no longer loops forever on NaN probabilities or when rounding leaves the cumulative sum just below the last pointer.
- `FpsWithWindowing` and `SigmaScaling` return a uniform distribution on a converged population instead of NaN.
- `ReplaceWorst` no longer crashes when fewer offspring than `offspring_size` are available (e.g. after pruning), no longer shrinks the population on failure, and is a no-op with no offspring.
- `SelfAdaptiveSingleStep` no longer crashes on individuals built without a mask.
- `Population.get_full_genome` supports populations of individuals with different dimensions, which fixes `Static.train` for variable-dimension populations.
- `auxein.fitness` exports `MaximumLikelihood` (and `MultipleLinearRegression` only once).
- `SigmaScaling` computes the population mean and standard deviation once per call instead of once per individual.

### Tooling and packaging

- Python 3.11+ is required; tested on 3.11, 3.12, 3.13 and 3.14. numpy is `>=1.26` (it was pinned to `1.24.3`).
- Poetry replaced by uv and hatchling; flake8 replaced by ruff (`E, F, W, B, UP, I`); pyright is clean and enforced in CI.
- The package ships a `py.typed` marker.
- GitHub Actions replaces Travis. CI runs the tests on every supported Python version, a lowest-direct-dependencies job, linting, type checking and the example notebooks.
- The example notebooks were fixed, re-executed and are now run in CI.
