# Auxein core design

**Status:** Draft for review · **Date:** 2026-10-07 · **Target version:** 0.3.0

---

## 1. Purpose

**Auxein is a Python framework for evolving agents that act in environments.** It searches over agent designs (genomes), which can be numeric parameters, structures or text. It does this by proposing candidates, evaluating them in an environment across a set of scenarios, and selecting on the outcomes. It works with any type of genome and any environment, and treats evaluation as the expensive step: evaluations are budgeted and batched, and may be asynchronous or noisy. Function optimisation, regression and model fitting are supported directly.

This document defines the core architecture of the next version of Auxein. It replaces the 0.x design entirely: **no backward compatibility is kept**. The fixed 0.x engine is tagged `v0.2.0` and kept only as the reference point for benchmark comparisons.

### 1.1 Design principles

1. **Algorithms don't evaluate.** Strategies propose candidates and learn from results. Evaluation is somebody else's job (§3).
2. **Evaluation is the expensive step.** Budgets, batching, concurrency, caching, timeouts and failures are first-class concerns of the framework, not of each algorithm.
3. **Any genome, any environment.** The core never inspects genomes and never assumes how an environment runs.
4. **Fast where it matters, simple everywhere else.** Numeric strategies work on arrays and can run on a GPU. Everything else stays plain Python.
5. **Reproducible by default.** Same seed, same backend and precision: same run.
6. **Everything is recorded.** Every candidate's origin and every evaluation's outcome can be inspected after the fact.

---

## 2. Concepts at a glance

| Concept | What it is |
|---|---|
| **Genome** | The design of an agent: an array, a structure, text. Opaque to the core and immutable. |
| **Search space** | Describes the genomes a problem admits (e.g. a bounded real vector). Used for sampling and by operators. |
| **Candidate** | A genome plus identity and lineage: id, parents, the operator that produced it, the step it was proposed at. |
| **Batch** | A sequence of candidates, optionally backed by an array (the fast path). |
| **Strategy** | The algorithm. `ask` proposes candidates; `tell` receives their evaluations. Owns its internal state. |
| **Evaluator** | Turns a batch of candidates into evaluations. A plain function, a vectorised function, or a full episode evaluator. |
| **Evaluation** | The result for one candidate: objectives, constraints, descriptors, cost, raw measurements and status. |
| **Decoder** | Turns a genome into a runnable agent. Lives on the evaluation side. |
| **Environment** | Runs episodes: given agents (by role) and a scenario, returns raw measurements. |
| **Scenario set** | Fixed, seeded instances the agents are evaluated on, split into selection and held-out sets. |
| **Aggregator** | Turns per-scenario measurements into objectives, constraints and descriptors. |
| **Driver** | Owns the loop: ask, evaluate, tell. Also owns budget, concurrency, determinism, failures, recording and checkpoints. |
| **Recorder** | Persists the run: metadata, event log (SQLite), genomes, artifacts, checkpoints. |
| **Backend** | Array namespace, device, precision and random-number generation for numeric work. |

```
          ┌──────────── Driver ─────────────┐
          │ budget · concurrency · determinism · failures · recorder · checkpoints
          │                                  │
  Strategy ──ask──▶ Batch ──▶ Evaluator ──▶ EvaluationBatch ──tell──▶ Strategy
                                  │
                       (episode evaluator)
                  Decoder → Environment × ScenarioSet → Aggregator
```

---

## 3. The central contract: ask/tell

### 3.1 Why

In 0.x the playground owned the loop and called the fitness function from several places (population building, replacement, `Population.update()`). The algorithm decided when evaluation happened and waited for each one.

In the new core, **strategies never call evaluators**. A strategy proposes candidates (`ask`) and receives evaluations (`tell`). A separate **driver** runs the loop. This separation is what makes the following possible without changes to the core:

- synchronous generations, steady-state, asynchronous and distributed evaluation
- evaluations of very different cost: microsecond functions, GPU batches, long simulations, rate-limited LLM calls
- budgets, caching, re-evaluation, lineage and checkpoints handled once, in the driver
- wrapping external algorithms (pycma, evosax, Nevergrad) as strategies
- composite strategies (islands, restarts, portfolios) and, later, co-evolution, multi-fidelity evaluation and Population Based Training

### 3.2 Strategy interface

Strategies are **plain synchronous code**: no I/O and no `async`.

```python
class Strategy(Protocol[G]):
    capabilities: StrategyCapabilities

    def bind(self, problem: ProblemSpec[G], ctx: StrategyContext) -> None:
        """Called once by the driver before the first ask.
        Validates the problem (e.g. a single-objective strategy rejects two objectives)."""

    def ask(self, n: int) -> Batch[G]:
        """Propose candidates. `n` is the driver's suggestion; generation-based
        strategies may return their own generation size."""

    def tell(self, results: EvaluationBatch[G]) -> None:
        """Receive evaluations for previously asked candidates."""

    def should_stop(self) -> bool:
        """Optional strategy-specific termination (e.g. converged)."""

    def state_dict(self) -> StateDict: ...
    def load_state_dict(self, state: StateDict) -> None: ...


@dataclass(frozen=True)
class StrategyCapabilities:
    max_objectives: int | None      # 1 = single-objective; None = any number
    supports_constraints: bool
    tell_mode: Literal["generation", "steady_state", "both"]


@dataclass(frozen=True)
class ProblemSpec(Generic[G]):
    space: Space[G]
    objectives: tuple[Objective, ...]
    constraints: tuple[str, ...]
    descriptors: tuple[str, ...]


@dataclass(frozen=True)
class StrategyContext:
    rng: RandomStream               # the strategy's own random stream (§8)
    backend: Backend                # array namespace, device, precision (§7)
    new_id: Callable[[], CandidateId]   # deterministic id issuer (§4.2)
```

- **`StateDict`** is the type of everything saved by `state_dict()`: a dict with string keys whose values are JSON scalars (`None`, `bool`, `int`, `float`, `str`), lists, dicts with string keys, or arrays of a supported backend (numpy arrays that aren't of dtype `object`, torch tensors). Tuples, sets, numpy scalars and any other object are rejected by `validate_state_dict()`, so a state survives a JSON round trip unchanged. There is no pickle (§10.4).
- **`tell_mode`** tells the driver how to deliver results:
  - `generation`: all results of a batch together (e.g. CMA-ES, generational GA)
  - `steady_state`: results one at a time as they arrive
  - `both`: either
- **Single-objective strategies** accept exactly one objective, or a user-supplied scalarisation (§5.4). Otherwise they fail at `bind` with a clear error, never silently.

### 3.3 The composable genetic-algorithm strategy

Auxein keeps its character as a toolkit of interchangeable operators. A `GeneticAlgorithm` strategy is assembled from:

- **selection**: e.g. stochastic universal sampling with sigma scaling, tournament
- **variation**: mutation (Gaussian, self-adaptive), recombination (arithmetic, uniform)
- **replacement**: generational with elitism, steady-state replace-worst
- **initialisation**: sampling from the search space

Operators are written against the array API (§7) and work on whole populations at once. The operator ideas from 0.x are re-implemented, not ported. The bugs found in 0.x are kept fixed through the tests written for them.

### 3.4 Strategies in the first implementation

- `RandomSearch`: the floor.
- `GeneticAlgorithm`: composable, as above, supporting both tell modes.
- `PycmaStrategy`: a wrapper around pycma, as an optional extra. Its purpose is to prove that external algorithms fit the contract.

More algorithms (native CMA-ES, NSGA-II, MAP-Elites and others) are added later on top of the same contract.

---

## 4. Genomes, candidates and batches

### 4.1 Genomes

- A genome is **whatever the user defines**: a 1-D array, a dataclass, a tree, a string. The core never inspects it.
- **Genomes are immutable.** Operators always create new genomes. Arrays handed out by the framework are marked read-only where the backend allows, so accidental modification fails loudly. No deep copies are needed anywhere.
- **Strategy parameters aren't part of the genome.** Self-adaptive step sizes, covariance matrices and the like live in the strategy's state, keyed by candidate id. Evaluators only ever see what defines the agent.

### 4.2 Candidates

```python
@dataclass(frozen=True)
class Candidate(Generic[G]):
    id: CandidateId                 # deterministic: issued by the driver's id counter
    genome: G
    parents: tuple[CandidateId, ...]
    origin: str                     # e.g. "init", "mutation:self_adaptive", "crossover:arithmetic", "reevaluation"
    step: int                       # the ask round it was proposed in
```

Candidate ids are **deterministic** (a run-scoped counter, `IdIssuer`, not random UUIDs), because evaluation randomness is derived from them (§8). Ids are limited to 32 bits (they are stream keys), so a run can issue about four billion candidates.

### 4.3 Batches and the array fast path

```python
class Batch(Protocol[G]):
    @property
    def candidates(self) -> Sequence[Candidate[G]]: ...

    def as_array(self) -> Array | None:
        """An (n, d) array view of the genomes if the batch is array-backed; otherwise None."""
```

- **Numeric strategies** return array-backed batches (`ArrayBatch`). The genomes *are* rows of an `(n, d)` array on the backend's device, with no per-candidate copies: `candidates` builds `Candidate` objects lazily, and each genome is a row view of the array. The array is exposed read-only where the backend allows. The batch holds a *view* of the array it is given, so the caller's own array keeps its flags. Ids, parents and origins are per-candidate metadata parallel to the rows; `step` is the ask round of the whole batch. `ListBatch` is the general path for structured genomes.
- **Evaluators choose their view.** A vectorised function takes `as_array()` in one call. An agent evaluator iterates over `candidates`.
- `EvaluationBatch` mirrors this. It can be read as a sequence of `Evaluation` records, or as columnar arrays on a backend for strategies that work on arrays: an `(n, k)` objectives matrix (in natural units, or in **minimisation form**, where maximised columns are negated by their declared direction, so strategies never handle signs), constraint violations, total violation, a feasibility mask and status masks. Failed and timed-out evaluations have no usable values: they appear as NaN in the objective and descriptor columns and as infinite violation in the constraint columns, so they are infeasible and rank last (§6.6). A missing value in an `OK` evaluation is an error.

### 4.4 Variable-length structure

- **Numeric problems with structure** (e.g. polynomial degree, number of active rules or sensors) use a **fixed maximum size with structure genes**: binary switches or an integer gene that decides which components are active. The genome length stays constant, so the array fast path applies. A complexity objective (e.g. number of active terms, minimised) turns the problem into a trade-off between accuracy and complexity, which multi-objective strategies can explore.
- **Open-ended structure** (growing networks, program trees, rule lists without a natural maximum, prompts) uses **structured genomes** through the general (non-array) path, with operators that understand the structure.

### 4.5 Search spaces

```python
class Space(Protocol[G]):
    def sample_genomes(self, n: int, rng: RandomStream, backend: Backend) -> Sequence[G] | Array: ...
    def contains(self, genome: G) -> bool: ...
```

Spaces return **genomes**, not batches. Building a batch requires candidate ids, which are issued through the strategy context, so strategies wrap sampled genomes into batches themselves. Array spaces such as `Box` return an `(n, d)` array.

- **First implementation:** `Box`, a bounded real vector with lower and upper bounds per dimension and an optional log scale per dimension (log-scale dimensions are sampled log-uniformly and need a positive lower bound). Samples are guaranteed to lie within `[lower, upper]` in float32 as well as float64: float32 bounds are rounded *inward* (a bound that isn't a float32 number moves to the next float32 inside the box) and samples are clipped to them. A box too narrow or too wide for float32 to represent is an error when sampling in float32. `Box.contains` and `Box.clip` are the membership test and a vectorised repair helper for operators.
- **Later:** integer, categorical, mixed and conditional spaces (e.g. hyperparameter spaces), plus structured spaces for text and trees, added when a use case needs them. The concept lives in the core from the start, so operators and strategies can rely on it.

---

## 5. Evaluations

### 5.1 The evaluation record

```python
class Status(Enum):
    OK = "ok"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class Objective:
    name: str
    direction: Literal["minimise", "maximise"] = "minimise"


@dataclass(frozen=True)
class Evaluation(Generic[G]):
    candidate: Candidate[G]
    status: Status
    objectives: Mapping[str, float]     # natural units; direction declared in ProblemSpec
    constraints: Mapping[str, float]    # violation amount; 0 = satisfied, > 0 = violated
    descriptors: Mapping[str, float]    # how the agent behaved (not optimised)
    cost: Cost                          # wall time plus user-defined units (tokens, money, sim-seconds)
    raw: RawRef | None                  # reference to per-scenario measurements (§6.4)
    error: str | None                   # message for FAILED / TIMEOUT
```

Validation: objective values must be finite when the status is `OK`, and may be non-finite for `FAILED` or `TIMEOUT` evaluations, which strategies never read. Constraint violations are always finite and at least 0. The mappings are copied and made read-only. A `ProblemSpec` needs at least one objective, and names must be non-empty and unique within objectives, constraints and descriptors.

### 5.2 Conventions

- **Directions are explicit per objective.** A **bare number** returned by a fitness function is a single objective that is **minimised**. Users never negate values; strategies convert internally to whatever convention they use.
- **Constraints are violation amounts.** Strategies rank **feasible before infeasible**, and among infeasible candidates prefer smaller total violation. Penalty terms mixed into objectives are discouraged.
- **Descriptors** feed quality-diversity methods and analysis. They're never optimised directly.

### 5.3 Evaluators

```python
class Evaluator(Protocol[G]):
    async def evaluate(self, batch: Batch[G], ctx: EvalContext) -> EvaluationBatch[G]: ...
```

The evaluator interface is asynchronous, but users rarely implement it directly. Built-in evaluators wrap ordinary code:

- `FunctionEvaluator(f)`: `f(genome) -> number | dict`, called per candidate. Synchronous functions are offloaded to a thread or process pool automatically; `async def` functions run natively.
- `VectorisedEvaluator(f)`: `f(X) -> (n,)` or `(n, k)` values on the batch's array. This is the fast path for numeric problems, and runs on GPU when the arrays are on GPU.
- `EpisodeEvaluator(decoder, environment, scenarios, aggregator)`: the agent evaluator (§6).

`EvalContext` carries the backend, deadlines and timeouts, and a factory that derives each candidate's evaluation random stream from its id (§8). It is a factory rather than a list of streams so that a batch of thousands of candidates doesn't create thousands of generators up front.

### 5.4 Scalarisation

A single-objective strategy applied to a multi-objective problem requires an explicit scalarisation supplied by the user (e.g. weighted sum, or a target-based scalarisation). The recorder stores the original objectives regardless.

---

## 6. Agents, environments and scenarios

### 6.1 Episode evaluator

To the core, an evaluator is anything that produces evaluation records. For agents, Auxein provides the **episode evaluator**, composed of:

1. a **decoder**: `genome -> Agent`
2. an **environment** that runs episodes
3. a **scenario set**
4. an **aggregator**: per-scenario measurements → objectives, constraints, descriptors

### 6.2 Environment interface: whole episodes

```python
class Environment(Protocol):
    roles: tuple[str, ...]          # e.g. ("own_ship",) or ("buyer", "seller")

    def run_episode(
        self, agents: Mapping[str, Agent], scenario: Scenario, rng: RandomStream
    ) -> EpisodeResult: ...

    # Optional, for simulators that batch many agents and scenarios in one call:
    # def run_episodes(self, agents: Sequence[Mapping[str, Agent]], scenarios: Sequence[Scenario], rngs) -> Sequence[EpisodeResult]
    # Async variants (`async def`) are accepted for environments that do I/O.


@dataclass(frozen=True)
class EpisodeResult:
    measurements: Mapping[str, float]   # raw: fuel, time, closest approach, success, tokens...
    status: Status
    artifacts: ArtifactRef | None       # optional trajectory, transcript, log
    error: str | None
```

- **The core contract is a whole episode**, plus an optional batched variant. Real simulators often own their own loop (external processes, co-simulation, ROS), LLM agents run their own multi-turn loops, and batched GPU simulators run a population in one call.
- **Step-level environments** (`reset`/`step`) plug in through an adapter that runs the step loop. A Gymnasium adapter is provided as an optional extra; Gymnasium isn't a core dependency.
- **Environments return raw measurements, not scores.** What counts as good is decided by the aggregator, so runs can be re-judged without re-simulating.

### 6.3 Multiple agents and roles

An episode receives agents **by role**. A single-agent problem has one role.

- **Evolved agents among scripted ones** (e.g. other traffic around an evolved vessel) are part of the scenario.
- **Several evolved agents in one episode** (competition, cooperation, co-evolution) are supported by the interface. Deciding who meets whom is the evaluator's responsibility. Full co-evolution support is future work, and needs no interface change.

### 6.4 Scenarios

- A **scenario** is an id plus parameters (initial geometry, sea state, traffic, task instance…).
- A **scenario set** is a fixed list of seeded instances, generated once per run or loaded from a file.
- Every candidate in a run is evaluated on the **same scenarios** (common random numbers), so differences in results reflect the agents, not luck.
- Scenario sets are split into a **selection set** (used for evolution) and a **held-out set** (used only for reporting), to detect overfitting and reward hacking.
- Per-scenario measurements are kept (§9), so results can be re-aggregated later.

### 6.5 Aggregators

An aggregator maps the per-scenario `EpisodeResult`s of one candidate to objectives, constraints and descriptors. It's declarative and swappable. Built-in reductions include mean, worst case, quantile and CVaR (mean of the worst α fraction). For example:

```python
aggregator = Aggregator(
    objectives={"fuel": mean("fuel_used"), "time": mean("time_to_waypoint")},
    constraints={"cpa": worst(lambda m: max(0.0, 0.5 - m["closest_approach_nm"]))},
    descriptors={"mean_speed": mean("mean_speed")},
)
```

### 6.6 Failures

- A failed or timed-out episode or evaluation is **a result, not a crash**. It's recorded with its status and error, and the run continues.
- **Default policy: infeasible.** A failed candidate ranks below every feasible candidate, is kept in the lineage for inspection, and is never silently retried.
- **`fail_fast`** stops the run at the first failure (useful when developing an environment). Custom policies are possible.
- Auxein provides **timeouts** and **process isolation** for evaluations. Sandboxing what an agent is allowed to *do* is the environment's responsibility.
- **Reserved for later:** episodes may report intermediate measurements, enabling multi-fidelity evaluation and early stopping. The interface leaves room for this; it isn't implemented in the first version.

---

## 7. Numeric backend

### 7.1 Array API standard

- Numeric strategies, operators and vectorised evaluators are written against the **Python array API standard** (via `array-api-compat`), using the namespace of their inputs.
- **numpy** is the reference backend and the only required numeric dependency.
- **PyTorch** is the second supported and tested backend. It covers **CUDA** and **Apple Metal (MPS)**.
- **JAX** (CUDA/TPU, JIT) and **MLX** (Apple Silicon, once it implements the standard) are possible optional extras later. The core never depends on them.

### 7.2 Rules for numeric code

1. **No in-place mutation** of arrays.
2. **No global random state.** All randomness goes through the backend's random layer (§8).
3. **Arrays stay on their device** through ask → evaluate → tell. Only metadata (ids, lineage, scalar summaries) moves to the host.
4. **Float32-safe.** Metal doesn't support float64. Every numeric strategy must behave correctly in float32. Numerically delicate code is written with float32 in mind.

### 7.3 Backend configuration

```python
@dataclass(frozen=True)
class Backend:
    name: Literal["numpy", "torch"]
    device: str                         # "cpu", "cuda", "cuda:1", "mps"
    precision: Literal["float64", "float32"]
```

- Default: `numpy`, `cpu`, `float64`. `Backend.for_device(name, device)` picks the default precision of a device (float32 on GPU devices, float64 on the CPU) unless one is given.
- The configuration is validated when the `Backend` is constructed, with clear messages: torch requested but not installed, numpy with a device other than `cpu`, unknown device strings (`cpu`, `mps`, `cuda`, `cuda:<index>` are accepted), `cuda` or `mps` requested but not available (or a CUDA index out of range), and `float64` on `mps`. The availability probes are replaceable, so the validation is tested without a GPU.
- `Backend` offers the array namespace (`xp`, through `array-api-compat`), `dtype`, `int_dtype`, `asarray` (to the backend's namespace, device and dtype), `to_numpy` (a host copy) and `readonly`. `readonly` sets the numpy read-only flag and is a documented no-op for torch, which has no such flag: there immutability is by convention and by tests. `backend_of(array)` infers a `Backend` from an existing array.

### 7.4 Testing

- CI runs numeric tests on **numpy and PyTorch (CPU)**, in **float64 and float32**. On Linux, torch comes from PyTorch's CPU-only wheel index (configured in `pyproject.toml`), which avoids gigabytes of CUDA libraries that the tests never use.
- Real GPU runs (CUDA and Metal) are verified by a small **GPU smoke suite**, run manually or on a self-hosted runner before releases. Standard hosted CI runners don't provide them.

---

## 8. Randomness and reproducibility

- **One seed per run.** The user supplies one seed (`RunSeed`). Independent streams are derived from it with `numpy.random.SeedSequence(entropy=seed, spawn_key=(name_id, *keys))`:
  - one for the strategy (`"strategy"`)
  - one for scenario generation (`"scenarios"`)
  - one per candidate evaluation, derived from the candidate's (deterministic) id (`"evaluation", candidate_id`)
- **Stable names.** A stream name is mapped to an integer with CRC-32 of its UTF-8 bytes, never with Python's `hash()` (randomised per process), so the same seed, name and keys give the same stream in any process on any machine. Keys are integers in `[0, 2**32)`. Two names could in principle collide in CRC-32 (about one chance in four billion); a test pins the names Auxein uses.
- **Evaluation randomness follows the candidate**, not the worker or the time of evaluation. Evaluations are reproducible in any parallel or distributed setup.
- **Backend-native generation.** The random layer (`RandomStream`) wraps numpy's `Generator(PCG64)` on CPU and `torch.Generator` on the target device, seeded from the derived streams. Random numbers are produced where the arrays live. Streams offer `uniform`, `normal`, `integers`, `permutation` and `choice`, returning arrays in the backend's namespace, device and dtype (integers as int64).
- **Torch seed width.** A `torch.Generator` on the CPU is an MT19937 that accepts only 32 bits of seed, so two derived seed sequences can give the same torch stream; the chance is negligible for the few streams a run needs but becomes likely around 65,000 torch streams, e.g. one per candidate evaluation. numpy streams use the full 128 bits, and generators on CUDA and Metal use the full 64-bit seed. **Decision (step 1): the limit is accepted for now** and revisited if a use case needs more than about 65,000 torch-CPU evaluation streams in one run (§15). Evaluators that need many independent streams can draw from numpy streams, which have no such limit.
- **Scope of reproducibility:** identical results for the same seed, backend and precision. Different backends or precisions give different (equally valid) runs.
- **No global random state** anywhere in the package, enforced by a test that forbids numpy's legacy global functions.
- **Checkpoints include all generator states.** `RandomStream.state_dict()` is JSON-serialisable: numpy's bit-generator state dict, or the bytes of `torch.Generator.get_state()` in base64.

### 8.1 Deterministic mode (default)

In steady-state asynchronous runs, results arrive in an order that depends on timing. If the strategy received them in arrival order, two runs with the same seed could diverge.

- **Deterministic mode (default on):** evaluations still run concurrently, but results are delivered to the strategy **in the order the candidates were asked for**. Same seed → identical event log, regardless of worker count or timing. The strategy may occasionally wait for a slow candidate.
- **Throughput mode (opt-in):** results are delivered as they arrive. It's faster under uneven evaluation times, but not reproducible run to run.

---

## 9. Execution: the driver

### 9.1 Asynchronous inside, synchronous outside

- The **driver is asynchronous internally** (asyncio). It manages in-flight evaluations, concurrency limits, timeouts, budgets, delivery order and recording.
- **Strategies are synchronous** (§3.2). **Evaluators** may be synchronous (offloaded to thread or process pools) or asynchronous (run natively). CPU-heavy synchronous evaluations go to process pools; asyncio alone doesn't parallelise CPU work.
- **Users get a synchronous entry point** that works in scripts and in notebooks (where an event loop is already running), plus an async variant for embedding Auxein in async applications.

```python
result = auxein.run(
    strategy=GeneticAlgorithm(...),
    evaluator=VectorisedEvaluator(rastrigin),
    space=Box(lower=-5.0, upper=5.0, dim=10),
    objectives=[Objective("value")],          # default: one minimised objective
    budget=Budget(evaluations=20_000),
    seed=42,
    backend=Backend("numpy", "cpu", "float64"),
    concurrency=8,
    deterministic=True,
    failure_policy="infeasible",
    run_dir="runs/rastrigin-ga",
)

result = await auxein.arun(...)  # same arguments
```

### 9.2 Delivery modes

The driver respects each strategy's `tell_mode`:

- **Generation:** ask a batch, evaluate it concurrently, tell all results together.
- **Steady-state:** keep `concurrency` evaluations in flight. When one completes, tell it (in ask order in deterministic mode) and ask for a replacement.

### 9.3 Budgets and termination

- **Budgets** are measured in **evaluations** (the primary unit), and optionally wall time and cost units (e.g. tokens, money). The run stops when any budget is exhausted, the strategy's `should_stop()` returns true, or the user interrupts.
- Every evaluation is counted, including initialisation and re-evaluations.

### 9.4 Caching and re-evaluation

- **Optional result cache**, keyed by genome content hash, for deterministic evaluators. Off by default for evaluators declared noisy.
- **Re-evaluation is explicit.** A strategy that wants a fresh evaluation of an existing genome (e.g. under noise) proposes it again with `origin="reevaluation"`. It counts against the budget.

### 9.5 Result

`RunResult` gives access to the best candidates (per objective, or the non-dominated set for multi-objective), budget usage, the run directory, and a handle to the recorded run for analysis.

---

## 10. Persistence and observability

### 10.1 Run directory

```
runs/<name>/
  metadata.json          # config, seed, versions, git SHA, backend, precision, problem spec
  events.sqlite          # event log: candidates, lineage, evaluations, checkpoints
  genomes/               # content-addressed genome store (large genomes)
  artifacts/             # optional heavy outputs: trajectories, transcripts, logs
  checkpoints/           # periodic snapshots of strategy, driver and RNG state
```

### 10.2 Event log (SQLite)

- **Append-only.** Written by the driver, which is the single writer, in transactions, so it's safe against crashes.
- Tables (indicative):
  - `candidates`: id, step, origin, genome (inline if small, else a hash reference), created_at
  - `lineage`: parent → child edges
  - `evaluations`: candidate id, status, objectives, constraints, descriptors, cost, raw reference, error, started/finished timestamps
  - `checkpoints`
- **Lineage queries** ("all descendants of X", "the ancestry of the best agent") are single recursive SQL queries.
- **Export** to JSONL or Parquet with one command.

### 10.3 Genomes and artifacts

- Small genomes are stored inline. Large genomes (network weights, long prompts) go into the **content-addressed store**: each distinct genome is saved once under its hash.
- **Heavy artifacts** (trajectories, transcripts) are optional and stored by reference. **Default: kept for failed evaluations only.** Options keep them for the best candidates or a sample.

### 10.4 Checkpoints and resume

- Checkpoints are written **periodically** (default: time-based, every few minutes) and at the end of the run.
- A checkpoint contains strategy state, driver state (counters, budget usage, in-flight candidate ids) and all random-generator states.
- **No pickle.** Strategies implement `state_dict()` / `load_state_dict()`. Arrays are saved in a standard array format, loaded with pickling disabled. Everything else is saved as JSON.
- **On resume**, in-flight candidates are re-issued. Their evaluation randomness is tied to their ids, so a resumed run in deterministic mode produces the same event log as an uninterrupted one.

### 10.5 Recorder interface

The recorder is pluggable. The SQLite run directory is the built-in implementation. Integrations (e.g. MLflow, Weights & Biases) can be added later as additional sinks.

---

## 11. Validation plan

The design is validated against **two deliberately different toy domains**, plus function optimisation and regression problems.

### 11.1 Toy domain A: two-ship encounter (numeric, vectorised)

- **Dynamics:** own ship with first-order Nomoto yaw dynamics (`T·ṙ + r = K·δ`) at constant speed, steering towards a waypoint, with one target vessel on a crossing course.
- **Genome:** a fixed-size controller (heading-controller gains plus avoidance parameters, or a small fixed-size network) in a `Box` space.
- **Scenarios:** encounter geometries, target speeds and headings, currents. Selection and held-out sets.
- **Objectives:** time to waypoint (minimise), rudder effort (minimise).
- **Constraint:** closest point of approach ≥ threshold.
- **Descriptor:** passing side, mean rudder angle.
- **Implementation:** vectorised over candidates × scenarios. It exercises the array fast path, the batched episode interface, the PyTorch backend and float32.

### 11.2 Toy domain B: prompt evolution with a mock LLM (structured, asynchronous)

- **Genome:** a structured prompt configuration (instruction list, example slots, tool flags).
- **Evaluation:** a deterministic mock "LLM" whose answer quality depends on genome features. It injects random latency, occasional failures and timeouts, so it exercises async evaluation, steady-state delivery, deterministic mode and failure policies, at no cost.
- **Operators:** structure-aware text mutations, plus an "LLM-driven mutation" operator interface implemented with the mock.
- **Objectives:** task success (maximise), token cost (minimise).

### 11.3 Function optimisation and regression

- Function optimisation (e.g. Rastrigin) through `VectorisedEvaluator`.
- Linear and logistic regression.
- Polynomial regression with a fixed maximum degree, structure genes and a complexity objective (§4.4).

These are rewritten as notebooks and documentation for the new core.

### 11.4 Acceptance criteria

1. **Generality:** both toy domains and all problems in §11.3 run on the same core, with no special cases in the core.
2. **Benchmark:** with a harness adapter for the new core, the `GeneticAlgorithm` strategy at equal evaluation budgets is **not statistically worse than the `v0.2.0` baseline** on any problem and dimension of the benchmark suite. Its **overhead per evaluation is lower**.
3. **Determinism:** in deterministic mode, the same seed produces an identical event log across synchronous and asynchronous evaluators and any worker count.
4. **Resume:** a run killed and resumed from a checkpoint produces the same event log as an uninterrupted run.
5. **Backends:** numeric tests pass on numpy and PyTorch (CPU) in float64 and float32. The GPU smoke suite passes on CUDA and Metal.
6. **Failures:** runs with injected failures and timeouts complete, with every failure recorded and handled per the configured policy.

---

## 12. Package layout (indicative)

```
auxein/
  core/            # Candidate, Batch, Evaluation, Objective, ProblemSpec, protocols
  spaces/          # Space protocol, Box
  backend/         # Backend, array-API helpers, precision
  random/          # seed streams, backend-native generators
  driver/          # run/arun, budgets, delivery modes, failure policies, caching
  strategies/
    random_search.py
    ga/            # GeneticAlgorithm + operators (selection, mutation, recombination, replacement)
    external/      # PycmaStrategy (optional extra)
  evaluators/      # FunctionEvaluator, VectorisedEvaluator, EpisodeEvaluator
  environments/    # Environment protocol, step-level adapter, Gymnasium adapter (optional extra)
  aggregators/     # Aggregator, reductions (mean, worst, quantile, CVaR)
  recording/       # Recorder protocol, SQLite run directory, genome store, checkpoints, export
examples/          # toy domains A and B, function optimisation, regression
docs/design/core.md
```

---

## 13. Versioning

- **No backward compatibility** with 0.x. Old modules are replaced in place after the `v0.2.0` tag.
- The new core is versioned **0.3.0** onwards, and stays on **0.x** until the acceptance criteria (§11.4) are met and the API has settled.

---

## 14. Suggested implementation order

Each step ends with passing tests and is a candidate for its own Claude Code prompt and PR.

1. **Foundations:** core types, `Space`/`Box`, backend, random streams, deterministic ids. The backend and random layers are tested on numpy and PyTorch (CPU) from the start, so the abstraction is proven on two backends early.
2. **Minimal driver:** synchronous generation mode, `FunctionEvaluator`, `VectorisedEvaluator`, budgets, SQLite recorder (metadata + event log).
3. **Strategies:** `RandomSearch`, `GeneticAlgorithm` with array-based operators, plus a benchmark-harness adapter. First comparison with the `v0.2.0` baseline.
4. **Asynchrony:** async driver internals, steady-state delivery, deterministic mode, timeouts, failure policies, process isolation.
5. **Checkpoints and resume:** `state_dict` for strategies, driver and RNG state, the genome store.
6. **Agents:** `EpisodeEvaluator`, `Environment` protocol, scenario sets, aggregators. Toy domain A.
7. **Structured genomes:** structure-aware operators, toy domain B with the mock LLM.
8. **PyTorch everywhere:** extend numpy and PyTorch coverage in float64 and float32 to every numeric strategy, operator and evaluator, and add the GPU smoke suite.
9. **External strategies and examples:** `PycmaStrategy`, step-level and Gymnasium adapters, rewritten function-optimisation and regression notebooks.

---

## 15. Open questions

- **Late results in throughput mode:** how strategies should treat results for candidates asked several rounds earlier (accept, discount, or ignore). It will probably be declared per strategy.
- **Scalarisation API:** which scalarisations to provide built in, beyond weighted sums.
- **Distributed execution:** whether to offer Ray or Dask as evaluator execution backends, and when.
- **Quality-diversity:** the archive interface needed for MAP-Elites-style strategies (descriptor bins, insertion rules), and whether archives become a core concept.
- **Multi-fidelity:** the shape of partial-result reporting and how strategies decide to stop or continue an evaluation.
- **Co-evolution:** how matchmaking between populations is configured.
- **Noise handling:** built-in support for repeated evaluation and averaging, and how it interacts with caching.
- **GPU CI:** whether to set up a self-hosted runner for CUDA and Metal.
- **Torch CPU stream independence:** the 32-bit seed limit of torch CPU generators is accepted for now (§8). If it becomes a problem: keep evaluation streams on numpy, or generate on the host and transfer.
