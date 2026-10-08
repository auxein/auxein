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
6. **Everything can be recorded.** A run is designed to be fully reconstructable: every candidate's origin and every evaluation's outcome can be inspected after the fact. Recording is opt-in (a run only writes to disk when it is given a `run_dir`), and a run without one says so with a warning, because it can't be reconstructed.

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

Auxein keeps its character as a toolkit of interchangeable operators. A `GeneticAlgorithm` strategy is assembled from operator objects, each a small class with a protocol of its own (in `auxein.strategies.ga`), all working on whole arrays through the backend's array namespace, with no Python loops over individuals or genes:

- **parent selection**: tournament (size `k`, default 2), and stochastic universal sampling with sigma scaling
- **recombination**: intermediate, uniform, and none (asexual), plus a crossover probability `p_c`
- **mutation**: Gaussian with a fixed step, and self-adaptive with one step size per individual or per gene
- **bounds repair**: clip and reflect
- **initialisation**: sampling from the search space

The operator ideas from 0.x are re-implemented, not ported, and the limitations of 0.x are fixed by design: the number of offspring is exact, the two parents of a child are distinct members, every candidate is evaluated exactly once (the population is never re-scored), and step sizes have a lower bound.

**Failed candidates (§6.6)** have infinite violation and rank last, so a failed child never displaces a member that did not fail, and **a member that failed is never chosen as a parent while any other exists**: tournaments are drawn from the members that did not fail, SUS gives failed members no weight, and a partner is always another such member. With fewer than two of them, the GA samples random candidates instead of breeding, as it does at the start. Failed members never count in the convergence test.

**Survivor selection is "plus" only.** The population of size μ always holds the best μ of everything evaluated so far, parents and offspring pooled. In a steady-state view each child replaces the current worst member if it is better; inserting children one at a time and taking the best μ of μ + λ give the same population, so one mechanism serves any way of telling results (generations, or one at a time). Comma selection and plain generational replacement are not in this version.

**One ranking is used everywhere** (survivors, tournaments, elites): feasible before infeasible, then a lower total violation, then a lower objective in minimisation form, then a lower candidate id. Failed and timed-out evaluations appear as NaN objectives and infinite violation, so they rank last without a special case.

**Breeding.** Parent selection, recombination, mutation, bounds repair, in that order: recombine first, then mutate (unlike 0.x).

- *Tournament:* each parent is the best of `k` members drawn uniformly at random. *SUS:* the weight of a feasible member is `max(g − (mean(g) − c·std(g)), 0)` with goodness `g = −value` and `c = 2`; infeasible members have weight 0 unless none is feasible, when the weights come from a lower violation in the same way; if the weights are all zero or not finite the selection is uniform.
- *Distinct parents:* when the second parent of a child equals the first, it is replaced by a uniformly random other member (at least two members are needed to breed).
- *Recombination* mixes genes as `w·a + (1 − w)·b`: intermediate draws `w ~ U(0, 1)` per child (or per gene), uniform draws each gene from either parent with probability ½, none copies the first parent. A child that does not cross over (probability `1 − p_c`, or always with no recombination) copies its first parent and records one parent; otherwise two.
- *Mutation* steps are relative to the box width, per dimension, so the operators are scale-free; they act in linear space, also on log-scale dimensions. Self-adaptive mutation updates the step first (`σ' = σ·exp(τ·N(0, 1))`, `τ = 1/√d`; per gene, `σᵢ' = σᵢ·exp(τ'·N(0, 1) + τ·Nᵢ(0, 1))` with `τ' = 1/√(2d)` and `τ = 1/√(2√d)`), bounded below by `σ_min` (default 10⁻¹² of the box width), then moves the genes.
- *Bounds repair:* `clip` (via `Box.clip`) or `reflect` (a true reflection, however far out a gene is).

**Step sizes are strategy state, not genome** (§4.1). They are kept per candidate inside the strategy, as arrays aligned with the population. A child inherits them by the same recombination as its genes, as a weighted geometric mean with the same weights (the geometric mean when the weights are equal; one parent's exactly with weights 0 or 1), then they mutate; the steps of non-survivors are discarded.

**The ask/tell behaviour.** The first `ask` returns the whole initial population (origin `"init"`), whatever `n` the driver suggests. Afterwards `ask` returns exactly λ children (`offspring_size`), or exactly `n` when it is `None`, with their parents recorded and origins that name the operators, e.g. `"tournament+intermediate+self_adaptive"` (`copy` stands for the recombination of a child that copies a parent). Children that were asked for and not yet told are pending, with their parents' step sizes inherited; asking again before results arrive is allowed, and children are bred from the population as it is. Children that are never told (budget truncation) are dropped. If fewer than two members exist, because the initial candidates are still being evaluated, `ask` returns more random candidates. `tell` accepts any grouping of pending children. The strategy is single-objective, supports constraints, needs a `Box` space, and `should_stop` is false unless a convergence tolerance is given (the objective spread of a full population is below it, and every step size is at its floor).

**The default configuration** (`GeneticAlgorithm()`) was chosen by benchmark among three candidates on instances other than those of the comparison with 0.2.0: μ = λ = 50, tournament selection (k = 2), intermediate recombination, one self-adaptive step size per individual (initial step 0.1 of the box width, floor 10⁻¹²) and clipping. The candidates (A: this one; B: as A with one step size per gene; C: as A with SUS and sigma scaling) ended within 0.5 of each other in mean rank of the median final error (C 1.70, A 2.10, B 2.20; an earlier run of the same selection gave A 1.80, B 2.10, C 2.10), a near tie, so the simplest configuration was chosen. The results are in `benchmarks/reports/ga-default-selection/`.

### 3.4 Strategies in the first implementation

- `RandomSearch`: the floor (done in step 2).
- `GeneticAlgorithm`: composable, as above, supporting both tell modes (done in step 3a).
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

Validation: objective values must be finite when the status is `OK`. A `FAILED` or `TIMEOUT` evaluation carries none (the batch views fill them in as NaN, with infinite constraint violation, which strategies never read).

**Building a non-`OK` evaluation.** There is one way: `Evaluation.failed(candidate, status, error, wall_time, units)` with `status` `FAILED` or `TIMEOUT`. `error` is always a non-empty text that says what happened:

- an **exception** in user code: its type, message and the full formatted traceback, including the chain of causes (for a worker process, the traceback of the worker, which is sent as text);
- a **timeout**: that the evaluation timed out, and the limit (`wall_time` is the limit);
- a **crash**: that the worker process died, and its exit code or the signal that killed it;
- a **non-finite objective**: which objective was NaN or infinite, since that is how a diverged simulation shows up.

The same tools build and read these records everywhere: `describe_exception(error)` formats the text, `EvaluationBatch` and the result tracker already skip non-`OK` rows, and the recorder stores the status and the error. `RunResult.status_counts` and the run's summary count the evaluations per status. Constraint violations are always finite and at least 0. The mappings are copied and made read-only. A `ProblemSpec` needs at least one objective, and names must be non-empty and unique within objectives, constraints and descriptors.

### 5.2 Conventions

- **Directions are explicit per objective.** A **bare number** returned by a fitness function is a single objective that is **minimised**. Users never negate values; strategies convert internally to whatever convention they use.
- **Constraints are violation amounts.** Strategies rank **feasible before infeasible**, and among infeasible candidates prefer smaller total violation. Penalty terms mixed into objectives are discouraged.
- **Descriptors** feed quality-diversity methods and analysis. They're never optimised directly.

### 5.3 Evaluators

```python
class Evaluator(Protocol[G]):
    async def evaluate(self, batch: Batch[G], ctx: EvalContext[G]) -> EvaluationBatch[G]: ...
```

The evaluator interface is asynchronous, but users rarely implement it directly. Built-in evaluators wrap ordinary code:

- `FunctionEvaluator(f, uses_rng=False)`: `f(genome)`, or `f(genome, rng)` with `uses_rng=True`, called per candidate, where `rng` is the candidate's own evaluation stream (§8). `f` may be a plain function or an `async def`. Concurrency and executors are described below. Each candidate's wall time is recorded in its `Cost`.
- `VectorisedEvaluator(f, uses_rng=False)`: `f(X)`, or `f(X, rng)`, once per batch with `X = batch.as_array()`; it needs an array-backed batch. This is the fast path for numeric problems, and runs on GPU when the arrays are on GPU. The batch's wall time is split equally across its candidates. With `uses_rng=True` it receives one stream per batch (§8). **`concurrency` and `executor` do not apply to it**: there is one call per batch, on the driver's thread, and the parallelism is inside the array operation. Under steady-state delivery (§9.2) it would be called with one candidate at a time, which works but defeats its purpose, so the driver emits a `SteadyStateVectorisationWarning` once per run.
- `EpisodeEvaluator(decoder, environment, scenarios, aggregator)`: the agent evaluator (§6).

**What a fitness function returns.** The rules are the same for every evaluator, and are implemented in one normalisation function:

- A **bare number** (a Python int or float, a numpy scalar, or a 0-d array of any supported backend) is the single objective of a problem with **exactly one objective**, minimised or maximised as declared. A bare number for a problem with several objectives, or with declared constraints or descriptors, is an error that says so.
- Anything richer **must** use an explicit result object: `Result(objectives, constraints, descriptors, cost)` per candidate, or `BatchResult` for a vectorised function (the same fields, each a mapping of name to an array of shape `(n,)`). A result must match the problem exactly: every declared objective, constraint and descriptor present, and no unknown name, which catches typos. `cost` holds user-defined units (tokens, money, simulated seconds).
- **Plain dicts are rejected** with a `TypeError` that points to `Result`: a dict can't say which keys are objectives, constraints or descriptors, so there is no flat-dict format.
- A vectorised function may return an array of shape `(n,)` (a single objective) or `(n, k)` (`k` declared objectives, columns in declared order) on any supported backend, or a `BatchResult`.
- A **non-finite objective value** is not an error in the contract but a **failed evaluation** (`FAILED`, with a message that names the objective): a diverged simulation typically shows up this way, and it should rank last, not stop the run. In a vectorised batch only the rows concerned fail.

**Failures.** The run's `failure_policy` (§6.6) decides what the built-in evaluators do when user code fails, and they read it from `EvalContext.failure_policy`:

- under **`infeasible`** (the default) an exception in the function, a timeout, or a worker process that died becomes a `FAILED` or `TIMEOUT` evaluation with an informative `error`, and the other candidates are unaffected. For a `VectorisedEvaluator` an exception fails **every candidate of the batch**, each with the same error.
- under **`fail_fast`** the first failure raises an `EvaluationError` that names the candidate id(s) and chains the original exception; the other evaluations in progress in the same batch are cancelled first.
- **misconfiguration** raises whatever the policy (§6.6). The driver is a backstop for both policies, so custom evaluators are covered too.

**Concurrency and executors.** `run(..., concurrency=N, executor=...)`:

- `concurrency` (default 1) is the maximum number of evaluations in progress at once. It is a resource setting: in deterministic mode it never changes what the strategy sees (§8.1).
- `executor` says where **synchronous** user functions run: `"inline"` (the driver's own thread, with no hand-off, and the least overhead), `"thread"` (daemon threads), `"process"` (up to `concurrency` worker processes) or `"auto"` (the default: inline when `concurrency == 1` and there is no `timeout`, threads otherwise). **Processes are never chosen automatically**, because they change what user code may do.
- **`async def` functions always run natively** on the driver's event loop, limited by `concurrency`, whatever the executor. `executor="process"` with an `async def` function is an error. An inline executor does not stop `async def` functions from overlapping: it only means synchronous ones run one after the other.
- **Pools belong to a run**: they are created when it starts and shut down when it ends, whatever ends it (normal end, an exception, `KeyboardInterrupt`, task cancellation). Worker processes are stopped, and killed if they are busy or do not stop promptly, so **no orphan process remains**. Threads are daemons, and Python cannot stop one, so a function still running in a thread when the run ends is left to finish in the background: it never blocks the end of the run or of the interpreter.
- **Processes use `spawn` on every platform**, which is what macOS and Windows do and which does not inherit the parent's state. The function and its arguments must be picklable, and the function importable by name in the worker, so lambdas, local functions and functions defined in a notebook or in a script's `__main__` fail (scripts also need the usual `if __name__ == "__main__":` guard). That is detected and reported as an `ExecutorError` that explains the fixes: define the function at the top level of a module, or use `executor="thread"`. A candidate's `RandomStream` is picklable, so `uses_rng=True` works in workers and gives the same numbers as everywhere else. What the function returns is sent back and turned into an `Evaluation` **in the parent**, so validation errors look the same with every executor.
- **The process pool is Auxein's own**, not `concurrent.futures.ProcessPoolExecutor`, which cannot kill one task and breaks the whole pool when one worker dies. It has up to `concurrency` workers, started with `spawn` **lazily** (when work first needs one, and again to replace a worker that was killed or died), each evaluating **one call at a time** and talking to the parent over its own pipe. A worker says `ready` once it is up: one that cannot start (typically a script without the `if __name__ == "__main__":` guard) is a misconfiguration and raises an `ExecutorError`, not a failed candidate. An exception in the function is re-raised in the parent with the worker's traceback chained as text. `ctx.call` is unchanged for user-written evaluators. On the same machine the pool costs about 56 µs per call against 149 µs for `ProcessPoolExecutor` at `concurrency=1` (30 against 89 µs at 4).
- **Timeouts** (`run(..., timeout=seconds)`, per evaluation, default none; `EvalContext.timeout` carries it). How hard a timeout is depends on where the function runs, because Python cannot stop a thread:
  - **`async def`: hard.** The evaluation is cancelled, and recorded as `TIMEOUT`. A `TimeoutError` raised by the function itself is an ordinary failure.
  - **`executor="process"`: hard.** The worker is killed (terminated, then killed if needed) and replaced; evaluations on the other workers are unaffected. The clock starts when the call is handed to a started worker, so spawning a worker is not counted.
  - **`executor="thread"`: soft.** The evaluation is recorded as `TIMEOUT` and its result, if it ever arrives, is discarded; the thread cannot be stopped and goes on in the background (it is a daemon, so it never blocks the interpreter from exiting). An `AbandonedEvaluationWarning` explains this once per run, and a second one at the end reports how many abandoned evaluations were still running (also in the summary as `abandoned_evaluations`). A concurrency slot is freed as soon as an evaluation times out, so `concurrency` keeps meaning "evaluations the run is waiting for"; abandoned threads do not count, so more threads than `concurrency` can exist temporarily.
  - **Inline: not possible** for a synchronous function, since it blocks the event loop. With `executor="auto"` a `timeout` resolves to threads even for `concurrency == 1`; asking for `executor="inline"` together with a `timeout` is an error at start-up that lists the options.
  - **`VectorisedEvaluator` does not support timeouts** in this version (setting one is an error at start-up): it runs on the driver's thread, and a vectorised call is normally cheap.
  - A timed-out evaluation counts towards the evaluation budget, and its wall time is the timeout. `EvalContext.deadline` (an absolute deadline for a batch) was removed: nothing needs it, since every limit is per evaluation.
- **A worker that dies** (a segfault, `os._exit`, being killed by the OS for memory) fails the candidate it was evaluating, as `FAILED` with an error that says the worker process died and gives the exit code or the signal. The worker is replaced and the other evaluations carry on.
- `FunctionEvaluator` with `concurrency == 1`, the inline executor and no timeout keeps the sequential fast path. Otherwise the candidates of a batch are evaluated concurrently and returned **in ask order**, whatever order they finish in.
- **For user-written evaluators**, `EvalContext` offers `await ctx.call(fn, *args)`, which runs a synchronous function where the run's executor says, and `ctx.concurrency`, the limit an evaluator must respect. `async def` code needs neither: it simply awaits.

`EvalContext` carries the problem specification (so that what user code returns can be checked against it), the backend, the executor with `ctx.call` and `ctx.concurrency`, the failure policy and the timeout, and two factories that derive evaluation random streams from candidate ids: `rng_for(id)` for a candidate, and `batch_rng_for(first_id)` for a vectorised batch (§8). They are factories rather than lists of streams so that a batch of thousands of candidates doesn't create thousands of generators up front.

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

- A failed or timed-out episode or evaluation is **a result, not a crash**. It's recorded with its status and error (§5.1), and the run continues.
- **`failure_policy`** (a `run` argument):
  - **`"infeasible"` (the default).** An exception in user code, a timeout or a worker crash produces an `Evaluation` with status `FAILED` or `TIMEOUT` and an informative `error`. Strategies receive it in `tell`; it ranks below every feasible candidate (NaN objectives, infinite violation), is kept in the lineage for inspection, and is **never retried** (there are no retries of any kind).
  - **`"fail_fast"`** stops the run at the first `FAILED` or `TIMEOUT`, with an error that names the candidate and chains the original exception (useful when developing an environment). Custom policies are possible later.
- **Misconfiguration always fails the run, whatever the policy**, because it is not a result and means nothing will work: executor errors (a function or arguments that cannot be pickled, results that cannot be sent back, workers that cannot start), return-value contract violations (wrong type, a plain dict, missing or unknown names), driver validation errors (`StrategyError`, `EvaluatorError`), and `KeyboardInterrupt` and other `BaseException`s that are not `Exception`s.
- **A non-finite objective value** is an evaluation failure (`FAILED`, with a message saying which objective), not misconfiguration.
- **Safety net, under `infeasible`.** A policy that swallows errors needs a guard against hiding a bug:
  - **The first failure is shown at once**: an `EvaluationFailureWarning` (a `UserWarning`), once per run, with the candidate id, the status and the full traceback of the original exception (or the timeout or crash details). It is emitted by the driver, so it covers every evaluator.
  - **The run stops if everything fails at the start.** If the first `N` evaluations to be told are all not `OK`, the run stops with an `AllEvaluationsFailedError` that includes the first failure's details and explains that this is almost certainly a bug in the evaluation, not a result, and how to turn the check off. `N` is `initial_failure_guard` (default 10; `None` disables it). The check ends as soon as one of the first `N` succeeds. "First to be told" is ask order in deterministic mode, so the outcome does not depend on `concurrency` or the executor. A run too short to reach `N` in which every evaluation failed is stopped at its end. The recording is finalised with status `failed`.
- **Process isolation and crashes.** With `executor="process"` an evaluation runs in a worker process of Auxein's own pool (§5.3). A timeout kills that worker; a crash fails only the candidate it was evaluating; both are replaced, and the other evaluations are unaffected. Sandboxing what an agent is allowed to *do* is the environment's responsibility, and there are no memory or CPU limits on workers.
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
  - one per candidate evaluation, derived from the candidate's (deterministic) id (`"evaluation", candidate_id`); **always numpy-backed**, whatever the run's backend (see below)
- **Stable names.** A stream name is mapped to an integer with CRC-32 of its UTF-8 bytes, never with Python's `hash()` (randomised per process), so the same seed, name and keys give the same stream in any process on any machine. Keys are integers in `[0, 2**32)`. Two names could in principle collide in CRC-32 (about one chance in four billion); a test pins the names Auxein uses.
- **Evaluation randomness follows the candidate**, not the worker or the time of evaluation. Evaluations are reproducible in any parallel or distributed setup.
- **Exception for vectorised evaluators.** A vectorised function draws its randomness for the whole batch at once, so per-candidate streams would be unusable. It receives **one stream per batch**, derived from the id of the batch's first candidate (`SeedSequence` key `("evaluation-batch", first_id)`). That is deterministic because the composition of a batch is. Per-candidate streams remain the rule for per-candidate evaluators.
- **Backend-native generation.** The random layer (`RandomStream`) wraps numpy's `Generator(PCG64)` on CPU and `torch.Generator` on the target device, seeded from the derived streams. Random numbers are produced where the arrays live. Streams offer `uniform`, `normal`, `integers`, `permutation` and `choice`, returning arrays in the backend's namespace, device and dtype (integers as int64).
- **Per-candidate evaluation streams are numpy-backed.** `ctx.rng_for(id)` returns a numpy stream (with the run's float precision) on every backend. Evaluating one candidate is host-side Python anyway, and these are the only streams a run creates by the tens of thousands. numpy streams use the full 128 bits of the derived seed, which settles the limit of torch CPU generators, an MT19937 that accepts only 32 bits of seed and would make two derived streams collide around 65,000 of them (decided in step 4a). numpy streams are also picklable, so a candidate's stream can be sent to a worker process. The strategy stream and the vectorised per-batch stream (`batch_rng_for`) stay backend-native: a run creates only a few, and they feed array operations on the target device.
- **Scope of reproducibility:** identical results for the same seed, backend and precision. Different backends or precisions give different (equally valid) runs.
- **No global random state** anywhere in the package, enforced by a test that forbids numpy's legacy global functions.
- **Checkpoints include all generator states.** `RandomStream.state_dict()` is JSON-serialisable: numpy's bit-generator state dict, or the bytes of `torch.Generator.get_state()` in base64.

### 8.1 Deterministic mode (default)

In steady-state runs, results arrive in an order that depends on timing. If the strategy received them in arrival order, two runs with the same seed could diverge. For the event log to be identical whatever the number of workers (§11.4, criterion 3), the **algorithmic** behaviour must not depend on it, so two things are kept apart:

- **The in-flight window `W`**: the number of candidates asked for but not yet told. It is an algorithmic setting and equals **`batch_size`**, fixed per run.
- **`concurrency`**: how many of those candidates are being evaluated at once. It is a resource setting. If `concurrency < W`, the rest wait in a queue.

**Deterministic mode (default, `deterministic=True`)** in steady-state delivery:

- Results are told **one at a time, in ask order**: a finished result is held until every earlier-asked candidate has been told.
- The driver asks **only right after a tell**, whenever the window has room. It tells results one by one rather than "whatever is ready", because how many results are ready at a given moment depends on timing, and the number of candidates a strategy is asked for would depend on it.
- The sequence of `ask(n)` and `tell(...)` calls the strategy sees, and so the event log, therefore depends only on the seed and `batch_size`, never on `concurrency`, the executor, sync or async functions, or timing. Evaluations still run concurrently, and the strategy may occasionally wait for a slow candidate (head-of-line blocking): the accepted cost.
- Generation delivery needs no such rule: it is deterministic by construction.
- The guarantee holds **given the same evaluation outcomes**: failures that depend on the candidate (an exception for a given genome or stream, a non-finite objective) are part of the log and stay deterministic, and the guard (§6.6) stops the same run everywhere. Outside it are **timeouts**, which depend on timing, budgets measured in wall time or cost units (§9.3), and the text of an error, whose traceback shows different frames inline, in a thread and in a worker process.

**Throughput mode (`deterministic=False`)** in steady-state delivery: results are told **as they finish**, and the window is refilled as soon as it has room. It is faster when evaluation times vary, but not reproducible run to run. Results that finish at the same moment are told in ask order. Generation delivery has nothing to reorder, so it is identical in both modes.

---

## 9. Execution: the driver

### 9.1 Asynchronous inside, synchronous outside

- The **driver is asynchronous internally** (asyncio). It manages in-flight evaluations, concurrency limits, timeouts, budgets, delivery order and recording.
- **Strategies are synchronous** (§3.2). **Evaluators** may be synchronous (offloaded to thread or process pools) or asynchronous (run natively). CPU-heavy synchronous evaluations go to process pools; asyncio alone doesn't parallelise CPU work.
- **Users get a synchronous entry point** that works in scripts and in notebooks (where an event loop is already running: the driver then runs on a fresh loop in a dedicated thread, and `run()` waits for it), plus an async variant for embedding Auxein in async applications.

```python
import auxein

result = auxein.run(
    strategy=auxein.GeneticAlgorithm(),
    evaluator=auxein.VectorisedEvaluator(rastrigin),
    space=auxein.Box(-5.12, 5.12, dim=10),
    objectives=[auxein.Objective("value")],   # default: one minimised objective called "value"
    constraints=(), descriptors=(),
    budget=auxein.Budget(evaluations=20_000), # also wall_time (seconds) and cost (limits per cost unit)
    seed=42,
    backend=auxein.Backend("numpy", "cpu", "float64"),
    batch_size=64,                            # candidates per ask; in steady-state delivery, the in-flight window (§8.1)
    concurrency=1,                            # evaluations in progress at once (a resource setting, §5.3)
    executor="auto",                           # "inline" | "thread" | "process" | "auto" (inline for 1, else threads)
    delivery=None,                            # "generation" | "steady_state" | None (the strategy's tell mode decides)
    deterministic=True,                       # steady-state: tell in ask order (False: as results finish, §8.1)
    failure_policy="infeasible",              # or "fail_fast": what an exception, a timeout or a crash does (§6.6)
    timeout=None,                             # seconds per evaluation; hard for async def and processes, soft for threads (§5.3)
    initial_failure_guard=10,                 # stop if the first 10 evaluations all fail (None: off, §6.6)
    run_dir="runs/rastrigin-ga",              # opt-in recording; without it nothing is written and a warning is emitted
    name="rastrigin-ga",                      # defaults to the directory name
)

result = await auxein.arun(...)  # same arguments
```

The common case needs the one import. The top-level API is deliberately short: `run`, `arun`, `Budget`, `RunResult`, `Objective`, `Result`, `BatchResult`, `Status`, `Box`, `Backend`, `FunctionEvaluator`, `VectorisedEvaluator`, `RandomSearch`, `GeneticAlgorithm`, `open_run` and `RecordingDisabledWarning` (plus `__version__`). Everything else is imported from its subpackage: `auxein.core` (candidates, batches, evaluations, the protocols), `auxein.strategies.ga` (the operators), `auxein.backend`, `auxein.random`, `auxein.spaces`, `auxein.driver`, `auxein.evaluators`, `auxein.recording`.

- **Recording is opt-in.** Without `run_dir` nothing is written to disk, and the run emits a `RecordingDisabledWarning` (a `UserWarning`) once per run, with a stack level that points at the user's call. `warnings.filterwarnings("ignore", category=RecordingDisabledWarning)` silences it. A `run_dir` that already exists and isn't empty is refused, so runs never mix.
- Results are delivered by generation or in steady state (§9.2), with up to `concurrency` evaluations in progress. The defaults (`concurrency=1`, generation delivery) keep today's sequential behaviour and cost. Failures are handled by `failure_policy` (§6.6), and `timeout` limits one evaluation.
- `run()` also accepts a `clock` (the time source of the wall-time budget, injectable for tests).

### 9.2 Delivery modes

`delivery` is `"generation"`, `"steady_state"` or `None`. `None` means generation, unless the strategy's `tell_mode` is `steady_state`. Asking for a mode the strategy does not support (its `tell_mode` is neither `both` nor that mode) is an error at start-up.

- **Generation:** ask a batch (`min(batch_size, remaining budget)`), evaluate it with up to `concurrency` evaluations in progress, tell all results together in ask order.
- **Steady-state:** the driver keeps up to **`W = batch_size`** candidates in flight (asked, not yet told).
  - Asked candidates go into a queue; at most `concurrency` are evaluated at once, each as a **one-candidate batch** through `evaluator.evaluate`, so every evaluator works unchanged.
  - **When to ask:** whenever the window has room, for `min(room, remaining budget − in flight)` candidates. If a strategy returns more than asked (a generation-sized strategy such as the GA with a fixed `offspring_size`), the extra candidates stay in the queue: they count towards the window and the budget, and are evaluated in ask order. Nothing is asked while the window is full.
  - **Told** one at a time: in ask order in deterministic mode, as they finish in throughput mode (§8.1).
  - Each asked batch is validated as in §9.3 (ids issued by this run and new, steps that never go back).
- **Cancellation.** On `KeyboardInterrupt`, an exception or task cancellation, evaluations in progress are cancelled, the pools are shut down, the recording is finalised with status `interrupted` or `failed`, and the exception is re-raised. No task, thread or process outlives the run.

### 9.3 Budgets and termination

- **Budgets** (`Budget(evaluations, wall_time, cost)`) are measured in **evaluations** (the primary unit), and optionally wall time (seconds) and cost units (limits on the summed user-defined units, e.g. tokens, money). At least one limit is required. The run stops when any budget is exhausted, the strategy's `should_stop()` returns true, or the user interrupts.
- Every evaluation is counted, including initialisation and re-evaluations.
- **The evaluation budget is a hard limit.** The driver asks for `min(batch_size, remaining)` candidates; if a strategy returns more than remain, only the remaining candidates are evaluated (in ask order) and recorded, and the run ends **without telling the strategy** about the incomplete batch. Every evaluated candidate counts towards the result.
- **Under steady-state delivery** the evaluation limit is just as hard: the driver never starts more evaluations than remain, and a surplus returned by the strategy beyond the remaining budget is dropped before it is queued: never evaluated, never told, no row in the recording. Told evaluations are what count towards the limit.
- **A timed-out evaluation counts towards the evaluation budget** (and so does any failed one), and its wall time is the timeout.
- **Wall-time and cost budgets are checked between batches** (generation delivery): the batch in progress completes, so they can be exceeded by up to one batch. **Under steady-state delivery** they are checked before every ask and every start: once one is exhausted (or the strategy asks to stop) the driver stops asking, **evaluations already in progress finish and are told** under the delivery rules, and candidates still queued are dropped (never evaluated, told or recorded). Which candidates were in progress at that moment depends on the clock, so a run that ends on a wall-time or cost budget is not reproducible.
- **Stop reasons:** `budget:evaluations`, `budget:wall_time`, `budget:cost:<unit>` and `strategy`, checked in that order before each ask. On `KeyboardInterrupt` or any exception the recording is finalised with status `interrupted` or `failed`, and the exception is re-raised.
- **The driver validates what it is handed.** An asked batch must be non-empty, with ids that this run issued and has not seen before (ids are never reused, not even for re-evaluations) and a single step that never goes back. The results of an evaluator must be one evaluation per candidate, in ask order. Violations raise `StrategyError` and `EvaluatorError`.

### 9.4 Caching and re-evaluation

- **Optional result cache**, keyed by genome content hash, for deterministic evaluators. Off by default for evaluators declared noisy.
- **Re-evaluation is explicit.** A strategy that wants a fresh evaluation of an existing genome (e.g. under noise) proposes it again with `origin="reevaluation"`. It counts against the budget.

### 9.5 Result

`RunResult` holds the stop reason, the evaluations used, the wall time and the run directory, plus:

- **`best`** (single objective only): the best `OK` evaluation. Feasible beats infeasible, then a lower total violation, then a lower objective in minimisation form (the declared direction is respected), with ties going to the earliest id. `None` for several objectives, or if no `OK` evaluation exists.
- **`pareto_front`**: the non-dominated feasible `OK` evaluations in minimisation form, maintained incrementally as an archive, sorted by candidate id. For one objective it is the best feasible evaluation.
- **`trace`** (single objective only): `(evaluations_used, best_value)` at each change of `best`, in natural units, for quick plots.

Results are tracked incrementally, so memory doesn't grow with the length of a single-objective run. A handle to the recorded run for analysis is `auxein.recording.open_run(result.run_dir)`.

---

## 10. Persistence and observability

### 10.1 Run directory

```
runs/<name>/
  metadata.json          # name, seed, versions, git SHA and dirty flag, backend, precision, problem spec, budget, batch size,
                         # strategy and evaluator, start and end times, final status, stop reason and summary (written atomically)
  events.sqlite          # event log: candidates, lineage, evaluations, checkpoints
  genomes/               # content-addressed genome store (large genomes)
  artifacts/             # optional heavy outputs: trajectories, transcripts, logs
  checkpoints/           # periodic snapshots of strategy, driver and RNG state
```

### 10.2 Event log (SQLite)

- **Append-only.** Written by the driver, which is the single writer, one transaction per batch (per told candidate in steady-state delivery), so it's safe against crashes. The database is in WAL mode and is checkpointed when the run ends.
- **Schema (version 1)**, as implemented; a `schema_version` table holds the version number:
  - `candidates`: `id` (primary key), `step`, `origin`, `genome_kind` (`array` or `json`), `genome` (raw bytes or JSON text), `genome_dtype` and `genome_shape` (arrays only), `created_at`
  - `lineage`: `parent_id`, `child_id`, both indexed
  - `evaluations`: `candidate_id` (primary key), `status`, `objectives`, `constraints`, `descriptors` and `cost_units` (JSON objects of name to value), `wall_time`, `error`, `finished_at`
  - `events`: `seq`, `kind` (`ask`, `tell` or `stop`), `step`, a JSON `payload`
  - `checkpoints` is added with checkpoints (step 5), together with the raw-measurement references of §6.4.
- **Candidates are recorded when they are told**, together with their evaluation, and a candidate that is never told has no row: one dropped by the budget truncation or by a wall-time stop (§9.3) was never evaluated. In generation delivery that is once per batch, in ask order. In steady-state delivery it is once per candidate, so the `ask` and `tell` events come per candidate (with a count of 1), in ask order in deterministic mode and in completion order in throughput mode. The tables are keyed by candidate id; the order of telling is in the `events` table.
- **`metadata.json`** records, besides the seed, backend, budget and components, how the run was scheduled: `batch_size`, `in_flight_window`, `concurrency`, `executor` (as resolved, never `auto`), `delivery` (as resolved), `deterministic`, `failure_policy`, `timeout` and `initial_failure_guard`. The **summary** written when the run ends holds the totals (`evaluations_used`, `wall_time`, the best candidate), the **failure counts** `status_counts` (`ok`, `failed`, `timeout`) and `abandoned_evaluations` (timed-out thread evaluations still running when the run ended).
- **Determinism:** the contents of `events.sqlite` are identical for the same seed, backend, precision and `batch_size`, apart from `created_at`, `finished_at` and `wall_time`, which are timestamps and measured times, **whatever `concurrency`, the executor and sync or async user code** (in generation delivery, and in steady-state delivery with `deterministic=True`; §8.1).
- **Lineage queries** ("all descendants of X", "the ancestry of the best agent") are single recursive SQL queries; `auxein.recording.open_run(path)` offers `ancestry(id)` and `descendants(id)`, and iterates over the evaluations with their genomes decoded.
- **Export** to JSONL or Parquet with one command comes later.

### 10.3 Genomes and artifacts

- In the first implementation genomes are stored inline in `candidates`, without pickle: array genomes (numpy arrays, or torch tensors copied to the host) as raw C-order bytes plus dtype and shape, other genomes as JSON when they are JSON-serialisable. A genome that is neither is a clear error; structured genomes get proper storage in a later step.
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
   *Outcome (step 3a, [`benchmarks/reports/core-ga-0.3.0-dev/`](../../benchmarks/reports/core-ga-0.3.0-dev/report.md)):* **met.** The default `GeneticAlgorithm` is better than the 0.2.0 default on all 15 problem × dimension cells of the suite (large effect, Holm-adjusted p ≤ 10⁻⁷), and its time per evaluation is lower at every population size and dimension of the overhead benchmark (4.9 to 7.6 µs against 9.2 to 12.1 µs), after an optimisation of the driver and the strategy that the first run of the comparison called for.
3. **Determinism:** in deterministic mode, the same seed produces an identical event log across synchronous and asynchronous evaluators and any worker count.
4. **Resume:** a run killed and resumed from a checkpoint produces the same event log as an uninterrupted run.
5. **Backends:** numeric tests pass on numpy and PyTorch (CPU) in float64 and float32. The GPU smoke suite passes on CUDA and Metal.
6. **Failures:** runs with injected failures and timeouts complete, with every failure recorded and handled per the configured policy.
   *Outcome (step 4b):* **met.** Tests inject exceptions, non-finite objectives, timeouts (async, process, thread), worker crashes (`os._exit`, `SIGKILL`) and custom evaluators that report failures, under both policies, both deliveries and every executor: runs complete, each failure is recorded with its status and error, the counts are in `RunResult` and the summary, and no worker process or non-daemon thread is left behind. The deterministic event log with candidate-driven failures is identical across `concurrency` and executors. A `GeneticAlgorithm` with 30% random failures still beats random search on the 10-D sphere at the same budget.

---

## 12. Package layout

What exists after step 3 (the packages marked "later" are planned):

```
auxein/
  __init__.py      # the public API (§9.1) and __version__
  core/            # ids, Candidate, batches (ListBatch, ArrayBatch), Evaluation and EvaluationBatch, Result and BatchResult,
                   # the normalisation of what user code returns, ProblemSpec, StateDict, the Strategy/Evaluator protocols
  spaces/          # Space protocol, Box
  backend/         # Backend, array-API helpers, precision, device validation
  random/          # RunSeed, backend-native RandomStream
  driver/          # run/arun, Budget, RunResult and its tracking, driver errors and warnings
  strategies/
    random_search.py
    ga/            # GeneticAlgorithm and its operators (selection, recombination, mutation, bounds repair)
  evaluators/      # FunctionEvaluator, VectorisedEvaluator
  recording/       # Recorder protocol, SQLiteRecorder run directory, genome encoding, a minimal reader
  # later: strategies/external/ (PycmaStrategy), evaluators (EpisodeEvaluator), environments/, aggregators/,
  #        recording (genome store, checkpoints, export)
benchmarks/        # the benchmark harness, its configs, and the committed reports (frozen results are history)
tests/             # tests of the package, parametrised over the available backends and precisions
docs/design/core.md
# later: examples/ (toy domains A and B, function optimisation, regression), rewritten notebooks
```

---

## 13. Versioning

- **No backward compatibility** with 0.x. The 0.x engine was removed in step 3b, and the git tag **`v0.2.0`** is its reference: the fixed engine, its notebooks and its documentation stay available there, and its benchmark results are kept, frozen, in `benchmarks/reports/baseline-0.2.0/`.
- The new core is versioned **0.3.0** onwards (`0.3.0.dev0` until it is released), and stays on **0.x** until the acceptance criteria (§11.4) are met and the API has settled. Nothing is published to PyPI yet, and there is no release workflow.

---

## 14. Suggested implementation order

Each step ends with passing tests and is a candidate for its own Claude Code prompt and PR.

1. **Foundations (done):** core types, `Space`/`Box`, backend, random streams, deterministic ids. The backend and random layers are tested on numpy and PyTorch (CPU) from the start, so the abstraction is proven on two backends early.
2. **Minimal driver (done):** synchronous generation mode, `FunctionEvaluator`, `VectorisedEvaluator`, budgets, SQLite recorder (metadata + event log), and `RandomSearch`, cross-checked against the benchmark harness's own random search.
3. **Strategies (done):**
   - **3a:** `GeneticAlgorithm` with array-based operators, its benchmark-harness adapter, the choice of its default configuration, and the first comparison with the `v0.2.0` baseline.
   - **3b:** removing the 0.x engine, and switching the public API (`auxein/__init__.py`) to the new core.
4. **Asynchrony (done):**
   - **4a (done):** concurrent evaluation (`concurrency`, `executor`), steady-state delivery, deterministic mode and throughput mode, numpy-backed per-candidate streams.
   - **4b (done):** failure policies, the first-failure warning and the all-failures guard, timeouts (hard for `async def` and processes, soft for threads), Auxein's own process pool with crash handling, and failure-aware parent selection in the GA.
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
