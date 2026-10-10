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
    operators: OperatorLog          # recorded and replayed calls of external proposal operators (§3.5); a no-op log when not recording
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

- *Tournament:* each parent is the best of `k` members drawn uniformly at random. *SUS:* the weight of a feasible member is `max(g − (mean(g) − c·std(g)), 0)` with goodness `g = −value` and `c = 2`; infeasible members have weight 0 unless none is feasible, when the weights come from a lower violation in the same way; if the weights are all zero or not finite the selection is uniform. A feasible member whose objective is not finite *in the precision of the run* (a value beyond float32's range) has no goodness to weigh and gets weight 0 too, instead of making the scale infinite and every weight NaN (step 7).
- *Distinct parents:* when the second parent of a child equals the first, it is replaced by a uniformly random other member (at least two members are needed to breed).
- *Recombination* mixes genes as `w·a + (1 − w)·b`: intermediate draws `w ~ U(0, 1)` per child (or per gene), uniform draws each gene from either parent with probability ½, none copies the first parent. A child that does not cross over (probability `1 − p_c`, or always with no recombination) copies its first parent and records one parent; otherwise two.
- *Mutation* steps are relative to the box width, per dimension, so the operators are scale-free; they act in linear space, also on log-scale dimensions. Self-adaptive mutation updates the step first (`σ' = σ·exp(τ·N(0, 1))`, `τ = 1/√d`; per gene, `σᵢ' = σᵢ·exp(τ'·N(0, 1) + τ·Nᵢ(0, 1))` with `τ' = 1/√(2d)` and `τ = 1/√(2√d)`), bounded below by `σ_min` (default 10⁻¹² of the box width), then moves the genes.
- *Bounds repair:* `clip` (via `Box.clip`) or `reflect` (a true reflection, however far out a gene is).

**Step sizes are strategy state, not genome** (§4.1). They are kept per candidate inside the strategy, as arrays aligned with the population. A child inherits them by the same recombination as its genes, as a weighted geometric mean with the same weights (the geometric mean when the weights are equal; one parent's exactly with weights 0 or 1), then they mutate; the steps of non-survivors are discarded.

**The ask/tell behaviour.** The first `ask` returns the whole initial population (origin `"init"`), whatever `n` the driver suggests. Afterwards `ask` returns exactly λ children (`offspring_size`), or exactly `n` when it is `None`, with their parents recorded and origins that name the operators, e.g. `"tournament+intermediate+self_adaptive"` (`copy` stands for the recombination of a child that copies a parent). Children that were asked for and not yet told are pending, with their parents' step sizes inherited; asking again before results arrive is allowed, and children are bred from the population as it is. Children that are never told (budget truncation) are dropped. If fewer than two members exist, because the initial candidates are still being evaluated, `ask` returns more random candidates. `tell` accepts any grouping of pending children. The strategy is single-objective, supports constraints, needs a `Box` or a `MixedSpace` (below), and `should_stop` is false unless a convergence tolerance is given (the objective spread of a full population is below it, and every step size is at its floor).

**The default configuration** (`GeneticAlgorithm()`) was chosen by benchmark among three candidates on instances other than those of the comparison with 0.2.0: μ = λ = 50, tournament selection (k = 2), intermediate recombination, one self-adaptive step size per individual (initial step 0.1 of the box width, floor 10⁻¹²) and clipping. The candidates (A: this one; B: as A with one step size per gene; C: as A with SUS and sigma scaling) ended within 0.5 of each other in mean rank of the median final error (C 1.70, A 2.10, B 2.20; an earlier run of the same selection gave A 1.80, B 2.10, C 2.10), a near tie, so the simplest configuration was chosen. The results are in `benchmarks/reports/ga-default-selection/`.

**Mixed spaces (step 8a).** On a `MixedSpace` (§4.5) the genome is one float array with real, integer, binary and categorical columns, and the GA applies **type-aware operators**, following mixed-integer evolution strategies (MIES, Li et al.). On a `Box` the GA runs exactly the code described above: its results for a given seed are byte-identical to before, pinned by the golden test, and the mixed support is additive (a `MixedVariation` object that exists only when the space is mixed).

- *Real columns:* the operators above, unchanged (`recombination`, `mutation`, `repair`), applied to the sub-array of the real columns, with steps relative to the width of the real dimensions and a log scale as on a `Box`.
- *Integer columns:* **`IntegerMutation`** adds to a mutated gene the *difference of two geometric random variables* of mean `m`, a symmetric integer step (zero with probability `1/(1+2m)`); each gene mutates with probability `1/k` for `k` integer genes by default (`probability=`), and the result is clipped to the bounds. The mean step `m` is a strategy parameter, one per individual, adapted like a real step size (`m' = m·exp(τ·N(0,1))`, `τ = 1/√k`) and kept in `[min_step, range]` (defaults: initial 1, floor 0.1): **the floor is what keeps every integer gene able to change for good**, since at the floor a mutated gene still moves with probability `2m/(1+2m)` = 17 %. `adaptive=False` keeps `m` fixed.
- *Binary columns:* **`BitFlipMutation`** flips each gene with probability `1/k`. *Categorical columns:* **`CategoricalMutation`** replaces the index with a *different* category drawn uniformly, with probability `1/k`. For both the default rate is `1/max(k, 2)`: with a single binary or categorical gene the rate `1/k` would be 1, every child would change it, and no child could ever inherit its parent's value (measured: it stalled the mixed fixture in 2 of 10 runs).
- *Recombination of discrete columns* is always **discrete**: each gene comes from exactly one of the two parents (a fair coin per gene); the real columns use the real recombination. With no recombination a child copies its first parent.
- *Strategy state:* the step sizes of the real columns (none, one per individual, or one per real gene, as the real mutation says) and the integer mean step are packed in one array aligned with the population, inherited as the weighted geometric mean with the weights of the genes they belong to, checkpointed in `state_dict` like the step sizes of a `Box` run, and discarded with their non-survivors. Convergence (`convergence_tolerance`) requires every adapted step, real and integer, to be at its own floor.
- *Defaults come from the space; every operator can be replaced:* `mutation`, `recombination` and `repair` are the real operators, and `integer_mutation`, `binary_mutation` and `categorical_mutation` are new keyword arguments. Origins name the operators by type, compactly, e.g. `"tournament+intermediate/discrete+self_adaptive/geometric/bitflip/resample"` (only the types the space has). The description (`repr`) of a GA that was not given the new arguments is unchanged, so a recorded run still resumes. Everything is vectorised over the population and goes through the array namespace, on both backends and both precisions; ranking, survivor selection, parent selection and failure handling are the shared ones.

**SBX and polynomial mutation (step 8b).** Two standard real-coded operators, added to the numeric operator set and usable by `GeneticAlgorithm` too, which keeps its own defaults.

- **`SimulatedBinaryCrossover(eta=15, variable_probability=0.5)`** (Deb and Agarwal). The children of a pair sit symmetrically around the parents' midpoint, spread by a factor β whose distribution is set by `eta`. It fits the recombination-weight protocol (a child gene is `w·a + (1 − w)·b`, so `w = (1 ± β)/2`): **a weight outside [0, 1] is how SBX extrapolates**, and the strategy's bounds repair brings the child back. Each gene is crossed with probability `variable_probability`; in a crossed gene a fair coin picks the side of the midpoint (this is what makes SBX contract a population: a child that always sat on its first parent's side would not), and the genes that are not crossed come from one parent, the same for the whole child. It is the *unbounded* form; the reference code's bounded form differs near the bounds.
- **`PolynomialMutation(eta=20, probability=None)`** (Deb): bounded, non-adaptive. A mutated gene moves by a fraction of the box width drawn from a polynomial distribution built from the gene's distance to each bound (so nothing needs clipping, and a gene on a wall has no room to move out, which makes half the draws no-ops there); `probability` per gene defaults to `1/d` for the `d` real genes. It needs the bounds, which a mutation call does not carry (it gets the width), so strategies bind them once with `with_bounds`, which returns a copy: operator objects are never mutated and can be shared. On a `MixedSpace` both apply to the real genes only, through `MixedVariation`.

**NSGA-II (`NSGA2`, step 8b)** is the multi-objective strategy (Deb, Pratap, Agarwal and Meyarivan, 2002), a separate class that works on a `Box`, a `MixedSpace` and a `SequenceSpace` (any space with a codec, with operators of your own).

- **Survivors: "plus" selection by non-dominated sorting and crowding.** The population (μ) is pooled with the told children and the best μ are kept: whole fronts first, then, in the last front that fits, the members with the largest crowding distance. **Constrained domination** is Deb's rule: a feasible member dominates every infeasible one; among infeasible members the lower total violation dominates; among feasible ones, Pareto domination on the objectives **in minimisation form** (the `EvaluationBatch` converts the declared directions once, nothing is negated by hand). **Failed evaluations** (NaN objectives, infinite violation) are dominated by every member that did not fail, do not dominate each other, and so fill the last front; they are never parents while there is an alternative. **Crowding distance** is computed per front, every objective normalised by its range in the front, the two extreme members infinite, and a front with no range in an objective gets nothing from it (no NaN).
- **Parents: the crowded tournament through the unchanged `TournamentSelection`.** The crowded-comparison order (lower front, then larger crowding distance, then lower candidate id) is a total order, so it is the `order` and `rank` of a `PopulationView`, and the binary tournament is the GAs' own. As in Deb's algorithm the order is that of the **pool** the survivors were chosen from (the fronts and distances of the pool, restricted to the survivors), which depends on the pool and not on the survivors alone, so it is part of the strategy's state (`state_dict` has an `order` array). `SigmaScalingSUS` needs a scalar fitness and is an error.
- **Contract:** like the GAs: exactly λ children (or `n`), no self-mating, every candidate evaluated once, pending children kept, `state_dict` continuing identically, `tell_mode="both"`, any number of objectives (with one it is a GA with plus selection) and constraints. **With steady-state delivery the result depends on how many children are told at once**, because the fronts and crowding distances are recomputed at every `tell`; in deterministic mode (§8.1) that is fixed by the seed and the batch size, so the event log is still identical whatever the concurrency.
- **Defaults are the textbook ones:** population 100, offspring 100, binary crowded tournament, and on real genes SBX (η = 15, probability 0.9 per pair) and polynomial mutation (η = 20, 1/d per gene), then clipping; on the discrete genes of a `MixedSpace` the operators of step 8a (§3.3 above); on a `SequenceSpace` the structured defaults.
- **How it is built.** `NSGA2` is a thin class that, when bound, builds an *engine* for the space: a subclass of `GeneticAlgorithm` for `Box` and `MixedSpace`, a subclass of `StructuredGeneticAlgorithm` for spaces with a codec. Both algorithms received a few protected **ranking hooks** whose defaults are the single-objective code (how the objectives of the population are stored, how a pool is ordered, how the population's order is recomputed, extra state, and whether a `tell` with no surviving child can skip the update), and the engines override them; breeding, pending children, survivor gathering and checkpointing are the GAs' own code, not copies of it. The hooks were proved neutral: the golden test, every existing test and the benchmark adapters give the same results (the 60 quick-benchmark runs of `auxein-core-ga` and `auxein-core-random` have identical final errors and traces), and the overhead per evaluation is unchanged.
- **Limit.** The domination relation is an `(n, n)` boolean matrix over the pool (population plus offspring), built from an `(n, n, k)` comparison: memory grows with the square of the pool, about 50 MB for 4,000 members and 3 objectives. Sorting and crowding run on the backend, and the only Python loop is over the fronts. The driver's Pareto archive (`RunResult.pareto_front`) compares each new point with the whole archive in one numpy expression (it was a Python loop per point, which made the archive of a converged three-objective run, thousands of points, cost more than the strategy).

**The structured genetic algorithm** (`StructuredGeneticAlgorithm`, step 6b) is a separate strategy for genomes that are not arrays, generic over the genome type, not a generalisation of the numeric one (which is unchanged: its results for a given seed are byte-identical, pinned by a test). It has the same contract (single-objective, constraints supported, both tell modes, exactly λ children or `n`, no self-mating, every candidate evaluated once, pending children, exact `state_dict` continuation) and **reuses what only looks at objective values**: the ranking (`rank_order`), "plus" survivor selection (the population is stored in rank order and re-ranked on every `tell`), `PopulationView` and the parent-selection operators with their failure-aware behaviour (failed members are never parents while any other member exists; fewer than two valid members means random sampling). Only variation differs: **per-genome operators** instead of array operators.

- `StructuredMutation.mutate(genome, rng, ctx) -> genome` and `StructuredRecombination.recombine(first, second, rng, ctx) -> child`, with a `VariationContext` (the space, its codec, the run's operator log, the id of the child being made). `mutation` may be a list of `(operator, probability)` pairs, mixed by those probabilities; without a `recombination` a child copies its first parent. Origins name the operators, e.g. `"tournament+one_point+sequence"`.
- For a `SequenceSpace` the defaults are chosen automatically: **`SequenceMutation`** (insert, delete, replace or swap one item, each respecting the length bounds and `unique`, with configurable probabilities and number of `edits` per child) and **`SequenceCrossover`** (one-point or two-point cut-and-splice for variable lengths; the result is repaired: duplicates removed in a `unique` space, cut at `max_length`, filled up to `min_length` first with items of the parents that are not in it, then with random vocabulary items). Any other space needs a mutation supplied by the user.
- The space must have a codec (§4.1): the population goes into checkpoints through it, and genomes are recorded through it.
- Variation may call **external proposal operators** (§3.5).

### 3.4 Strategies in the first implementation

- `RandomSearch`: the floor (done in step 2).
- `GeneticAlgorithm`: composable, as above, supporting both tell modes (done in step 3a).
- `StructuredGeneticAlgorithm`: the same algorithm for structured genomes, with `SequenceSpace` as the built-in space (done in step 6b).
- `NSGA2`: the multi-objective strategy, on `Box`, `MixedSpace` and `SequenceSpace`; and `Scalarised(strategy, scalarisation)`, which runs any single-objective strategy on a multi-objective problem (§5.4) (done in step 8b).
- `PycmaStrategy`: a wrapper around pycma, as an optional extra. Its purpose is to prove that external algorithms fit the contract.

More algorithms (native CMA-ES, MAP-Elites and others) are added later on top of the same contract.

### 3.5 External proposal operators

Some variation is not a function of Auxein's random streams: asking a language model to rewrite a prompt is the typical case. The slot for it is the **`ProposalOperator`** protocol: `name` and `propose(parents, rng) -> Proposal(genome, cost, metadata)`, where `cost` holds user-defined units (tokens, money) and `metadata` is JSON-serialisable. `ExternalMutation(operator)` and `ExternalRecombination(operator)` plug one into `StructuredGeneticAlgorithm` (one parent or two), alone or mixed by probability with built-in operators.

- **Every call is recorded and replayed.** A call is keyed by the SHA-256 of the operator's name, the canonical encodings of its inputs and **a value drawn from the strategy's stream at that point** (so asking twice for the same input gives two keys, and the stream advances by exactly one draw whether the call is live or replayed). With a `run_dir`, the proposed genome (canonical JSON), cost units, wall time, metadata and the event sequence number go into the `operator_calls` table (§10.2) as soon as the call returns, in a transaction of its own, because the call has been paid for. **On replay** (§10.4) the recorded output is returned without calling the operator, so a resumed or extended run regenerates the identical candidates and never pays twice; that includes the calls made for children the old budget dropped. A live call for a candidate that the recording already holds means the run **diverged** (the operator's name, its inputs or the draw differ, so the strategy's configuration or code changed): that is a `ReplayMismatchError` and the operator is not called. Without recording, calls are always live, a `OperatorNotRecordedWarning` says so, and the run cannot be reproduced or resumed.
- **Synchronous, inside `ask`.** A call blocks the driver's event loop while it runs. This is a documented limitation of this version; concurrent, asynchronous variation is an open question (§15).
- **Failures fail the run.** An exception in the operator, a result that is not a `Proposal`, or a genome outside the space raises an `OperatorError` naming the operator (the original exception chained), and the run ends as `failed`. Failure policies for variation are future work.
- **Cost is recorded, not budgeted.** The cost units of operator calls are kept in `operator_calls` but are **not counted against the run's budget** (§9.3) in this version; that is an open question too (§15).
- The operator receives a stream derived from the draw, so an operator that uses it is deterministic; one that does not is made reproducible by recording.

---

## 4. Genomes, candidates and batches

### 4.1 Genomes

- A genome is **whatever the user defines**: a 1-D array, a dataclass, a tree, a string. The core never inspects it.
- **Genomes are immutable.** Operators always create new genomes. Arrays handed out by the framework are marked read-only where the backend allows, so accidental modification fails loudly. No deep copies are needed anywhere.
- **Structured genomes are encoded through their space (step 6b).** A structured space has a **codec**: `encode(genome)` to a JSON-serialisable value and `decode(value)` back, with `decode(encode(g)) == g`. The **canonical encoding** is the one function `canonical_json`: UTF-8 JSON with sorted keys, no insignificant whitespace, only string keys and no NaN or infinity; values keep their JSON type (`1` and `1.0` differ), floats are written by Python's shortest round-trip `repr`, tuples encode as lists, and anything JSON cannot encode (numpy scalars other than `float64`, sets, bytes) is a clear `CanonicalEncodingError`. Equal genomes therefore always encode to identical bytes, which is what recording, replay's byte-identical check, the genome store's hashes and checkpoints rely on. Structured genomes must be **immutable by convention** (tuples, frozen dataclasses) and **picklable** (they are sent to worker processes). Spaces without a codec keep the earlier behaviour: arrays as raw bytes, JSON-serialisable values as JSON.
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

- **Numeric problems with structure** (e.g. polynomial degree, number of active rules or sensors) use a **fixed maximum size with structure genes**: binary switches or an integer gene that decides which components are active. The genome length stays constant, so the array fast path applies. A complexity objective (e.g. number of active terms, minimised) turns the problem into a trade-off between accuracy and complexity, which `NSGA2` explores (step 8b). **Structure genes exist since step 8a**, in a `MixedSpace` (§4.5). The test fixture previewing the polynomial notebook is the example: a genome of seven real coefficients and seven binary switches (maximum degree 6), the data error plus 0.02 per active term as the single objective, noise-free data from `2 + 3x² − 1.5x⁵`. The GA recovers the three true terms and their coefficients (to 0.02 in float64) in 60 to 100 % of the seeds at 15,000 evaluations, depending on the backend and precision; the others end in a near optimum with one spurious term (the switches converge prematurely, which a larger population does not fix).
  **The trade-off with two objectives** (the data error and the number of active terms, both minimised, `NSGA2` with a polynomial mutation of η = 50, 40,000 evaluations) gives a front that is a staircase over the number of terms, and **contains the true model** (terms 0, 2 and 5, an error of about zero). How reliably: over seeds 0 to 9 on the four configurations, the true set of terms is on the front in 6 to 7 seeds of 10, and with an error under 0.05 (a quarter of a percent of the data variance, 19.5) in 2 to 5 of 10. NSGA-II spreads its population over the eight levels of complexity and has no step-size adaptation, so the three coefficients of the true model are refined slowly; the single-objective version, which has the complexity price built in, did better (§4.4 above). The test pins a seed that recovers the true set on all four configurations.
- **Open-ended structure** (growing networks, program trees, rule lists without a natural maximum, prompts) uses **structured genomes** through the general (non-array) path, with operators that understand the structure. This is built end to end (step 6b): a space with a codec (§4.5), `ListBatch`, recording through the codec, and `StructuredGeneticAlgorithm` (§3.3). The built-in variable-length space is **`SequenceSpace(items, min_length, max_length, unique=False)`**: genomes are tuples of items from a finite vocabulary of JSON-serialisable values (instructions, rule identifiers, tool names, tokens; two items are the same when their canonical encodings are), sampled by drawing a length uniformly and then the items (distinct if `unique`). Trees, graphs and free text are not built in: users bring their own space, codec and operators through the protocols.

### 4.5 Search spaces

```python
class Space(Protocol[G]):
    def sample_genomes(self, n: int, rng: RandomStream, backend: Backend) -> Sequence[G] | Array: ...
    def contains(self, genome: G) -> bool: ...
```

A **structured space** also exposes a `codec` (a `GenomeCodec`: `encode`, `decode`, §4.1) and a `describe()`; `auxein.spaces.codec_of(space)` returns the codec of a space, or None.

Spaces return **genomes**, not batches. Building a batch requires candidate ids, which are issued through the strategy context, so strategies wrap sampled genomes into batches themselves. Array spaces such as `Box` return an `(n, d)` array.

- **First implementation:** `Box`, a bounded real vector with lower and upper bounds per dimension and an optional log scale per dimension (log-scale dimensions are sampled log-uniformly and need a positive lower bound). Samples are guaranteed to lie within `[lower, upper]` in float32 as well as float64: float32 bounds are rounded *inward* (a bound that isn't a float32 number moves to the next float32 inside the box) and samples are clipped to them. A box too narrow or too wide for float32 to represent is an error when sampling in float32. `Box.contains` and `Box.clip` are the membership test and a vectorised repair helper for operators.
- **`SequenceSpace`** (step 6b): variable-length sequences from a finite vocabulary, with `describe()` for the metadata and resume validation (the vocabulary and the bounds are part of the run's identity) and a codec.
- **`MixedSpace`** (step 8a): named dimensions of four types, declared in order,

  ```python
  MixedSpace({
      "lr": Real(1e-5, 1e-1, log=True),
      "layers": Integer(1, 8),
      "dropout": Binary(),
      "optimiser": Categorical(["sgd", "adam"]),
  })
  ```

  - **One array genome.** The genome is a single array of the backend's float dtype, like a `Box` genome, so the array fast path, `VectorisedEvaluator`, GPUs, recording as raw bytes, the genome store and replay work unchanged (the space has no codec; the reader returns arrays). An integer is stored as itself, a binary as 0 or 1 and a categorical as the **index** of its choice (choices are JSON-serialisable, at least two, distinct by canonical encoding). `Real`, `Integer` (inclusive bounds, `lower < upper`), `Binary` and `Categorical` validate themselves; names must be distinct non-empty strings.
  - **Float32.** Every integer up to 2²⁴ is a float32, so integer bounds beyond that are an error *when the run starts* (`MixedSpace.check_backend`, called by `bind` of both strategies; the message says to narrow the range or use float64). Real bounds are rounded inward as in `Box`. Sampling an integer range wider than 2²³ in float32 is uniform only to the resolution of float32's unit interval.
  - **Sampling** is uniform per type (log-uniform for log-scale reals, uniform over the integers of a range, over {0, 1} and over the category indices) and on the backend; every sample is valid in both precisions. `contains` checks bounds **and** integrality **and** valid category indices.
  - **Decoding is the space's job, done by user code:** `values(genome)` returns a dict of name → Python value (`float`, `int`, `bool`, the choice itself) and `columns(genomes)` one array per dimension on the genomes' backend (floats, int64, bool, and the int64 index for a categorical, with `categories(name)` mapping indices to choices) for vectorised objectives. `describe()` goes into the run metadata, and a changed space is refused on resume.
  - `IntegerSpace(lower, upper, dim)` and `BinarySpace(dim)` are thin subclasses (dimensions `x0 … x{dim-1}`).
  - `Box` stays the all-real case and is **not** reimplemented on top of `MixedSpace`: its code path and results are untouched.
- **Later:** conditional dimensions (§15), and spaces for text and trees, added when a use case needs them. The concept lives in the core from the start, so operators and strategies can rely on it.

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
- `EpisodeEvaluator(decoder, environment, scenarios, aggregator, role=None)`: the agent evaluator (§6). It has a per-episode path, which obeys `concurrency`, `executor` and `timeout` (one executor call per episode), and a batched path that, like `VectorisedEvaluator`, runs on the driver's thread, rejects a timeout at start-up and warns under steady-state delivery.

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
- **The process pool is Auxein's own**, not `concurrent.futures.ProcessPoolExecutor`, which cannot kill one task and breaks the whole pool when one worker dies. It has up to `concurrency` workers, started with `spawn` **lazily** (when work first needs one, and again to replace a worker that was killed or died), each evaluating **one call at a time** and talking to the parent over its own pipe. A worker says `ready` once it is up: one that cannot start (typically a script without the `if __name__ == "__main__":` guard) is a misconfiguration and raises an `ExecutorError`, not a failed candidate. An exception in the function is re-raised in the parent with the worker's traceback chained as text. Workers are daemon processes that exit when their pipe closes, so they cannot outlive the parent (a consequence: a function running in a worker cannot start child processes of its own). `ctx.call` is unchanged for user-written evaluators. On the same machine the pool costs about 56 µs per call against 149 µs for `ProcessPoolExecutor` at `concurrency=1` (30 against 89 µs at 4).
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

`EvalContext` carries the problem specification (so that what user code returns can be checked against it), the backend, the executor with `ctx.call` and `ctx.concurrency`, the failure policy and the timeout, and factories that derive evaluation random streams from candidate ids: `rng_for(id)` for a candidate, `batch_rng_for(first_id)` for a vectorised batch, and for the agent layer `episode_rng_for(id, scenario_index)` and `episode_batch_rng_for(first_id)` (§8). They are factories rather than lists of streams so that a batch of thousands of candidates doesn't create thousands of generators up front.

### 5.4 Scalarisation

A single-objective strategy applied to a multi-objective problem requires an explicit scalarisation supplied by the user (step 8b). **`Scalarised(strategy, scalarisation)`** is the wrapper (`auxein.Scalarised`):

- **The inner strategy sees one minimised objective**, `"scalarised"`, with the problem's space, constraints and descriptors; each time it is told, every evaluation carries that one value instead of the original objectives. A failed evaluation passes through as a failure, and a scalarisation that is not finite fails the evaluation. **The driver, the recorder and `RunResult` keep all the original objectives**: the wrapper only changes what the inner strategy is told, so the recording has every objective, `RunResult.pareto_front` is the Pareto archive of the original ones, and `best` is None, as for any several-objective run.
- **Built-in scalarisations** work on the objectives **in minimisation form**:
  - `WeightedSum({"cost": 1.0, "time": 2.0})`: `Σ wᵢ·fᵢ`. It can only find points on the convex part of a Pareto front.
  - `Chebyshev(weights, reference=None)`: the weighted Chebyshev (Tchebycheff) scalarisation `maxᵢ wᵢ·(fᵢ − zᵢ)`, which can reach every point of a front, convex or not. The **reference point `z` must be explicit** (it is given in natural units and converted with each objective's direction; 0 for every objective when omitted). The usual "ideal point seen so far" is not allowed: it would change during the run, so a candidate would score differently at different times, and a strategy that compares scores across generations (a plus-selection GA does) would compare numbers on different scales.
  - Weights are **named by objective**, validated against the `ProblemSpec` when the strategy is bound (every objective needs a weight, 0 to ignore it; unknown names are an error), must be finite and non-negative, and not all zero. A custom scalarisation is any object with `validate(objectives)` and `apply(minimised, objectives, backend)`.
- **The wrapper's `repr` includes the inner strategy and the scalarisation**, so a resume with a changed weight is refused by the configuration check; `state_dict` is the inner strategy's.
- **`best_by_scalarisation(run, scalarisation)`** returns the winner by the scalarisation from a `RunResult` (with `objectives=`, searching its Pareto archive, which holds the minimiser of any scalarisation that does not decrease in an objective) or from a run directory (searching everything recorded; the directions come from its metadata): feasible first, then a lower total violation, then the lower value, then the lower id.

---

## 6. Agents, environments and scenarios

The agent layer (step 6a) lives in `auxein.environments` (scenarios, episode results, the environment, decoder and step-adapter interfaces), `auxein.aggregators` (reductions and the `Aggregator`) and `auxein.evaluators` (`EpisodeEvaluator`). From the top level: `EpisodeEvaluator`, `Scenario`, `ScenarioSet`, `Aggregator` and `EpisodeResult`.

### 6.1 Episode evaluator

To the core, an evaluator is anything that produces evaluation records. For agents, Auxein provides the **episode evaluator**, composed of:

1. a **decoder**: `genome -> Agent`
2. an **environment** that runs episodes
3. a **scenario set**
4. an **aggregator**: per-scenario measurements → objectives, constraints, descriptors

```python
evaluator = auxein.EpisodeEvaluator(decoder, environment, scenarios, aggregator, role=None)
```

`role` names the evolved role and can be omitted when the environment has one. **One evolved role per episode** is supported; the other participants are part of the scenario. Agents are handed to the environment as a mapping by role, so several evolved roles will need no change of interface.

**Two paths, one aggregator.** The **batched path** applies when the environment has `run_batch`, the decoder has `decode_batch` and the batch is array-backed: the batch is decoded at once, `run_batch` is called once with all scenarios, and the `(n, s)` arrays it returns are aggregated. It runs on the driver's thread, on the backend's device. Otherwise the **per-episode path** decodes each candidate once and runs every scenario as one call of `run_episode`, through `ctx.call` for a synchronous environment (so `executor=` applies, and with processes the environment, the decoded agents and the scenarios must be picklable) and natively for an `async def` one, with **at most `ctx.concurrency` episodes in progress across the whole batch** (candidates times scenarios, not per candidate), returned in ask order. The per-episode results are stacked into the same `(n, s)` arrays, so one aggregator code path serves both. On the fixture environment the framework adds about 8 µs per episode inline and about 35 µs with four threads, against a per-episode cost of microseconds to seconds in a real simulator.

### 6.2 Environment interface: whole episodes

```python
class Environment(Protocol):
    roles: tuple[str, ...]          # e.g. ("own_ship",) or ("buyer", "seller")

    def run_episode(
        self, agents: Mapping[str, Agent], scenario: Scenario, rng: RandomStream
    ) -> EpisodeResult | Awaitable[EpisodeResult]: ...      # plain or `async def`


class BatchedEnvironment(Protocol):                          # optional capability
    def run_batch(self, agents: AgentBatch, scenarios: Sequence[Scenario], rng: RandomStream) -> EpisodeBatchResult: ...


class Decoder(Protocol[G]):
    def decode(self, genome: G) -> Agent: ...
    # optional: def decode_batch(self, genomes: Array) -> AgentBatch    (opaque to Auxein; passed to run_batch)


@dataclass(frozen=True)
class EpisodeResult:
    measurements: Mapping[str, float]   # raw: fuel, time, closest approach, success, tokens...
    status: Status = Status.OK
    error: str | None = None
    artifacts: ArtifactRef | None = None    # reserved: trajectories and transcripts are not stored yet


@dataclass(frozen=True)
class EpisodeBatchResult:
    measurements: Mapping[str, Array]   # name -> array of shape (n_candidates, n_scenarios), on the backend
    failures: Mapping[tuple[int, int], EpisodeFailure]   # sparse status/error grid: the episodes that did not succeed
```

- **The core contract is a whole episode**, plus an optional batched variant. Real simulators often own their own loop (external processes, co-simulation, ROS), LLM agents run their own multi-turn loops, and batched GPU simulators run a population in one call. `IdentityDecoder` is the decoder for environments that take the genome directly (a batch of agents is then the genome array).
- **Environments return raw measurements, not scores.** What counts as good is decided by the aggregator, so runs can be re-judged without re-simulating. A failed or timed-out `EpisodeResult` has a status other than `OK` and an `error`; a measurement that is NaN in a successful episode shows up as a non-finite objective (a failed evaluation).
- **Step-level environments** (`reset`/`step`) plug in through `StepEnvironment(world_factory, role=, max_steps=, last=)`. The world has `reset(scenario, rng) -> observation` and `step(action) -> (observation, measurement_updates, done)`; the agent has `act(observation) -> action` and optionally `reset(rng)` (called with its episode stream). The adapter builds a new world for every episode, loops until `done` or `max_steps`, **sums** the measurement updates over the episode (except the names listed in `last`, whose last value is kept) and adds two measurements itself, `steps` and `done`. It is small on purpose; the Gymnasium adapter (step 9) builds on it, and Gymnasium is not a core dependency.
- **Which randomness is which.** `run_episode` receives the **agent's** stream for this candidate and scenario. The **world's** randomness must come from `scenario.rng()`, which depends on the scenario's seed alone (§6.4, §8).

### 6.3 Multiple agents and roles

An episode receives agents **by role**. A single-agent problem has one role.

- **Evolved agents among scripted ones** (e.g. other traffic around an evolved vessel) are part of the scenario.
- **Several evolved agents in one episode** (competition, cooperation, co-evolution) are supported by the interface, which passes a mapping by role. `EpisodeEvaluator` evolves one role in this version; deciding who meets whom is the evaluator's responsibility, and matchmaking is future work that needs no interface change.

### 6.4 Scenarios

- A **`Scenario`** is frozen and hashed by id: an `id` (string), its `index` within its set, a `seed` (an int, for the world's randomness) and `params`, a read-only mapping of JSON values (initial geometry, sea state, traffic, a task instance...; a class of its own so that it pickles to worker processes). `scenario.rng(*keys)` is the world's stream, derived from the seed alone.
- A **`ScenarioSet`** is an ordered, immutable collection with a `fingerprint`: a SHA-256 of the ids, seeds and params in order. Build one from a list of params (`from_params`), or generate it with a function `(index, rng) -> params` and a seed (`generate`); `split(n_selection, n_held_out)` partitions a set into two disjoint ones (re-indexed from 0, ids and seeds kept), and `generate_split` produces both from one seed without overlap. `save` / `load` use JSON and check the fingerprint on loading. The fingerprint is part of the `EpisodeEvaluator`'s description, so **resuming a run with another scenario set is refused** (§10.4).
- Every candidate in a run is evaluated on the **same scenarios** (**common random numbers**), so differences in results reflect the agents, not luck: the world's randomness is derived from the scenario's own seed, the agent's from a per-(candidate, scenario) stream (§8).
- Scenario sets are split into a **selection set** (used for evolution) and a **held-out set** (used only for reporting), to detect overfitting and reward hacking. **Held-out evaluation** is a helper that runs *after* a run: `auxein.driver.evaluate_held_out(run_dir_or_result, evaluator, scenarios, candidates="best")` (and `aevaluate_held_out`) evaluates the chosen candidates (the run's best, its Pareto front, or a list of ids) with the same machinery but **outside** the run: no strategy, no budget, nothing added to the event log, and agent streams from a seed derived from the run's so that they are independent of the ones used in evolution. It returns a `HeldOutReport` with the per-scenario measurements and the aggregated values of each candidate, and writes it to `held_out.json` in the run directory unless `write=False`. The strategy never sees the held-out scenarios during evolution.
- **Per-scenario measurements are recorded** with the evaluation (§10.2), so results can be re-aggregated later without re-simulating (`RunReader.reaggregate`).

### 6.5 Aggregators

An aggregator maps the measurements of every candidate on every scenario to objectives, constraints and descriptors (and optionally user-defined cost units). It is declarative and swappable: each name maps to a **reduction**, a source (a measurement name, or a function of the dict of measurement arrays) plus a reducer applied **across scenarios**. For example:

```python
aggregator = Aggregator(
    objectives={"fuel": mean("fuel_used"), "time": mean("time_to_waypoint")},
    constraints={"cpa": maximum(lambda m: relu(0.5 - m["closest_approach_nm"]))},
    descriptors={"mean_speed": mean("mean_speed")},
)
```

- **It always works on arrays of shape `(n, s)`** (`n` candidates, `s` scenarios) in the backend's array namespace: the per-episode path stacks its results into the same shape, so one vectorised, backend-generic code path serves both paths and the reader's re-aggregation.
- **Reducers** (`auxein.aggregators`): `mean`, `minimum`, `maximum`, `total` (sum), `quantile(source, q)` (linear interpolation of the sorted values, as numpy's default) and the two tails of CVaR. `cvar_upper(source, alpha)` is the mean of the largest `ceil(alpha * s)` values, the "worst α fraction" of something where **higher is worse** (fuel, time, a violation); `cvar_lower(source, alpha)` is the mean of the smallest, the worst fraction where **lower is worse** (a reward, a success rate). `alpha = 1` is the mean for both. The tail size is `ceil(alpha * s)` computed with a tolerance of 1e-9, because `alpha * s` in binary floating point can land a hair above an integer (`0.07 * 100` is `7.000000000000001`), which put one scenario too many in the tail before step 7.
- **Its output must match the `ProblemSpec` exactly**: every declared objective, constraint and descriptor present and no unknown name, so a typo is caught like a typo in a `Result`. A constraint value that is negative is an error (violation amounts are at least 0); a non-finite objective, constraint, descriptor or cost value fails the candidate.

### 6.6 Failures

- A failed or timed-out episode or evaluation is **a result, not a crash**. It's recorded with its status and error (§5.1), and the run continues.
- **A candidate fails if any of its episodes fails.** If an episode returns a failure or raises, the candidate's evaluation is `FAILED` (or `TIMEOUT`, if every failing episode timed out), and its `error` lists the failing scenarios and their errors (the first three in full, the rest named). The run's `failure_policy` then applies as for any evaluator. The measurements of the episodes that succeeded are still recorded. A genome that cannot be decoded fails only its candidate.
- **`failure_policy`** (a `run` argument):
  - **`"infeasible"` (the default).** An exception in user code, a timeout or a worker crash produces an `Evaluation` with status `FAILED` or `TIMEOUT` and an informative `error`. Strategies receive it in `tell`; it ranks below every feasible candidate (NaN objectives, infinite violation), is kept in the lineage for inspection, and is **never retried** (there are no retries of any kind). For an episode evaluator, an exception in the environment is a failed episode and so a failed candidate.
  - **`"fail_fast"`** stops the run at the first `FAILED` or `TIMEOUT`, with an error that names the candidate and chains the original exception (useful when developing an environment). For an exception in the environment the evaluator raises at once, cancelling the other episodes; for a failure that the environment *returned*, the evaluator returns the `FAILED` evaluation and the driver's backstop stops the run.
- **Timeouts apply per episode** on the per-episode path, each episode being one executor call (§5.3): a timed-out episode makes the candidate `TIMEOUT`, and an episode timeout's wall time is the limit. The batched path runs on the driver's thread and cannot time out; a `timeout` with it is an error at start-up, as for `VectorisedEvaluator`.
- **Misconfiguration always fails the run, whatever the policy**, because it is not a result and means nothing will work: executor errors (a function or arguments that cannot be pickled, results that cannot be sent back, workers that cannot start), return-value contract violations (wrong type, a plain dict, missing or unknown names; an environment returning something that is not an `EpisodeResult`; episodes of one batch reporting different measurement names; an aggregator that does not match the problem or reads an unknown measurement), driver validation errors (`StrategyError`, `EvaluatorError`), and `KeyboardInterrupt` and other `BaseException`s that are not `Exception`s.
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

**What the rules cover, and how they are enforced (step 7).** A *numeric code path* is anything that computes on genomes or objective values: the spaces, both strategies' ranking and selection (and the numeric GA entirely), the operators, the three evaluators, the aggregators, the `EvaluationBatch` columns and the result tracker, the checkpoints of array state, and the genome store for array genomes. Each runs on numpy and PyTorch, in float64 and float32. Four things enforce it: the `backend` fixture runs every unit test of these on all four configurations (§7.4); the integration tests run on two corners of that grid; `tests/device_placement_test.py` asserts that what leaves ask, tell, a stream, a space, an evaluator or an aggregator is an array of the backend's namespace, device and dtype; and the GPU smoke suite checks the same on real devices.

**Host-side metadata is a deliberate exception, not a leak.** These stay on the host, as numpy or Python values, and each place says so in a comment:

- ids, lineage (the parents' ids), statuses and origins: they are recorded and compared as Python integers and strings. The GA copies `mu` or `lambda` integers per `ask` or `tell` to the host for them, never a genome, and never once per candidate;
- the survivors' indices of a `tell` (how many children survive depends on the data, and the array API has no boolean-mask indexing), a scalar convergence check, the one scalar draw that picks between mutation operators;
- the `Evaluation` records: a result is a dict of Python floats, because that is what user code returns and what the recorder writes. The `EvaluationBatch` matrices turn them back into arrays on the backend, once per batch;
- the aggregator's output: the reductions run on the device over `(n, s)` arrays, and the per-candidate columns that come out (one float64 vector per name) are the numbers of the `Evaluation`s. The per-episode path of the episode evaluator collects its Python floats on the host and moves them to the backend once per measurement name; the per-scenario measurements go back to the host once per batch for the recorder;
- a space's bounds and their validation (`d` numbers, float64, rounded inward to float32 on demand); `contains`;
- everything recorded, replayed or checkpointed: files are host files, and arrays come back to the run's device when they are loaded (§10.4).

Reading a device array on the host goes through `Backend.to_numpy`, never `np.asarray`, which raises for a CUDA or MPS tensor. The audit of step 7 found exactly one such accidental path: the structured GA's choice among mutation operators read its draw with `np.asarray` (it worked on CPU tensors and failed on a device; fixed, and covered by the GPU suite).

**Float32 and the strategy's arrays.** The strategy's arrays have the backend's precision, so an objective value that float64 holds and float32 does not (beyond about 3.4e38) is infinite in the arrays, and such a member simply ranks last (ties by id) and gets no weight in sigma-scaled selection. The `Evaluation`, the recording and the `RunResult` keep the exact Python float.

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

- CI runs numeric tests on **numpy and PyTorch (CPU)**, in **float64 and float32**. Torch is part of the `dev` dependency group, so every CI job that runs tests (and the benchmark job) has it, on all four Python versions and with the lowest direct dependencies. On Linux, torch comes from PyTorch's CPU-only wheel index (configured in `pyproject.toml`), which avoids gigabytes of CUDA libraries that the tests never use.
- **Two fixtures, two matrices.** `backend` (`tests/support/fixtures.py`) runs a unit test on the four configurations numpy and torch × float64 and float32. `corner_backend` runs an *integration* test (an end-to-end run, kill and resume, failures and timeouts, episode runs, external operators, the genome store) on two: **numpy-float64** and **torch-float32**, the reference and the most different pair. A bug that depends on the backend or the precision shows on a corner, and the integration tests are the minutes of the suite, so trimming them keeps CI affordable: the mixed configurations are covered by the unit tests and by a few integration tests that name a backend themselves. A module opts in with `pytestmark = pytest.mark.usefixtures("use_corner_backend")`, and its shared helpers pass `integration_backend()` to `run` and `resume` (and to the subprocess of a kill test, through its JSON configuration).
- **The torch backend must search as well as numpy.** The streams differ, so the runs do too, but the distribution of results must not. `benchmarks/tests/crosscheck_test.py` runs `auxein-core-ga` on torch-float32 and on numpy-float64 on the sphere and Rastrigin in 2 and 10 dimensions (and random search on the sphere), 30 paired runs each with the quick budget, and requires the Vargha–Delaney A₁₂ of the final errors to be in [0.35, 0.65]. The seeds are fixed, so the test is deterministic. The band is about two standard errors with 30 runs, hence a wide net for gross shifts (a precision bug, a broken operator on tensors) and blind to a 10% one; the block of seeds is chosen so that no cell is near its edge, on arm64 and on x86-64 (float32 trajectories differ in the last bits between architectures). The adapters take `backend`, `precision` and `device` parameters, and `quick.toml` has an `auxein-core-ga-torch` entry so that the CI benchmark job runs torch end to end.
- Real GPU runs (CUDA and Metal) are verified by a small **GPU smoke suite** (`tests/gpu/`), run by hand before releases and whenever the numeric layer changes. Standard hosted CI runners don't provide GPUs. The suite is selected by the `gpu` marker and a device (`uv run pytest -m gpu --device mps`, or `cuda`, `cuda:1`, or `cpu` as a dry run, or `AUXEIN_GPU_DEVICE`) and is skipped everywhere else. It runs in float32, and in float64 too except on Metal, where float64 is a configuration error that a test checks. It covers `Backend` validation, random streams and their `state_dict` round trips, `Box` sampling, end-to-end runs of the three strategies with the results staying on the device, the batched point mass through `EpisodeEvaluator`, and checkpoint, extension and replay of a device run (identical on the same device). It also runs an informational performance probe (CPU against device, at three sizes, with a cheap and a heavy objective; nothing asserted). Each run writes `docs/gpu-smoke/<device>-<date>.md` (machine, OS, torch and device, the commit, pass or fail per test, the probe's table); the reports of real devices are committed, as the evidence for criterion 5 of §11.4.

---

## 8. Randomness and reproducibility

- **One seed per run.** The user supplies one seed (`RunSeed`). Independent streams are derived from it with `numpy.random.SeedSequence(entropy=seed, spawn_key=(name_id, *keys))`:
  - one for the strategy (`"strategy"`)
  - one for scenario generation (`"scenarios"`)
  - one per candidate evaluation, derived from the candidate's (deterministic) id (`"evaluation", candidate_id`); **always numpy-backed**, whatever the run's backend (see below)
  - for the agent layer (§6): the agent's stream per episode, `("episode", candidate_id, scenario_index)`, numpy-backed like the per-candidate stream, and one stream per batch for a batched environment, `("episode-batch", first_candidate_id)`, on the run's backend
  - for scenarios: the scenario set's own seed gives each scenario its `seed` (`("scenario-seed", index)`) and the parameters generator its stream (`("scenarios", index)`); a scenario's **world** stream is `("world", *keys)` derived from the scenario's seed alone; a held-out evaluation uses a seed derived from the run's (`"held-out"`)
- **Stable names.** A stream name is mapped to an integer with CRC-32 of its UTF-8 bytes, never with Python's `hash()` (randomised per process), so the same seed, name and keys give the same stream in any process on any machine. Keys are integers in `[0, 2**32)`. Two names could in principle collide in CRC-32 (about one chance in four billion); a test pins the names Auxein uses.
- **Evaluation randomness follows the candidate**, not the worker or the time of evaluation. Evaluations are reproducible in any parallel or distributed setup.
- **Exception for vectorised evaluators.** A vectorised function draws its randomness for the whole batch at once, so per-candidate streams would be unusable. It receives **one stream per batch**, derived from the id of the batch's first candidate (`SeedSequence` key `("evaluation-batch", first_id)`). That is deterministic because the composition of a batch is. Per-candidate streams remain the rule for per-candidate evaluators.
- **Backend-native generation.** The random layer (`RandomStream`) wraps numpy's `Generator(PCG64)` on CPU and `torch.Generator` on the target device, seeded from the derived streams. Random numbers are produced where the arrays live. Streams offer `uniform`, `normal`, `integers`, `permutation` and `choice`, returning arrays in the backend's namespace, device and dtype (integers as int64).
- **Common random numbers (§6.4).** *World* randomness (disturbances, noise) comes from the scenario's seed, so every candidate faces the identical realisation; *agent* randomness (a stochastic policy, later LLM sampling) comes from the per-(candidate, scenario) stream, so it follows the candidate and the scenario, never the worker or the time. The batched path's one stream per batch is the same kind of exception as the vectorised evaluator's (derived from the first candidate's id). A test pins all the stream names.
- **Per-candidate evaluation streams are numpy-backed.** `ctx.rng_for(id)` returns a numpy stream (with the run's float precision) on every backend. Evaluating one candidate is host-side Python anyway, and these are the only streams a run creates by the tens of thousands. numpy streams use the full 128 bits of the derived seed, which settles the limit of torch CPU generators, an MT19937 that accepts only 32 bits of seed and would make two derived streams collide around 65,000 of them (decided in step 4a). numpy streams are also picklable, so a candidate's stream can be sent to a worker process. The strategy stream and the vectorised per-batch stream (`batch_rng_for`) stay backend-native: a run creates only a few, and they feed array operations on the target device.
- **Scope of reproducibility:** identical results for the same seed, backend, precision **and CPU architecture** (and, on a GPU, the same device: the random streams and the reductions of a device are its own). Across architectures, float64 is stable in practice and float32 is not, and neither is *byte*-identical. Element-wise arithmetic is the same IEEE operation everywhere, but reductions (sums, dot products, means) add in an order that depends on the SIMD width, on fused multiply-add and on the library (numpy dispatches differently even between x86-64 machines), and the vectorised `exp`, `log` and `cos` behind the random generators differ in the last bit too. The step-3a golden test found it first: the objective values of a float64 numpy run differ in the last digits between arm64 and x86-64, so a digest of the run's bytes does not travel. What differs is tiny in float64 (a few units in the sixteenth digit), so which candidate wins a comparison almost never changes and the *trajectory* (ids, evaluation counts, the sequence of improvements) is the same; in float32 the differences are of the order of 1e-7 relative, close comparisons do flip, and a trajectory can diverge, so a float32 run's event log and any digest of it are tied to the architecture. Tests that compare a run with a pinned result therefore pin the trajectory with a tolerance, or the values of one architecture, never a digest across machines. A run recorded on one architecture and resumed on another replays by comparing its candidates byte for byte (§10.4), and a float32 run can fail that check, and say so, on the other machine. Different backends or precisions give different (equally valid) runs, and a test (§7.4) checks that they search equally well.
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
- **`NSGA2` and steady-state delivery.** Its fronts and crowding distances are recomputed at every `tell`, so with steady-state delivery the population depends on how many results are told at once. In deterministic mode that is fixed by the seed and the batch size (results are told one at a time, in ask order), so the event log is identical whatever the concurrency; in throughput mode it is not reproducible, as for every strategy.
- **The guarantee covers resumed runs.** A run killed at any point and resumed (§10.4) has the event log of the run that was never interrupted, and a finished run extended with a larger evaluation budget has the log of one run with that budget. Two things make it so. *Replay*: a resumed run restores its strategy and asks again, and the sequence of `ask` and `tell` calls depends only on the seed and `batch_size`, so it regenerates the recorded candidates exactly; their recorded evaluations are used instead of evaluating again. *The hard-budget rule*: candidates that were asked but dropped, or evaluated but not told, at the old evaluation limit (§9.3) are regenerated by replay and now evaluated and told, as in the longer run. For this to be exact, the driver keeps the last checkpoint taken before the budget first shaped what the run did (a smaller final ask, a surplus dropped) and takes none after it (§10.4).
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
    checkpoint_every=300.0,                   # seconds of run time between checkpoints (default 300; math.inf: never); needs run_dir
    checkpoint_every_evaluations=None,        # also every this many evaluations; needs run_dir
    keep_checkpoints=2,                       # how many to keep (0: none, and a resume replays from the start); needs run_dir
    genome_store_threshold=4096,              # genomes larger than this many encoded bytes are stored once by hash (None: never, §10.3)
)

result = await auxein.arun(...)  # same arguments

# continue a run that was interrupted, killed or finished (§10.4): the same script with `resume` instead of `run`
result = auxein.resume(..., budget=auxein.Budget(evaluations=40_000), run_dir="runs/rastrigin-ga")  # also `aresume`
```

The common case needs the one import. The top-level API is deliberately short: `run`, `arun`, `resume`, `aresume`, `Budget`, `RunResult`, `Objective`, `Result`, `BatchResult`, `Status`, `Box`, `Backend`, `FunctionEvaluator`, `VectorisedEvaluator`, `RandomSearch`, `GeneticAlgorithm`, `open_run` and `RecordingDisabledWarning` (plus `__version__`). Everything else is imported from its subpackage: `auxein.core` (candidates, batches, evaluations, the protocols), `auxein.strategies.ga` (the operators), `auxein.backend`, `auxein.random`, `auxein.spaces`, `auxein.driver`, `auxein.evaluators`, `auxein.recording`.

- **Recording is opt-in.** Without `run_dir` nothing is written to disk, and the run emits a `RecordingDisabledWarning` (a `UserWarning`) once per run, with a stack level that points at the user's call. `warnings.filterwarnings("ignore", category=RecordingDisabledWarning)` silences it. A `run_dir` that already exists and isn't empty is refused, so runs never mix.
- Results are delivered by generation or in steady state (§9.2), with up to `concurrency` evaluations in progress. The defaults (`concurrency=1`, generation delivery) keep today's sequential behaviour and cost. Failures are handled by `failure_policy` (§6.6), and `timeout` limits one evaluation.
- `run()` also accepts a `clock` (the time source of the wall-time budget and of the checkpoint interval, injectable for tests).
- **Checkpoint arguments** exist only for recorded runs: `checkpoint_every`, `checkpoint_every_evaluations` and `keep_checkpoints` without a `run_dir` are a `ValueError`. They are not part of the run's identity: a resume may use different ones.
- **`resume` / `aresume`** take the arguments of `run` / `arun`, with `run_dir` required, and continue the recorded run in that directory (§10.4). Only the budget may differ from the recorded run; any other difference is a `ConfigurationMismatchError` that lists each setting with its recorded and its given value.

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
- **Budgets are cumulative across the sessions of a resumed run** (§10.4): evaluations used and cost units are totals over the whole run, recomputed from the recording, so a cost unit that is new to the budget still counts what was spent before. **Wall time is cumulative *active* time**: the time between sessions does not count, and neither does time that elapsed in a session that was killed after its last checkpoint (nothing recorded it). Wall time is not checked while a resumed run replays its recording (the clock says nothing about a recorded decision), only once it goes live. A resumed run whose budget is already exhausted returns the result of the recording without evaluating anything. The evaluation budget of a resume cannot be smaller than the number of evaluations already recorded.
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
  events.sqlite          # event log: candidates, lineage, evaluations, events, checkpoints
  checkpoints/           # ckpt-<event seq>/state.json + arrays.npz: snapshots of strategy, driver and random-generator state
  writer.lock            # the writer's process id, while a process is writing (§10.4)
  held_out.json          # optional: the report of a held-out evaluation (§6.4)
  # later: artifacts/ (heavy outputs: trajectories, transcripts, logs)
  # (the content-addressed genome store is not a directory: it is a table of events.sqlite, §10.3)
```

### 10.2 Event log (SQLite)

- **Append-only.** Written by the driver, which is the single writer, one transaction per batch (per told candidate in steady-state delivery), so it's safe against crashes. The database is in WAL mode and is checkpointed when the run ends.
- **Schema (version 4)**, as implemented; a `schema_version` table holds the version number. Version 2 (step 5) added `candidates.event_seq` and the `checkpoints` table, version 3 (step 6a) the `episodes` table, version 4 (step 6b) the genome store (`blobs` and `candidates.genome_hash`) and `operator_calls`. **A run recorded with an earlier version cannot be resumed, and the reader refuses it** (there is nothing to migrate: no one has such runs).
  - `candidates`: `id` (primary key), `step`, `origin`, `genome_kind` (`array` or `json`), `genome` (raw bytes or JSON text), `genome_dtype` and `genome_shape` (arrays only), `created_at`, and **`event_seq`**: the sequence number of the `ask` event whose transaction recorded the candidate together with its evaluation. It tells exactly what was recorded after a given point, which truncation and replay need
  - `lineage`: `parent_id`, `child_id`, both indexed
  - `evaluations`: `candidate_id` (primary key), `status`, `objectives`, `constraints`, `descriptors` and `cost_units` (JSON objects of name to value), `wall_time`, `error`, `finished_at`
  - `events`: `seq`, `kind` (`ask`, `tell`, `stop` or `resume`), `step`, a JSON `payload`. A candidate's tell is its `ask` event's successor; an `ask` without a following `tell` was recorded but never told (a final batch cut by the budget, or a process killed between the two writes)
  - `checkpoints`: `id`, `event_seq` (the checkpoint is consistent with the events up to this one), `evaluations_used`, `path` (relative to the run directory), `created_at`
  - `blobs`: `hash` (primary key: the SHA-256 of the encoded genome, in hex) and `data`; and `candidates.genome_hash`, set when the genome is in the store (then `genome` is empty). See §10.3.
  - `operator_calls`: `key` (primary key: the SHA-256 of the operator's name, its encoded inputs and a draw from the strategy's stream), `operator`, `output` (the proposed genome as canonical JSON), `cost_units`, `wall_time`, `metadata`, `event_seq` (the last event written when the call was made) and `created_at`; see §3.5. They are written at once, in their own transaction, and **never deleted**: a call is a paid-for result addressed by its content, so keeping the ones made after a checkpoint that throughput mode truncates costs nothing and lets the redone work reuse them whenever it asks for the same call.
  - `episodes`: `candidate_id`, `scenario_index`, `scenario_id`, `status`, `measurements` (a JSON object of name to value; empty for an episode that did not succeed) and `error`; primary key (`candidate_id`, `scenario_index`). They are written **in the same transaction as the candidate's evaluation**, at tell time, in batched inserts (thousands of candidates by tens of scenarios record without a measurable slowdown beyond the rows themselves), and `Evaluation.raw` is a `RawRef` with the key `episodes/<candidate id>`. Evaluators other than the episode evaluator write none.
- **A `resume` event** is written for each resume. Its payload holds the `mode` (`replay` or `truncate`), the `checkpoint` used (its event sequence, or null when the run restarts from the beginning), the number of evaluations `replayed` or `truncated`, and the old and new budget. A run has **one `stop` event, at its end**: a resume deletes the `stop` of the earlier session (the session history in `metadata.json` keeps what it said).
- **Candidates are recorded when they are told**, together with their evaluation, and a candidate that is never told has no row: one dropped by the budget truncation or by a wall-time stop (§9.3) was never evaluated. In generation delivery that is once per batch, in ask order. In steady-state delivery it is once per candidate, so the `ask` and `tell` events come per candidate (with a count of 1), in ask order in deterministic mode and in completion order in throughput mode. The tables are keyed by candidate id; the order of telling is in the `events` table.
- **`metadata.json`** records, besides the seed, backend, budget and components, how the run was scheduled: `batch_size`, `in_flight_window`, `concurrency`, `executor` (as resolved, never `auto`), `delivery` (as resolved), `deterministic`, `failure_policy`, `timeout` and `initial_failure_guard`. **`metadata.json` also holds `sessions`**, oldest first: the first start and each resume, with `mode` (`start`, `replay` or `truncate`), the `budget` of that session, the `versions` and `git` state it ran with, and, once it ended, `ended_at`, `status`, `stop_reason`, `evaluations_used` and the cumulative `wall_time`. A session that was killed has `status: running` and no end. The configuration at the top (seed, budget, components...) is the original and never changes; the top-level `status`, `stop_reason` and `summary` describe the latest session. The **summary** written when the run ends holds the totals (`evaluations_used`, `wall_time`, the best candidate), the **failure counts** `status_counts` (`ok`, `failed`, `timeout`) and `abandoned_evaluations` (timed-out thread evaluations still running when the run ended).
- **Determinism:** the contents of `events.sqlite` are identical for the same seed, backend, precision and `batch_size`, apart from `created_at`, `finished_at` and `wall_time`, which are timestamps and measured times, apart from the `resume` events and `checkpoints` rows of a run that was resumed (§10.4), **whatever `concurrency`, the executor and sync or async user code** (in generation delivery, and in steady-state delivery with `deterministic=True`; §8.1).
- **Re-aggregation.** `RunReader.episodes(candidate_id)` returns a candidate's recorded episodes, and `RunReader.reaggregate(aggregator)` recomputes objectives, constraints, descriptors and cost units for every recorded candidate from the stored measurements, without re-simulating: the same run seen through another aggregator (a worst case instead of a mean). It returns the candidates in id order; one with a failed episode is reported with that status and no values.
- **Lineage queries** ("all descendants of X", "the ancestry of the best agent") are single recursive SQL queries; `auxein.recording.open_run(path)` offers `ancestry(id)` and `descendants(id)`, and iterates over the evaluations with their genomes decoded.
- **Export** to JSONL or Parquet with one command comes later.

### 10.3 Genomes and artifacts

- Genomes are stored without pickle: array genomes (numpy arrays, or torch tensors copied to the host) as raw C-order bytes plus dtype and shape; structured genomes as the **canonical JSON** of their codec encoding (§4.1); other genomes as JSON when they are JSON-serialisable. A genome that is none of these is a clear error that points to the codec.
- **The genome store** (step 6b) is a table of `events.sqlite`, not a directory of files, so that writing a blob is part of the batch's transaction (crash-safe, consistent with the single-writer design) and truncation and replay stay simple. A genome whose encoding is larger than `genome_store_threshold` bytes (a `run` argument; default 4096; `None` keeps everything inline) is stored **once** in `blobs`, keyed by the SHA-256 of its encoded bytes, and its `candidates` row has an empty `genome` and the hash in `genome_hash`; smaller genomes stay inline. It applies to array and structured genomes alike, and dtype and shape stay per candidate. The reader and replay resolve references transparently, so replay's byte-identical check is unchanged. Throughput-mode truncation (§10.4) deletes the blobs that no remaining candidate refers to. The threshold is a recording detail, not part of the run's identity: a resume may use another. On a run of 2,000 candidates with a few large genomes the database is 6 to 56 times smaller (measured in step 6b: 10.4 MB to 0.41 MB for five distinct 5 KB structured genomes, 33 MB to 0.59 MB for ten distinct 16 KB array genomes, 10.4 MB to 1.7 MB when 250 genomes are distinct).
- **Runs recorded with schema 3 or earlier cannot be resumed and the reader refuses them** (there is nothing to migrate: no one has such runs).
- **Heavy artifacts** (trajectories, transcripts) are optional and stored by reference. **Default: kept for failed evaluations only.** Options keep them for the best candidates or a sample.

### 10.4 Checkpoints and resume

**Checkpoints** exist only for recorded runs (`run_dir`), and make a long run survive being killed, interrupted or finished too early.

- **When they are written.** Every `checkpoint_every` seconds of run time (default 300), optionally every `checkpoint_every_evaluations` evaluations, when a run ends normally, and when it is interrupted (`KeyboardInterrupt` or cancellation) *if the state is consistent at that moment*. Setting any checkpoint option without `run_dir` is an error. The last `keep_checkpoints` (default 2) are kept; `0` writes none, and a resume then replays from the start.
- **Consistency.** A checkpoint is taken only where the strategy, the driver and the recording agree: between tells, with everything told so far already recorded. In steady-state delivery that is any time the driver waits, because the window (candidates asked and not told) is part of the checkpoint. In generation delivery it is between batches, so an interrupt in the middle of a batch writes no checkpoint (the strategy has asked candidates that no one has told) and a resume uses the one before. Evaluations in progress carry on while a checkpoint is written.
- **Contents.** A format version and the event sequence number it is consistent with, plus:
  - the strategy's `state_dict` (random-generator state included);
  - the driver's state: the id issuer, the ids issued but never carried by a batch, the last step, evaluations used, cost totals, the failure counters, the guard's state and the first failure, the cumulative wall time, the largest ask so far, and for steady-state delivery the window: the queued and in-flight candidates, in ask order, with their ids, lineage, origins and genomes.
- **Not in a checkpoint.** The result tracker (`best`, `pareto_front`, `trace`) is **rebuilt from the recording** on resume, which is fast and keeps one source of truth (cost totals for the units of the budget are recomputed the same way, so a new unit counts what was spent before). Results that finished but were not told yet (held back in deterministic steady-state mode, or waiting in throughput mode) are not saved: those candidates are evaluated again.
- **Format.** One directory per checkpoint, `checkpoints/ckpt-<event seq>/`, with `state.json` (everything but arrays, with a reference where an array was) and `arrays.npz` (the arrays; torch tensors go through numpy and come back on the run's device). **No pickle anywhere**: arrays are loaded with `allow_pickle=False`, and `validate_state_dict` rejects anything that could only be saved with it. A checkpoint is written to a temporary directory, flushed to disk and renamed into place, then registered in the `checkpoints` table; the oldest beyond `keep_checkpoints` are deleted. A crash while writing leaves the previous checkpoints untouched, and a leftover temporary directory is removed on the next resume. A damaged newest checkpoint is skipped with a warning, and the one before is used.
- **The budget shapes the run, so checkpoints stop.** A checkpoint can be continued with a larger budget into the run that budget would have made only if the budget had not yet changed anything the run did. It changes things at the end: the last ask is smaller than `batch_size`, a surplus is dropped (§9.3), queued candidates are dropped on a stop. So when the remaining evaluation budget comes within the size of one ask, the driver writes one more checkpoint, still clean, and once the budget has shaped the run it takes **no more** (not periodic, not at the end, not on interrupt). A run that ends exactly on its budget has not been shaped and checkpoints at its end. This is what makes extension exact.

**Resume** continues a recorded run with `auxein.resume(...)` / `auxein.aresume(...)`, which take the arguments of `run` / `arun` with `run_dir` required: the intended workflow is to run the same script with `resume` instead of `run`.

- **What can be resumed.** Runs recorded with the current schema whose status is `completed`, `interrupted` or `failed`, and runs that were killed (status still `running`). Two processes never write to one run: the recorder takes a lock file (`writer.lock`, with the writer's process id) for as long as it writes. A live writer makes a second one refuse with a message naming the process; the lock of a process that was killed is recognised and taken over.
- **Only the budget may change.** Everything else must equal the recorded run: the problem (space, objectives and directions, constraints, descriptors), seed, backend and precision, `batch_size`, `delivery`, `deterministic`, `concurrency`, `executor` (as resolved), `timeout`, `failure_policy`, the guard, and the strategy's and evaluator's class and `repr`. Any difference raises `ConfigurationMismatchError` listing every mismatching setting with its recorded and its given value, before anything is written. The checkpoint options and the `clock` are not part of the run. **Auxein cannot check that the evaluator's *code* is unchanged**: replay does not run it, so keeping it the same is the user's responsibility. The evaluation budget cannot go below what is already recorded.
- **The mode comes from the recorded `deterministic` setting.**
  - **Deterministic mode replays.** The strategy and driver are restored from the latest checkpoint (or built afresh if there is none) and asked again. The driver rebuilds the result tracker from the recording, then re-runs the same loop with the *same decisions as a live run*: the same ask sizes, window refills, truncation rules and stop checks, under the new budget. For each candidate it regenerates for which the recording holds an evaluation, it checks that the candidate is exactly the recorded one (same id, origin, step and parents, **byte-identical genome**), and tells the strategy the **recorded evaluation instead of evaluating it again**. **No recorded evaluation is ever repeated**, with one exception: an evaluator that draws its randomness once per batch (a `VectorisedEvaluator`, or an `EpisodeEvaluator` on its batched path; they say so with `batch_sensitive`) evaluates a final batch that a larger budget has made longer whole again (it was cut at the old limit, never told) to give what the longer run gives, and the recording replaces the shorter batch, episodes included. For any other evaluator the recorded candidates of that batch stay as they are, with their episodes, and the batch grows by the new ones. Replayed evaluations keep their recorded episodes: replay writes none twice. Once the recorded evaluations are used up, the run goes on live. Any difference stops the resume with a `ReplayMismatchError` that names the first candidate that differs and the likely cause (the strategy's configuration or code changed, a different seed, an edited recording), and records nothing: the session is written only when the resume first writes something.
  - **Throughput mode truncates and redoes.** The run restarts from the latest checkpoint; everything recorded after it (candidates, lineage, evaluations, episodes, events, and the genome blobs no remaining candidate refers to) is deleted in one transaction and that work is redone live. Recorded calls of external operators are kept (§3.5, §10.2). Candidates in flight at the checkpoint are re-issued with the same ids and genomes. With no checkpoint, everything recorded is deleted and the run restarts from the beginning.
- **Extending a finished run** with a larger evaluation budget gives, in deterministic mode, **the same event log and result as one uninterrupted run with the larger budget**, for both delivery modes. It follows from the hard-budget rule (§9.3): the candidates asked but dropped, or cut, at the old limit are regenerated by replay and now evaluated and told. A candidate that the old run evaluated but never told (the cut part of a final batch) is used from the recording and re-recorded as part of the longer batch. This needs the strategy to propose the same first candidates when asked for fewer (`ask(16)` is a prefix of `ask(64)`), which `RandomSearch` and the default `GeneticAlgorithm` do. A strategy that does not, such as a `GeneticAlgorithm` with `offspring_size=None` (it breeds exactly `n` children, drawing its random numbers per array) in generation delivery with a budget that is not a multiple of `batch_size`, cannot be extended: the first candidate to differ raises the replay mismatch. Steady-state delivery is not affected, because it asks one candidate at a time.
- **Resuming a complete run** with exactly the budget it finished with returns its result from the recording without evaluating anything, and warns (`ResumeWarning`); it does not even add a session. A run that was killed with its budget already used up replays its tail and then stops.
- **`fail_fast` runs.** The evaluation that stopped a `fail_fast` run was never recorded (the failure is raised before anything is written), so a resume evaluates it again: that is how to continue after fixing a bug in the evaluator. A run stopped by the all-failures guard replays the same recorded failures and stops again, because the guard is part of the configuration that cannot change.
- **Budgets and the result.** Evaluations used, cost totals and failure counts are cumulative, and **wall time is cumulative active time** (§9.3). The `RunResult` of a resumed run covers the whole run: `best`, `pareto_front`, `trace`, `status_counts` and `evaluations_used` are those of all sessions.
- **The record of a resume.** Each resume writes a `resume` event and adds a session to `metadata.json` (§10.2). A run killed twice has two sessions without an end.
- **Platforms.** The lock checks whether the writer's process exists with `os.kill(pid, 0)`, which terminates the process on Windows, so recording a run is refused there with a clear error. Only macOS and Linux are supported and tested (the kill-and-resume tests use `SIGKILL`).

### 10.5 Recorder interface

The recorder is pluggable. The SQLite run directory is the built-in implementation, and the only one that also serves as the run's `OperatorLog` (§3.5) and that takes the space's codec (`use_codec`, called by the driver) to record structured genomes. Integrations (e.g. MLflow, Weights & Biases) can be added later as additional sinks.

---

## 11. Validation plan

Features are validated by **small, purpose-built test fixtures** (tiny deterministic environments, short structured genomes, in `tests/`) plus the **benchmark harness** (`benchmarks/`), not by toy domains. A fixture exists to test a pipeline end to end and is as small as that allows; it is not an example for users. The two toy domains planned earlier are deferred (§11.2).

### 11.1 Fixtures and the benchmark harness

- **Test fixtures** live under `tests/` and are deterministic by construction. The agent layer's (step 6a, `tests/support/pointmass.py`) is a point mass on a line that a three-gain controller must bring to a target under a constant drift: drift and target come from the scenario's params, a little noise on the force from its seed. It reports raw measurements (final distance, steps, control effort, success, maximum overshoot) and is implemented three ways with the same maths, per episode, batched (in the run's array namespace) and step by step through the adapter; a test checks that they agree (exactly in float64 numpy, within tolerance in float32 and torch). A genetic algorithm solves it in a few hundred evaluations. Other fixtures: structured genomes of a few fields (step 6b), and scripted evaluators and strategies for the driver.
- **The benchmark harness** compares strategies on function-optimisation problems and measures the engine's overhead (§11.4, criterion 2). Since step 8b it also has a **multi-objective family** (§11.4, criterion 7): ZDT1, ZDT2, ZDT3 and DTLZ2 with their known fronts, the hypervolume (against a fixed reference point, 1.1 times the nadir of the true front) and the IGD+ of the non-dominated set of everything a run has evaluated, kept by the harness and not by the algorithm, and pymoo (a benchmark-only dependency, never imported by `auxein`) as the external reference.

### 11.2 Deferred: toy domains

**Not planned.** The two toy domains below were meant to validate the design end to end. They were deferred to an unspecified later time, and their descriptions are kept only for reference.

**Toy domain A: two-ship encounter (numeric, vectorised).**

- **Dynamics:** own ship with first-order Nomoto yaw dynamics (`T·ṙ + r = K·δ`) at constant speed, steering towards a waypoint, with one target vessel on a crossing course.
- **Genome:** a fixed-size controller (heading-controller gains plus avoidance parameters, or a small fixed-size network) in a `Box` space.
- **Scenarios:** encounter geometries, target speeds and headings, currents. Selection and held-out sets.
- **Objectives:** time to waypoint (minimise), rudder effort (minimise).
- **Constraint:** closest point of approach ≥ threshold.
- **Descriptor:** passing side, mean rudder angle.
- **Implementation:** vectorised over candidates × scenarios. It exercises the array fast path, the batched episode interface, the PyTorch backend and float32.

**Toy domain B: prompt evolution with a mock LLM (structured, asynchronous).**

- **Genome:** a structured prompt configuration (instruction list, example slots, tool flags).
- **Evaluation:** a deterministic mock "LLM" whose answer quality depends on genome features. It injects random latency, occasional failures and timeouts, so it exercises async evaluation, steady-state delivery, deterministic mode and failure policies, at no cost.
- **Operators:** structure-aware text mutations, plus an "LLM-driven mutation" operator interface implemented with the mock.
- **Objectives:** task success (maximise), token cost (minimise).

### 11.3 Function optimisation and regression

- Function optimisation (e.g. Rastrigin) through `VectorisedEvaluator`.
- Linear and logistic regression.
- Polynomial regression with a fixed maximum degree, structure genes and a complexity objective (§4.4).

These are rewritten as four notebooks for the new core (step 9): Rastrigin, linear regression, logistic regression and polynomial regression.

### 11.4 Acceptance criteria

1. **Generality:** *deferred* together with the toy domains (§11.2). It said that both toy domains and all problems in §11.3 run on the same core, with no special cases in the core. The agent layer (step 6a) and structured genomes (step 6b) are validated by test fixtures instead.
2. **Benchmark:** with a harness adapter for the new core, the `GeneticAlgorithm` strategy at equal evaluation budgets is **not statistically worse than the `v0.2.0` baseline** on any problem and dimension of the benchmark suite. Its **overhead per evaluation is lower**.
   *Outcome (step 3a, [`benchmarks/reports/core-ga-0.3.0-dev/`](../../benchmarks/reports/core-ga-0.3.0-dev/report.md)):* **met.** The default `GeneticAlgorithm` is better than the 0.2.0 default on all 15 problem × dimension cells of the suite (large effect, Holm-adjusted p ≤ 10⁻⁷), and its time per evaluation is lower at every population size and dimension of the overhead benchmark (4.9 to 7.6 µs against 9.2 to 12.1 µs), after an optimisation of the driver and the strategy that the first run of the comparison called for.
3. **Determinism:** in deterministic mode, the same seed produces an identical event log across synchronous and asynchronous evaluators and any worker count.
4. **Resume:** a run killed and resumed from a checkpoint produces the same event log as an uninterrupted run.
   *Outcome (step 5):* **met.** Tests kill runs with `SIGKILL` at three points (before the first checkpoint, between checkpoints, near the end) in both delivery modes, with inline, thread and process executors, `concurrency` 1 and 4, and `RandomSearch` and `GeneticAlgorithm`, and resume them: the event log equals the uninterrupted run's, and the evaluation function runs only for the evaluations that were not recorded. A run killed twice, an interrupted run, a crash between recording a batch and telling it, replay with and without checkpoints, and a run extended from 2,000 to 5,000 evaluations all give the same log as the run that was never stopped; extension is also tested on numpy float32 and PyTorch in both precisions.
5. **Backends:** numeric tests pass on numpy and PyTorch (CPU) in float64 and float32. The GPU smoke suite passes on CUDA and Metal.
   *Outcome (step 7):* **CPU parts met; GPU parts pending.** Unit tests run on the four configurations and integration tests on the two corners (§7.4); the torch backend's search quality is cross-checked against numpy's; every numeric code path was audited, and the one accidental numpy-only path found was fixed. The GPU smoke suite exists and passes on Apple Metal (`docs/gpu-smoke/mps-2026-10-09.md`, run by the author of the step on an Apple Silicon Mac) and on the CPU as a dry run. **A CUDA run is still pending** (no CUDA machine was available), and the Metal report is the author's, to be repeated by the project owner on the release machine; the criterion is complete when a CUDA report is committed.
6. **Failures:** runs with injected failures and timeouts complete, with every failure recorded and handled per the configured policy.
   *Outcome (step 4b):* **met.** Tests inject exceptions, non-finite objectives, timeouts (async, process, thread), worker crashes (`os._exit`, `SIGKILL`) and custom evaluators that report failures, under both policies, both deliveries and every executor: runs complete, each failure is recorded with its status and error, the counts are in `RunResult` and the summary, and no worker process or non-daemon thread is left behind. The deterministic event log with candidate-driven failures is identical across `concurrency` and executors. A `GeneticAlgorithm` with 30% random failures still beats random search on the 10-D sphere at the same budget.
7. **Multi-objective:** `NSGA2` is not statistically worse than pymoo's NSGA-II on the final hypervolume on any problem of the multi-objective benchmark, and is significantly better than random search on every problem.
   *Outcome (step 8b, [`benchmarks/reports/nsga2-0.3.0-dev/`](../../benchmarks/reports/nsga2-0.3.0-dev/report.md)):* **met.** On ZDT1, ZDT2, ZDT3 (25,000 evaluations) and DTLZ2 (30,000), 25 runs each, with identical population, operators and parameters, `NSGA2` has a higher final hypervolume than pymoo's NSGA-II on all four problems (Holm-adjusted p ≤ 2 × 10⁻⁴, A₁₂ between 0.81 and 1.00), by a small margin (about 0.01 to 0.1 % of the front's hypervolume), and is far above random search. The margin is not explained away in this document: the two implementations differ in the form of SBX (unbounded with repair against bounded), in duplicate elimination and in children per pair, and the comparison is the one stated, not an isolation of the cause.

---

## 12. Package layout

What exists after step 8b (the packages marked "later" are planned):

```
auxein/
  __init__.py      # the public API (§9.1) and __version__
  core/            # ids, Candidate, batches (ListBatch, ArrayBatch), Evaluation and EvaluationBatch, Result and BatchResult,
                   # the normalisation of what user code returns, ProblemSpec, StateDict, the Strategy/Evaluator protocols
  spaces/          # Space protocol, Box, MixedSpace (Real, Integer, Binary, Categorical), SequenceSpace, the codec protocol and canonical JSON
  backend/         # Backend, array-API helpers, precision, device validation
  random/          # RunSeed, backend-native RandomStream
  driver/          # run/arun, resume/aresume, Budget, RunResult and its tracking, held-out evaluation, driver errors and warnings
  strategies/
    random_search.py
    scalarisation.py  # Scalarised, WeightedSum, Chebyshev, best_by_scalarisation
    nsga2/         # NSGA2 (with its engines), non-dominated sorting and crowding distance
    ga/            # GeneticAlgorithm and its operators (selection, recombination incl. SBX, mutation incl. polynomial mutation, bounds repair; integer, binary and categorical mutation)
    structured/    # StructuredGeneticAlgorithm, the sequence operators, external proposal operators and their recorded calls
  environments/    # Scenario, ScenarioSet, EpisodeResult, the Environment/BatchedEnvironment/Decoder interfaces, StepEnvironment
  aggregators/     # Aggregator and its reductions (mean, minimum, maximum, total, quantile, cvar_upper, cvar_lower)
  evaluators/      # FunctionEvaluator, VectorisedEvaluator, EpisodeEvaluator
  recording/       # Recorder protocol, SQLiteRecorder run directory, genome encoding, checkpoint files, the writer lock, replay, a reader
  # later: strategies/external/ (PycmaStrategy), a multi-objective strategy (8), recording (export)
benchmarks/        # the benchmark harness (single- and multi-objective, with pymoo as a benchmark-only reference), its configs, and the committed
                   # reports (frozen results are history)
tests/             # tests of the package: unit tests on four backend configurations, integration tests on two corners (§7.4);
                   # tests/gpu is the GPU smoke suite (skipped unless --device is given), tests/support the shared fixtures
docs/gpu-smoke/    # the reports of GPU smoke runs, one file per device and day
docs/design/core.md
# later: examples/ (the four rewritten notebooks, step 9)
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
5. **Checkpoints and resume (done):** checkpoints of strategy, driver and random-generator state (JSON and arrays, no pickle, written atomically), `resume` / `aresume` by replay (deterministic mode) or truncate-and-redo (throughput mode), extension of finished runs, the single-writer lock, recording schema 2. The content-addressed genome store moved to step 6b, with the other genome storage.
6. **Agents and structured genomes:**
   - **6a, agent layer (done):** environment interface, step-level adapter, decoders, episode evaluator, scenario sets with held-out splits, aggregators, recording of per-scenario measurements (schema 3), held-out evaluation, and a point-mass test fixture implemented three ways.
   - **6b, structured genomes (done):** codecs and canonical encoding, `SequenceSpace`, `StructuredGeneticAlgorithm` with structure-aware operators, recorded and replayed external proposal operators (the slot for LLM-driven mutation), and the content-addressed genome store in `events.sqlite` (recording schema 4).
7. **PyTorch everywhere (done, GPU runs pending):** every numeric code path audited and proven on numpy and PyTorch in float64 and float32 (unit tests on four configurations, integration tests on two corners), a statistical cross-check that the torch backend searches as well as numpy, device-placement tests, focused float32 numerics tests (two bugs fixed: the CVaR tail size and sigma scaling with objectives beyond float32), the reproducibility scope across CPU architectures (§8), and the GPU smoke suite with its reports (§7.4). The suite has been run on Apple Metal; a CUDA report is pending (§11.4).
8. **Mixed spaces and multi-objective**, in two steps:
   - **8a, mixed search spaces (done):** `MixedSpace` with real, integer, binary and categorical dimensions in one array genome, type-aware operators in the numeric `GeneticAlgorithm` (MIES-style integer mutation, bit flips, categorical resampling, discrete recombination), `RandomSearch` on them, and two fixtures: a mixed-variable problem with a known optimum, and a polynomial regression with structure genes.
   - **8b, multi-objective (done):** `NSGA2` (non-dominated sorting with constrained domination, crowding distance, the crowded tournament through `PopulationView`) on `Box`, `MixedSpace` and `SequenceSpace`, SBX and polynomial mutation, scalarisation (`Scalarised`, `WeightedSum`, `Chebyshev`), and a multi-objective benchmark (ZDT1-3 and DTLZ2 against pymoo's NSGA-II and random search, by hypervolume and IGD+).
9. **External strategies, adapters and examples:** `PycmaStrategy`, the Gymnasium adapter, and the four rewritten notebooks: Rastrigin, linear regression, logistic regression, polynomial regression (formerly step 9; the polynomial one depends on step 8).

---

## 15. Open questions

- **Conditional dimensions:** "momentum only if the optimiser is SGD". Not in 8a. Options are an inactive-gene mask derived from the other genes (the genome stays a fixed-length array, the inactive genes are ignored by the objective and by mutation and carried by recombination), or a hierarchy of spaces. The fixed-length array is what the rest of the core relies on, so the mask is the likelier route.
- **Mixed dimensions in other strategies:** the `StructuredGeneticAlgorithm` and `SequenceSpace` have no numeric genes. A genome that is a structure plus parameters would need a product space and per-part operators.
- **Duplicate children on small discrete spaces:** with per-gene rates of `1/k`, a child of a purely discrete genome equals its parent with probability about 1/e (more when a kind has a single gene and no real gene exists), and the GA evaluates it anyway (every candidate is evaluated once; there is no cache of equal genomes). A cache keyed by the genome bytes would save those evaluations (§9.4).
- **A mixed benchmark suite:** the harness has only real-valued problems. A small mixed suite (the knapsack and the one-max/trap families for binary genes, a hyperparameter-style mixed function, the mixed-integer problems of the MIES papers) would be worth adding with step 8b's benchmarks, so that the type-aware operators are compared with a baseline on more than the two fixtures.
- **Premature convergence of structure switches:** on the polynomial fixture the default GA recovers the true terms in 60 to 100 % of the seeds; per-gene step sizes did better in a small experiment (10 of 10 on numpy float64, 7 of 10 on torch float32). Whether the default for mixed spaces should differ is a question for a benchmark, not for this step.
- **Process workers and torch:** a candidate evaluated in a worker process crosses the boundary as a pickled tensor, so each spawned worker imports torch on its first call (about a second), which counts towards a per-evaluation `timeout` of the first calls (the tests give the process-timeout test a longer timeout on torch). Tensors on a GPU in worker processes are untested and probably unwanted: process executors are for host-side, per-candidate work, and GPU work belongs in a `VectorisedEvaluator` or a batched environment. Whether the process executor should hand workers numpy genomes, or pre-import torch while it starts them, is open.
- **Host round trips of the GA on a device:** the GA copies a few small integer arrays to the host per `ask` and `tell` (parents' ids for the lineage, the survivors' indices, §7.2). It is not a hot loop, but each copy is a synchronisation, and the Metal probe shows a fixed cost of 6 to 8 ms per generation that makes a device 20 times slower than the CPU at population 100 × dimension 100, and worthwhile only from about 1,000 × 1,000 (`docs/gpu-smoke/`). Keeping the lineage on the device until the recorder needs it, computing the survivors' positions with a sort instead of a mask, and caching the space's bounds on the device (they are uploaded at every `ask`) would lower that floor. Not done: step 7 fixes numpy-only paths and does not rewrite algorithms for speed.
- **A CUDA smoke run:** the GPU smoke suite has been run on Metal and on the CPU (a dry run) but not on CUDA, and the performance probe's answer on CUDA is unknown until it is.

- **Late results in throughput mode:** how strategies should treat results for candidates asked several rounds earlier (accept, discount, or ignore). It will probably be declared per strategy.
- **Distributed execution:** whether to offer Ray or Dask as evaluator execution backends, and when.
- **Quality-diversity:** the archive interface needed for MAP-Elites-style strategies (descriptor bins, insertion rules), and whether archives become a core concept.
- **Multi-fidelity:** the shape of partial-result reporting and how strategies decide to stop or continue an evaluation.
- **Co-evolution:** how matchmaking between populations is configured.
- **Noise handling:** built-in support for repeated evaluation and averaging, and how it interacts with caching.
- **GPU CI:** whether to set up a self-hosted runner for CUDA and Metal.
- **Concurrent, asynchronous variation:** external proposal operators (§3.5) are called synchronously inside `ask`, which blocks the driver while a model answers. Whether `ask` should become asynchronous, or variation run ahead of evaluation, is open; so is a failure policy for variation (retry, skip, fall back to a built-in operator).
- **Operator costs and the budget:** the cost units of operator calls are recorded but not counted against the budget (§9.3). Whether, and how, they should be (a separate budget, or one with the evaluation costs), and how recorded calls reused by a replay count, is open.
- **Many objectives and other multi-objective algorithms:** `NSGA2` is built for two or three objectives (Pareto ranking loses its power as objectives grow, and the dominance matrix is `(n, n, k)`). NSGA-III, SMS-EMOA, MOEA/D, hypervolume-based selection and preference-based methods are not in the core.
- **Bounded SBX and duplicate elimination:** `SimulatedBinaryCrossover` is the unbounded form, with the bounds repair; the reference implementation's bounded form and its elimination of duplicate offspring are not reproduced. On the benchmark `NSGA2` is slightly ahead of pymoo's NSGA-II, and the cause was not isolated.
- **NSGA-II on structure genes:** on the polynomial fixture `NSGA2` recovers the true model's terms on the front in 6 to 7 seeds of 10 and refines its coefficients slowly (§4.4): no step-size adaptation, and a population spread over the levels of complexity. Whether a self-adaptive real mutation (the numeric GA's) or a larger population should be the default for mixed spaces is a question for a benchmark.
- **A mixed-variable multi-objective benchmark:** the benchmark has only real-valued multi-objective problems. A small set with discrete or mixed variables (a knapsack, a bi-objective feature selection) would test the discrete operators of `NSGA2`.
- **The driver's archive on very large fronts:** it is vectorised but still compares each new point with the whole archive, so a run that produces tens of thousands of non-dominated points pays for it; a spatial index or periodic pruning would help. The strategies do not depend on it.
