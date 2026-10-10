# Auxein core, step 8a: integer, binary, categorical and mixed search spaces

You are working in the `auxein` repository (https://github.com/auxein/auxein, default branch `master`). Auxein is **a Python framework for evolving agents that act in environments**. The design is in `docs/design/core.md`. **Read it completely before starting**, especially §4 (genomes, batches, variable-length structure, search spaces), §3.3 (the genetic algorithm), §7 (backends, precision, the test matrix) and §14.

Then read the current code, particularly:

- `auxein/spaces/` (`Box`, `SequenceSpace`, codecs)
- `auxein/strategies/ga/` (operators, ranking, how the GA uses `Box`: widths, step sizes relative to the box, clipping, log-scale dimensions)
- `auxein/strategies/random_search.py`
- `tests/support/fixtures.py` (the `backend` and `corner_backend` fixtures)
- `tests/strategies/numeric_ga_golden_test.py`

**The roadmap's step 8 is split in two.** This prompt is **8a: search spaces with integer, binary and categorical dimensions, and mixed spaces that combine them with real dimensions, supported end to end by `RandomSearch` and the numeric `GeneticAlgorithm`.** Step 8b, a separate later prompt, adds the multi-objective strategy (NSGA-II), a scalarisation helper and a multi-objective benchmark. Update §14 to show the split.

## Preconditions (stop and tell me if any is missing)

- Step 7 is merged on `master`, and all checks pass.

## Ground rules (as in previous steps)

- **One PR**, on branch `core/step-8a-mixed-spaces`, with small commits. Never push to `master`, and don't tag or publish anything.
- **The design doc wins.** If implementing reveals a problem with it, choose what best fits §1.1, update `docs/design/core.md` in the same PR, and list every change.
- **Quality bar:**
  - pyright strict
  - ruff clean
  - docstrings explaining *why*
  - no `Any` in public signatures unless justified
- **Numeric code** goes through the array namespace. Unit tests run on all four backend configurations, integration tests on the two corners, as step 7 established. All randomness comes from Auxein's streams.
- **The numeric GA must not change behaviour on `Box`.** The golden test, every existing test and the benchmark adapters must give the same results as before. Mixed-space support must be additive.
- **No performance regression.** Report the overhead of `auxein-core-ga` on `Box` before and after.
- **When done, stop and report:**
  - what you built
  - the mixed-variable fixture results
  - design-doc changes
  - open questions

## Decisions already made (don't revisit them)

1. **Array-based representation.** A mixed space's genome is **one numeric array of the backend's float dtype**, like `Box` genomes, so the array fast path, `VectorisedEvaluator`, GPU support, recording as raw bytes, the genome store and replay all keep working unchanged.
   - Each dimension has a **type**:
     - **real:** bounds, optional log scale, as in `Box`
     - **integer:** inclusive integer bounds, stored as integral float values
     - **binary:** 0 or 1
     - **categorical:** a finite list of JSON-serialisable choices, stored as the **index** of the choice
   - **Integer bounds must be exactly representable in the run's precision.** In float32 that's |value| ≤ 2²⁴. Exceeding it is an error when the run starts, with a clear message.
2. **A `MixedSpace` built from named dimensions**, e.g.:

   ```python
   MixedSpace({
       "lr": Real(1e-5, 1e-1, log=True),
       "layers": Integer(1, 8),
       "dropout": Binary(),
       "optimiser": Categorical(["sgd", "adam"]),
   })
   ```

   - Dimensions keep their declared order.
   - It provides `sample_genomes`, `contains` (bounds **and** integrality **and** valid category indices), `describe()` (for metadata and resume validation), and **decoding helpers for user code**:
     - `space.values(genome)`: a dict of name → Python value (float, int, bool, the category itself)
     - a batched equivalent returning per-dimension arrays, for vectorised objectives
   - **Convenience constructors** for homogeneous spaces, e.g. `IntegerSpace(lower, upper, dim)` and `BinarySpace(dim)`, are thin wrappers over `MixedSpace`.
   - `Box` stays as it is, the all-real case, and is **not** reimplemented on top of `MixedSpace` in this step: the golden test must stay byte-identical.
3. **Type-aware operators in the numeric `GeneticAlgorithm`**, following mixed-integer evolution strategies (e.g. Li et al., MIES):
   - **Real dimensions:** exactly today's operators (self-adaptive or Gaussian mutation relative to the width, intermediate or uniform recombination, clip or reflect repair, log scale as today).
   - **Integer dimensions:**
     - mutation by adding the **difference of two geometric random variables** (a symmetric integer step), applied with a per-dimension probability, with a self-adapted mean step size (one per individual, kept as strategy state like real step sizes) or a fixed one
     - **discrete recombination** (each gene from one of the two parents)
     - repair by clipping to the bounds
   - **Binary dimensions:** bit-flip mutation with probability `1/d_binary` by default; discrete recombination.
   - **Categorical dimensions:** with probability `1/d_categorical` by default, replace the index with a **different** category drawn uniformly; discrete recombination.
   - **No dimension may stop moving for good.** Integer step sizes have a lower bound, so integer genes can always still change.
   - **Everything vectorised over the population**, with no Python loops over individuals or genes, on both backends and both precisions.
   - **Defaults are selected automatically from the space**, and every operator can be replaced, as today.
   - **On a `Box`, the GA uses exactly its current code path**, with no behaviour change (ground rules).
4. **`RandomSearch` works on mixed spaces** through `sample_genomes`, with no special handling.
5. **No conditional dimensions** (e.g. "momentum only if the optimiser is SGD") in this step. Record them in §15 as future work.

## What to build

### 1. Spaces: `auxein/spaces/`

- **Dimension types** `Real`, `Integer`, `Binary` and `Categorical`, and **`MixedSpace`** (decision 2), with:
  - validation: bounds, empty categories, duplicate names, float32 representability (decision 1)
  - sampling that's uniform per dimension type (log-uniform for log-scale reals), on the backend, with results always valid in both precisions
  - `contains`
  - `describe`
  - `values` / batched values
- The convenience constructors.
- **Export** `MixedSpace`, `Real`, `Integer`, `Binary` and `Categorical` from `auxein`, and update the public-API test.

### 2. The numeric `GeneticAlgorithm`

- Accept `MixedSpace` in addition to `Box` (decision 3). Keep the `Box` path unchanged; route mixed spaces through type-aware operators.
- Step sizes:
  - real dimensions keep self-adaptive step sizes, relative to their width, as strategy state
  - integer dimensions get their own adapted mean step, also strategy state, with its lower bound
  - both are included in `state_dict` for exact continuation and resume
- Origins describe the operators used per type, compactly.
- Make sure `PopulationView`, ranking, survivor selection and failure handling are unchanged.

### 3. A mixed-variable test fixture: `tests/support/`

- A deterministic objective with a **known optimum** that needs all four dimension types to be right. For example, a shifted sphere on the real part, plus penalties for wrong integers, wrong bits and a wrong category, with interactions so the dimensions can't be solved independently by luck. Plus one constraint.
- **A polynomial-regression-style fixture with structure genes:**
  - real coefficients plus binary switches for which terms are active, up to a maximum degree
  - a single objective (data error plus a complexity penalty), since multi-objective comes in 8b
  - it must recover the true active terms on noise-free data

  This previews the polynomial notebook of step 9.

## Tests (at minimum)

- **Dimension types and `MixedSpace`:**
  - validation errors, including float32 integer bounds
  - sampling: every sample valid (integral, in bounds, valid category index) for 10⁵ samples on all four configurations
  - log-uniform reals
  - `contains` rejects non-integral values, out-of-range values and bad indices
  - `values` and batched values round trip
  - `describe` is stable
- **Operators**, on all four configurations:
  - integer mutation produces integral, in-bounds values, symmetric in distribution (statistical check, fixed seed), with the step-size lower bound respected
  - bit-flip and categorical mutations respect their probabilities (statistical check), and categorical never "mutates" to the same category
  - discrete recombination takes each gene from a parent
  - results always valid
  - device placement kept on torch
- **The numeric GA:**
  - the `Box` golden test and all existing GA tests unchanged
  - on the mixed fixture, the GA reaches the known optimum (or within a stated tolerance on the real part) within a CI-friendly budget, and beats `RandomSearch` at the same budget
  - `state_dict` round trip continues identically
  - kill-and-resume replay is identical (corner backends)
  - works under generation and steady-state delivery
- **The polynomial fixture:** the GA recovers the true active terms.
- **Recording:** mixed genomes record and replay byte-identically. The reader returns them as arrays; decoding to named values is the space's job, which is documented.

## Design-doc updates required in this PR

- **§4.4:** structure genes now exist (binary switches, integer counts), with the polynomial fixture as the example.
- **§4.5:** `MixedSpace` and its dimension types as built; the array representation; float32 integer limits; decoding helpers.
- **§3.3:** the type-aware operators and their defaults; the `Box` path unchanged.
- **§14:** step 8 split into 8a (done) and 8b (NSGA-II, scalarisation, multi-objective benchmark).
- **§15:** conditional dimensions; anything else you found.
- Anything else you decided.

## Out of scope (step 8b and later)

- Multi-objective strategies, scalarisation helpers, the multi-objective benchmark
- Conditional or hierarchical spaces
- Mixed dimensions inside `StructuredGeneticAlgorithm` or `SequenceSpace`
- Reimplementing `Box` on top of `MixedSpace`
- Benchmark-harness problems for mixed spaces (report whether you think a small mixed suite would be worth adding later)
