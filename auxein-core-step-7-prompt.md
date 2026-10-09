# Auxein core, step 7: PyTorch everywhere, and the GPU smoke suite

You are working in the `auxein` repository (https://github.com/auxein/auxein, default branch `master`). Auxein is **a Python framework for evolving agents that act in environments**. The design is in `docs/design/core.md`. **Read it completely before starting**, especially §7 (the numeric backend: the array API, numpy and PyTorch, precision, devices, testing), §8 (randomness and the scope of reproducibility) and §14 (step 7 is this step).

Then read:

- `tests/support/fixtures.py` (the `backend` fixture: numpy and torch × float64 and float32)
- the tests that already use it
- `tests/strategies/numeric_ga_golden_test.py`, which found that **float32 results differ between arm64 and x86_64**
- the numeric code paths: `auxein/strategies/ga/`, `auxein/aggregators/`, `auxein/evaluators/`, `auxein/spaces/box.py`, `auxein/core/evaluation_batch.py`, `auxein/driver/result.py`

This prompt implements **step 7: every numeric code path proven on numpy and PyTorch, in float64 and float32, including end-to-end runs, resume and the agent layer; a statistical cross-check that the torch backend searches as well as numpy; and a GPU smoke suite for CUDA and Apple Metal (MPS)**, run manually since hosted CI has no GPUs.

## What we already know (measured on `master` before this step)

- **Many unit tests already take the `backend` fixture**, especially core types, spaces, GA operators and evaluators.
- **These test files are numpy-only:** the integration tests (`tests/driver/`: resume, failures, timeouts, episode runs, structured runs, external operators, concurrency in parts) and `tests/recording/genome_store_test.py`. The GA has few end-to-end torch tests, and the episode evaluator has two.
- **Numeric code still uses numpy internally** in a few places. Check whether each one is correct and intended (host-side metadata) or an accidental numpy-only path that would break, or silently move data to the host, on torch or on a GPU. Known places:
  - `auxein/evaluators/episode.py` (`np.full` grids, `np.asarray` of values)
  - `auxein/aggregators/aggregator.py` (scanning for non-finite values)
  - `auxein/strategies/ga/genetic_algorithm.py` (survivor positions)
  - `auxein/spaces/box.py` (bounds validation, which is host-side by design)
- **Float32 isn't bit-reproducible across CPU architectures.** That's normal (different SIMD and summation orders), but the design doc's reproducibility scope doesn't mention it yet.

## Preconditions (stop and tell me if any is missing)

- Step 6b is merged on `master`, and all checks pass.

## Ground rules (as in previous steps)

- **One PR**, on branch `core/step-7-pytorch-everywhere`, with small commits. Never push to `master`, and don't tag or publish anything.
- **The design doc wins.** If implementing reveals a problem with it, choose what best fits §1.1, update `docs/design/core.md` in the same PR, and list every change.
- **Quality bar:**
  - pyright strict
  - ruff clean
  - docstrings explaining *why*
  - no `Any` in public signatures unless justified
- **No behaviour change on numpy.** The numeric GA's golden test, every existing test and the benchmark adapters must give the same results as before on numpy. Fixes for torch must not alter numpy results.
- **No performance regression on numpy.** Report the overhead of `auxein-core-ga` and `auxein-core-random` before and after.
- **When done, stop and report:**
  - what you built
  - every numpy-only path you found and what you did with it
  - the torch-versus-numpy cross-check results
  - the test-time impact on CI
  - design-doc changes
  - open questions

## Decisions already made (don't revisit them)

1. **Every numeric code path must work on numpy and PyTorch, in float64 and float32, on CPU in CI.** "Numeric code path" means:
   - spaces
   - both strategies' numeric parts (`GeneticAlgorithm` fully; `StructuredGeneticAlgorithm`'s ranking and selection, which work on objective arrays)
   - operators
   - all three evaluators
   - aggregators
   - `EvaluationBatch` columns
   - the result tracker
   - checkpoints and resume of array state
   - the genome store for array genomes
2. **Host-side metadata may stay numpy, deliberately.** Statuses, ids, bounds validation, recording and replay byte comparisons can use numpy. Each such place must be **documented as intentional** in a short comment, and it must never be an accidental device-to-host round trip inside a hot loop on the GPU path.
3. **The integration test matrix is trimmed to keep CI time reasonable.** Unit tests run on all four configurations, as now. **Integration tests** (end-to-end runs, resume, kill-and-resume, failures, timeouts, episode runs, external operators, genome store) run on two "corner" configurations: **numpy-float64** and **torch-float32**. That pairs the most different backend with the most different precision. Add a second fixture, e.g. `corner_backend`, for this, and document the choice in `tests/support/fixtures.py` and the design doc.
4. **Reproducibility scope**, written down in §8:
   - identical results for the same seed, backend, precision **and CPU architecture**
   - float64 is in practice stable across architectures, float32 isn't
   - different backends or precisions give different, equally valid runs

   The golden test keeps pinning per-architecture values where it needs to.
5. **The torch backend must search as well as numpy.** Not identically (different random streams), but statistically indistinguishable. Add a benchmark adapter option, e.g. a `backend` parameter on `auxein_core_ga` and `auxein_core_random`, and a cross-check test like the existing random-search one:
   - `auxein-core-ga` on torch-float32 against numpy-float64
   - Sphere and Rastrigin, d = 2 and 10, 30 seeds each, quick budget
   - Vargha–Delaney A₁₂ of final errors within `[0.35, 0.65]`
   - deterministic, with fixed seeds
6. **GPU testing is manual, not in CI** (§7.4). Hosted runners have neither CUDA nor Metal. The GPU smoke suite:
   - runs with a single command
   - is selected by a pytest marker (e.g. `gpu`) and an option or environment variable naming the device (`cuda`, `cuda:1`, `mps`)
   - is skipped cleanly everywhere else
   - **MPS runs float32 only.** Float64 on MPS is already a configuration error.
   - Its results are written to a report file (see below) by whoever runs it. The project owner will run it on an Apple Silicon Mac (MPS), and on CUDA when one is available.

## What to build

### 1. Close the numeric gaps

- **Audit every numeric module** (decision 1) for numpy-only code. For each case found, including the known list above:
  - make it backend-generic through the array namespace
  - or document it as intentional host-side metadata (decision 2)
  - or, if it's a real limitation, record it in §15 and explain it in your report
- **Check device placement:** no array silently ends up on the host on the GPU path, apart from the documented metadata. Add a test helper that, on a torch backend, asserts results stay on the backend's device, and use it in the evaluator, aggregator and GA tests.
- **Check float32 numerics** where they're delicate:
  - self-adaptive step sizes near `σ_min`
  - sigma scaling with tiny spreads
  - CVaR and quantile reducers
  - the minimisation-form conversion
  - `Box` sampling at the bounds (already guarded)

  Add focused float32 tests where coverage is missing.

### 2. Extend the tests

- **Add the `corner_backend` fixture** (decision 3), and parametrise the integration tests listed above with it.
- **Raise unit-test coverage on all four configurations** wherever a numeric module has numpy-only tests.
- Make sure torch is installed in **every CI job that runs tests**, not just one. Keep the CPU-only torch index for Linux, as today. Report the CI time before and after.

### 3. The torch cross-check (decision 5)

- Add the `backend` parameter to the core benchmark adapters.
- Add the cross-check test.
- Add one `auxein-core-ga-torch` entry to `quick.toml`, so the CI benchmark job exercises torch end to end. Don't change existing entries.

### 4. The GPU smoke suite: `tests/gpu/`

Each test runs on the configured device in float32 (float64 too on CUDA). It covers:

- `Backend` validation for the device
- random streams generating on the device, with `state_dict` round trips
- `Box` sampling on the device
- `GeneticAlgorithm` and `RandomSearch` runs end to end with `VectorisedEvaluator` and the GPU backend, results staying on the device
- the batched path of the point-mass fixture through `EpisodeEvaluator`, with aggregators on the device
- checkpoint and resume of a GPU run: array state restored to the device, replay identical **on the same device**
- **a performance probe, informational, not asserted.** For a large vectorised workload (e.g. population 1,000, dimension 1,000, a cheap vectorised objective, and a heavier one such as a matrix product per candidate), time per generation on the CPU against the GPU. This is where we learn whether the GPU pays off and from what size, as discussed in the design.

How to run and record it:

- **A single command**, documented in `tests/gpu/README.md`, e.g. `uv run pytest -m gpu --device mps`. It writes a Markdown report to `docs/gpu-smoke/<device>-<date>.md` with:
  - machine, OS, torch version and device name
  - pass/fail per test
  - the performance probe's table
- **Run it yourself** on whatever device your environment has. If none, run it with `--device cpu` to prove the suite works (it must support that as a dry run). Say clearly in your report that GPU runs are pending for the owner.

## Design-doc updates required in this PR

- **§7.2 and §7.4:**
  - the numeric-path rules as enforced (decision 1)
  - host-side metadata as an intentional exception (decision 2)
  - the test matrix (four configurations for unit tests, two corners for integration tests)
  - the torch cross-check
  - the GPU smoke suite and how its reports are kept
- **§8:** the reproducibility scope including CPU architecture (decision 4).
- **§11.4:** criterion 5 (backends). CPU parts met. GPU parts pending until the owner's MPS and CUDA runs are recorded.
- **§14:** step 7 done, with the GPU runs noted as pending if they are.
- **§15:** any real limitation you found.
- Anything else you decided.

## Out of scope

- JAX, MLX or CuPy backends
- `torch.compile` or other JIT compilation
- GPU runners in CI
- Multi-GPU or distributed execution
- New strategies, spaces or evaluators (multi-objective is step 8)
- Rewriting numeric algorithms for speed beyond fixing numpy-only paths (report opportunities instead)
