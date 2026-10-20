# Contributing to Auxein

Thank you for helping. This file sets the standard a change is held to before it is merged: what to discuss first, how to set
up, what the code, the tests and the documentation must do, and how a pull request is put together. The
[design document](docs/design/core.md) is the reference for how Auxein works; this file is about how changes to it are made.

## Before you start

- **Open an issue first for anything beyond a small fix**: a new strategy, evaluator or environment, a change to a public
  signature, to the recording format or to the ask/tell contract. Say what problem it solves, and which section of the design
  document it touches. A short discussion up front costs less than a rewritten pull request.
- **The design document comes first.** If a change alters behaviour the document describes, the document is updated in the
  same pull request. If it settles one of the open questions (§15), it says so there.
- **One concern per pull request.** A fix, a feature, a documentation change or a release, not several at once. Refactoring
  that a change needs goes in with it; refactoring for its own sake goes in its own pull request.

## Setting up

Auxein needs Python 3.11 or later and is developed with [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/auxein/auxein.git
cd auxein
uv sync                     # the package, the dev and bench dependency groups, torch included
```

On Linux, torch comes from PyTorch's CPU-only index, so the install stays small. On macOS the PyPI wheel is used, and it
includes Metal.

## Checks

Every pull request must pass what CI runs. Run them locally before pushing:

```sh
uv run pytest                                           # tests (and the benchmark harness tests)
uv run ruff check                                       # lint
uv run ruff format --check                              # formatting (`uv run ruff format` fixes it)
uv run pyright                                          # type check, strict on the package
uv build                                                # the package builds
uv run --group notebooks pytest --nbmake notebooks/     # the notebooks, if you touched them or the API they use
```

CI also runs the tests on Python 3.11 to 3.14 and with the lowest direct dependencies (`uv sync --resolution lowest-direct`),
and runs the quick benchmarks as a smoke test. A pull request is merged only when every job is green.

If a change touches the numeric layer (spaces, strategies, operators, evaluators, aggregators, the backend, random streams,
checkpoints of arrays), also run the GPU smoke suite on a device you have, and commit its report:

```sh
uv run pytest -m gpu --device mps     # or cuda, cuda:1; `cpu` is only a dry run
```

See [`tests/gpu/README.md`](tests/gpu/README.md). If you have no GPU, say so in the pull request, and a maintainer will run it.

## Code

- **Formatting and lint** are ruff's, with a line length of 140 (`pyproject.toml`). Don't silence a rule for a whole file
  unless there is a reason, written next to the ignore.
- **Types.** The package is checked by pyright in strict mode: every function is annotated, and `Any` and `cast` are kept to
  the boundaries where the type cannot be known statically (JSON, user callables, the namespace of an array), never used to
  quiet an error in code whose types are known. `# type: ignore` needs a comment saying why. The benchmark harness is checked
  at the standard level.
- **Dependencies.** The core depends on numpy and `array-api-compat` only. Anything else (torch, pycma, Gymnasium) is an
  optional extra, imported lazily where it is used, with a clear error when it is missing. Adding a required dependency needs
  a discussion in an issue first.
- **Numeric code follows the rules of design doc §7.2:** no in-place mutation of arrays, no global random state (all
  randomness goes through the run's streams), arrays stay on their device through ask, evaluate and tell, and the code is
  correct in float32. Host-side metadata is the only exception, and each place that moves something to the host says so in a
  comment. A device array is read on the host with `Backend.to_numpy`, never `np.asarray`.
- **Reproducibility.** The same seed and configuration must give the same event log. Nothing may depend on dict or set
  iteration order that is not guaranteed, on wall-clock time, or on the number of workers in deterministic mode.
- **The public API** is what `auxein/__init__.py` and the subpackages export in `__all__`. Adding, renaming or removing a
  name is a deliberate change: update `tests/api_test.py`, the design document and the changelog.
- **Docstrings and comments explain why**, not what the next line does: the contract a function keeps, the case it guards
  against, the reason a simpler approach was not taken. Write them as plain sentences, and name the design document section
  when there is one.
- **Errors are for the user.** A configuration error names the wrong value and what would be accepted, and is raised as
  early as possible, before a run starts.

## Tests

- **Every change comes with tests**: a fix with a test that fails without it, a feature with tests of its behaviour and of
  its edge cases. Tests live under `tests/`, mirroring the package layout, in files named `*_test.py`.
- **Numeric code is tested on every backend.** A unit test of numeric code takes the `backend` fixture, which runs it on
  numpy and torch in float64 and float32. An integration test (an end-to-end run, resume, failures, episodes) runs on the two
  corners, numpy-float64 and torch-float32, with `pytestmark = pytest.mark.usefixtures("use_corner_backend")`. See design doc
  §7.4 and `tests/support/fixtures.py`.
- **Tests are deterministic.** Fixed seeds, no reliance on timing beyond generous timeouts, no network. A statistical test
  fixes its seeds and documents the band it accepts.
- **Tests are fast.** The suite runs on every push on five configurations. A slow test needs a reason; if it only adds
  coverage another test already gives, make it smaller.
- **The README's examples are tested** (`tests/api_test.py`). If you change one, or the API it uses, keep the test passing.

## The recording format

A run directory is read back by `resume`, `open_run` and the reader, so its format is a compatibility promise.

- Any change to what `events.sqlite` stores, or how, bumps `SCHEMA_VERSION` (`auxein/recording/sqlite.py`), updates design
  doc §10.2, and is announced in the changelog in bold, saying what happens to runs recorded with the previous version.
- A change that alters a run's description (the strategy's, evaluator's or problem's configuration as recorded) changes what
  `resume` accepts. Say so in the changelog.

## Documentation

- **Changelog.** Every user-visible change adds an entry under `## Unreleased` at the top of [`CHANGELOG.md`](CHANGELOG.md)
  (create the heading if it is not there): a bold summary, then what changed and why, with the design document section.
  Fixes start with `Fixed:` and describe what the user saw before.
- **Design document.** Update the sections the change affects, in the same pull request.
- **README and notebooks.** Update them when the API they show changes. The notebooks are committed with their outputs and
  executed in CI.
- **Benchmark reports are history.** Never edit a committed report under `benchmarks/reports/`. A new comparison is a new
  report directory.

## Branches, commits and pull requests

- **Branch from `master`**, named by the kind of change: `feature/<name>`, `fix/<name>`, `docs/<name>`, `infra/<name>`,
  `benchmarks/<name>`, or `release/<version>`.
- **Commit messages** start with the area changed, then say what the commit does in one sentence:

  ```
  Driver: yield to the event loop after every tell, so one Ctrl-C stops a run whose evaluator computes on the driver's thread

  asyncio.run turns Ctrl-C into a cancellation of the driver's task, delivered only when the task yields; the live loop
  yielded only while replaying, so a batched or vectorised evaluator ran on to the end of the budget.
  ```

  The body, when there is one, says why and how the problem was found. Each commit should leave the tests passing.
- **The pull request description** says what changed and why, how it was tested, and anything a reviewer should look at
  closely. Link the issue it resolves.
- **Merging.** Pull requests are merged with a merge commit once CI is green and a maintainer has approved. Keep the branch
  up to date with `master` by merging it in; don't force-push a branch someone is reviewing.

## Releases

Releases are made by the maintainers. The version is `0.x` until the API settles (design doc §13).

1. A `release/<version>` pull request with one commit, `Version <version>`, which sets the version in `pyproject.toml`,
   updates `uv.lock` (`uv lock`), turns `## Unreleased` in the changelog into `## <version> (<date>)`, and updates the
   version in the design document's header and in §13. Historical records (earlier changelog entries, benchmark reports, GPU
   smoke reports) keep the versions they were written for.
2. Once it is merged, tag the merge commit `v<version>` and publish a GitHub release of the tag. The release workflow builds
   the package from that commit and publishes it to PyPI with trusted publishing, so the tag and `pyproject.toml` must agree.

## Licence

Auxein is released under the [Apache License 2.0](LICENSE). By contributing, you agree that your contributions are licensed
under it too.
