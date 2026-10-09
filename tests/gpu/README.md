# The GPU smoke suite

Hosted CI runners have neither CUDA nor Metal, so the numeric code is proven on a GPU by hand, before a release and whenever
something in the numeric layer changes (design doc §7.4). The suite is small: it checks that the things that run on a device
really do, and that a run on a device is what it should be. It does not benchmark anything, apart from one informational probe.

## Run it

One command, naming the device:

```sh
uv run pytest -m gpu --device mps          # Apple Silicon (Metal)
uv run pytest -m gpu --device cuda         # the first CUDA device
uv run pytest -m gpu --device cuda:1       # another one
uv run pytest -m gpu --device cpu          # a dry run: proves the suite itself works, says nothing about a GPU
```

`AUXEIN_GPU_DEVICE=mps uv run pytest -m gpu` does the same. Without a device the suite is skipped, with the reason, so
`uv run pytest` on any machine (and in CI) never touches it.

On macOS, `uv sync` installs the PyPI wheel of torch, which includes Metal. On Linux the project is configured to use PyTorch's
CPU-only index (so that CI does not download gigabytes of CUDA libraries); for CUDA, install a CUDA build in the environment
first (the selector at pytorch.org gives the command for your CUDA version), and run the tests with
`uv run --no-sync pytest -m gpu --device cuda` so that `uv` leaves it alone.

## What it checks

Every test runs in float32, and in float64 too except on Metal, which has no float64 (asking for it is a configuration error,
and a test checks that it is).

| File | What |
|---|---|
| `gpu_backend_test.py` | `Backend` validation for the device; arrays, integer arrays, arithmetic, sorting and gathering on it |
| `gpu_random_test.py` | streams generating on the device; moments; reproducibility on the same device; `state_dict` round trips; pickling |
| `gpu_box_test.py` | `Box` sampling on the device, on a log scale, and at bounds float32 cannot represent |
| `gpu_search_test.py` | `GeneticAlgorithm` (selections × mutations), `RandomSearch` and `StructuredGeneticAlgorithm` end to end with a `VectorisedEvaluator`; results stay on the device |
| `gpu_episodes_test.py` | the batched point mass through `EpisodeEvaluator` with aggregators reducing device arrays; agreement with numpy float64 |
| `gpu_resume_test.py` | checkpoints restore their arrays to the device; an extended run and a replay without checkpoints equal the uninterrupted run **on the same device** |
| `gpu_probe_test.py` | the performance probe (below) |

"Stays on the device" is checked with `Backend.matches`, which compares the namespace, the device and the dtype, and
`tests/support/fixtures.py:assert_on_device`. The only things allowed on the host are the metadata that design doc §7.2 lists.

## The performance probe

Informational: it asserts nothing about the numbers. A `GeneticAlgorithm` with population 1,000 in dimension 1,000 runs seven
generations, twice (a cheap objective, a sum of squares; and a heavy one, a 1,000 × 1,000 matrix product for the whole
population), on numpy float64 on the CPU, on torch float32 on the CPU, and on the device. The report gives the median
milliseconds per generation of the last five and the speed-up over numpy. The probe is where we learn whether the device pays
off, and from what size: change `POPULATION` and `DIMENSION` in `gpu_probe_test.py` to look for the crossover.

## The report

A run writes `docs/gpu-smoke/<device>-<date>.md` with the machine, OS, torch version and device name, the commit, the result of
every test, and the probe's table. Commit the file of a real run (CUDA, MPS) with the change that was tested; a `cpu` file is a
dry run and is not evidence of anything. A report of the same device and day is replaced.
