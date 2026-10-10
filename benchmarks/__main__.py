"""Command line: `python -m benchmarks run --config CONFIG` and `python -m benchmarks report RESULTS_DIR`."""

import os

# one thread per process: the runs are parallelised across processes, and BLAS threads would disturb the timings
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m benchmarks", description="Auxein benchmark harness.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run a benchmark config and write a results directory")
    run.add_argument("--config", required=True, type=Path, help="a TOML config, e.g. benchmarks/configs/full.toml")
    run.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="parallel worker processes (default: all cores)")
    run.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT, help="where results directories are created")
    run.add_argument("--skip-overhead", action="store_true", help="only run the quality benchmark")

    report = commands.add_parser("report", help="build report.md and plots from a results directory")
    report.add_argument("results_dir", type=Path)

    args = parser.parse_args(argv)
    if args.command == "run":
        from benchmarks.mo_config import KIND, config_kind

        if config_kind(args.config) == KIND:
            from benchmarks.mo_config import load_mo_config
            from benchmarks.mo_runner import run_mo_benchmark

            print(run_mo_benchmark(load_mo_config(args.config), args.results_root, args.workers))
        else:
            from benchmarks.config import load_config
            from benchmarks.runner import run_benchmark

            print(run_benchmark(load_config(args.config), args.results_root, args.workers, args.skip_overhead))
    else:
        import json

        metadata = json.loads((args.results_dir / "metadata.json").read_text())
        if metadata["config"].get("kind") == "multi-objective":
            from benchmarks.mo_report import build_mo_report

            print(build_mo_report(args.results_dir))
        else:
            from benchmarks.report import build_report

            print(build_report(args.results_dir))


if __name__ == "__main__":
    main()
