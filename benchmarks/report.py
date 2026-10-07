"""Build report.md and plots from a results directory: `python -m benchmarks report <results-dir>`."""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks import stats  # noqa: E402
from benchmarks.runner import read_jsonl  # noqa: E402

ERROR_FLOOR = 1e-10  # errors are clipped here in the plots, so that exact zeros fit on a log axis
REFERENCE = "auxein-default"

# categorical slots 1, 2, 3, 7, 5 of the reference palette (validated light-mode set), assigned by algorithm
COLORS = {
    "auxein-default": "#2a78d6",
    "auxein-fixedvar": "#eb6834",
    "auxein-windowing": "#1baf7a",
    "random-search": "#4a3aa7",
    "cma-es": "#e87ba4",
}
FALLBACK_COLORS = ["#008300", "#e34948", "#eda100", "#52514e"]
GRID = "#d9d8d2"
INK = "#0b0b0b"
MUTED = "#52514e"

PROBLEM_TABLE = """| Problem | Definition (z = transformed x) | Tests |
|---|---|---|
| sphere | `Σ z_i²`, z = x − x* | sanity: everything should solve it |
| ellipsoid | `Σ 10^(6(i−1)/(d−1)) z_i²` (condition number 10⁶), z = R(x − x*) with a random rotation R | step-size adaptation across very different scales; rotation defeats per-axis tricks |
| rosenbrock | `Σ_{i<d} 100(z_{i+1} − z_i²)² + (1 − z_i)²`, z = x − x* + 1 | following a narrow curved valley |
| rastrigin | `10d + Σ (z_i² − 10 cos(2π z_i))`, z = x − x* | many regularly spaced local optima |
| noisy_sphere | sphere × `(1 + 0.1·ε)`, ε ~ N(0, 1) per call | robustness to noisy fitness (the trace records the noise-free error) |"""


class Results:
    """A results directory, loaded."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.runs = read_jsonl(path / "runs.jsonl")
        overhead = path / "overhead.jsonl"
        self.overhead = read_jsonl(overhead) if overhead.exists() else []
        self.metadata: dict[str, Any] = json.loads((path / "metadata.json").read_text())
        configured = [a["name"] for a in self.metadata["config"]["algorithms"]]
        seen = list(dict.fromkeys(r["algorithm"] for r in self.runs))
        self.algorithms = [a for a in configured if a in seen] + [a for a in seen if a not in configured]
        self.targets: list[float] = [float(t) for t in self.metadata["config"]["targets"]]
        self.problems = list(dict.fromkeys(r["problem"] for r in self.runs))
        self.dims = sorted({r["dim"] for r in self.runs})
        groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for r in self.runs:
            groups[(r["problem"], r["dim"], r["algorithm"])].append(r)
        self.groups = {k: sorted(v, key=lambda r: r["instance"]) for k, v in groups.items()}

    def cell(self, problem: str, dim: int) -> dict[str, list[dict[str, Any]]]:
        return {a: self.groups[(problem, dim, a)] for a in self.algorithms if (problem, dim, a) in self.groups}

    def color(self, algorithm: str) -> str:
        if algorithm in COLORS:
            return COLORS[algorithm]
        extra = [a for a in self.algorithms if a not in COLORS]
        return FALLBACK_COLORS[extra.index(algorithm) % len(FALLBACK_COLORS)]


def key(target: float) -> str:
    return f"{target:g}"


def fmt(value: float) -> str:
    if math.isinf(value):
        return "∞"
    if value == 0:
        return "0"
    return f"{value:.2e}" if abs(value) < 1e-2 or abs(value) >= 1e5 else f"{value:.3g}"


def fmt_evals(value: float) -> str:
    return "∞" if math.isinf(value) else f"{value:,.0f}"


def final_errors(records: list[dict[str, Any]]) -> list[float]:
    return [r["final_error"] for r in records]


def style_axes(ax: Any) -> None:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def convergence_plot(results: Results, problem: str, dim: int, path: Path) -> None:
    cell = results.cell(problem, dim)
    grid = sorted({e for records in cell.values() for r in records for e, _ in r["trace"]})
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110, facecolor="#fcfcfb")
    for algorithm, records in cell.items():
        values = np.maximum(stats.resample_traces([r["trace"] for r in records], grid), ERROR_FLOOR)
        low, median, high = np.nanpercentile(values, [25, 50, 75], axis=0)
        color = results.color(algorithm)
        ax.fill_between(grid, low, high, color=color, alpha=0.18, linewidth=0)
        ax.plot(grid, median, color=color, linewidth=2, label=algorithm)
    for target in results.targets:
        ax.axhline(target, color=MUTED, linewidth=0.7, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=ERROR_FLOOR / 2)
    ax.set_xlabel("fitness evaluations", color=INK, fontsize=9)
    ax.set_ylabel("best error so far (median, IQR band)", color=INK, fontsize=9)
    ax.set_title(f"{problem}, d={dim}", color=INK, fontsize=11, loc="left")
    style_axes(ax)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def summary_table(results: Results, cell: dict[str, list[dict[str, Any]]]) -> str:
    keys = [key(t) for t in results.targets]
    header = ["Algorithm", "Final error, median [IQR]"] + [f"Success {t}" for t in keys] + [f"ERT {t}" for t in keys]
    rows = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for algorithm, records in cell.items():
        errors = final_errors(records)
        q1, median, q3 = np.percentile(errors, [25, 50, 75])
        rates = [stats.success_rate(records, k) for k in keys]
        erts = [stats.expected_running_time(records, k) for k in keys]
        cells = [algorithm, f"{fmt(median)} [{fmt(q1)}, {fmt(q3)}]"]
        cells += [f"{ok}/{n}" for ok, n in rates] + [fmt_evals(e) for e in erts]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def comparison_rows(cell: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """auxein-default against every other algorithm, with Holm correction over the comparisons of this cell."""
    if REFERENCE not in cell:
        return []
    reference = final_errors(cell[REFERENCE])
    rows = []
    for algorithm, records in cell.items():
        if algorithm == REFERENCE:
            continue
        other = final_errors(records)
        rows.append(
            {
                "algorithm": algorithm,
                "median_reference": statistics.median(reference),
                "median_other": statistics.median(other),
                "a12": stats.vargha_delaney(reference, other),
                "p": stats.mann_whitney_p(reference, other),
            }
        )
    for row, adjusted in zip(rows, stats.holm([r["p"] for r in rows])):
        row["p_holm"] = adjusted
        row["reading"] = stats.reading(REFERENCE, row["algorithm"], row["a12"], adjusted)
    return rows


def comparison_table(rows: list[dict[str, Any]]) -> str:
    header = f"| {REFERENCE} vs | median error ({REFERENCE}) | median error (other) | A12 | p | p (Holm) | Reading |"
    lines = [header, "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['algorithm']} | {fmt(r['median_reference'])} | {fmt(r['median_other'])} | {r['a12']:.2f} | {r['p']:.2g} | {r['p_holm']:.2g} | {r['reading']} |"
        )
    return "\n".join(lines)


def overhead_tables(results: Results) -> tuple[str, str]:
    """Median microseconds per evaluation, and Auxein's evaluations per generation."""
    records = results.overhead
    medians: dict[tuple[str, int, int | None], float] = {}
    cost: dict[tuple[str, int | None], float] = {}
    generations: dict[tuple[str, int | None], list[int]] = defaultdict(list)
    grouped: dict[tuple[str, int, int | None], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[(r["algorithm"], r["dim"], r["population_size"])].append(r)
    for k, rs in grouped.items():
        medians[k] = statistics.median(r["us_per_eval"] for r in rs)
        if rs[0]["evals_per_generation"] is not None and k[2] is not None:
            cost[(k[0], k[2])] = statistics.median(r["evals_per_generation"] for r in rs)
            generations[(k[0], k[2])].append(int(statistics.median(r["generations"] for r in rs)))

    dims = sorted({k[1] for k in medians})
    sizes = sorted({k[2] for k in medians if k[2] is not None})
    algorithms = [a for a in results.algorithms if any(k[0] == a for k in medians)]

    with_population = [a for a in algorithms if any(k[0] == a and k[2] is not None for k in medians)]
    without = [a for a in algorithms if a not in with_population]
    lines = ["Median time per fitness evaluation, in microseconds, on a negligible-cost objective.", ""]
    if with_population:
        lines += ["| Algorithm | Dimension | " + " | ".join(f"population {s}" for s in sizes) + " |", "|---|---|" + "---|" * len(sizes)]
        for a in with_population:
            for d in dims:
                lines.append(
                    f"| {a} | {d} | " + " | ".join(f"{medians[(a, d, s)]:.1f}" if (a, d, s) in medians else "" for s in sizes) + " |"
                )
        lines.append("")
    if without:
        lines += ["| Algorithm | " + " | ".join(f"d={d}" for d in dims) + " |", "|---|" + "---|" * len(dims)]
        for a in without:
            lines.append(f"| {a} | " + " | ".join(f"{medians[(a, d, None)]:.1f}" if (a, d, None) in medians else "" for d in dims) + " |")
    timing = "\n".join(lines)

    gen_lines = [
        "Fitness evaluations per generation, and generations completed within the overhead budget (median over dimensions).",
        "",
        "| Algorithm | Population | Evaluations per generation | Generations |",
        "|---|---|---|---|",
    ]
    for (a, s), c in sorted(cost.items(), key=lambda kv: (algorithms.index(kv[0][0]), kv[0][1])):
        gen_lines.append(f"| {a} | {s} | {c:g} | {statistics.median(generations[(a, s)]):g} |")
    return timing, "\n".join(gen_lines)


def overhead_plots(results: Results, out_dir: Path) -> list[str]:
    records = results.overhead
    if not records:
        return []
    grouped: dict[tuple[str, int, int | None], float] = {}
    by_key: dict[tuple[str, int, int | None], list[float]] = defaultdict(list)
    for r in records:
        by_key[(r["algorithm"], r["dim"], r["population_size"])].append(r["us_per_eval"])
    for k, v in by_key.items():
        grouped[k] = statistics.median(v)
    dims = sorted({k[1] for k in grouped})
    sizes = sorted({k[2] for k in grouped if k[2] is not None})
    with_population = [a for a in results.algorithms if any(k[0] == a and k[2] is not None for k in grouped)]
    without = [a for a in results.algorithms if any(k[0] == a and k[2] is None for k in grouped)]
    files = []

    if sizes and with_population:
        fig, axes = plt.subplots(1, len(dims), figsize=(3.6 * len(dims), 3.6), dpi=110, facecolor="#fcfcfb", sharey=True, squeeze=False)
        for ax, d in zip(axes[0], dims):
            for a in with_population:
                ys = [grouped.get((a, d, s)) for s in sizes]
                ax.plot(sizes, ys, color=results.color(a), linewidth=2, marker="o", markersize=5, label=a)
            for a in without:
                ax.axhline(grouped[(a, d, None)], color=results.color(a), linewidth=1.5, linestyle="--", label=a)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(sizes, [str(s) for s in sizes])
            ax.minorticks_off()
            ax.set_yscale("log")
            ax.set_xlabel("population size", color=INK, fontsize=9)
            ax.set_title(f"d={d}", color=INK, fontsize=10, loc="left")
            style_axes(ax)
        axes[0][0].set_ylabel("µs per evaluation (median)", color=INK, fontsize=9)
        axes[0][-1].legend(fontsize=7, frameon=False, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(out_dir / "overhead-population.png", facecolor=fig.get_facecolor())
        plt.close(fig)
        files.append("overhead-population.png")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=110, facecolor="#fcfcfb")
    middle = sizes[len(sizes) // 2] if sizes else None
    for a in results.algorithms:
        size = middle if a in with_population else None
        points = [(d, grouped[(a, d, size)]) for d in dims if (a, d, size) in grouped]
        if points:
            label = f"{a} (population {size})" if size is not None else a
            ax.plot(*zip(*points), color=results.color(a), linewidth=2, marker="o", markersize=5, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(dims, [str(d) for d in dims])
    ax.minorticks_off()
    ax.set_yscale("log")
    ax.set_xlabel("dimension", color=INK, fontsize=9)
    ax.set_ylabel("µs per evaluation (median)", color=INK, fontsize=9)
    style_axes(ax)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "overhead-dimension.png", facecolor=fig.get_facecolor())
    plt.close(fig)
    files.append("overhead-dimension.png")
    return files


def sanity_checks(results: Results) -> list[str]:
    """Checks on the harness itself: if one fails, suspect the harness before the algorithms."""
    lines = []

    def check(title: str, problem: str, algorithm: str, dim: int, need_success: bool) -> None:
        records = results.groups.get((problem, dim, algorithm))
        if not records:
            lines.append(f"- {title}: not run in this benchmark")
        elif records[0]["budget"] < 20000:
            lines.append(
                f"- {title}: not applicable, the budget ({records[0]['budget']}) is below the 20000 evaluations of the full benchmark"
            )
        else:
            ok, n = stats.success_rate(records, key(1e-6 if need_success else 1e-3))
            if need_success:
                passed, detail = ok / n >= 0.9, f"{ok}/{n} runs reached 1e-6 (need at least 90%)"
            else:
                passed, detail = ok == 0, f"{ok}/{n} runs reached 1e-3 (need none)"
            lines.append(f"- {'PASS' if passed else 'FAIL'}: {title}: {detail}")

    check("CMA-ES reaches 1e-6 on 10-D sphere", "sphere", "cma-es", 10, True)
    check("CMA-ES reaches 1e-6 on 10-D ellipsoid", "ellipsoid", "cma-es", 10, True)
    check("random search does not reach 1e-3 on 10-D sphere", "sphere", "random-search", 10, False)
    return lines


def methodology(results: Results) -> str:
    meta, config = results.metadata, results.metadata["config"]
    versions, cpu = meta["versions"], meta["cpu"]
    algorithms = "\n".join(
        f"  - `{a['name']}` (adapter `{a['adapter']}`): `{json.dumps(a.get('params', {}))}`" for a in config["algorithms"]
    )
    overhead = config.get("overhead")
    overhead_text = (
        f"Plain sphere, {overhead['budget']} evaluations, {overhead['repeats']} repeats (median reported), dimensions {overhead['dims']}, "
        f"population sizes {overhead['population_sizes']} for the algorithms with a population. Run serially, after the quality runs, "
        "with one numpy thread per process. The time covers everything the algorithm does, including building the initial population "
        "and the budget-counting wrapper (identical for every algorithm)."
        if overhead
        else "not run"
    )
    commit = (meta["git_sha"] or "unknown")[:10] + (" (dirty working tree)" if meta["git_dirty"] else "")
    return f"""- **Config**: `{config["name"]}`, commit `{commit}`, run at {meta["timestamp"]} on {meta["workers"]} worker processes.
- **Software**: Python {versions["python"]}, auxein {versions["auxein"]}, numpy {versions["numpy"]}, pycma {versions["cma"]}, scipy {versions["scipy"]}.
- **Machine**: {cpu["model"]} ({cpu["logical_cpus"]} logical CPUs), {cpu["system"]}.
- **Budget**: {config["budget_per_dim"]} × d fitness evaluations for every algorithm, counted outside the algorithms by `CountingObjective`. The initial population counts, and a partial generation is fine. Progress is always against evaluations, never generations.
- **Domain and conventions**: search domain [-5, 5]^d for every problem, boundaries not enforced. Optimum value 0, results are errors f(x) − f*. Algorithms see the noisy value on the noisy problem, the trace records the noise-free error of the evaluated point.
- **Instances**: each instance has its own random shift x* ~ U[-4, 4]^d (and, for the ellipsoid, a uniform random rotation from a QR decomposition with sign correction), drawn from a generator keyed by the instance id and the dimension, separate from algorithm randomness. Run k of every algorithm uses instance {config.get("instance_offset", 0)} + k, so comparisons are paired. {config["runs"]} runs per combination.
- **Seeds**: run k uses seed {config.get("base_seed", 0)} + k. Auxein is seeded with `np.random.seed(seed)` (it uses global numpy randomness), random search with `np.random.default_rng(seed)`, CMA-ES with seed + 1.
- **Precision targets**: {", ".join(f"{t:g}" for t in results.targets)}.
- **Traces**: best-so-far error at about 20 log-spaced evaluation counts per decade, plus the first and last evaluation. Plots show the median and the interquartile band over runs at each checkpoint. Errors below {ERROR_FLOOR:g} are drawn at {ERROR_FLOOR:g}; dotted lines mark the targets.
- **Statistics**: two-sided Mann-Whitney U test on the final error of `{REFERENCE}` against every other algorithm (scipy), with Holm correction over the comparisons of each problem × dimension table. A12 is the probability that a `{REFERENCE}` run ends with a lower error than a run of the other algorithm (ties count half); 0.5 is no difference. Effect size labels follow Vargha and Delaney: negligible below |A12 − 0.5| = 0.06, small below 0.14, medium below 0.21, large above. Significance level 0.05.
- **ERT (expected running time)**: total evaluations spent over all runs, divided by the number of successful runs, infinity if there is none. A successful run spends the evaluations it needed to first reach the target, an unsuccessful run spends everything it evaluated. It estimates the evaluations needed to reach the target if failed runs were restarted from scratch.
- **Algorithms**:
{algorithms}
- **Overhead benchmark**: {overhead_text}

### Algorithms and problems

{PROBLEM_TABLE}

### Reproducing

```
uv run --group bench python -m benchmarks run --config benchmarks/configs/{config["name"]}.toml
uv run --group bench python -m benchmarks report <results-dir>
```

The same config, commit and seeds give identical `runs.jsonl` contents, except for the `wall_time` fields."""


def build_report(results_dir: Path) -> Path:
    results = Results(Path(results_dir))
    out = results.path
    sections: list[str] = [f"# Auxein benchmark report: {results.metadata['config']['name']}", ""]

    findings = out / "findings.md"
    if findings.exists():
        sections += [findings.read_text().strip(), ""]

    sections += [
        "## 1. Convergence",
        "",
        "Median best-so-far error against fitness evaluations, with the interquartile band over runs.",
        "",
    ]
    for problem in results.problems:
        for dim in results.dims:
            if not results.cell(problem, dim):
                continue
            name = f"convergence-{problem}-d{dim}.png"
            convergence_plot(results, problem, dim, out / name)
            sections += [f"![{problem}, d={dim}]({name})", ""]

    sections += [
        "## 2. Summary",
        "",
        "Final error, success rate per precision target (runs that reached it) and expected running time (ERT, in evaluations).",
        "",
    ]
    comparisons = []
    for problem in results.problems:
        for dim in results.dims:
            cell = results.cell(problem, dim)
            if not cell:
                continue
            sections += [f"### {problem}, d={dim}", "", summary_table(results, cell), ""]
            rows = comparison_rows(cell)
            if rows:
                comparisons += [f"### {problem}, d={dim}", "", comparison_table(rows), ""]

    sections += ["## 3. Statistical comparison", ""]
    sections += [
        f"Final error of `{REFERENCE}` against every other algorithm: two-sided Mann-Whitney U with Holm correction within each table, and the Vargha-Delaney A12 effect size (the probability that `{REFERENCE}` wins).",
        "",
    ]
    sections += comparisons or [f"No `{REFERENCE}` runs in this benchmark.", ""]

    sections += ["## 4. Overhead", ""]
    if results.overhead:
        timing, per_generation = overhead_tables(results)
        files = overhead_plots(results, out)
        sections += [timing, ""] + [f"![{f}]({f})" for f in files] + ["", per_generation, ""]
    else:
        sections += ["The overhead benchmark was not run.", ""]

    sections += ["## 5. Sanity checks", "", *sanity_checks(results), ""]
    sections += ["## 6. Methodology", "", methodology(results), ""]

    report = out / "report.md"
    report.write_text("\n".join(sections))
    return report
