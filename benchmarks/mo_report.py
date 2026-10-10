"""Build report.md and plots from a multi-objective results directory: `python -m benchmarks report <results-dir>`."""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from pymoo.indicators.hv import HV

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks import stats  # noqa: E402
from benchmarks.mo_problems import make_mo_problem  # noqa: E402
from benchmarks.report import COLORS, FALLBACK_COLORS, GRID, INK, MUTED, fmt, style_axes  # noqa: E402
from benchmarks.runner import read_jsonl  # noqa: E402

MO_COLORS = {"auxein-nsga2": COLORS["auxein-default"], "pymoo-nsga2": COLORS["auxein-fixedvar"], "random-search": COLORS["random-search"]}
TRUE_FRONT = "#52514e"
MIN_RUNS_FOR_TESTS = 10


class MOResults:
    """A multi-objective results directory, loaded."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.runs = read_jsonl(path / "runs.jsonl")
        self.metadata: dict[str, Any] = json.loads((path / "metadata.json").read_text())
        configured = [a["name"] for a in self.metadata["config"]["algorithms"]]
        seen = list(dict.fromkeys(r["algorithm"] for r in self.runs))
        self.algorithms = [a for a in configured if a in seen] + [a for a in seen if a not in configured]
        references = self.metadata["config"].get("report", {}).get("references", self.algorithms[:1])
        self.references: list[str] = [r for r in references if r in self.algorithms] or self.algorithms[:1]
        self.problems = list(dict.fromkeys(r["problem"] for r in self.runs))
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in self.runs:
            groups[(r["problem"], r["algorithm"])].append(r)
        self.groups = {k: sorted(v, key=lambda r: r["seed"]) for k, v in groups.items()}

    def cell(self, problem: str) -> dict[str, list[dict[str, Any]]]:
        return {a: self.groups[(problem, a)] for a in self.algorithms if (problem, a) in self.groups}

    def color(self, algorithm: str) -> str:
        if algorithm in MO_COLORS:
            return MO_COLORS[algorithm]
        extra = [a for a in self.algorithms if a not in MO_COLORS]
        return FALLBACK_COLORS[extra.index(algorithm) % len(FALLBACK_COLORS)]


def hypervolumes(records: list[dict[str, Any]]) -> list[float]:
    return [r["final_hv"] for r in records]


def igd_values(records: list[dict[str, Any]]) -> list[float]:
    return [r["final_igd_plus"] for r in records]


def true_front_hypervolume(problem_name: str) -> float:
    problem = make_mo_problem(problem_name)
    value = HV(ref_point=problem.reference_point)(problem.pareto_front())
    assert value is not None
    return float(value)


def resample(traces: list[list[list[float]]], grid: list[int]) -> np.ndarray:
    """The indicator of every run at every evaluation count of `grid` (a step function; NaN before the first checkpoint)."""
    out = np.empty((len(traces), len(grid)))
    for i, trace in enumerate(traces):
        evals = np.array([e for e, _ in trace])
        values = np.array([v for _, v in trace])
        index = np.searchsorted(evals, grid, side="right") - 1
        out[i] = np.where(index >= 0, values[np.maximum(index, 0)], np.nan)
    return out


def convergence_plot(results: MOResults, problem: str, path: Path) -> None:
    cell = results.cell(problem)
    grid = sorted({e for records in cell.values() for r in records for e, _ in r["trace_hv"]})
    best = true_front_hypervolume(problem)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110, facecolor="#fcfcfb")
    for algorithm, records in cell.items():
        values = resample([r["trace_hv"] for r in records], grid) / best
        low, median, high = np.nanpercentile(values, [25, 50, 75], axis=0)
        color = results.color(algorithm)
        ax.fill_between(grid, low, high, color=color, alpha=0.18, linewidth=0)
        ax.plot(grid, median, color=color, linewidth=2, label=algorithm)
    ax.axhline(1.0, color=MUTED, linewidth=0.7, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlim(left=50)  # nothing of interest happens before the first generations
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("fitness evaluations", color=INK, fontsize=9)
    ax.set_ylabel("hypervolume / hypervolume of the true front (median, IQR band)", color=INK, fontsize=8)
    ax.set_title(problem, color=INK, fontsize=11, loc="left")
    style_axes(ax)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def fronts_plot(results: MOResults, problem: str, path: Path) -> None:
    """The final non-dominated set of the first run (seed 0) of every algorithm, against the true front: one panel with all the
    algorithms for two objectives, and one 3-D panel per algorithm for three (a projection of a surface onto a plane hides it)."""
    true_front = make_mo_problem(problem).pareto_front()
    cell = results.cell(problem)
    n_obj = true_front.shape[1]
    limit = 1.5 * np.max(true_front, axis=0) + 0.5
    if n_obj == 2:
        fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=110, facecolor="#fcfcfb")
        ax.scatter(true_front[:, 0], true_front[:, 1], s=3, color=TRUE_FRONT, alpha=0.5, label="true front", zorder=1)
        for rank, (algorithm, records) in enumerate(cell.items()):
            front = np.array(records[0]["front"])
            front = front[np.all(front <= limit, axis=1)]
            if len(front):  # the first algorithms are drawn smaller and on top, so that fronts that coincide are all visible
                ax.scatter(
                    front[:, 0],
                    front[:, 1],
                    s=6 + 8 * rank,
                    color=results.color(algorithm),
                    alpha=0.85,
                    label=algorithm,
                    zorder=10 - rank,
                    linewidths=0,
                )
        ax.set_xlabel("f1", color=INK, fontsize=9)
        ax.set_ylabel("f2", color=INK, fontsize=9)
        ax.set_title(f"{problem}: final fronts of the first run", color=INK, fontsize=10, loc="left")
        style_axes(ax)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK, markerscale=2)
    else:
        fig = plt.figure(figsize=(4.4 * len(cell), 4.4), dpi=110, facecolor="#fcfcfb")
        for position, (algorithm, records) in enumerate(cell.items(), start=1):
            ax: Any = fig.add_subplot(1, len(cell), position, projection="3d")  # an Axes3D, which matplotlib's stubs do not model
            front = np.array(records[0]["front"])
            front = front[np.all(front <= limit, axis=1)]
            if len(front) > 2500:  # a front of thousands of points is a cloud: a sample of it shows the surface
                front = front[np.random.default_rng(0).choice(len(front), 2500, replace=False)]
            ax.scatter(true_front[:, 0], true_front[:, 1], true_front[:, 2], s=2, color=TRUE_FRONT, alpha=0.25)
            ax.scatter(front[:, 0], front[:, 1], front[:, 2], s=4, color=results.color(algorithm), alpha=0.7, linewidths=0)
            ax.view_init(elev=22, azim=40)
            ax.set_title(algorithm, color=INK, fontsize=10)
            for label, setter in (("f1", ax.set_xlabel), ("f2", ax.set_ylabel), ("f3", ax.set_zlabel)):
                setter(label, color=INK, fontsize=8)
            ax.tick_params(colors=MUTED, labelsize=7)
        fig.suptitle(f"{problem}: final fronts of the first run, against the true front (grey)", color=INK, fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def summary_table(results: MOResults, problem: str) -> str:
    best = true_front_hypervolume(problem)
    header = [
        "Algorithm",
        "Final hypervolume, median [IQR]",
        "Share of the true front's hypervolume",
        "IGD+, median [IQR]",
        "Non-dominated points, median",
    ]
    rows = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for algorithm, records in results.cell(problem).items():
        hv, igd = hypervolumes(records), igd_values(records)
        h1, hm, h3 = np.percentile(hv, [25, 50, 75])
        g1, gm, g3 = np.percentile(igd, [25, 50, 75])
        rows.append(
            f"| {algorithm} | {hm:.4f} [{h1:.4f}, {h3:.4f}] | {hm / best:.1%} | {fmt(gm)} [{fmt(g1)}, {fmt(g3)}] | "
            f"{statistics.median(len(r['front']) for r in records):g} |"
        )
    return "\n".join(rows)


def comparison_rows(cell: dict[str, list[dict[str, Any]]], reference_name: str) -> list[dict[str, Any]]:
    """The reference against every other algorithm on final hypervolume (higher is better), with Holm correction in the cell."""
    if reference_name not in cell:
        return []
    reference = hypervolumes(cell[reference_name])
    rows = []
    for algorithm, records in cell.items():
        if algorithm == reference_name:
            continue
        other = hypervolumes(records)
        rows.append(
            {
                "algorithm": algorithm,
                "median_reference": statistics.median(reference),
                "median_other": statistics.median(other),
                "a12": stats.vargha_delaney([-v for v in reference], [-v for v in other]),  # the probability that the reference is higher
                "p": stats.mann_whitney_p(reference, other),
            }
        )
    for row, adjusted in zip(rows, stats.holm([r["p"] for r in rows]), strict=True):
        row["p_holm"] = adjusted
        row["reading"] = stats.reading(reference_name, row["algorithm"], row["a12"], adjusted)
    return rows


def comparison_table(rows: list[dict[str, Any]], reference_name: str) -> str:
    lines = [
        f"| {reference_name} vs | median hypervolume ({reference_name}) | median hypervolume (other) | A12 | p | p (Holm) | Reading |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['algorithm']} | {r['median_reference']:.4f} | {r['median_other']:.4f} | {r['a12']:.2f} | {r['p']:.2g} | {r['p_holm']:.2g} | {r['reading']} |"
        )
    return "\n".join(lines)


def acceptance(results: MOResults) -> list[str]:
    """The acceptance criteria of the benchmark, one line each per problem (the reference is the first of the config)."""
    reference = results.references[0]
    lines = []
    for problem in results.problems:
        cell = results.cell(problem)
        if reference not in cell:
            continue
        runs = len(cell[reference])
        if runs < MIN_RUNS_FOR_TESTS:
            lines.append(
                f"- not assessed: {problem}: {runs} runs are too few for a significance test (a smoke run); the full config has 25"
            )
            continue
        rows = {r["algorithm"]: r for r in comparison_rows(cell, reference)}
        if "pymoo-nsga2" in rows:
            r = rows["pymoo-nsga2"]
            worse = r["p_holm"] < stats.ALPHA and r["a12"] < 0.5
            lines.append(
                f"- {'FAIL' if worse else 'PASS'}: {problem}: `{reference}` is {'significantly worse than' if worse else 'not significantly worse than'} "
                f"pymoo's NSGA-II on final hypervolume (A12 = {r['a12']:.2f}, Holm p = {r['p_holm']:.2g})"
            )
        if "random-search" in rows:
            r = rows["random-search"]
            better = r["p_holm"] < stats.ALPHA and r["a12"] > 0.5
            lines.append(
                f"- {'PASS' if better else 'FAIL'}: {problem}: `{reference}` is {'significantly better than' if better else 'not significantly better than'} "
                f"random search on final hypervolume (A12 = {r['a12']:.2f}, Holm p = {r['p_holm']:.2g})"
            )
    return lines


PROBLEM_TABLE = """| Problem | Objectives, variables | Front | Reference point for hypervolume |
|---|---|---|---|
| zdt1 | 2, 30 in [0, 1] | convex, `f2 = 1 − √f1` | (1.1, 1.1) |
| zdt2 | 2, 30 in [0, 1] | concave, `f2 = 1 − f1²` | (1.1, 1.1) |
| zdt3 | 2, 30 in [0, 1] | five disconnected pieces of `f2 = 1 − √f1 − f1 sin(10π f1)` | (1.1 × 0.8518, 1.1 × 1) |
| dtlz2 | 3, 12 in [0, 1] | the positive octant of the unit sphere | (1.1, 1.1, 1.1) |"""


def methodology(results: MOResults) -> str:
    meta, config = results.metadata, results.metadata["config"]
    versions, cpu = meta["versions"], meta["cpu"]
    algorithms = "\n".join(
        f"  - `{a['name']}` (adapter `{a['adapter']}`): `{json.dumps(a.get('params', {}))}`" for a in config["algorithms"]
    )
    budgets = ", ".join(f"{p['name']}: {p['budget']:,}" for p in config["problems"])
    commit = (meta["git_sha"] or "unknown")[:10] + (" (dirty working tree)" if meta["git_dirty"] else "")
    return f"""- **Config**: `{config["name"]}`, commit `{commit}`, run at {meta["timestamp"]} on {meta["workers"]} worker processes.
- **Software**: Python {versions["python"]}, auxein {versions["auxein"]}, numpy {versions["numpy"]}, pymoo {versions.get("pymoo")}, scipy {versions["scipy"]}.
- **Machine**: {cpu["model"]} ({cpu["logical_cpus"]} logical CPUs), {cpu["system"]}.
- **Budget** (fitness evaluations per run, counted outside the algorithms by `MOCountingObjective`): {budgets}. The initial population counts and a partial generation is fine. Progress is always against evaluations.
- **Runs and seeds**: {config["runs"]} runs per problem and algorithm; run k uses seed {config.get("base_seed", 0)} + k. The problems have no instances: runs differ by the seed of the algorithm alone.
- **What is measured**: the **non-dominated set of everything a run has evaluated so far** (an archive kept by the harness, not the algorithm's population). The **hypervolume** of that set against a fixed reference point (1.1 times the nadir point of the true front, componentwise) is the main indicator (higher is better), and **IGD+** to a dense sample of the true front the secondary one (lower is better). Both are computed with pymoo's `HV` and `IGDPlus`. Hypervolume is also shown as a share of the hypervolume of the true front itself (the same dense sample), so that 100% is a perfect front (it can exceed it by a hair: the sample of the true front is finite, and a front of thousands of points covers slightly more). Traces are recorded at about 20 log-spaced evaluation counts per decade.
- **Statistics**: two-sided Mann-Whitney U on the final hypervolume of the reference (`{"`, `".join(results.references)}`) against every other algorithm, with Holm correction over the comparisons of each problem. A12 is the probability that a run of the reference ends with a *higher* hypervolume than a run of the other algorithm (ties count half). Effect size labels follow Vargha and Delaney (negligible below |A12 − 0.5| = 0.06, small below 0.14, medium below 0.21, large above). Significance level 0.05.
- **Algorithms**:
{algorithms}
- **Differences from pymoo's NSGA-II that the parameters do not remove**: Auxein's simulated binary crossover is the *unbounded* form (children may leave the box and are clipped back by the bounds repair), pymoo's is the *bounded* form of Deb's code; pymoo eliminates duplicate offspring and Auxein does not; Auxein makes one child per pair of parents and pymoo two.

### Problems

{PROBLEM_TABLE}

### Reproducing

```
uv run --group bench python -m benchmarks run --config benchmarks/configs/{config["name"]}.toml
uv run --group bench python -m benchmarks report <results-dir>
```

The same config, commit and seeds give identical `runs.jsonl` contents, except for the `wall_time` fields."""


def build_mo_report(results_dir: Path) -> Path:
    results = MOResults(Path(results_dir))
    out = results.path
    sections: list[str] = [f"# Auxein multi-objective benchmark report: {results.metadata['config']['name']}", ""]

    findings = out / "findings.md"
    if findings.exists():
        sections += [findings.read_text().strip(), ""]

    sections += [
        "## 1. Hypervolume against evaluations",
        "",
        "Median and interquartile band over runs, as a share of the hypervolume of the true front.",
        "",
    ]
    for problem in results.problems:
        name = f"hypervolume-{problem}.png"
        convergence_plot(results, problem, out / name)
        sections += [f"![{problem}]({name})", ""]

    sections += ["## 2. Final fronts", "", "The non-dominated set of the first run of every algorithm, against the true front.", ""]
    for problem in results.problems:
        name = f"fronts-{problem}.png"
        fronts_plot(results, problem, out / name)
        sections += [f"![{problem}]({name})", ""]

    sections += ["## 3. Summary", ""]
    for problem in results.problems:
        sections += [f"### {problem}", "", summary_table(results, problem), ""]

    sections += ["## 4. Statistical comparison", ""]
    several = len(results.references) > 1
    for reference_name in results.references:
        if several:
            sections += [f"### Reference: `{reference_name}`", ""]
        for problem in results.problems:
            rows = comparison_rows(results.cell(problem), reference_name)
            if rows:
                sections += [f"{'####' if several else '###'} {problem}", "", comparison_table(rows, reference_name), ""]

    sections += ["## 5. Acceptance", "", *acceptance(results), ""]
    sections += ["## 6. Methodology", "", methodology(results), ""]

    report = out / "report.md"
    report.write_text("\n".join(sections))
    _ = (GRID, math)
    return report
