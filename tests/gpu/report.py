"""What the performance probe measured, collected during the session and written into the report."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeRow:
    size: str
    objective: str
    configuration: str
    milliseconds_per_generation: float


PROBE_ROWS: list[ProbeRow] = []


def probe_section() -> list[str]:
    """The probe's table: per size and objective, the time of one generation on each configuration and the speed-up over numpy."""
    if not PROBE_ROWS:
        return ["## Performance probe", "", "The probe did not run."]
    lines = [
        "## Performance probe (informational, nothing is asserted)",
        "",
        "One generation = `ask` + evaluation + `tell` of a `GeneticAlgorithm`, median of 5 after a warm-up. "
        "`cheap` is a sum of squares per candidate; `heavy` multiplies the whole population (population × dimension) by a fixed "
        "dimension × dimension matrix, then sums the squares. The 1,000 × 1,000 size is the reference one; the others show how the "
        "result moves with the size.",
        "",
        "| Population × dimension | Objective | Configuration | ms per generation | Speed-up over numpy float64 (CPU) |",
        "|---|---|---|---|---|",
    ]
    for size, objective in dict.fromkeys((row.size, row.objective) for row in PROBE_ROWS):
        rows = [r for r in PROBE_ROWS if r.size == size and r.objective == objective]
        reference = next((r.milliseconds_per_generation for r in rows if r.configuration.startswith("numpy")), None)
        for row in rows:
            ratio = "" if reference is None else f"{reference / row.milliseconds_per_generation:.2f}×"
            lines.append(f"| {size} | {objective} | {row.configuration} | {row.milliseconds_per_generation:.1f} | {ratio} |")
    return lines
