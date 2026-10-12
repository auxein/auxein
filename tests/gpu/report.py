"""What the performance probe measured, collected during the session and written into the report."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeRow:
    size: str
    objective: str
    configuration: str
    milliseconds_per_generation: float


PROBE_ROWS: list[ProbeRow] = []


@dataclass(frozen=True)
class PolicyRow:
    network: str
    configuration: str
    milliseconds_per_decode: float
    milliseconds_per_act: float


POLICY_ROWS: list[PolicyRow] = []


def policy_section() -> list[str]:
    """The `MLPPolicy` probe's table: time per `decode_batch` and per `act_batch` call, and the speed-up over numpy."""
    if not POLICY_ROWS:
        return ["## MLP policy probe", "", "The probe did not run."]
    lines = [
        "## MLP policy probe (informational, nothing is asserted)",
        "",
        "One call of `MLPPolicy.decode_batch` and of `act_batch` for 100 candidates, each acting on 10 scenarios × 40 agents "
        "(400 observations of size 16, 2 outputs), median of 20 calls after a warm-up and after the device has finished its queue.",
        "",
        "| Network | Configuration | ms per `decode_batch` | ms per `act_batch` | Speed-up of `act_batch` over numpy float64 (CPU) |",
        "|---|---|---|---|---|",
    ]
    for network in dict.fromkeys(row.network for row in POLICY_ROWS):
        rows = [r for r in POLICY_ROWS if r.network == network]
        reference = next((r.milliseconds_per_act for r in rows if r.configuration.startswith("numpy")), None)
        for row in rows:
            ratio = "" if reference is None else f"{reference / row.milliseconds_per_act:.2f}×"
            lines.append(
                f"| {network} | {row.configuration} | {row.milliseconds_per_decode:.3f} | {row.milliseconds_per_act:.3f} | {ratio} |"
            )
    return lines


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
