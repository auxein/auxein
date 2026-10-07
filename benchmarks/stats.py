"""The statistics behind the report: success rates, expected running time, Mann-Whitney with Holm, Vargha-Delaney."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats

ALPHA = 0.05


def success_rate(records: Sequence[dict[str, Any]], target_key: str) -> tuple[int, int]:
    """(runs that reached the target, runs)."""
    return sum(1 for r in records if r["hits"][target_key] is not None), len(records)


def expected_running_time(records: Sequence[dict[str, Any]], target_key: str) -> float:
    """Total evaluations spent over all runs divided by the number of successful runs (infinity if none succeeded).

    A successful run spends the evaluations it needed to reach the target; an unsuccessful run spends all of
    the evaluations it made. This is the usual definition (COCO), and it estimates the evaluations needed to reach the
    target with restarts from scratch.
    """
    spent = 0
    successes = 0
    for r in records:
        hit = r["hits"][target_key]
        if hit is None:
            spent += r["evals"]
        else:
            spent += hit
            successes += 1
    return spent / successes if successes else math.inf


def vargha_delaney(better: Sequence[float], other: Sequence[float]) -> float:
    """A12 for errors (lower is better): the probability that a run of `better` ends with a lower error than a run of `other`.

    Ties count half. 0.5 means no difference, 1 means `better` always wins, 0 means it always loses.
    """
    x, y = np.asarray(better, dtype=float), np.asarray(other, dtype=float)
    lower = (x[:, None] < y[None, :]).sum()
    ties = (x[:, None] == y[None, :]).sum()
    return float((lower + 0.5 * ties) / (len(x) * len(y)))


def effect_size_label(a12: float) -> str:
    """Vargha and Delaney's thresholds on |A12 - 0.5|: 0.06 small, 0.14 medium, 0.21 large."""
    distance = abs(a12 - 0.5)
    if distance < 0.06:
        return "negligible"
    if distance < 0.14:
        return "small"
    if distance < 0.21:
        return "medium"
    return "large"


def mann_whitney_p(x: Sequence[float], y: Sequence[float]) -> float:
    """Two-sided Mann-Whitney U p-value. 1 when the two samples are identical, where the test is undefined."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.all(a == a[0]) and np.all(b == a[0]):
        return 1.0
    result: Any = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(result.pvalue)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the order given."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def reading(reference: str, other: str, a12: float, p_adjusted: float) -> str:
    """A one-line plain-English reading of one comparison; `a12` is the probability that `reference` wins."""
    effect = effect_size_label(a12)
    if p_adjusted >= ALPHA:
        return f"no significant difference ({effect} effect)"
    winner = reference if a12 > 0.5 else other
    return f"{winner} better, {effect} effect"


def resample_traces(traces: Sequence[Sequence[Sequence[float]]], grid: Sequence[int]) -> np.ndarray:
    """Best-so-far error of every run at every evaluation count of `grid` (a step function; a stopped run stays at its last value)."""
    out = np.empty((len(traces), len(grid)))
    for i, trace in enumerate(traces):
        evals = np.array([e for e, _ in trace])
        errors = np.array([err for _, err in trace])
        index = np.searchsorted(evals, grid, side="right") - 1
        out[i] = np.where(index >= 0, errors[np.maximum(index, 0)], np.nan)
    return out
