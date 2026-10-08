import math
from itertools import pairwise

import numpy as np
import pytest

from benchmarks import stats


def record(evals, hits, final=1.0):
    return {"evals": evals, "hits": hits, "final_error": final}


def test_holm_matches_the_textbook_example():
    # sorted p: 0.01, 0.03, 0.04 -> times 3, 2, 1 -> 0.03, 0.06, 0.04 -> running maximum 0.03, 0.06, 0.06
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([0.5]) == [0.5]
    assert stats.holm([0.4, 0.9]) == pytest.approx([0.8, 0.9])
    assert stats.holm([0.001, 0.2]) == pytest.approx([0.002, 0.2])
    assert max(stats.holm([0.6, 0.7, 0.9])) <= 1.0


def test_holm_is_monotone_in_the_raw_p_values():
    raw = [0.002, 0.04, 0.01, 0.3, 0.2]
    adjusted = stats.holm(raw)
    order = np.argsort(raw)
    assert all(adjusted[a] <= adjusted[b] for a, b in pairwise(order))
    assert all(a >= r for a, r in zip(adjusted, raw, strict=True))


def test_vargha_delaney_extremes_ties_and_symmetry():
    assert stats.vargha_delaney([1, 2, 3], [4, 5, 6]) == 1.0  # lower error always wins
    assert stats.vargha_delaney([4, 5, 6], [1, 2, 3]) == 0.0
    assert stats.vargha_delaney([1, 2, 3], [1, 2, 3]) == 0.5
    assert stats.vargha_delaney([1, 1], [1, 1]) == 0.5
    a, b = [1.0, 5.0, 2.0], [3.0, 4.0, 0.5, 6.0]
    assert stats.vargha_delaney(a, b) + stats.vargha_delaney(b, a) == pytest.approx(1.0)
    assert stats.vargha_delaney([1, 3], [2]) == 0.5


def test_effect_size_labels():
    assert stats.effect_size_label(0.5) == "negligible"
    assert stats.effect_size_label(0.55) == "negligible"
    assert stats.effect_size_label(0.4) == "small"
    assert stats.effect_size_label(0.66) == "medium"
    assert stats.effect_size_label(0.1) == "large"
    assert stats.effect_size_label(1.0) == "large"


def test_mann_whitney():
    assert stats.mann_whitney_p([1, 1, 1], [1, 1, 1]) == 1.0
    assert stats.mann_whitney_p(list(range(20)), list(range(100, 120))) < 1e-5
    assert stats.mann_whitney_p([1, 2, 3, 4, 5], [1.5, 2.5, 3.5, 4.5, 5.5]) > 0.5


def test_reading():
    assert stats.reading("auxein-default", "cma-es", 0.02, 1e-6) == "cma-es better, large effect"
    assert stats.reading("auxein-default", "random-search", 0.95, 1e-6) == "auxein-default better, large effect"
    assert stats.reading("auxein-default", "random-search", 0.6, 0.3) == "no significant difference (small effect)"
    assert stats.reading("auxein-default", "random-search", 0.5, 1.0) == "no significant difference (negligible effect)"


def test_ert_counts_hit_time_for_successes_and_all_evaluations_for_failures():
    runs = [record(1000, {"t": 100}), record(1000, {"t": 300}), record(1000, {"t": None}), record(500, {"t": None})]
    # spent: 100 + 300 + 1000 + 500 over 2 successes
    assert stats.expected_running_time(runs, "t") == 950
    assert stats.success_rate(runs, "t") == (2, 4)


def test_ert_is_infinite_without_a_success():
    runs = [record(1000, {"t": None}), record(1000, {"t": None})]
    assert math.isinf(stats.expected_running_time(runs, "t"))
    assert stats.success_rate(runs, "t") == (0, 2)


def test_resample_traces_is_a_step_function_and_a_stopped_run_stays_at_its_last_value():
    trace = [[1, 10.0], [5, 4.0], [20, 1.0]]
    values = stats.resample_traces([trace], [1, 2, 4, 5, 19, 20, 100])
    assert values[0].tolist() == [10.0, 10.0, 10.0, 4.0, 4.0, 1.0, 1.0]
