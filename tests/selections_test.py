from collections import Counter

import numpy as np
import pytest

from auxein.parents.selections import StochasticUniversalSampling
from auxein.parents.selections import cumulative_probability_distribution as cpd


def test_cumulative_probability_distribution_with_known_values():
    probabilities = [0.15, 0.15, 0.25, 0.1, 0.35]
    assert cpd(0, probabilities) == 0.15
    assert cpd(1, probabilities) == 0.30
    assert cpd(2, probabilities) == 0.55
    assert cpd(3, probabilities) == 0.65
    assert cpd(4, probabilities) == 1.0


def test_stochastic_universal_sampling():
    individuals_ids = ["a", "b", "c", "d", "e"]
    probabilities = [0.15, 0.15, 0.25, 0.1, 0.35]
    selection = StochasticUniversalSampling(4096)
    ids = selection.select(individuals_ids, probabilities)

    n = int(selection.parents_to_select)
    assert len(ids) == n

    counts = Counter(ids)
    for individual_id, probability in zip(individuals_ids, probabilities):
        # equally spaced pointers: each count is within one of its expected value
        assert abs(counts[individual_id] - n * probability) <= 1
        np.testing.assert_allclose(counts[individual_id] / n, probability, atol=1 / n)


IDS = ["a", "b", "c", "d", "e"]


@pytest.fixture
def bounded_sampling(monkeypatch):
    """Turns an endless sampling loop into a fast failure, without relying on a signal.

    pytest-timeout's signal method can deadlock pytest-cov when it interrupts a busy loop, so the timeout marks
    below are only a backstop.
    """
    calls = 0

    def bounded(index, probabilities):
        nonlocal calls
        calls += 1
        if calls > 10_000:
            pytest.fail("StochasticUniversalSampling.select did not terminate")
        return sum(probabilities[: index + 1])

    monkeypatch.setattr("auxein.parents.selections.core.cumulative_probability_distribution", bounded, raising=False)


@pytest.mark.timeout(2)
def test_sus_nan_probabilities_do_not_hang(bounded_sampling):
    with pytest.raises(ValueError):
        StochasticUniversalSampling(4096).select(IDS, [np.nan] * 5)


@pytest.mark.timeout(2)
def test_sus_terminates_when_cumulative_sum_rounds_below_one(bounded_sampling, monkeypatch):
    selection = StochasticUniversalSampling(180)
    n = int(selection.parents_to_select)
    assert n == 10
    probabilities = [0.1] * 9 + [0.1 - 1e-12]
    assert sum(probabilities) < 1  # the cumulative sum ends just below the last pointer
    # put the first pointer as far right as it can go, so that the last pointer lands beyond the cumulative sum
    monkeypatch.setattr(np.random, "uniform", lambda low, high: high)
    ids = selection.select([str(i) for i in range(10)], probabilities)
    assert len(ids) == n


@pytest.mark.timeout(2)
@pytest.mark.parametrize(
    "probabilities",
    [
        [0.5, 0.5],  # length mismatch
        [0.5, 0.5, -0.1, 0.1, 0.0],  # negative
        [0.5, np.inf, 0.0, 0.0, 0.0],  # not finite
        [0.0] * 5,  # zero sum
    ],
    ids=["length_mismatch", "negative", "not_finite", "zero_sum"],
)
def test_sus_rejects_invalid_probabilities(bounded_sampling, probabilities):
    with pytest.raises(ValueError):
        StochasticUniversalSampling(4096).select(IDS, probabilities)


@pytest.mark.timeout(2)
def test_sus_normalises_probabilities(bounded_sampling):
    selection = StochasticUniversalSampling(4096)
    n = int(selection.parents_to_select)
    ids = selection.select(IDS, [1, 1, 2, 1, 5])  # weights summing to 10
    counts = Counter(ids)
    assert len(ids) == n
    assert counts["e"] == pytest.approx(n * 0.5, abs=1)
    assert counts["c"] == pytest.approx(n * 0.2, abs=1)
