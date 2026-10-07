from auxein.parents.selections import cumulative_probability_distribution as cpd, StochasticUniversalSampling

from collections import Counter

import numpy as np


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
