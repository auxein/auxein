from auxein.parents.selections import cumulative_probability_distribution as cpd, StochasticUniversalSampling

import numpy as np


def test_cumulative_probability_distribution_with_known_values():
    probabilities = [0.15, 0.15, 0.25, 0.1, 0.35]
    assert cpd(0, probabilities) == 0.15
    assert cpd(1, probabilities) == 0.30
    assert cpd(2, probabilities) == 0.55
    assert cpd(3, probabilities) == 0.65
    assert cpd(4, probabilities) == 1.0


def test_stochastic_universal_sampling():
    individuals_ids = ['a', 'b', 'c', 'd', 'e']
    probabilities = [0.15, 0.15, 0.25, 0.1, 0.35]
    ids = StochasticUniversalSampling(4096).select(individuals_ids, probabilities)

    from collections import Counter
    counts = Counter(ids)

    np.testing.assert_almost_equal(counts['a'] / 46, 0.15, 2)
    np.testing.assert_almost_equal(counts['b'] / 46, 0.15, 2)
    np.testing.assert_almost_equal(counts['c'] / 46, 0.25, 2)
    np.testing.assert_almost_equal(counts['d'] / 46, 0.10, 2)
    np.testing.assert_almost_equal(counts['e'] / 46, 0.35, 2)
