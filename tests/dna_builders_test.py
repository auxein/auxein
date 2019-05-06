from unittest.mock import Mock, patch
import numpy as np

from auxein.population.dna_builders import UniformRandomDnaBuilder, NormalRandomDnaBuilder

def test_uniform_random_dna_builder_instantiation():
    builder = UniformRandomDnaBuilder(interval = (-5, 0))
    assert builder.get_distribution() == 'uniform'
    assert len(builder.get(10)) == 10


def test_uniform_random_dna_builder_values():
    builder = UniformRandomDnaBuilder()
    for _ in range(0, 100):
        dna: np.ndarray = builder.get(2)
        assert -1 < dna[0] < 1
        assert -1 < dna[1] < 1

@patch('numpy.random.normal')
def test_normal_random_dna_builder_instantiation(mock_np_normal):

    mock_np_normal.return_value = [0.5, -1.3]

    builder = NormalRandomDnaBuilder()
    assert builder.get_distribution() == 'normal'
    assert len(builder.get(2)) == 2
    mock_np_normal.assert_called_once_with(0.0, 1.0, 2)
