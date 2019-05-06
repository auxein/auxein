import numpy as np

from auxein.population.dna_builders import UniformRandomDnaBuilder

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