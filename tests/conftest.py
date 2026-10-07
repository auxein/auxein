import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_numpy_random() -> None:
    np.random.seed(42)
