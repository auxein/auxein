"""No global random state in the new core (design doc §7.2 and §8).

Later steps widen COVERED_SUBPACKAGES; step 3 widens it to the whole package once the 0.x code is gone.
"""

from pathlib import Path

import pytest

from tests.support.global_random import NUMPY_LEGACY, find_violations, scan_package

PACKAGE = Path(__file__).resolve().parents[2] / "auxein"
COVERED_SUBPACKAGES = ["core", "spaces", "backend", "random"]


@pytest.mark.parametrize("subpackage", COVERED_SUBPACKAGES)
def test_subpackage_uses_no_global_random_state(subpackage: str):
    assert (PACKAGE / subpackage).is_dir()
    assert scan_package(PACKAGE / subpackage) == []


def test_the_legacy_numpy_names_cover_the_global_api():
    for name in ("seed", "rand", "randn", "random", "uniform", "normal", "randint", "choice", "shuffle", "permutation", "random_sample"):
        assert name in NUMPY_LEGACY
    for name in ("Generator", "PCG64", "SeedSequence", "default_rng"):
        assert name not in NUMPY_LEGACY


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np\nnp.random.seed(0)",
        "import numpy\nx = numpy.random.rand(3)",
        "import numpy as np\nx = np.random.normal(0, 1)",
        "import numpy as np\nnp.random.shuffle(a)",
        "import numpy as np\nf = np.random.choice",
        "import numpy.random as npr\nnpr.randint(0, 5)",
        "from numpy import random as r\nr.permutation(4)",
        "from numpy.random import seed",
        "from numpy.random import uniform as u",
        "from numpy.random import *",
        "import torch\ntorch.manual_seed(0)",
        "import torch\ntorch.cuda.manual_seed_all(0)",
        "import torch\ntorch.random.manual_seed(0)",
        "import torch\ntorch.mps.manual_seed(0)",
        "import torch as t\nt.seed()",
        "from torch import manual_seed",
        "import torch\nx = torch.rand(3)",
        "import torch\nx = torch.randn(3, device='cpu')",
        "import torch\nx = torch.randperm(3)",
        "t = import_torch()\nt.manual_seed(1)",
        "from auxein.backend.devices import import_torch\nimport_torch().rand(3)",
        "import random",
        "import random as r",
        "from random import choice",
    ],
)
def test_the_scan_flags_global_random_state(source: str):
    assert find_violations(source) != []


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np\nrng = np.random.default_rng(1)",
        "import numpy as np\nrng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(3)))",
        "from numpy.random import Generator, PCG64, SeedSequence",
        "import torch\ng = torch.Generator()\ng.manual_seed(3)\nx = torch.rand(3, generator=g)",
        "t = import_torch()\ng = t.Generator()\nx = t.randn((2,), generator=g, device='cpu')",
        "import_torch().randint(0, 3, (2,), generator=g)",
        "from auxein.random import RunSeed",
        "from auxein import random",
        "from . import random",
        "import numpy as np\nx = rng.random(3)\ny = np.random.Generator",
        "import time\nrandom = 3\nx = random + 1",
    ],
)
def test_the_scan_allows_explicit_local_state(source: str):
    assert find_violations(source) == []
