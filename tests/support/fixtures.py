"""Fixtures shared by the tests of the new core: every test that takes `backend` runs on each available backend and precision."""

import pytest

from auxein.backend import Array, Backend, backend_of
from auxein.backend.devices import torch_installed

CONFIGURATIONS = [("numpy", "float64"), ("numpy", "float32"), ("torch", "float64"), ("torch", "float32")]


def _parameters() -> list[object]:
    skip_torch = pytest.mark.skipif(not torch_installed(), reason="torch is not installed")
    return [pytest.param(c, id=f"{c[0]}-{c[1]}", marks=[skip_torch] if c[0] == "torch" else []) for c in CONFIGURATIONS]


@pytest.fixture(params=_parameters())
def backend(request: pytest.FixtureRequest) -> Backend:
    name, precision = request.param
    return Backend(name, "cpu", precision)


def assert_on_backend(array: Array, backend: Backend, dtype: object | None = None) -> None:
    """The array is in the backend's namespace and device, with its dtype (the float dtype unless `dtype` is given)."""
    assert backend.matches(array, dtype), (
        f"expected an array of {backend} (dtype {dtype or backend.dtype}), got {type(array)} {getattr(array, 'dtype', None)}"
    )
    assert backend_of(array).name == backend.name
