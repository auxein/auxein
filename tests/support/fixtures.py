"""Fixtures shared by the tests of the new core: every test that takes `backend` runs on each available backend and precision."""

from collections.abc import Iterator

import pytest

from auxein.backend import Array, Backend, backend_of
from auxein.backend.devices import torch_installed

CONFIGURATIONS = [("numpy", "float64"), ("numpy", "float32"), ("torch", "float64"), ("torch", "float32")]
"""The four configurations that unit tests run on: each backend in each precision."""

CORNERS = [("numpy", "float64"), ("torch", "float32")]
"""The two configurations that integration tests run on (design doc §7.4): the reference backend in the reference precision,
and the most different backend in the most different precision. A bug that depends on the backend or on the precision shows
on one of them, and half the configurations are half the minutes of a suite whose end-to-end runs, kills and resumes cost seconds each.
The two mixed configurations are covered by the unit tests, which run on all four."""


def _parameters(configurations: list[tuple[str, str]]) -> list[object]:
    skip_torch = pytest.mark.skipif(not torch_installed(), reason="torch is not installed")
    return [pytest.param(c, id=f"{c[0]}-{c[1]}", marks=[skip_torch] if c[0] == "torch" else []) for c in configurations]


@pytest.fixture(params=_parameters(CONFIGURATIONS))
def backend(request: pytest.FixtureRequest) -> Backend:
    """Unit tests: every backend in every precision (CPU)."""
    name, precision = request.param
    return Backend(name, "cpu", precision)


@pytest.fixture(params=_parameters(CORNERS))
def corner_backend(request: pytest.FixtureRequest) -> Backend:
    """Integration tests (end-to-end runs, resume, failures, timeouts, episode runs): the two corner configurations."""
    name, precision = request.param
    return Backend(name, "cpu", precision)


_CURRENT: list[Backend] = [Backend()]


def integration_backend() -> Backend:
    """The backend of the integration test that is running: the numpy-float64 default outside of one.

    Integration tests build their runs through shared helpers (`arguments`, `go`, `settings`) in many places, and passing the
    fixture through every one of them would touch every test for nothing. A module opts in with
    `pytestmark = pytest.mark.usefixtures("use_corner_backend")`, which runs each of its tests once per corner and makes this
    function return that corner's backend; the helpers pass it to `run` and `resume`.
    """
    return _CURRENT[0]


@pytest.fixture
def use_corner_backend(corner_backend: Backend) -> Iterator[Backend]:
    """Run the test on each corner configuration (see `integration_backend`)."""
    _CURRENT[0] = corner_backend
    try:
        yield corner_backend
    finally:
        _CURRENT[0] = Backend()


def assert_on_backend(array: Array, backend: Backend, dtype: object | None = None) -> None:
    """The array is in the backend's namespace and device, with its dtype (the float dtype unless `dtype` is given)."""
    assert backend.matches(array, dtype), (
        f"expected an array of {backend} (dtype {dtype or backend.dtype}), got {type(array)} {getattr(array, 'dtype', None)}"
    )
    assert backend_of(array).name == backend.name


def assert_on_device(array: Array, backend: Backend) -> None:
    """The array is in the backend's namespace and on its device, whatever its dtype (ids and masks are not floats).

    This is the check for results that must not have been moved to the host on the way (design doc §7.2, rule 3).
    On a CPU-only machine a torch tensor on the host looks like one on the "device", so the check only has teeth on a GPU,
    where the smoke suite (`tests/gpu/`) runs it; the unit tests run it on CPU so that the code path is exercised
    everywhere and a regression to numpy, which has no device, is caught.
    """
    assert backend.matches(array, getattr(array, "dtype", None)), (
        f"expected an array on {backend.name}:{backend.device}, got {type(array)} {getattr(array, 'device', 'on the host')}"
    )
