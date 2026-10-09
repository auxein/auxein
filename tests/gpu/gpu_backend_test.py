"""`Backend` on the device: validation, and arrays that really live there."""

import numpy as np
import pytest

from auxein.backend import Backend, BackendError, backend_of
from tests.support.fixtures import assert_on_backend, assert_on_device


def test_the_device_is_a_valid_backend_and_its_arrays_live_on_it(gpu_backend: Backend, device: str):
    import torch

    array = gpu_backend.asarray(np.arange(6, dtype=np.float64).reshape(2, 3))
    assert isinstance(array, torch.Tensor) and array.device.type == torch.device(device).type
    assert_on_backend(array, gpu_backend)
    assert backend_of(array).device.split(":")[0] == device.split(":")[0]
    ints = gpu_backend.asarray([1, 2, 3], dtype=gpu_backend.int_dtype)
    assert_on_device(ints, gpu_backend)
    np.testing.assert_array_equal(gpu_backend.to_numpy(array), np.arange(6).reshape(2, 3))  # and comes back to the host intact


def test_for_device_picks_float32_on_a_gpu_and_float64_on_the_cpu(device: str):
    chosen = Backend.for_device("torch", device)
    assert chosen.precision == ("float64" if device == "cpu" else "float32")


def test_float64_on_metal_is_a_configuration_error(device: str):
    if not device.startswith("mps"):
        pytest.skip("only Metal has no float64")
    with pytest.raises(BackendError, match="float64"):
        Backend("torch", device, "float64")


def test_arithmetic_runs_on_the_device_in_the_backends_precision(gpu_backend: Backend):
    xp = gpu_backend.xp
    a = gpu_backend.asarray(np.linspace(0.0, 1.0, 12).reshape(3, 4))
    result = xp.sum(a * a, axis=1)
    assert_on_backend(result, gpu_backend)
    tolerance = 1e-5 if gpu_backend.precision == "float32" else 1e-12
    np.testing.assert_allclose(gpu_backend.to_numpy(result), (np.linspace(0, 1, 12).reshape(3, 4) ** 2).sum(axis=1), rtol=tolerance)


def test_sorting_and_gathering_run_on_the_device(gpu_backend: Backend):
    """Ranking is `argsort` + `take`, the heart of survivor selection."""
    xp = gpu_backend.xp
    values = gpu_backend.asarray([3.0, 1.0, 2.0, 1.0])
    order = xp.argsort(values, stable=True)
    assert_on_device(order, gpu_backend)
    assert gpu_backend.to_numpy(order).tolist() == [1, 3, 2, 0]
    assert_on_backend(xp.take(values, order, axis=0), gpu_backend)
