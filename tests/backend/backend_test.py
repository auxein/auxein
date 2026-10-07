import warnings

import numpy as np
import pytest

from auxein.backend import Backend, BackendError, backend_of, default_precision, devices
from auxein.backend.devices import parse_device, torch_installed
from tests.support.fixtures import assert_on_backend

needs_torch = pytest.mark.skipif(not torch_installed(), reason="torch is not installed")


@pytest.fixture
def fake_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend torch is installed, CUDA has two devices and Metal is available."""
    monkeypatch.setattr(devices, "torch_installed", lambda: True)
    monkeypatch.setattr(devices, "cuda_available", lambda: True)
    monkeypatch.setattr(devices, "cuda_device_count", lambda: 2)
    monkeypatch.setattr(devices, "mps_available", lambda: True)


def test_default_backend_is_numpy_cpu_float64():
    backend = Backend()
    assert (backend.name, backend.device, backend.precision) == ("numpy", "cpu", "float64")


def test_backend_is_frozen_and_hashable():
    backend = Backend()
    with pytest.raises(AttributeError):
        backend.precision = "float32"  # type: ignore[misc]
    assert {Backend(): 1}[Backend()] == 1
    assert Backend() != Backend(precision="float32")


def test_default_precision_per_device():
    assert default_precision("cpu") == "float64"
    for device in ("cuda", "cuda:1", "mps"):
        assert default_precision(device) == "float32"


def test_for_device_picks_the_default_precision(fake_gpus: None):
    assert Backend.for_device().precision == "float64"
    assert Backend.for_device("torch", "cpu").precision == "float64"
    assert Backend.for_device("torch", "cuda").precision == "float32"
    assert Backend.for_device("torch", "cuda:1").precision == "float32"
    assert Backend.for_device("torch", "mps").precision == "float32"


def test_for_device_honours_an_explicit_precision(fake_gpus: None):
    assert Backend.for_device("torch", "cuda", "float64").precision == "float64"
    assert Backend.for_device("torch", "cpu", "float32").precision == "float32"
    assert Backend.for_device("numpy", "cpu", "float32").precision == "float32"


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "cuda:0", "cuda:1", "cuda:10"])
def test_parse_device_accepts_supported_devices(device: str):
    assert parse_device(device) is not None


@pytest.mark.parametrize("device", ["", "gpu", "CPU", "cuda:", "cuda:-1", "cuda:01", "cuda:x", "cpu:0", "mps:0", "xpu", " cpu", "cuda "])
def test_parse_device_rejects_everything_else(device: str):
    assert parse_device(device) is None


@pytest.mark.parametrize("device", ["gpu", "cuda:", "cpu:0", "mps:0", "", "tpu"])
def test_unknown_device_strings_are_rejected(device: str, fake_gpus: None):
    with pytest.raises(BackendError, match="unknown device"):
        Backend("torch", device, "float32")
    with pytest.raises(BackendError, match="unknown device"):
        Backend("numpy", device)


def test_unknown_backend_name_and_precision():
    with pytest.raises(BackendError, match="unknown backend 'jax'"):
        Backend("jax")  # type: ignore[arg-type]
    with pytest.raises(BackendError, match="unknown precision 'float16'"):
        Backend("numpy", "cpu", "float16")  # type: ignore[arg-type]


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "mps"])
def test_numpy_only_runs_on_the_cpu(device: str, fake_gpus: None):
    with pytest.raises(BackendError, match="numpy backend only runs on 'cpu'"):
        Backend("numpy", device)


def test_torch_requested_but_not_installed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(devices, "torch_installed", lambda: False)
    with pytest.raises(BackendError, match=r"torch is not installed.*auxein\[torch\]"):
        Backend("torch")
    assert Backend("numpy").name == "numpy"  # numpy never needs torch


def test_cuda_requested_but_not_available(fake_gpus: None, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(devices, "cuda_available", lambda: False)
    for device in ("cuda", "cuda:0"):
        with pytest.raises(BackendError, match="CUDA is not available"):
            Backend("torch", device, "float32")
    assert Backend("torch", "cpu").device == "cpu"  # the other devices are unaffected
    assert Backend("torch", "mps", "float32").device == "mps"


def test_cuda_device_index_out_of_range(fake_gpus: None):
    assert Backend("torch", "cuda:1", "float32").device == "cuda:1"
    with pytest.raises(BackendError, match=r"cuda:2.*only 2 CUDA device"):
        Backend("torch", "cuda:2", "float32")


def test_mps_requested_but_not_available(fake_gpus: None, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(devices, "mps_available", lambda: False)
    with pytest.raises(BackendError, match="Metal .* is not available"):
        Backend("torch", "mps", "float32")


def test_float64_is_rejected_on_mps(fake_gpus: None):
    with pytest.raises(BackendError, match="float64 is not supported on 'mps'"):
        Backend("torch", "mps", "float64")
    assert Backend("torch", "mps", "float32").precision == "float32"
    assert Backend("torch", "cuda", "float64").precision == "float64"  # only Metal lacks float64


def test_float64_on_mps_is_a_configuration_error_even_if_metal_is_unavailable(fake_gpus: None, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(devices, "mps_available", lambda: False)
    with pytest.raises(BackendError, match="float64 is not supported"):
        Backend("torch", "mps", "float64")


def test_backend_errors_are_value_errors():
    assert issubclass(BackendError, ValueError)


def test_namespace_and_dtypes(backend: Backend):
    assert backend.xp.__name__ == f"array_api_compat.{backend.name}"
    assert backend.dtype == getattr(backend.xp, backend.precision)
    assert backend.int_dtype == backend.xp.int64
    assert backend.bool_dtype == backend.xp.bool


def test_asarray_converts_to_the_backend(backend: Backend):
    x = backend.asarray([[1, 2], [3, 4]])
    assert_on_backend(x, backend)
    assert tuple(x.shape) == (2, 2)
    assert_on_backend(backend.asarray([1, 2, 3], dtype=backend.int_dtype), backend, backend.int_dtype)
    assert_on_backend(backend.asarray([True, False], dtype=backend.bool_dtype), backend, backend.bool_dtype)


def test_asarray_converts_between_backends_and_precisions(backend: Backend):
    source = np.linspace(0, 1, 5, dtype=np.float64)
    x = backend.asarray(source)
    assert_on_backend(x, backend)
    np.testing.assert_allclose(backend.to_numpy(x), source, rtol=1e-6)
    # converting an array of this backend again keeps it where it is
    assert_on_backend(backend.asarray(x), backend)
    other = Backend("numpy", "cpu", "float32" if backend.precision == "float64" else "float64")
    converted = other.asarray(x)
    assert_on_backend(converted, other)


def test_asarray_does_not_copy_a_matching_numpy_array():
    backend = Backend()
    x = np.zeros(3)
    assert backend.asarray(x) is x


@needs_torch
def test_asarray_converts_torch_tensors_to_numpy():
    import torch

    x = Backend().asarray(torch.arange(3, dtype=torch.float32))
    assert_on_backend(x, Backend())
    np.testing.assert_array_equal(x, [0.0, 1.0, 2.0])


@needs_torch
def test_asarray_accepts_read_only_numpy_arrays_for_torch_without_warning():
    x = np.arange(3, dtype=np.float64)
    x.setflags(write=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = Backend("torch", "cpu", "float64").asarray(x)
    assert y.tolist() == [0.0, 1.0, 2.0]


def test_to_numpy_round_trip_returns_a_host_copy(backend: Backend):
    x = backend.asarray([1.5, 2.5, 3.5])
    host = backend.to_numpy(x)
    assert isinstance(host, np.ndarray)
    np.testing.assert_allclose(host, [1.5, 2.5, 3.5])
    host[0] = 99.0  # a copy: the original is untouched
    assert float(backend.to_numpy(x)[0]) == 1.5
    assert not np.shares_memory(host, backend.to_numpy(x))


def test_to_numpy_copies_a_numpy_array():
    x = np.arange(3.0)
    host = Backend().to_numpy(x)
    host[0] = 5.0
    assert x[0] == 0.0


def test_readonly_on_numpy_blocks_writes():
    backend = Backend()
    x = np.zeros(3)
    assert backend.readonly(x) is x
    assert not x.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        x[0] = 1.0


def test_readonly_is_a_documented_no_op_for_torch():
    if not torch_installed():
        pytest.skip("torch is not installed")
    backend = Backend("torch", "cpu", "float64")
    x = backend.asarray([1.0, 2.0])
    assert backend.readonly(x) is x
    x[0] = 7.0  # torch has no read-only flag: immutability there is by convention
    assert float(x[0]) == 7.0


def test_readonly_protects_views_of_a_read_only_array():
    backend = Backend()
    x = backend.readonly(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        x[0][1] = 1.0


def test_backend_of_infers_the_backend(backend: Backend):
    assert backend_of(backend.asarray([1.0, 2.0])) == backend


def test_backend_of_defaults_the_precision_of_integer_arrays(backend: Backend):
    inferred = backend_of(backend.asarray([1, 2], dtype=backend.int_dtype))
    assert inferred.name == backend.name
    assert inferred.device == "cpu"
    assert inferred.precision == "float64"


def test_backend_of_rejects_other_types():
    with pytest.raises(TypeError, match="cannot infer a backend from list"):
        backend_of([1.0, 2.0])


def test_matches_checks_namespace_and_dtype(backend: Backend):
    x = backend.asarray([1.0])
    assert backend.matches(x)
    assert not backend.matches([1.0])
    assert not backend.matches(backend.asarray([1], dtype=backend.int_dtype))
    assert backend.matches(backend.asarray([1], dtype=backend.int_dtype), backend.int_dtype)
    other_precision = Backend(backend.name, "cpu", "float32" if backend.precision == "float64" else "float64")
    assert not backend.matches(other_precision.asarray([1.0]))


@needs_torch
def test_matches_distinguishes_numpy_from_torch():
    assert not Backend("numpy").matches(Backend("torch").asarray([1.0]))
    assert not Backend("torch").matches(np.zeros(1))


@pytest.mark.skipif(not (torch_installed() and devices.mps_available()), reason="needs a machine with Metal")
def test_mps_smoke():
    """Runs only on a Mac with Metal (a manual check: hosted CI has no GPU)."""
    backend = Backend.for_device("torch", "mps")
    assert backend.precision == "float32"
    x = backend.asarray([1.0, 2.0])
    assert backend.matches(x)
    assert backend_of(x) == backend
    np.testing.assert_allclose(backend.to_numpy(x), [1.0, 2.0])
