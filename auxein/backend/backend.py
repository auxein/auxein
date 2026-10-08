"""The numeric backend: array namespace, device and precision (design doc §7)."""

import importlib
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, cast

import numpy as np

from auxein.backend import devices
from auxein.backend.types import Array, ArrayNamespace, HostArray

BackendName = Literal["numpy", "torch"]
Precision = Literal["float64", "float32"]

_NAMES = ("numpy", "torch")
_PRECISIONS = ("float64", "float32")


class BackendError(ValueError):
    """An invalid or unavailable backend configuration. Raised when the `Backend` is constructed, never later."""


def default_precision(device: str) -> Precision:
    """float32 on GPU devices (Metal has no float64, and consumer GPUs are slow at it), float64 on the CPU."""
    return "float64" if device == "cpu" else "float32"


def _validate(name: str, device: str, precision: str) -> None:
    if name not in _NAMES:
        raise BackendError(f"unknown backend {name!r}: expected 'numpy' or 'torch'")
    if precision not in _PRECISIONS:
        raise BackendError(f"unknown precision {precision!r}: expected 'float64' or 'float32'")
    parsed = devices.parse_device(device)
    if parsed is None:
        raise BackendError(f"unknown device {device!r}: expected {devices.SUPPORTED_DEVICES}")
    kind, index = parsed

    if name == "numpy":
        if kind != "cpu":
            raise BackendError(f"the numpy backend only runs on 'cpu', not {device!r}: use the torch backend for GPU devices")
        return

    if not devices.torch_installed():
        raise BackendError("the torch backend was requested but torch is not installed: install it with `pip install auxein[torch]`")
    if kind == "mps" and precision == "float64":
        raise BackendError("float64 is not supported on 'mps': Metal has no double precision, use precision='float32'")
    if kind == "cuda":
        if not devices.cuda_available():
            raise BackendError(f"device {device!r} was requested but CUDA is not available")
        if index is not None and index >= devices.cuda_device_count():
            raise BackendError(f"device {device!r} was requested but only {devices.cuda_device_count()} CUDA device(s) are available")
    elif kind == "mps" and not devices.mps_available():
        raise BackendError("device 'mps' was requested but Metal (MPS) is not available")


@dataclass(frozen=True)
class Backend:
    """Array namespace, device and precision for numeric work.

    `Backend()` is numpy on the CPU in float64. The configuration is validated when the object is constructed, so an
    unusable backend (torch missing, a GPU that isn't there, float64 on Metal) fails at once with a clear message,
    not in the middle of a run. Use `Backend.for_device` to get the default precision of a device.
    """

    name: BackendName = "numpy"
    device: str = "cpu"
    precision: Precision = "float64"

    def __post_init__(self) -> None:
        _validate(self.name, self.device, self.precision)

    def __reduce__(self) -> tuple[type["Backend"], tuple[str, str, str]]:
        """Pickle by configuration: the cached array namespace is a module, which cannot be pickled."""
        return (Backend, (self.name, self.device, self.precision))

    @classmethod
    def for_device(cls, name: BackendName = "numpy", device: str = "cpu", precision: Precision | None = None) -> "Backend":
        """Build a backend, choosing float32 for GPU devices and float64 for the CPU unless `precision` is given."""
        return cls(name, device, default_precision(device) if precision is None else precision)

    @cached_property
    def xp(self) -> ArrayNamespace:
        """The array API namespace (`array_api_compat.numpy` or `array_api_compat.torch`)."""
        return importlib.import_module(f"array_api_compat.{self.name}")

    @cached_property
    def dtype(self) -> object:
        """The floating-point dtype of this backend's namespace."""
        return getattr(self.xp, self.precision)

    @cached_property
    def int_dtype(self) -> object:
        """The integer dtype used for indices, ids and random integers (int64)."""
        return self.xp.int64

    @cached_property
    def bool_dtype(self) -> object:
        return self.xp.bool

    def asarray(self, x: Array, dtype: object | None = None) -> Array:
        """Convert `x` to an array of this backend's namespace, device and dtype (`dtype` defaults to the float dtype).

        Arrays that already match are returned as they are, without a copy. Pass `dtype=backend.int_dtype` or
        `backend.bool_dtype` for index and mask arrays.
        """
        target = self.dtype if dtype is None else dtype
        if self.name == "numpy":
            if _is_torch_tensor(x):
                x = x.detach().cpu().numpy()
            return np.asarray(x, dtype=cast("np.dtype[np.generic]", target))

        torch = devices.import_torch()
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=target)
        if _is_numpy_array(x) and not cast("HostArray", x).flags.writeable:
            x = cast("HostArray", x).copy()  # torch cannot share a read-only buffer, and warns when asked to
        return torch.as_tensor(x, dtype=target, device=self.device)

    def to_numpy(self, x: Array) -> HostArray:
        """Copy an array of any supported backend to a new host numpy array."""
        if _is_torch_tensor(x):
            return np.array(x.detach().cpu().numpy(), copy=True)
        return np.array(x, copy=True)

    def readonly(self, x: Array) -> Array:
        """Mark a numpy array read-only (`setflags(write=False)`) and return it.

        This is a no-op for torch tensors: torch has no read-only flag, so there immutability is by convention (the
        design forbids in-place mutation, §7.2) and is checked by tests, not enforced by the library. Note that a
        read-only flag protects only the array it is set on, not other views of the same memory.
        """
        if _is_numpy_array(x):
            cast("HostArray", x).setflags(write=False)
        return x

    def matches(self, x: Array, dtype: object | None = None) -> bool:
        """Whether `x` is an array of this backend's namespace and device, with this dtype (the float dtype by default)."""
        target = self.dtype if dtype is None else dtype
        if self.name == "numpy":
            return _is_numpy_array(x) and bool(cast("HostArray", x).dtype == target)
        torch = devices.import_torch()
        if not isinstance(x, torch.Tensor) or x.dtype != target:
            return False
        parsed = devices.parse_device(self.device)
        assert parsed is not None  # validated at construction
        kind, index = parsed
        return bool(x.device.type == kind and (index is None or x.device.index == index))


def _is_numpy_array(x: Array) -> bool:
    return isinstance(x, np.ndarray)


def _is_torch_tensor(x: Array) -> bool:
    if not devices.torch_installed():
        return False
    return isinstance(x, devices.import_torch().Tensor)


def is_array(x: Array) -> bool:
    """Whether `x` is an array of a supported backend: a numpy array or a torch tensor."""
    return _is_numpy_array(x) or _is_torch_tensor(x)


def backend_of(array: Array) -> Backend:
    """Infer a `Backend` from an existing array: its namespace, device and (float) precision.

    Integer or boolean arrays say nothing about the precision, which then defaults per device. The device of a torch
    tensor is reported with its index (`cuda:0`).
    """
    if _is_numpy_array(array):
        return Backend("numpy", "cpu", "float32" if cast("HostArray", array).dtype == np.float32 else "float64")
    if _is_torch_tensor(array):
        torch = devices.import_torch()
        kind, index = array.device.type, array.device.index
        device = f"cuda:{index}" if kind == "cuda" and index is not None else kind
        precision: Precision = (
            "float32" if array.dtype == torch.float32 else "float64" if array.dtype == torch.float64 else default_precision(device)
        )
        return Backend("torch", device, precision)
    raise TypeError(f"cannot infer a backend from {type(array).__name__}: expected a numpy array or a torch tensor")
