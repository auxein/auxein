"""The numeric backend (design doc §7): array namespace, device and precision."""

from auxein.backend.backend import Backend, BackendError, BackendName, Precision, backend_of, default_precision, is_array
from auxein.backend.types import Array, ArrayNamespace, HostArray

__all__ = [
    "Array",
    "ArrayNamespace",
    "Backend",
    "BackendError",
    "BackendName",
    "HostArray",
    "Precision",
    "backend_of",
    "default_precision",
    "is_array",
]
