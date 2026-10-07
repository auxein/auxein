"""The numeric backend (design doc §7): array namespace, device and precision."""

from auxein.backend.backend import Backend, BackendError, BackendName, Precision, backend_of, default_precision
from auxein.backend.types import Array, ArrayNamespace

__all__ = [
    "Array",
    "ArrayNamespace",
    "Backend",
    "BackendError",
    "BackendName",
    "Precision",
    "backend_of",
    "default_precision",
]
