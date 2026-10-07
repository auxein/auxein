"""Type aliases for arrays.

numpy arrays and torch tensors have no common static type, and the array API standard is a runtime contract rather
than a class hierarchy. Both aliases are therefore `Any`. This is the one place in the new core where `Any` is
unavoidable; everything else is typed in terms of these aliases, which keeps the exception visible and in one place.
"""

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

Array: TypeAlias = Any
"""An array of a supported backend: a `numpy.ndarray` or a `torch.Tensor`."""

ArrayNamespace: TypeAlias = Any
"""An array API namespace, as returned by `array_api_compat` (e.g. `array_api_compat.numpy`)."""

HostArray: TypeAlias = npt.NDArray[Any]
"""A numpy array in host memory. `NDArray[Any]` is numpy's own idiom for "an ndarray of some dtype"."""

__all__ = ["Array", "ArrayNamespace", "HostArray", "np"]
