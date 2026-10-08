"""Backend-native random streams (design doc §8)."""

import base64
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from auxein.backend import Array, Backend, devices

Shape = int | Sequence[int]
Scalar = float | Array  # a float, or an array broadcastable to the shape being drawn

StreamState = dict[str, Any]
# the state of a stream is JSON data (numpy: a dict of ints and strings, torch: a base64 string); `Any` is the honest
# value type of a JSON document.


def _restore(seed_sequence: np.random.SeedSequence, backend: Backend, state: StreamState) -> "RandomStream":
    stream = RandomStream(seed_sequence, backend)
    stream.load_state_dict(state)
    return stream


def _shape(shape: Shape) -> tuple[int, ...]:
    result = (shape,) if isinstance(shape, int) else tuple(shape)
    if any(d < 0 for d in result):
        raise ValueError(f"shape must not have negative dimensions, got {result}")
    return result


class RandomStream:
    """A random generator that lives where the arrays live: numpy's `Generator(PCG64)` or a `torch.Generator`.

    Streams are never global. Each is seeded from its own `numpy.random.SeedSequence` (see `RunSeed`), so that
    independent streams can be derived from one run seed. Every method returns arrays in the backend's namespace and
    device, with its float dtype (integer and boolean results use the backend's integer dtype).

    Reproducibility holds for the same seed, backend and precision. numpy and torch streams produce different numbers.

    A caveat of torch on the CPU: its generator is an MT19937 that accepts only 32 bits of seed, so two seed
    sequences can end up with the same torch stream. The chance is negligible for the handful of streams a run
    needs, but it becomes likely around 65,000 torch streams (e.g. one per candidate evaluation). numpy streams use the
    full 128 bits of entropy. Generators on CUDA and Metal use the full 64-bit seed.
    """

    def __init__(self, seed_sequence: np.random.SeedSequence, backend: Backend | None = None) -> None:
        self.backend = Backend() if backend is None else backend
        self.seed_sequence = seed_sequence
        self._numpy: np.random.Generator | None = None
        self._torch: Any = None
        if self.backend.name == "numpy":
            self._numpy = np.random.Generator(np.random.PCG64(seed_sequence))
        else:
            self._torch = devices.import_torch().Generator(device=self.backend.device)
            self._torch.manual_seed(int(seed_sequence.generate_state(1, dtype=np.uint64)[0]))

    def __reduce__(self) -> tuple[Callable[..., "RandomStream"], tuple[np.random.SeedSequence, Backend, StreamState]]:
        """Pickle through the state, so that a stream (e.g. a candidate's) can be sent to a worker process and continue there."""
        return (_restore, (self.seed_sequence, self.backend, self.state_dict()))

    @property
    def _generator(self) -> np.random.Generator:
        assert self._numpy is not None
        return self._numpy

    def _scalar(self, value: Scalar) -> Scalar:
        return value if isinstance(value, float | int) else self.backend.asarray(value)

    def uniform(self, shape: Shape, low: Scalar = 0.0, high: Scalar = 1.0) -> Array:
        """Uniform draws in [low, high). Rounding can make a float32 draw equal to `high`; use `Box` for hard bounds."""
        shape = _shape(shape)
        if isinstance(low, float | int) and isinstance(high, float | int) and not low <= high:
            raise ValueError(f"low must not exceed high, got low={low}, high={high}")
        if self._numpy is not None:
            unit = self._numpy.random(shape, dtype=np.float32 if self.backend.precision == "float32" else np.float64)
        else:
            unit = devices.import_torch().rand(shape, generator=self._torch, device=self.backend.device, dtype=self.backend.dtype)
        low_, high_ = self._scalar(low), self._scalar(high)
        return low_ + (high_ - low_) * unit

    def normal(self, shape: Shape, mean: Scalar = 0.0, std: Scalar = 1.0) -> Array:
        """Normal draws with the given mean and standard deviation."""
        shape = _shape(shape)
        if isinstance(std, float | int) and std < 0:
            raise ValueError(f"std must not be negative, got {std}")
        if self._numpy is not None:
            standard = self._numpy.standard_normal(shape, dtype=np.float32 if self.backend.precision == "float32" else np.float64)
        else:
            standard = devices.import_torch().randn(shape, generator=self._torch, device=self.backend.device, dtype=self.backend.dtype)
        return self._scalar(mean) + self._scalar(std) * standard

    def integers(self, low: int, high: int, shape: Shape = ()) -> Array:
        """Uniform integers in [low, high), as the backend's integer dtype (int64)."""
        shape = _shape(shape)
        if not low < high:
            raise ValueError(f"low must be below high, got low={low}, high={high}")
        if self._numpy is not None:
            return np.asarray(self._numpy.integers(low, high, size=shape, dtype=np.int64))
        return devices.import_torch().randint(
            low, high, shape, generator=self._torch, device=self.backend.device, dtype=self.backend.int_dtype
        )

    def permutation(self, n: int) -> Array:
        """A random permutation of 0 .. n-1."""
        if n < 0:
            raise ValueError(f"n must not be negative, got {n}")
        if self._numpy is not None:
            generator: Any = self._numpy  # numpy's overloads for permutation and choice are only partly typed
            return np.asarray(generator.permutation(n), dtype=np.int64)
        return devices.import_torch().randperm(n, generator=self._torch, device=self.backend.device, dtype=self.backend.int_dtype)

    def choice(self, n: int, size: int, p: Array | Sequence[float] | None = None, replace: bool = True) -> Array:
        """`size` indices drawn from 0 .. n-1, with probabilities `p` (normalised, uniform by default).

        With `replace=False` the indices are distinct. On torch, a non-uniform `p` is limited to 2**24 categories.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if size < 0:
            raise ValueError(f"size must not be negative, got {size}")
        if not replace and size > n:
            raise ValueError(f"cannot draw {size} distinct indices out of {n}")

        weights: npt.NDArray[np.float64] | None = None
        if p is not None:
            raw: npt.NDArray[np.float64] = np.asarray(self.backend.to_numpy(p), dtype=np.float64)
            if raw.shape != (n,):
                raise ValueError(f"p must have shape ({n},), got {raw.shape}")
            if not bool(np.isfinite(raw).all()) or bool((raw < 0).any()) or not float(raw.sum()) > 0:
                raise ValueError("p must be finite, non-negative and have a positive sum")
            positive = int((raw > 0).sum())
            if not replace and positive < size:
                raise ValueError(f"cannot draw {size} distinct indices from {positive} with non-zero probability")
            weights = raw / raw.sum()

        if self._numpy is not None:
            generator: Any = self._numpy
            return np.asarray(generator.choice(n, size=size, p=weights, replace=replace), dtype=np.int64)

        torch = devices.import_torch()
        device, int_dtype = self.backend.device, self.backend.int_dtype
        if weights is not None:
            return torch.multinomial(self.backend.asarray(weights), size, replacement=replace, generator=self._torch).to(dtype=int_dtype)
        if replace:
            return torch.randint(0, n, (size,), generator=self._torch, device=device, dtype=int_dtype)
        return torch.randperm(n, generator=self._torch, device=device, dtype=int_dtype)[:size]

    def state_dict(self) -> StreamState:
        """The generator state as JSON-serialisable data, to checkpoint a run.

        numpy: the bit generator's state dict. torch: the bytes of `Generator.get_state()`, base64-encoded.
        """
        if self._numpy is not None:
            return {"backend": "numpy", "bit_generator": self._numpy.bit_generator.state}
        raw = bytes(self._torch.get_state().numpy().tobytes())
        return {"backend": "torch", "state": base64.b64encode(raw).decode("ascii")}

    def load_state_dict(self, state: StreamState) -> None:
        """Restore a state saved by `state_dict`: the stream then continues exactly where it was saved."""
        if state.get("backend") != self.backend.name:
            raise ValueError(f"cannot load a {state.get('backend')!r} state into a {self.backend.name!r} stream")
        if self._numpy is not None:
            self._numpy.bit_generator.state = state["bit_generator"]
        else:
            torch = devices.import_torch()
            self._torch.set_state(torch.frombuffer(bytearray(base64.b64decode(state["state"])), dtype=torch.uint8))
