"""A multilayer-perceptron policy decoder (design doc §6.1–6.2): genome to feed-forward network, one genome per team.

Neuroevolution scenarios all need the same thing: a genome that is the weights of a small network, decoded per episode for a
simulator that owns its loop, or for a whole population at once for a simulator that steps every candidate, scenario and agent
together on a backend. `MLPPolicy` is that decoder, so that no scenario reinvents it. It is built from sizes, not from an
environment, and knows nothing about the task: only `MLPPolicy.for_spaces` reads Gymnasium spaces, and imports Gymnasium only
then.

**Genome layout** (documented and stable: recordings and checkpoints depend on it). Layer by layer, from the input side, each
layer's weight matrix (`out × in`, row-major, so the weights of one output neuron are contiguous) followed by its bias (`out`).
A network with `observation_size = 3`, `hidden = (4,)` and `outputs = 2` has the genome

    W1 (4×3 = 12 values) | b1 (4) | W2 (2×4 = 8) | b2 (2)         dim = 26

`hidden = ()` is a linear policy (a single `out × in` layer), laid out like `LinearPolicy`'s genome.

**Parameter sharing.** The batched path is built for teams of identical agents: one genome is one network, shared by every agent
of the team, and each agent feeds it its own local observation. Across candidates the networks differ.

    batch = policy.decode_batch(genomes)                 # genomes: (n, dim), on the backend and device of the run
    actions = batch.act_batch(observations)              # observations: (n, scenarios, agents, observation_size)
    # actions: (n, scenarios, agents, outputs), every agent of candidate i acting with candidate i's network

**Backends.** `decode_batch` and `act_batch` work in the namespace, device and dtype of the genome array (the layers are views of
it, and the products are batched matrix products), so nothing goes to the host and there are no Python loops over candidates,
agents or scenarios. `decode` builds the agent of the per-episode path on the host in float64 numpy: it pickles into worker
processes whatever the run's backend, and it computes through the same code as the batched path. On numpy float64 the two give
the same numbers: bit for bit when a candidate's observations reach the matrix product in the same grouping (BLAS may sum a
different number of rows in a different order), and otherwise within a few units in the last place. On float32 or torch they agree
to float32 rounding.
"""

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, cast, runtime_checkable

import numpy as np

from auxein.backend import Array, ArrayNamespace, Backend, backend_of
from auxein.spaces import Box

Activation: TypeAlias = Literal["tanh", "relu", "identity"]
"""The nonlinearity of the hidden layers."""

_ACTIVATIONS = ("tanh", "relu", "identity")
_PLAIN_OUTPUTS = ("linear", "tanh", "argmax")

Bound: TypeAlias = "float | Sequence[float]"


@dataclass(frozen=True)
class Clip:
    """Output mode: the raw outputs, clipped to `[low, high]` (a number for all outputs, or one per output).

    Clipping keeps the network's own scale (an output beyond the bound is simply saturated), unlike `Tanh`, which squashes. A
    bound may be infinite on one side.
    """

    low: Bound
    high: Bound


@dataclass(frozen=True)
class Tanh:
    """Output mode: `tanh` rescaled to `[low, high]`, per output. `Tanh()` is the plain `"tanh"` mode, outputs in [-1, 1].

    The bounds must be finite: a smooth squashing needs a range to squash into (use `Clip` for a one-sided bound).
    """

    low: Bound = -1.0
    high: Bound = 1.0


OutputMode: TypeAlias = "Literal['linear', 'tanh', 'argmax'] | Clip | Tanh"
"""`"linear"`: raw outputs. `"tanh"`: in [-1, 1]. `Clip(low, high)` / `Tanh(low, high)`: per-output bounds. `"argmax"`: the
index of the largest output, for discrete actions."""


@runtime_checkable
class BatchPolicy(Protocol):
    """What a batched environment needs of the agents it is given: `act_batch`, so it does not depend on `MLPPolicy` itself.

    `observations` has shape `(n, ..., observation_size)`: the candidate first, then any batch dimensions (scenarios, agents).
    The result has the same leading shape, with the last dimension replaced by the actions.
    """

    def act_batch(self, observations: Array) -> Array: ...


@dataclass(frozen=True)
class _Output:
    """A validated output mode: `kind` is `linear`, `tanh`, `argmax`, `clip` or `scaled` (tanh into bounds)."""

    kind: str
    low: tuple[float, ...] | None = None
    high: tuple[float, ...] | None = None

    def describe(self) -> str:
        if self.low is None or self.high is None:
            return repr(self.kind)
        return f"{'Clip' if self.kind == 'clip' else 'Tanh'}(low={list(self.low)}, high={list(self.high)})"


def _per_output(value: Bound, outputs: int, name: str) -> tuple[float, ...]:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim > 1 or (values.ndim == 1 and values.shape[0] != outputs):
        raise ValueError(f"{name} must be a number or one value per output ({outputs}), got shape {values.shape}")
    if np.isnan(values).any():
        raise ValueError(f"{name} contains NaN")
    if values.ndim == 0:
        return (float(values),) * outputs
    return tuple(float(v) for v in values.tolist())


def _resolve_output(mode: OutputMode, outputs: int) -> _Output:
    if isinstance(mode, str):
        if mode not in _PLAIN_OUTPUTS:
            raise ValueError(f"unknown output mode {mode!r}: use one of {list(_PLAIN_OUTPUTS)}, Clip(low, high) or Tanh(low, high)")
        return _Output(mode)
    low, high = _per_output(mode.low, outputs, "low"), _per_output(mode.high, outputs, "high")
    if any(lo > hi for lo, hi in zip(low, high, strict=True)):
        raise ValueError(f"every low bound must be at most its high bound, got low={list(low)}, high={list(high)}")
    if isinstance(mode, Tanh):
        if not all(math.isfinite(v) for v in (*low, *high)):
            raise ValueError("Tanh needs finite bounds: use Clip(low, high) for a bound that is infinite on one side")
        return _Output("scaled", low, high)
    return _Output("clip", low, high)


def _activate(xp: ArrayNamespace, activation: str, values: Array) -> Array:
    if activation == "tanh":
        return xp.tanh(values)
    if activation == "relu":
        return xp.where(values > 0, values, xp.zeros_like(values))
    return values


def _bound(backend: Backend, values: tuple[float, ...]) -> Array:
    return backend.asarray(np.asarray(values, dtype=np.float64))


class MLPAgentBatch:
    """The networks of a whole batch of genomes, ready to act: parameter sharing within a candidate (see the module docstring).

    `weights[k]` is the `(n, out, in)` weight array of layer `k` and `biases[k]` its `(n, out)` bias: views of the genome array,
    on its backend and device. Implements `BatchPolicy`.
    """

    def __init__(
        self,
        weights: Sequence[Array],
        biases: Sequence[Array],
        observation_size: int,
        activation: str,
        output: _Output,
    ) -> None:
        self.weights = tuple(weights)
        self.biases = tuple(biases)
        self.observation_size = observation_size
        self.outputs = int(self.weights[-1].shape[1])
        self.n = int(self.weights[0].shape[0])
        self._activation = activation
        self._output = output
        self._backend = backend_of(self.weights[0])

    def __repr__(self) -> str:
        return f"MLPAgentBatch(n={self.n}, layers={len(self.weights)}, backend={self._backend})"

    def act_batch(self, observations: Array) -> Array:
        """Each candidate's network applied to all of its observations.

        `observations` is `(n, ..., observation_size)` (at least `(n, observation_size)`, one per candidate) and is converted to
        the genomes' backend, device and dtype; the result is `(n, ..., outputs)`, or `(n, ...)` integer indices in `argmax`
        mode.
        """
        backend, xp = self._backend, self._backend.xp
        observations = backend.asarray(observations)
        shape = tuple(int(s) for s in observations.shape)
        if len(shape) < 2 or shape[0] != self.n or shape[-1] != self.observation_size:
            raise ValueError(
                f"observations must have shape (n, ..., observation_size) with n={self.n} candidates and a last dimension of "
                f"{self.observation_size}, got {shape}"
            )
        middle = shape[1:-1]
        values = xp.reshape(observations, (self.n, math.prod(middle), self.observation_size))
        last = len(self.weights) - 1
        for k, (weights, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            values = xp.matmul(values, xp.matrix_transpose(weights)) + bias[:, None, :]
            if k < last:
                values = _activate(xp, self._activation, values)
        result = self._finish(values)
        return xp.reshape(result, (self.n, *middle) if self._output.kind == "argmax" else (self.n, *middle, self.outputs))

    def _finish(self, values: Array) -> Array:
        xp, output = self._backend.xp, self._output
        if output.kind == "argmax":
            return xp.argmax(values, axis=-1)
        if output.kind == "tanh":
            return xp.tanh(values)
        if output.low is None or output.high is None:
            return values
        low, high = _bound(self._backend, output.low), _bound(self._backend, output.high)
        if output.kind == "clip":
            return xp.minimum(xp.maximum(values, low), high)
        return low + (xp.tanh(values) + 1.0) * 0.5 * (high - low)


class MLPAgent:
    """One decoded network, for the per-episode path and `StepEnvironment`: `act(observation) -> action`.

    It holds float64 host arrays (it pickles into worker processes) and computes through the batched code with a batch of one.
    An observation `(observation_size,)` gives the action of one agent; a team's `(agents, observation_size)` gives one action
    per agent, with the same network. The actions are `(outputs,)` or `(agents, outputs)` arrays, or an `int` or an integer
    array in `argmax` mode.
    """

    def __init__(self, batch: MLPAgentBatch) -> None:
        self._batch = batch
        self.observation_size = batch.observation_size
        self.outputs = batch.outputs
        self.weights = [w[0] for w in batch.weights]
        self.biases = [b[0] for b in batch.biases]

    def __repr__(self) -> str:
        return f"MLPAgent(observation_size={self.observation_size}, outputs={self.outputs}, layers={len(self.weights)})"

    def act(self, observation: object) -> object:
        values = np.asarray(observation, dtype=np.float64)
        if values.ndim not in (1, 2) or values.shape[-1] != self.observation_size:
            raise ValueError(
                f"an observation must have shape ({self.observation_size},) or (agents, {self.observation_size}), got {tuple(values.shape)}"
            )
        action = np.asarray(self._batch.act_batch(values[None]))[0]
        return int(action) if action.ndim == 0 else action


class MLPPolicy:
    """A decoder from genomes to feed-forward networks (see the module docstring for the genome layout and parameter sharing).

    - `observation_size`: inputs. `hidden`: the widths of the hidden layers, possibly empty (a linear policy). `outputs`: the
      size of the output layer, which for `argmax` is the number of actions.
    - `activation`: the hidden layers' nonlinearity, `"tanh"` (default), `"relu"` or `"identity"`.
    - `output`: `"linear"` (raw), `"tanh"` (in [-1, 1], the usual choice for forces and velocities), `Clip(low, high)` or
      `Tanh(low, high)` (per-output bounds, clipped or squashed into the range) or `"argmax"` (a discrete action index).

    Sizes must be positive. `dim` is the genome length; `space(scale)` the box of genomes; `explain(genome)` the layers by name.
    """

    def __init__(
        self,
        observation_size: int,
        hidden: Sequence[int] = (),
        outputs: int = 1,
        *,
        activation: Activation = "tanh",
        output: OutputMode = "linear",
    ) -> None:
        widths = tuple(int(w) for w in hidden)
        if observation_size < 1 or outputs < 1:
            raise ValueError(f"observation_size and outputs must be positive, got {observation_size} and {outputs}")
        if any(w < 1 for w in widths):
            raise ValueError(f"the hidden layer widths must be positive, got {list(widths)}")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}: use one of {list(_ACTIVATIONS)}")
        self.observation_size = int(observation_size)
        self.hidden = widths
        self.outputs = int(outputs)
        self.activation: Activation = activation
        self.output = output
        self._output = _resolve_output(output, self.outputs)
        sizes = (self.observation_size, *widths, self.outputs)
        self.layers: tuple[tuple[int, int], ...] = tuple(zip(sizes[:-1], sizes[1:], strict=True))
        """`(in, out)` of every layer, from the input side."""
        self.dim = sum(out * (size + 1) for size, out in self.layers)

    @classmethod
    def for_spaces(
        cls,
        observation_space: object,
        action_space: object,
        hidden: Sequence[int] = (),
        *,
        activation: Activation = "tanh",
        continuous: Literal["tanh", "clip"] = "tanh",
    ) -> "MLPPolicy":
        """A policy for Gymnasium spaces, mirroring `LinearPolicy`: a `Box` observation (flattened), a `Discrete` or `Box` action.

        A `Discrete(n)` action space gets `n` outputs and `argmax`. A `Box` action space gets one output per component and, by
        default, `Tanh` rescaled to the action bounds (`continuous="tanh"`): smooth, and always in range, which a random network
        needs (a clipped one starts mostly saturated at a bound). The bounds must then be finite; `continuous="clip"` clips the
        raw outputs instead, and accepts infinite bounds. Gymnasium is imported here, not before; the extra is named if missing.
        """
        spaces: Any = _import_gymnasium().spaces  # Gymnasium is optional and untyped for strict checking
        observation: Any = observation_space
        action: Any = action_space
        if not isinstance(observation, spaces.Box):
            raise TypeError(f"MLPPolicy needs a Box observation space (flattened), got {type(observation).__name__}")
        size = int(np.prod(tuple(cast("Sequence[int]", observation.shape))))
        if isinstance(action, spaces.Discrete):
            return cls(size, hidden, int(cast("int", action.n)), activation=activation, output="argmax")
        if isinstance(action, spaces.Box):
            if continuous not in ("tanh", "clip"):
                raise ValueError(f"continuous must be 'tanh' or 'clip', got {continuous!r}")
            outputs = int(np.prod(tuple(cast("Sequence[int]", action.shape))))
            low = np.asarray(action.low, dtype=np.float64).reshape(-1).tolist()
            high = np.asarray(action.high, dtype=np.float64).reshape(-1).tolist()
            mode: OutputMode = Tanh(low, high) if continuous == "tanh" else Clip(low, high)
            return cls(size, hidden, outputs, activation=activation, output=mode)
        raise TypeError(f"MLPPolicy needs a Discrete or a Box action space, got {type(action).__name__}")

    def __repr__(self) -> str:
        return (
            f"MLPPolicy(observation_size={self.observation_size}, hidden={self.hidden}, outputs={self.outputs}, "
            f"activation={self.activation!r}, output={self._output.describe()})"
        )

    def space(self, scale: float = 1.0) -> Box:
        """The box of genomes, with bounds scaled by fan-in: weights in `±scale/√fan_in`, biases in `±scale`.

        Uniform sampling in this box is a Glorot/He-like initialisation: for inputs of unit variance the pre-activations of a
        random network stay of order one, so it neither saturates a `tanh` nor vanishes layer after layer. The bounds also
        **limit the search** (repair clips genomes to them), and a weight beyond `scale/√fan_in` can never be reached, so `scale`
        is the lever: raise it to allow sharper or larger-gain networks at the price of a larger, wilder space and a more
        saturated random start. The GA's mutation steps are relative to the box width, so each layer's mutations scale with its
        own weights.
        """
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"scale must be positive and finite, got {scale}")
        bounds: list[float] = []
        for size, out in self.layers:
            bounds += [scale / math.sqrt(size)] * (out * size)
            bounds += [float(scale)] * out
        upper = np.asarray(bounds, dtype=np.float64)
        return Box(-upper, upper)

    def decode(self, genome: Array) -> MLPAgent:
        """One genome to an agent (float64, on the host)."""
        values = np.asarray(Backend().to_numpy(genome), dtype=np.float64)
        if values.shape != (self.dim,):
            raise ValueError(f"a genome of this policy has {self.dim} values, got shape {tuple(values.shape)}")
        return MLPAgent(self._batch(values[None]))

    def decode_batch(self, genomes: Array) -> MLPAgentBatch:
        """An `(n, dim)` genome array to the networks of the batch, on the array's backend and device, with no host round trip."""
        shape = tuple(int(s) for s in genomes.shape)
        if len(shape) != 2 or shape[1] != self.dim:
            raise ValueError(f"genomes of this policy have {self.dim} values each: expected shape (n, {self.dim}), got {shape}")
        return self._batch(genomes)

    def _batch(self, genomes: Array) -> MLPAgentBatch:
        xp, n = backend_of(genomes).xp, int(genomes.shape[0])
        weights: list[Array] = []
        biases: list[Array] = []
        start = 0
        for size, out in self.layers:
            weights.append(xp.reshape(genomes[:, start : start + out * size], (n, out, size)))
            start += out * size
            biases.append(genomes[:, start : start + out])
            start += out
        return MLPAgentBatch(weights, biases, self.observation_size, self.activation, self._output)

    def explain(self, genome: Array) -> Mapping[str, object]:
        """The layers of a genome by name, for reading: `hidden1`, `hidden2`, ... and `output`, each with `weights` and `bias`."""
        agent = self.decode(genome)
        names = [*(f"hidden{k + 1}" for k in range(len(self.hidden))), "output"]
        return {
            name: {"weights": weights.tolist(), "bias": bias.tolist()}
            for name, weights, bias in zip(names, agent.weights, agent.biases, strict=True)
        }


def _import_gymnasium() -> Any:
    try:
        return importlib.import_module("gymnasium")
    except ImportError as error:
        raise ImportError("MLPPolicy.for_spaces needs the gymnasium package: install it with `pip install auxein[gymnasium]`") from error


__all__ = ["Activation", "BatchPolicy", "Clip", "MLPAgent", "MLPAgentBatch", "MLPPolicy", "OutputMode", "Tanh"]
