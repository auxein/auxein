"""Type-aware variation for mixed spaces (design doc §3.3): integer, binary and categorical operators, and the glue that
applies the right one to each column of a `MixedSpace` genome.

The integer operator follows mixed-integer evolution strategies (Li et al., MIES): an integer gene moves by the *difference
of two geometric random variables*, which is symmetric around zero, has integer steps of any size and is zero often when the
mean step is small. The mean step is a strategy parameter adapted per individual, like a real step size, and has a lower
bound, so an integer gene can always still change. Binary genes flip and categorical genes jump to a *different* category,
each with a probability that defaults to one over the number of genes of that kind, and discrete genes are recombined by
taking each from one of the parents. Everything is vectorised over the population and goes through the array namespace.

Real columns use the unchanged real operators of the numeric GA, on the sub-array of the real columns.
"""

import math
from dataclasses import dataclass

from auxein.backend import Array, Backend
from auxein.random import RandomStream
from auxein.spaces import Box
from auxein.spaces.mixed import BINARY, CATEGORICAL, INTEGER, REAL, MixedSpace
from auxein.strategies.ga.base import BoundsRepair, Mutation, Recombination


def _check_probability(probability: float | None) -> None:
    if probability is not None and not 0.0 < probability <= 1.0:
        raise ValueError(f"a mutation probability must be in (0, 1], got {probability}")


class IntegerMutation:
    """Adds the difference of two geometric random variables to an integer gene, with a probability per gene.

    Each geometric variable counts failures before the first success and has mean `m`, the *mean step*; their difference is
    symmetric, zero with probability `1/(1+2m)` and otherwise of either sign. With `adaptive=True` (the default) every
    individual carries its own `m`, strategy state like a real step size: it mutates first (`m' = m·exp(τ·N(0, 1))`,
    `τ = 1/√k` for `k` integer genes) and is kept in `[min_step, range]`. The lower bound is what keeps an integer gene
    from ever freezing: at `min_step` a mutated gene still moves with probability `2·min_step/(1 + 2·min_step)`.
    The result is clipped to the bounds. `probability` defaults to `1/k`, the usual MIES rate.
    """

    name = "geometric"

    def __init__(self, probability: float | None = None, initial_step: float = 1.0, min_step: float = 0.1, adaptive: bool = True) -> None:
        _check_probability(probability)
        if not 0 < min_step <= initial_step:
            raise ValueError(f"min_step must be positive and at most initial_step, got min_step={min_step}, initial_step={initial_step}")
        self.probability = probability
        self.initial_step = initial_step
        self.min_step = min_step
        self.adaptive = adaptive

    def __repr__(self) -> str:
        return (
            f"IntegerMutation(probability={self.probability}, initial_step={self.initial_step}, "
            f"min_step={self.min_step}, adaptive={self.adaptive})"
        )

    def initial_steps(self, count: int, backend: Backend) -> Array | None:
        """The mean steps of `count` new individuals, shape `(count,)`, or None when the step is fixed."""
        if not self.adaptive:
            return None
        return backend.xp.full((count,), self.initial_step, dtype=backend.dtype, device=backend.device)

    def mutate(
        self, values: Array, steps: Array | None, lower: Array, upper: Array, rng: RandomStream, backend: Backend
    ) -> tuple[Array, Array | None]:
        """`values` is `(n, k)`, the integer columns; `lower` and `upper` are their bounds `(k,)`. Returns the mutated values and
        the new mean steps `(n,)` (None when not adaptive)."""
        xp = backend.xp
        count, k = int(values.shape[0]), int(values.shape[1])
        if self.adaptive:
            assert steps is not None, "an adaptive integer mutation needs the mean steps of the individuals"
            cap = xp.maximum(xp.max(upper - lower), xp.asarray(self.min_step, dtype=backend.dtype, device=backend.device))
            grown = steps * xp.exp((1.0 / math.sqrt(k)) * rng.normal((count,)))
            new_steps: Array | None = xp.minimum(xp.clip(grown, self.min_step, None), cap)
            assert new_steps is not None
            mean = new_steps[:, None]
        else:
            new_steps = None
            mean = xp.asarray(self.initial_step, dtype=backend.dtype, device=backend.device)
        # a geometric variable with mean m (failures before a success of probability 1/(1+m)): floor(log(1-u) / log(m/(1+m)))
        log_ratio = xp.log(mean) - xp.log1p(mean)
        first = xp.floor(xp.log1p(-rng.uniform((count, k))) / log_ratio)
        second = xp.floor(xp.log1p(-rng.uniform((count, k))) / log_ratio)
        probability = 1.0 / k if self.probability is None else self.probability
        mutated = rng.uniform((count, k)) < probability
        moved = xp.minimum(xp.maximum(values + first - second, lower), upper)
        return xp.where(mutated, moved, values), new_steps


def _default_rate(count: int) -> float:
    """`1/k` for `k` genes of a kind, but never above one half: with a single binary or categorical gene a rate of 1 would
    change it in *every* child, so no child could ever inherit the parent's good value and the gene would oscillate."""
    return 1.0 / max(count, 2)


class BitFlipMutation:
    """Flips each binary gene with a probability, `1/k` for `k` binary genes by default (at most one half)."""

    name = "bitflip"

    def __init__(self, probability: float | None = None) -> None:
        _check_probability(probability)
        self.probability = probability

    def __repr__(self) -> str:
        return f"BitFlipMutation(probability={self.probability})"

    def mutate(self, values: Array, lower: Array, upper: Array, rng: RandomStream, backend: Backend) -> Array:
        probability = _default_rate(int(values.shape[1])) if self.probability is None else self.probability
        flipped = rng.uniform(tuple(values.shape)) < probability
        return backend.xp.where(flipped, 1.0 - values, values)


class CategoricalMutation:
    """Replaces the index of a categorical gene with a *different* category drawn uniformly, with a probability that is `1/k`
    for `k` categorical genes by default (at most one half). A gene never "mutates" to the category it already has."""

    name = "resample"

    def __init__(self, probability: float | None = None) -> None:
        _check_probability(probability)
        self.probability = probability

    def __repr__(self) -> str:
        return f"CategoricalMutation(probability={self.probability})"

    def mutate(self, values: Array, lower: Array, upper: Array, rng: RandomStream, backend: Backend) -> Array:
        xp = backend.xp
        probability = _default_rate(int(values.shape[1])) if self.probability is None else self.probability
        mutated = rng.uniform(tuple(values.shape)) < probability
        # a draw among the other `choices - 1` categories: skip the current one by shifting the draws that reach it
        others = upper  # the last index, which is `choices - 1`
        draw = xp.minimum(xp.floor(rng.uniform(tuple(values.shape)) * others), others - 1.0)
        different = draw + xp.astype(draw >= values, backend.dtype)
        return xp.where(mutated, different, values)


@dataclass(frozen=True)
class _Kind:
    """The columns of one kind of dimension, as arrays on the backend, and their bounds."""

    columns: Array
    count: int
    lower: Array
    upper: Array


class MixedVariation:
    """Recombination, mutation and repair of the genomes of a `MixedSpace`, column by column according to its types.

    It is built by `GeneticAlgorithm` when it is bound to a mixed space, from the operators the user gave (or the defaults)
    for each type. The *strategy parameters* of an individual are packed in one array `(n, w)`, so that the GA can keep,
    inherit and checkpoint them like the step sizes of a `Box` run: first the real step sizes (none, one, or one per real
    gene, as the real mutation says), then the integer mean step (when adaptive).
    """

    def __init__(
        self,
        space: MixedSpace,
        backend: Backend,
        *,
        real_mutation: Mutation,
        real_recombination: Recombination,
        integer_mutation: IntegerMutation,
        binary_mutation: BitFlipMutation,
        categorical_mutation: CategoricalMutation,
        repair: BoundsRepair,
    ) -> None:
        self.backend = backend
        self.space = space
        self.real_mutation, self.real_recombination = real_mutation, real_recombination
        self.integer_mutation, self.binary_mutation, self.categorical_mutation = integer_mutation, binary_mutation, categorical_mutation
        self.repair_operator = repair
        self._kinds: dict[int, _Kind] = {}
        order: list[int] = []
        low, high = space.lower, space.upper
        for kind in (REAL, INTEGER, BINARY, CATEGORICAL):
            columns = space.indices(kind)
            if columns:
                order.extend(columns)
                self._kinds[kind] = _Kind(
                    backend.asarray(list(columns), dtype=backend.int_dtype),
                    len(columns),
                    backend.asarray(low[list(columns)]),
                    backend.asarray(high[list(columns)]),
                )
        self._identity = order == list(range(space.dim))
        self._inverse = backend.asarray(
            sorted(range(len(order)), key=order.__getitem__), dtype=backend.int_dtype
        )  # the inverse permutation
        self._real_box: Box | None = None
        self._real_width: Array | None = None
        if REAL in self._kinds:
            columns = list(space.indices(REAL))
            self._real_box = Box(low[columns], high[columns], log_scale=space.log_scale[columns].tolist())
            self._real_width = backend.asarray(high[columns] - low[columns])
        self._real_steps = 0  # the width of the real step sizes in the packed array: 0 (none), 1 (per individual) or the real genes
        if REAL in self._kinds:
            sample = real_mutation.initial_steps(0, self._kinds[REAL].count, backend)
            self._real_steps = 0 if sample is None else (1 if sample.ndim == 1 else int(sample.shape[1]))
        self._integer_adaptive = INTEGER in self._kinds and integer_mutation.adaptive
        self.adaptive = self._real_steps > 0 or self._integer_adaptive

    # --- names, for the origins ---

    @property
    def recombination_name(self) -> str:
        names = ([self.real_recombination.name] if REAL in self._kinds else []) + (
            ["discrete"] if len(self._kinds) > (REAL in self._kinds) else []
        )
        return "/".join(names)

    @property
    def mutation_name(self) -> str:
        names: list[str] = []
        if REAL in self._kinds:
            names.append(self.real_mutation.name)
        for kind, operator in ((INTEGER, self.integer_mutation), (BINARY, self.binary_mutation), (CATEGORICAL, self.categorical_mutation)):
            if kind in self._kinds:
                names.append(operator.name)
        return "/".join(names)

    # --- assembling and splitting genomes ---

    def _take(self, genomes: Array, kind: int) -> Array:
        return self.backend.xp.take(genomes, self._kinds[kind].columns, axis=1)

    def _assemble(self, parts: dict[int, Array]) -> Array:
        xp = self.backend.xp
        ordered = [parts[kind] for kind in (REAL, INTEGER, BINARY, CATEGORICAL) if kind in parts]
        joined = ordered[0] if len(ordered) == 1 else xp.concat(ordered, axis=1)
        return joined if self._identity else xp.take(joined, self._inverse, axis=1)

    # --- strategy parameters ---

    def initial_steps(self, count: int) -> Array | None:
        """The packed strategy parameters of `count` new individuals, or None when nothing is adapted."""
        if not self.adaptive:
            return None
        backend = self.backend
        parts: list[Array] = []
        if self._real_steps:
            real = self.real_mutation.initial_steps(count, self._kinds[REAL].count, backend)
            assert real is not None
            parts.append(real[:, None] if real.ndim == 1 else real)
        if self._integer_adaptive:
            integer = self.integer_mutation.initial_steps(count, backend)
            assert integer is not None
            parts.append(integer[:, None])
        return parts[0] if len(parts) == 1 else backend.xp.concat(parts, axis=1)

    def floors(self) -> Array | None:
        """The lower bound of each packed parameter, to tell whether every step has reached its floor."""
        if not self.adaptive:
            return None
        floor: list[float] = []
        if self._real_steps:
            minimum = self.real_mutation.min_step
            floor += [float("-inf") if minimum is None else minimum] * self._real_steps
        if self._integer_adaptive:
            floor.append(self.integer_mutation.min_step)
        return self.backend.asarray(floor)

    # --- recombination ---

    def weights(self, count: int, rng: RandomStream) -> Array:
        """The weight of the first parent in each gene, `(count, d)` (or `(count, 1)` without recombination): the real
        recombination's on the real columns, and a fair coin (0 or 1) on the discrete ones, so a discrete gene always
        comes from exactly one parent."""
        backend = self.backend
        xp = backend.xp
        if not self.real_recombination.sexual:
            return self.real_recombination.weights(count, self.space.dim, rng, backend)
        parts: dict[int, Array] = {}
        if REAL in self._kinds:
            n_real = self._kinds[REAL].count
            weights = self.real_recombination.weights(count, n_real, rng, backend)
            parts[REAL] = (
                weights if weights.shape[1] == n_real else weights * xp.ones((1, n_real), dtype=backend.dtype, device=backend.device)
            )
        for kind in (INTEGER, BINARY, CATEGORICAL):
            if kind in self._kinds:
                parts[kind] = xp.astype(rng.uniform((count, self._kinds[kind].count)) < 0.5, backend.dtype)
        return self._assemble(parts)

    def mix_steps(self, weights: Array, first: Array, second: Array) -> Array:
        """The packed parameters of a child: the weighted geometric mean of the parents', with the weights of the genes they
        belong to (the mean over the real genes for a single real step, over the integer genes for the integer step)."""
        backend = self.backend
        xp = backend.xp
        if weights.shape[1] == 1:
            weights = weights * xp.ones((1, self.space.dim), dtype=backend.dtype, device=backend.device)
        parts: list[Array] = []
        if self._real_steps:
            real = self._take(weights, REAL)
            parts.append(xp.mean(real, axis=1, keepdims=True) if self._real_steps == 1 else real)
        if self._integer_adaptive:
            parts.append(xp.mean(self._take(weights, INTEGER), axis=1, keepdims=True))
        w = parts[0] if len(parts) == 1 else xp.concat(parts, axis=1)
        return xp.pow(first, w) * xp.pow(second, 1.0 - w)

    # --- mutation and repair ---

    def mutate(self, genomes: Array, steps: Array | None, rng: RandomStream) -> tuple[Array, Array | None]:
        """Mutate every column with the operator of its type. Returns the genomes and the new packed parameters."""
        backend = self.backend
        xp = backend.xp
        parts: dict[int, Array] = {}
        new_steps: list[Array] = []
        if REAL in self._kinds:
            kind = self._kinds[REAL]
            assert self._real_width is not None
            real_steps = None
            if self._real_steps:
                assert steps is not None
                real_steps = steps[:, 0] if self._real_steps == 1 else steps[:, : self._real_steps]
            mutated, stepped = self.real_mutation.mutate(self._take(genomes, REAL), real_steps, self._real_width, rng, backend)
            parts[REAL] = mutated
            if stepped is not None:
                new_steps.append(stepped[:, None] if stepped.ndim == 1 else stepped)
        if INTEGER in self._kinds:
            kind = self._kinds[INTEGER]
            integer_steps = None
            if self._integer_adaptive:
                assert steps is not None
                integer_steps = steps[:, -1]
            parts[INTEGER], stepped = self.integer_mutation.mutate(
                self._take(genomes, INTEGER), integer_steps, kind.lower, kind.upper, rng, backend
            )
            if stepped is not None:
                new_steps.append(stepped[:, None])
        if BINARY in self._kinds:
            kind = self._kinds[BINARY]
            parts[BINARY] = self.binary_mutation.mutate(self._take(genomes, BINARY), kind.lower, kind.upper, rng, backend)
        if CATEGORICAL in self._kinds:
            kind = self._kinds[CATEGORICAL]
            parts[CATEGORICAL] = self.categorical_mutation.mutate(self._take(genomes, CATEGORICAL), kind.lower, kind.upper, rng, backend)
        packed = None if not new_steps else (new_steps[0] if len(new_steps) == 1 else xp.concat(new_steps, axis=1))
        return self._assemble(parts), packed

    def repair(self, genomes: Array) -> Array:
        """Real genes go through the GA's bounds repair (on a box of the real columns); discrete genes are already valid, since
        every discrete operator keeps them integral and within their bounds."""
        if self._real_box is None:
            return genomes
        repaired = self.repair_operator.repair(self._take(genomes, REAL), self._real_box)
        if len(self._kinds) == 1:
            return self._assemble({REAL: repaired})
        parts = {kind: self._take(genomes, kind) for kind in self._kinds if kind != REAL}
        parts[REAL] = repaired
        return self._assemble(parts)
