"""Mutation: Gaussian with a fixed step, and self-adaptive with one step size per individual or per gene.

All steps are relative to the width of the box, per dimension, so the operators are scale-free. Mutation is done in
linear space, also on the dimensions of a `Box` that is log-scale.
"""

import math

from auxein.backend import Array, Backend
from auxein.random import RandomStream
from auxein.strategies.ga.base import Mutation


class GaussianMutation:
    """Adds `N(0, (step * width)^2)` to every gene, with a fixed `step` relative to the box width."""

    name = "gaussian"
    adaptive = False
    min_step = None

    def __init__(self, step: float = 0.1) -> None:
        if not step > 0:
            raise ValueError(f"the step must be positive, got {step}")
        self.step = step

    def __repr__(self) -> str:
        return f"GaussianMutation(step={self.step})"

    def initial_steps(self, count: int, dim: int, backend: Backend) -> None:
        return None

    def mutate(self, genomes: Array, steps: Array | None, width: Array, rng: RandomStream, backend: Backend) -> tuple[Array, None]:
        return genomes + self.step * width * rng.normal(tuple(genomes.shape)), None


class SelfAdaptiveMutation:
    """Self-adaptive Gaussian mutation: each individual carries its own step size(s), which mutate before the genes do.

    - One step size per individual (`per_gene=False`): `sigma' = sigma * exp(tau * N(0, 1))`, `tau = 1/sqrt(d)`.
    - One per gene (`per_gene=True`): `sigma_i' = sigma_i * exp(tau' * N(0, 1) + tau * N_i(0, 1))`, with the usual
      learning rates `tau' = 1/sqrt(2d)` (global) and `tau = 1/sqrt(2 sqrt(d))` (per gene).

    The genes then move by `sigma' * width * N(0, 1)`. Step sizes are relative to the box width and bounded below by
    `min_step`, so that they can't collapse to zero (the 0.x implementation had no bound). They are strategy state keyed
    by candidate id, not part of the genome (design doc §4.1).
    """

    adaptive = True

    def __init__(
        self,
        per_gene: bool = False,
        initial_step: float = 0.1,
        min_step: float = 1e-12,
        tau: float | None = None,
        tau_prime: float | None = None,
    ) -> None:
        if not initial_step > 0:
            raise ValueError(f"the initial step must be positive, got {initial_step}")
        if not 0 < min_step <= initial_step:
            raise ValueError(f"min_step must be positive and at most the initial step, got {min_step}")
        self.per_gene = per_gene
        self.initial_step = initial_step
        self.min_step = min_step
        self._tau, self._tau_prime = tau, tau_prime
        self.name = "self_adaptive_per_gene" if per_gene else "self_adaptive"

    def __repr__(self) -> str:
        return f"SelfAdaptiveMutation(per_gene={self.per_gene}, initial_step={self.initial_step}, min_step={self.min_step})"

    def tau(self, dim: int) -> float:
        """The learning rate of one step size per individual (`1/sqrt(d)`), or the per-gene rate (`1/sqrt(2 sqrt(d))`)."""
        if self._tau is not None:
            return self._tau
        return 1.0 / math.sqrt(2.0 * math.sqrt(dim)) if self.per_gene else 1.0 / math.sqrt(dim)

    def tau_prime(self, dim: int) -> float:
        """The global learning rate of per-gene step sizes (`1/sqrt(2d)`)."""
        return self._tau_prime if self._tau_prime is not None else 1.0 / math.sqrt(2.0 * dim)

    def initial_steps(self, count: int, dim: int, backend: Backend) -> Array:
        shape = (count, dim) if self.per_gene else (count,)
        return backend.xp.full(shape, self.initial_step, dtype=backend.dtype, device=backend.device)

    def mutate(self, genomes: Array, steps: Array | None, width: Array, rng: RandomStream, backend: Backend) -> tuple[Array, Array]:
        assert steps is not None, "a self-adaptive mutation needs the step sizes of the individuals"
        xp = backend.xp
        count, dim = int(genomes.shape[0]), int(genomes.shape[1])
        if self.per_gene:
            noise = self.tau_prime(dim) * rng.normal((count, 1)) + self.tau(dim) * rng.normal((count, dim))
            new_steps = xp.clip(steps * xp.exp(noise), self.min_step, None)
            moves = new_steps * width
        else:
            new_steps = xp.clip(steps * xp.exp(self.tau(dim) * rng.normal((count,))), self.min_step, None)
            moves = new_steps[:, None] * width
        return genomes + moves * rng.normal((count, dim)), new_steps


class PolynomialMutation:
    """Bounded polynomial mutation (Deb), the standard real-coded mutation of NSGA-II: non-adaptive, with a distribution index.

    A mutated gene moves by a random fraction of the box width drawn from a polynomial distribution around its current value,
    **truncated at the bounds**: the distribution is built from the gene's distance to each bound, so a gene near a wall
    mostly moves away from it and nothing needs clipping (unlike a Gaussian step). The index `eta` (default 20) sets how local
    the moves are: the larger, the smaller the moves. Each gene mutates with probability `probability`, by default `1/d` for
    the `d` real genes of the space.

    It needs the bounds, which a mutation protocol call does not carry (it gets the width only), so the strategy binds them
    with `bounded(lower, upper)`, which returns a copy: an operator object is never changed, and can be shared.
    """

    name = "polynomial"
    adaptive = False
    min_step = None

    def __init__(self, eta: float = 20.0, probability: float | None = None) -> None:
        if not eta >= 0:
            raise ValueError(f"the distribution index eta must not be negative, got {eta}")
        if probability is not None and not 0.0 < probability <= 1.0:
            raise ValueError(f"the mutation probability must be in (0, 1], got {probability}")
        self.eta = eta
        self.probability = probability
        self._lower: Array | None = None
        self._upper: Array | None = None

    def __repr__(self) -> str:
        return f"PolynomialMutation(eta={self.eta}, probability={self.probability})"

    def bounded(self, lower: Array, upper: Array) -> "PolynomialMutation":
        """A copy that knows the bounds of the (real) genes it will mutate, as arrays of shape `(d,)` on the backend."""
        bound = PolynomialMutation(self.eta, self.probability)
        bound._lower, bound._upper = lower, upper
        return bound

    def initial_steps(self, count: int, dim: int, backend: Backend) -> None:
        return None

    def mutate(self, genomes: Array, steps: Array | None, width: Array, rng: RandomStream, backend: Backend) -> tuple[Array, None]:
        if self._lower is None or self._upper is None:
            raise RuntimeError("PolynomialMutation needs the bounds of the genes: a strategy binds them with bounded(lower, upper)")
        xp = backend.xp
        lower, upper = self._lower, self._upper
        span = upper - lower
        count, dim = int(genomes.shape[0]), int(genomes.shape[1])
        to_lower = xp.clip((genomes - lower) / span, 0.0, 1.0)  # the share of the box below the gene
        to_upper = xp.clip((upper - genomes) / span, 0.0, 1.0)
        draw = rng.uniform((count, dim))
        power = 1.0 / (self.eta + 1.0)
        down = draw < 0.5
        base = xp.where(down, 1.0 - to_lower, 1.0 - to_upper)
        value = xp.where(
            down,
            2.0 * draw + (1.0 - 2.0 * draw) * xp.pow(base, self.eta + 1.0),
            2.0 * (1.0 - draw) + 2.0 * (draw - 0.5) * xp.pow(base, self.eta + 1.0),
        )
        step = xp.where(down, xp.pow(value, power) - 1.0, 1.0 - xp.pow(value, power))
        moved = xp.minimum(xp.maximum(genomes + step * span, lower), upper)
        probability = 1.0 / dim if self.probability is None else self.probability
        mutated = rng.uniform((count, dim)) < probability
        return xp.where(mutated, moved, genomes), None


def with_bounds(mutation: Mutation, lower: Array, upper: Array) -> Mutation:
    """The mutation to call for genes in `[lower, upper]`: the operator itself, or a bounded copy for those built from the
    bounds (`PolynomialMutation`). Strategies call this once when they are bound to a space."""
    return mutation.bounded(lower, upper) if isinstance(mutation, PolynomialMutation) else mutation
