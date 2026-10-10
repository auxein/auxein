"""`PycmaStrategy`: CMA-ES from pycma, wrapped as an Auxein strategy (design doc §3.4)."""

import importlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from auxein.backend import Array
from auxein.core import (
    ArrayBatch,
    Batch,
    CandidateId,
    EvaluationBatch,
    ProblemSpec,
    StateDict,
    StrategyCapabilities,
    StrategyContext,
)
from auxein.spaces import Box

FAILED_GENERATION_VALUE = 1e30
"""What every candidate of a generation is told when every one of them failed: pycma cannot be told NaN, and with nothing to
rank by, equal values are the honest answer (pycma then sees a flat generation and may stop, which `should_stop` reports)."""

_OWNED = {
    "seed": "that option seeds numpy's global generator, and Auxein never uses global random state: its own stream drives pycma",
    "randn": "the strategy passes its own random stream through this option",
    "bounds": "the bounds are the box of the search space",
    "popsize": "use the population_size argument",
    "CMA_stds": "the strategy sets the per-variable scaling from the box",
    "verbose": "the strategy keeps pycma quiet",
}


def _import_cma() -> Any:
    try:
        return importlib.import_module("cma")
    except ImportError as error:
        raise ImportError("PycmaStrategy needs the pycma package: install it with `pip install auxein[cma]`") from error


class PycmaStrategy:
    """CMA-ES (pycma's `CMAEvolutionStrategy`) through its ask/tell interface: a thin, faithful wrapper, and the proof that an
    external algorithm fits the strategy contract.

    **Restrictions, all checked when the strategy is bound.** It is single-objective (put several objectives through
    `Scalarised`), needs a `Box` with no log-scale dimension (CMA-ES adapts a Gaussian in the genome's own coordinates; for a
    `MixedSpace` use `GeneticAlgorithm`) and supports no constraints (pycma has no constraint handling of its own). It takes results in
    generations (`tell_mode="generation"`): pycma needs a whole generation, in the order it was asked.

    **Generation size.** `ask` returns pycma's own population (`population_size`, or pycma's default `4 + floor(3 ln d)`),
    whatever `n` the driver suggests, like the initial population of the GAs.

    **Start and step size.** The initial mean `x0` defaults to the centre of the box and the initial step size `sigma0` to
    `sigma_fraction` (0.3, the usual recommendation of a quarter to a third of the domain) of its mean width. When the
    widths of the box differ pycma is given a per-variable scaling (`CMA_stds`), so that the step is the same fraction of every
    width. **Bounds** use pycma's own bound handling (its `bounds` option, a boundary transformation), so every candidate pycma
    asks for lies in the box; the candidates are clipped once more to the box rounded inward to the run's precision, so that
    they are members of the space in float32 too.

    **Options.** `options` passes any other pycma option through (`tolfun`, `CMA_diagonal`, ...), except the few the strategy owns
    (`seed`, `randn`, `bounds`, `popsize`, `CMA_stds`, `verbose`), which are an error. `should_stop` is true when pycma's own
    stopping criteria say so (`es.stop()`), which is how the raw pycma loop ends too.

    **Randomness.** All of pycma's randomness comes from the strategy's Auxein stream, through pycma's `randn` option. The
    `seed` option is never used: it seeds numpy's *global* generator, which nothing in Auxein touches (a test checks that a run
    leaves it untouched, and is identical for the same seed).

    **Failed evaluations.** pycma cannot be told NaN or infinity, and a failed or timed-out candidate (or a non-finite
    objective) must rank last. It is told the worst finite value of its generation plus a margin, the spread of the
    finite values of that generation (or `max(1, |worst|)` when they are all equal). If *every* candidate of a generation failed,
    each is told the same large constant, `FAILED_GENERATION_VALUE`.

    **No checkpoints.** pycma's internal state (a covariance matrix, evolution paths, a history) cannot be saved exactly without
    pickle, which Auxein forbids, so `supports_checkpoints` is false and the driver writes none. A resumed run is rebuilt by
    replaying the recording from the start (deterministic mode: the strategy is fed the recorded evaluations, so no evaluation is
    repeated, but pycma's internal computation runs again, which takes a while for a long run) or restarted (throughput mode).

    **Precision.** pycma computes in float64. Candidates are converted to the run's backend and precision for evaluation, and
    pycma is told the float64 points it asked for, with the values of the (rounded) candidates that were evaluated.
    """

    capabilities = StrategyCapabilities(max_objectives=1, supports_constraints=False, tell_mode="generation", supports_checkpoints=False)

    def __init__(
        self,
        population_size: int | None = None,
        *,
        x0: float | Sequence[float] | npt.NDArray[np.float64] | None = None,
        sigma0: float | None = None,
        sigma_fraction: float = 0.3,
        options: Mapping[str, object] | None = None,
    ) -> None:
        _import_cma()
        if population_size is not None and population_size < 2:
            raise ValueError(f"population_size must be at least 2, got {population_size}")
        if sigma0 is not None and not (sigma0 > 0 and math.isfinite(sigma0)):
            raise ValueError(f"sigma0 must be positive, got {sigma0}")
        if not 0.0 < sigma_fraction <= 1.0:
            raise ValueError(f"sigma_fraction must be in (0, 1], got {sigma_fraction}")
        given = dict(options or {})
        for name, reason in _OWNED.items():
            if name in given:
                raise ValueError(f"the pycma option {name!r} cannot be set: {reason}")
        self.population_size = population_size
        self.x0 = x0
        self.sigma0 = sigma0
        self.sigma_fraction = sigma_fraction
        self.options = given
        self._problem: ProblemSpec[Array] | None = None
        self._ctx: StrategyContext | None = None
        self._box: Box | None = None
        self._es: Any = None
        self._step = 0
        self._pending: tuple[list[int], list[Any]] | None = None
        self._told = False

    def __repr__(self) -> str:
        x0 = self.x0 if self.x0 is None or isinstance(self.x0, (int, float)) else list(self.x0)
        return (
            f"PycmaStrategy(population_size={self.population_size}, x0={x0}, sigma0={self.sigma0}, "
            f"sigma_fraction={self.sigma_fraction}, options={dict(sorted(self.options.items()))})"
        )

    def bind(self, problem: ProblemSpec[Array], ctx: StrategyContext) -> None:
        if len(problem.objectives) != 1:
            names = ", ".join(repr(o.name) for o in problem.objectives)
            raise ValueError(
                f"PycmaStrategy is single-objective but the problem has {len(problem.objectives)} objectives ({names}): "
                "wrap it in Scalarised(strategy, scalarisation) to run it on several"
            )
        if problem.constraints:
            raise ValueError(
                f"PycmaStrategy does not support constraints (the problem declares {list(problem.constraints)}): pycma has no "
                "constraint handling; use GeneticAlgorithm or NSGA2, which rank feasible candidates first"
            )
        space = problem.space
        if not isinstance(space, Box):
            raise TypeError(
                f"PycmaStrategy needs a Box search space, got {type(space).__name__}: CMA-ES adapts a Gaussian over real "
                "coordinates; use GeneticAlgorithm for a MixedSpace"
            )
        if bool(space.log_scale.any()):
            raise ValueError(
                "PycmaStrategy does not support log-scale dimensions: CMA-ES adapts a Gaussian in the genome's own coordinates, "
                "where a log-scale dimension is badly scaled (use GeneticAlgorithm, or search the logarithm in a linear Box)"
            )
        cma = _import_cma()
        lower, upper = space.lower, space.upper
        width = upper - lower
        sigma0 = self.sigma0 if self.sigma0 is not None else self.sigma_fraction * float(np.mean(width))
        if self.x0 is None:
            x0 = (lower + upper) / 2.0
        else:
            x0 = np.asarray(self.x0, dtype=np.float64).reshape(-1)
            x0 = np.full(space.dim, float(x0[0])) if x0.size == 1 else x0
            if x0.shape != (space.dim,):
                raise ValueError(f"x0 must be a number or {space.dim} numbers, got {x0.size}")
            if not bool(((x0 >= lower) & (x0 <= upper)).all()):
                raise ValueError(f"x0 lies outside the box: {x0.tolist()}")
        rng, backend = ctx.rng, ctx.backend

        def randn(*shape: int) -> np.ndarray:
            """pycma's N(0, 1) source (the signature of `numpy.random.randn`), drawn from the strategy's own stream."""
            return np.asarray(backend.to_numpy(rng.normal(tuple(shape) or (1,))), dtype=np.float64).reshape(shape or ())

        options: dict[str, Any] = {"bounds": [lower.tolist(), upper.tolist()], "randn": randn, "verbose": -9}
        if self.population_size is not None:
            options["popsize"] = self.population_size
        if float(width.max() - width.min()) > 1e-12 * float(width.max()):
            options["CMA_stds"] = (width / float(np.mean(width))).tolist()  # the same fraction of every width
        options.update(self.options)
        self._es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, options)
        self._problem, self._ctx, self._box = problem, ctx, space
        self._step, self._pending, self._told = 0, None, False

    def _bound(self) -> tuple[ProblemSpec[Array], StrategyContext, Box]:
        if self._problem is None or self._ctx is None or self._box is None:
            raise RuntimeError("PycmaStrategy must be bound to a problem before use: the driver calls bind() first")
        return self._problem, self._ctx, self._box

    def ask(self, n: int) -> Batch[Array]:
        _, ctx, box = self._bound()
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if self._pending is not None:
            raise RuntimeError("PycmaStrategy was asked again before the previous generation was told: pycma needs the whole generation")
        solutions = list(self._es.ask())
        backend = ctx.backend
        genomes = box.clip(backend.asarray(np.asarray(solutions, dtype=np.float64)))
        ids = [int(ctx.new_id()) for _ in solutions]
        self._pending = (ids, solutions)
        step, self._step = self._step, self._step + 1
        return cast("Batch[Array]", ArrayBatch(genomes, [CandidateId(i) for i in ids], step, "pycma"))

    def tell(self, results: EvaluationBatch[Array]) -> None:
        problem, ctx, _ = self._bound()
        if self._pending is None:
            raise ValueError("tell() was called but no generation is pending: nothing was asked")
        ids, solutions = self._pending
        told = [int(e.candidate.id) for e in results]
        if sorted(told) != sorted(ids):
            raise ValueError(
                f"pycma needs the whole generation it asked for, {len(ids)} candidates, but tell() got {len(told)} "
                f"({'some are not pending' if set(told) - set(ids) else 'some are missing'})"
            )
        position = {cid: row for row, cid in enumerate(told)}
        column = np.asarray(ctx.backend.to_numpy(results.minimisation_matrix(problem.objectives, ctx.backend)[:, 0]), dtype=np.float64)
        values = column[[position[cid] for cid in ids]]  # in the order of the solutions pycma asked for
        self._es.tell(solutions, substitute_failures(values).tolist())
        self._pending, self._told = None, True

    def should_stop(self) -> bool:
        """True when pycma's own stopping criteria say so (its `stop()`), once a generation has been told."""
        return self._told and bool(self._es.stop())

    def state_dict(self) -> StateDict:
        raise NotImplementedError(
            "PycmaStrategy cannot be checkpointed: pycma's state cannot be saved exactly without pickle. The driver writes no "
            "checkpoints for it (supports_checkpoints=False), and a resumed run replays the recording from the start"
        )

    def load_state_dict(self, state: StateDict) -> None:
        raise NotImplementedError("PycmaStrategy has no checkpoints to load: a resumed run replays the recording from the start")


def substitute_failures(values: np.ndarray) -> np.ndarray:
    """The values pycma is told: finite ones unchanged; NaN and infinite ones (failed candidates) become the worst finite value
    of the generation plus a margin (the spread of the finite values, or `max(1, |worst|)` if they are all equal), and a
    generation with no finite value becomes `FAILED_GENERATION_VALUE` throughout."""
    finite = np.isfinite(values)
    if not bool(finite.any()):
        return np.full(values.shape, FAILED_GENERATION_VALUE)
    worst, best = float(values[finite].max()), float(values[finite].min())
    spread = worst - best
    margin = spread if spread > 0 else max(1.0, abs(worst))
    out = values.copy()
    out[~finite] = worst + margin
    return out
