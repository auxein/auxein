"""`GeneticAlgorithm`: a composable genetic algorithm with plus-survivor selection (design doc §3.3)."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from auxein.backend import Array, Backend
from auxein.core import (
    ArrayBatch,
    Batch,
    EvaluationBatch,
    ProblemSpec,
    StateDict,
    StrategyCapabilities,
    StrategyContext,
)
from auxein.core.evaluation_batch import INFEASIBLE
from auxein.core.ids import CandidateId
from auxein.spaces import Box, MixedSpace
from auxein.strategies.ga.base import BoundsRepair, Mutation, ParentSelection, PopulationView, Recombination
from auxein.strategies.ga.mixed import BitFlipMutation, CategoricalMutation, IntegerMutation, MixedVariation
from auxein.strategies.ga.mutation import SelfAdaptiveMutation, with_bounds
from auxein.strategies.ga.ranking import rank_order
from auxein.strategies.ga.recombination import IntermediateRecombination, mix_genes, mix_steps
from auxein.strategies.ga.repair import ClipRepair
from auxein.strategies.ga.selection import TournamentSelection

DEFAULT_POPULATION_SIZE = 50


@dataclass
class _Block:
    """Children (or initial candidates) that were asked for and are waiting for their results."""

    ids: list[int]
    genomes: Array
    steps: Array | None


# Host-side by design (design doc §7.2), and once per `ask` or `tell`, never per candidate: the survivors' indices and the
# parents' ids. A population is selected by indices whose number depends on the data (how many children survive), which the
# array API cannot express without a boolean mask index, and the ids are lineage metadata that goes to the recorder as Python
# ints. They are `mu` or `lambda` integers, not genomes: the genomes themselves are only ever gathered on the device.


def _survivor_positions(keep: npt.NDArray[np.int64], members: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Where the survivors sit in the new population, in rank order, and the inverse of that permutation.

    `keep` lists the survivors best first, as indices into the old members followed by the children. They are stored as
    [the kept old members, then the kept children], each in the order of `keep`, so that the position of a survivor is
    its running count among those of its kind (plus the number of kept old members, for children).
    """
    from_old = keep < members
    old_count = int(from_old.sum())
    among_old = from_old.cumsum() - 1
    among_new = (~from_old).cumsum() - 1
    position = from_old * among_old + (~from_old) * (old_count + among_new)
    inverse = np.empty_like(position)
    inverse[position] = np.arange(position.size)
    return position, inverse


class GeneticAlgorithm:
    """A genetic algorithm assembled from operators: parent selection, recombination, mutation and bounds repair.

    **Survivors.** The population of size `population_size` (mu) always holds the best mu of everything evaluated so far,
    parents and children pooled ("plus" selection). The ranking is: feasible before infeasible, then a lower total
    violation, then a lower objective in minimisation form, then a lower candidate id. Failed evaluations rank last.
    In a steady-state view, each child replaces the worst member if it is better: inserting children one at a time and
    taking the best mu of mu + lambda give the same population, so one mechanism serves any way of telling results.

    **Breeding.** `ask` first returns the random initial candidates (origin `"init"`), then exactly `offspring_size`
    (lambda) children, or exactly `n` when it is `None`. A child is made by selecting two distinct parents, recombining
    them (with probability `crossover_probability`, otherwise it copies the first parent), mutating, and repairing the
    bounds: recombine first, then mutate. Children are bred from the population as it is when `ask` is called, so asking
    again before results arrive is allowed. Every candidate is evaluated exactly once; the population is never re-scored.

    **Step sizes** of a self-adaptive mutation are strategy state, kept per candidate (relative to the box width), never
    in the genome (design doc §4.1). Children inherit them by the same recombination as their genes (weighted geometric
    mean), then they mutate; the steps of non-survivors are discarded.

    **Defaults.** `GeneticAlgorithm()` uses 50 parents and 50 children per generation, tournament selection (size 2),
    intermediate recombination, self-adaptive mutation with one step size per individual (initial step 10% of the box,
    floor 1e-12) and clipping to the bounds. See the design document (§3.3) for how this default was chosen by benchmark.

    It is single-objective, supports constraints, and needs a `Box` or a `MixedSpace` search space. **On a `Box` nothing
    changes** (the code path is the one described above). **On a `MixedSpace`** the genome mixes real, integer, binary and
    categorical genes, and each type gets its own operators: the real genes use `recombination`, `mutation` and `repair`
    exactly as on a `Box` (on the real columns), integer genes `integer_mutation` (the difference of two geometric variables,
    MIES), binary genes `binary_mutation` (bit flips) and categorical genes `categorical_mutation` (a jump to a different
    category), and every discrete gene is recombined by taking it from one of the parents. The defaults are chosen from the
    space; every operator can be replaced. The step sizes of the real genes and the integer mean step are strategy state
    (design doc §3.3).
    """

    capabilities = StrategyCapabilities(max_objectives=1, supports_constraints=True, tell_mode="both")

    _extra_keys: tuple[str, ...] = ()
    """The keys a subclass adds to the state (see `_extra_state`)."""

    _skip_unchanged_tell = True
    """Whether a `tell` after which no child survives leaves the population untouched. A subclass whose ranking depends on the
    whole pool (NSGA-II's fronts and crowding distances) must refresh the ranking even then."""

    def __init__(
        self,
        population_size: int = DEFAULT_POPULATION_SIZE,
        offspring_size: int | None = DEFAULT_POPULATION_SIZE,
        *,
        selection: ParentSelection | None = None,
        recombination: Recombination | None = None,
        mutation: Mutation | None = None,
        repair: BoundsRepair | None = None,
        crossover_probability: float = 1.0,
        convergence_tolerance: float | None = None,
        integer_mutation: IntegerMutation | None = None,
        binary_mutation: BitFlipMutation | None = None,
        categorical_mutation: CategoricalMutation | None = None,
    ) -> None:
        if population_size < 2:
            raise ValueError(f"population_size must be at least 2, got {population_size}")
        if offspring_size is not None and offspring_size < 1:
            raise ValueError(f"offspring_size must be at least 1 (or None for the size the driver asks for), got {offspring_size}")
        if not 0.0 <= crossover_probability <= 1.0:
            raise ValueError(f"crossover_probability must be in [0, 1], got {crossover_probability}")
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.selection: ParentSelection = TournamentSelection() if selection is None else selection
        self.recombination: Recombination = IntermediateRecombination() if recombination is None else recombination
        self.mutation: Mutation = SelfAdaptiveMutation() if mutation is None else mutation
        self.repair: BoundsRepair = ClipRepair() if repair is None else repair
        self.crossover_probability = crossover_probability
        self.convergence_tolerance = convergence_tolerance
        self._mixed_options = (integer_mutation, binary_mutation, categorical_mutation)
        self.integer_mutation = IntegerMutation() if integer_mutation is None else integer_mutation
        self.binary_mutation = BitFlipMutation() if binary_mutation is None else binary_mutation
        self.categorical_mutation = CategoricalMutation() if categorical_mutation is None else categorical_mutation
        self._reset()

    def __repr__(self) -> str:
        # the operators for the discrete types appear only when given: the description of a default GA, which a recorded run
        # compares on resume, is the one it always had
        names = ("integer_mutation", "binary_mutation", "categorical_mutation")
        mixed = "".join(f", {name}={op!r}" for name, op in zip(names, self._mixed_options, strict=True) if op is not None)
        return (
            f"GeneticAlgorithm(population_size={self.population_size}, offspring_size={self.offspring_size}, selection={self.selection!r}, "
            f"recombination={self.recombination!r}, mutation={self.mutation!r}, repair={self.repair!r}, "
            f"crossover_probability={self.crossover_probability}{mixed})"
        )

    # --- state ---

    def _reset(self) -> None:
        self._problem: ProblemSpec[Array] | None = None
        self._ctx: StrategyContext | None = None
        self._box: Box | MixedSpace | None = None
        self._variation: MixedVariation | None = None  # the per-type operators, only on a MixedSpace
        self._real_operator: Mutation = self.mutation  # the real mutation, with the box's bounds if it needs them
        self._step = 0
        self._initial_asked = 0
        self._blocks: dict[int, _Block] = {}
        self._pending: dict[int, tuple[int, int]] = {}  # candidate id -> (block, row)
        self._next_block = 0
        # the population, in no particular order: a member is a row of each array
        self._ids_array: Array | None = None
        self._genomes: Array | None = None
        self._values: Array | None = None
        self._violation: Array | None = None
        self._steps: Array | None = None
        # the ranking of the population, kept up to date by tell so that ask doesn't sort it again
        self._order: Array | None = None  # member indices, best first
        self._rank: Array | None = None  # the rank of each member: the inverse permutation of the order
        self._ever_failed = False  # whether any evaluation told so far failed: only then do selections need to look for it

    def _bound(self) -> tuple[ProblemSpec[Array], StrategyContext, Box | MixedSpace]:
        if self._problem is None or self._ctx is None or self._box is None:
            raise RuntimeError("GeneticAlgorithm must be bound to a problem before use: the driver calls bind() first")
        return self._problem, self._ctx, self._box

    @property
    def size(self) -> int:
        """How many members the population holds now."""
        return 0 if self._ids_array is None else int(self._ids_array.shape[0])

    def ranked_ids(self) -> list[int]:
        """The ids of the population's members, best first (by the ranking in the class documentation)."""
        if self._ctx is None or self._ids_array is None or self._values is None or self._violation is None:
            return []
        backend = self._ctx.backend
        order = self._population_order()
        return [int(i) for i in backend.to_numpy(backend.xp.take(self._ids_array, order, axis=0)).tolist()]

    # --- the ranking policy: what `NSGA2` replaces (design doc §3.3) ---
    #
    # The algorithm's machinery (breeding, pending children, plus survivor selection, state) only needs a few answers from the
    # ranking: how the objectives of the population are stored, the order of a pool of members, and the order of the population.
    # The defaults are the single-objective ones, and the numeric results are exactly what they were before these hooks existed.

    def _check_objectives(self, problem: ProblemSpec[Array]) -> None:
        if len(problem.objectives) != 1:
            names = ", ".join(repr(o.name) for o in problem.objectives)
            raise ValueError(
                f"GeneticAlgorithm is single-objective but the problem has {len(problem.objectives)} objectives ({names}): "
                "use a single objective or a scalarisation"
            )

    def _empty_values(self, backend: Backend) -> Array:
        return backend.xp.zeros((0,), dtype=backend.dtype, device=backend.device)

    def _told_values(self, results: EvaluationBatch[Array], problem: ProblemSpec[Array], backend: Backend) -> Array:
        return results.minimisation_matrix(problem.objectives, backend)[:, 0]

    def _pool_order(self, values: Array, violation: Array, ids: Array, backend: Backend) -> Array:
        """Member indices of a pool (the population followed by the told children), best first."""
        return rank_order(values, violation, ids, backend)

    def _population_order(self) -> Array:
        """Member indices of the current population, best first (recomputed, when a state is loaded)."""
        assert self._values is not None and self._violation is not None and self._ids_array is not None and self._ctx is not None
        return rank_order(self._values, self._violation, self._ids_array, self._ctx.backend)

    def _extra_state(self) -> StateDict:
        return {}

    def _load_extra_state(self, state: StateDict) -> None:
        return None

    def bind(self, problem: ProblemSpec[Array], ctx: StrategyContext) -> None:
        self._check_objectives(problem)
        if not isinstance(problem.space, (Box, MixedSpace)):
            raise TypeError(
                "GeneticAlgorithm needs a Box search space (a bounded real vector) or a MixedSpace (real, integer, binary and "
                f"categorical genes), got {type(problem.space).__name__}"
            )
        self._reset()
        self._problem, self._ctx, self._box = problem, ctx, problem.space
        backend, dim = ctx.backend, problem.space.dim
        xp = backend.xp
        if isinstance(problem.space, Box):
            self._real_operator = with_bounds(self.mutation, backend.asarray(problem.space.lower), backend.asarray(problem.space.upper))
        if isinstance(problem.space, MixedSpace):
            problem.space.check_backend(backend)  # integer bounds beyond what the precision holds exactly are refused now
            self._variation = MixedVariation(
                problem.space,
                backend,
                real_mutation=self.mutation,
                real_recombination=self.recombination,
                integer_mutation=self.integer_mutation,
                binary_mutation=self.binary_mutation,
                categorical_mutation=self.categorical_mutation,
                repair=self.repair,
            )
        self._ids_array = xp.zeros((0,), dtype=backend.int_dtype, device=backend.device)
        self._genomes = xp.zeros((0, dim), dtype=backend.dtype, device=backend.device)
        self._values = self._empty_values(backend)
        self._violation = xp.zeros((0,), dtype=backend.dtype, device=backend.device)
        self._steps = self._initial_steps(0, dim, backend)
        self._order = self._rank = xp.zeros((0,), dtype=backend.int_dtype, device=backend.device)

    # --- ask ---

    def ask(self, n: int) -> Batch[Array]:
        _, ctx, box = self._bound()
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        backend = ctx.backend
        if self._initial_asked < self.population_size:
            count, origin = self.population_size - self._initial_asked, "init"
            self._initial_asked += count
            genomes, parents = self._sample(box, ctx, count), [()] * count
            steps = self._initial_steps(count, box.dim, backend)
            origins = [origin] * count
        elif self._breedable() < 2:
            # nothing to breed from yet (the initial candidates are still being evaluated, or all but one failed): sample at random
            count = n if self.offspring_size is None else self.offspring_size
            genomes, parents = self._sample(box, ctx, count), [()] * count
            steps = self._initial_steps(count, box.dim, backend)
            origins = ["init"] * count
        else:
            count = n if self.offspring_size is None else self.offspring_size
            genomes, steps, parents, origins = self._breed(box, ctx, count)

        ids = [int(ctx.new_id()) for _ in range(count)]
        block = self._next_block
        self._next_block += 1
        self._blocks[block] = _Block(ids, genomes, steps)
        self._pending.update({cid: (block, row) for row, cid in enumerate(ids)})
        step, self._step = self._step, self._step + 1
        parent_ids = [tuple(CandidateId(p) for p in ps) for ps in parents]
        return cast("Batch[Array]", ArrayBatch(genomes, [CandidateId(i) for i in ids], step, origins, parent_ids))

    def _breedable(self) -> int:
        """How many members can be parents: those that did not fail. Failed members rank last, so they are never chosen
        while there is an alternative; this only costs anything once an evaluation has failed."""
        if not self._ever_failed:
            return self.size
        assert self._ctx is not None and self._violation is not None
        xp = self._ctx.backend.xp
        return int(xp.sum(xp.astype(xp.isfinite(self._violation), self._ctx.backend.int_dtype)))

    def _initial_steps(self, count: int, dim: int, backend: Backend) -> Array | None:
        if self._variation is not None:
            return self._variation.initial_steps(count)
        return self.mutation.initial_steps(count, dim, backend)

    def _sample(self, box: Box | MixedSpace, ctx: StrategyContext, count: int) -> Array:
        return box.sample_genomes(count, ctx.rng, ctx.backend)

    def _breed(
        self, box: Box | MixedSpace, ctx: StrategyContext, count: int
    ) -> tuple[Array, Array | None, list[tuple[int, ...]], list[str]]:
        backend, rng = ctx.backend, ctx.rng
        xp = backend.xp
        assert self._genomes is not None and self._values is not None and self._violation is not None and self._ids_array is not None
        assert self._order is not None and self._rank is not None
        view = PopulationView(
            self._values, self._violation, self._order, self._rank, backend, self._breedable() if self._ever_failed else None
        )

        first, second = self.selection.select(view, count, rng)
        parents_a, parents_b = xp.take(self._genomes, first, axis=0), xp.take(self._genomes, second, axis=0)
        variation = self._variation
        if variation is None:
            weights = self.recombination.weights(count, box.dim, rng, backend)
        else:
            weights = variation.weights(count, rng)
        if not self.recombination.sexual:
            copied = [True] * count
        elif self.crossover_probability < 1.0:  # the children that don't cross over copy their first parent
            crossed = rng.uniform((count, 1)) < self.crossover_probability
            weights = xp.where(crossed, weights, 1.0)
            copied = backend.to_numpy(~crossed[:, 0]).tolist()
        else:
            copied = [False] * count
        genomes = mix_genes(weights, parents_a, parents_b)

        steps = None
        if variation is not None:
            if variation.adaptive:
                assert self._steps is not None
                steps = variation.mix_steps(weights, xp.take(self._steps, first, axis=0), xp.take(self._steps, second, axis=0))
            genomes, steps = variation.mutate(genomes, steps, rng)
            genomes = variation.repair(genomes)
        else:
            assert isinstance(box, Box)
            if self.mutation.adaptive:
                assert self._steps is not None
                steps = mix_steps(weights, xp.take(self._steps, first, axis=0), xp.take(self._steps, second, axis=0), backend)
            width = backend.asarray(box.upper - box.lower)
            genomes, steps = self._real_operator.mutate(genomes, steps, width, rng, backend)
            genomes = self.repair.repair(genomes, box)

        first_ids = backend.to_numpy(xp.take(self._ids_array, first, axis=0)).tolist()
        second_ids = backend.to_numpy(xp.take(self._ids_array, second, axis=0)).tolist()
        parents: list[tuple[int, ...]] = [
            (int(a),) if c else (int(a), int(b)) for a, b, c in zip(first_ids, second_ids, copied, strict=True)
        ]
        mutation_name = self.mutation.name if variation is None else variation.mutation_name
        recombination_name = self.recombination.name if variation is None else variation.recombination_name
        base = f"{self.selection.name}+%s+{mutation_name}"
        origins = [base % ("copy" if c else recombination_name) for c in copied]
        return genomes, steps, parents, origins

    # --- tell ---

    def tell(self, results: EvaluationBatch[Array]) -> None:
        problem, ctx, _ = self._bound()
        backend = ctx.backend
        xp = backend.xp
        told = [int(e.candidate.id) for e in results]
        if not told:
            raise ValueError("tell() was called with no results")
        if len(set(told)) != len(told):
            raise ValueError("tell() was called with the same candidate more than once")
        unknown = [i for i in told if i not in self._pending]
        if unknown:
            raise ValueError(f"tell() was called with candidates that are not pending (never asked for, or already told): {unknown[:5]}")

        genomes, steps, positions = self._take_pending(told, backend)
        values = self._told_values(results, problem, backend)
        totals = results.violation_list()
        violation = backend.asarray(totals)
        if not self._ever_failed and INFEASIBLE in totals:  # a failed evaluation has infinite violation
            self._ever_failed = True
        if positions is not None:  # the children came from several blocks: put the values in the order of the genomes
            index = backend.asarray(positions, dtype=backend.int_dtype)
            values, violation = xp.take(values, index, axis=0), xp.take(violation, index, axis=0)
            told = [told[i] for i in positions]
        new_ids = backend.asarray(told, dtype=backend.int_dtype)

        assert self._genomes is not None and self._values is not None and self._violation is not None and self._ids_array is not None
        members = self.size
        ranking_ids = xp.concat([self._ids_array, new_ids], axis=0)
        ranking_values = xp.concat([self._values, values], axis=0)
        ranking_violation = xp.concat([self._violation, violation], axis=0)
        keep = backend.to_numpy(self._pool_order(ranking_values, ranking_violation, ranking_ids, backend)[: self.population_size])
        kept_old, kept_new = keep[keep < members], keep[keep >= members] - members
        if self._skip_unchanged_tell and kept_new.size == 0 and kept_old.size == members:
            return  # no child survives: the population is unchanged, and nothing needs copying

        # gather the survivors straight from the population and from the children: the big arrays are copied once
        old, new = backend.asarray(kept_old, dtype=backend.int_dtype), backend.asarray(kept_new, dtype=backend.int_dtype)

        def survivors(current: Array, incoming: Array) -> Array:
            return xp.concat([xp.take(current, old, axis=0), xp.take(incoming, new, axis=0)], axis=0)

        # the survivors are stored as [kept old members, kept children], each in rank order: the position of every survivor
        # follows from `keep`, which is the ranking, so the ranking of the new population needs no sorting
        position, inverse = _survivor_positions(keep, members)
        self._order = backend.asarray(position, dtype=backend.int_dtype)
        self._rank = backend.asarray(inverse, dtype=backend.int_dtype)
        self._ids_array = survivors(self._ids_array, new_ids)
        self._values = survivors(self._values, values)
        self._violation = survivors(self._violation, violation)
        self._genomes = survivors(self._genomes, genomes)
        if steps is not None:
            assert self._steps is not None
            self._steps = survivors(self._steps, steps)

    def _take_pending(self, told: list[int], backend: Backend) -> tuple[Array, Array | None, list[int] | None]:
        """The genomes and step sizes of the told children, in the order of the blocks they came from, and the positions
        (in `told`) that this order corresponds to, or None if it is the order of `told` already."""
        xp = backend.xp
        by_block: dict[int, tuple[list[int], list[int]]] = {}
        for position, cid in enumerate(told):
            block, row = self._pending.pop(cid)
            rows, positions = by_block.setdefault(block, ([], []))
            rows.append(row)
            positions.append(position)

        genome_parts: list[Array] = []
        step_parts: list[Array] = []
        positions_in_order: list[int] = []
        for block_id, (rows, positions) in by_block.items():
            block = self._blocks[block_id]
            whole = len(rows) == len(block.ids) and rows == list(range(len(rows)))
            index = None if whole else backend.asarray(rows, dtype=backend.int_dtype)
            genome_parts.append(block.genomes if index is None else xp.take(block.genomes, index, axis=0))
            if block.steps is not None:
                step_parts.append(block.steps if index is None else xp.take(block.steps, index, axis=0))
            positions_in_order.extend(positions)
            alive = [i for i in block.ids if i in self._pending]
            if not alive:
                del self._blocks[block_id]
        genomes = genome_parts[0] if len(genome_parts) == 1 else xp.concat(genome_parts, axis=0)
        steps = None if not step_parts else (step_parts[0] if len(step_parts) == 1 else xp.concat(step_parts, axis=0))
        return genomes, steps, None if positions_in_order == list(range(len(told))) else positions_in_order

    # --- stopping ---

    def should_stop(self) -> bool:
        """False, unless `convergence_tolerance` is set: then true when the objective spread of a full population is below
        the tolerance and (for an adaptive mutation) every step size has reached its floor."""
        if self.convergence_tolerance is None or self._ctx is None or self.size < self.population_size:
            return False
        backend = self._ctx.backend
        xp = backend.xp
        assert self._values is not None
        assert self._violation is not None
        valid = xp.isfinite(self._violation)  # members that failed have no objective value: they say nothing about convergence
        if not bool(xp.any(valid)):
            return False
        spread = xp.max(xp.where(valid, self._values, -xp.inf)) - xp.min(xp.where(valid, self._values, xp.inf))
        if float(spread) >= self.convergence_tolerance:
            return False
        if self._variation is not None:
            floors = self._variation.floors()
            if floors is not None:  # every adapted step (real and integer) at its own floor
                assert self._steps is not None
                return bool(xp.all(self._steps <= floors * (1.0 + 1e-9)))
            return True
        if self.mutation.adaptive and self.mutation.min_step is not None:
            assert self._steps is not None
            return bool(xp.all(self._steps <= self.mutation.min_step * (1.0 + 1e-9)))
        return True

    # --- state_dict ---

    def state_dict(self) -> StateDict:
        """Everything needed to continue identically: the population, the step sizes, the pending children, the counters
        and the stream state."""
        _, ctx, _ = self._bound()
        backend = ctx.backend
        xp = backend.xp
        blocks: list[object] = []
        for block in self._blocks.values():
            alive = [row for row, cid in enumerate(block.ids) if cid in self._pending]
            index = backend.asarray(alive, dtype=backend.int_dtype)
            blocks.append(
                {
                    "ids": [block.ids[row] for row in alive],
                    "genomes": xp.take(block.genomes, index, axis=0),
                    "steps": None if block.steps is None else xp.take(block.steps, index, axis=0),
                }
            )
        return {
            "stream": ctx.rng.state_dict(),
            "step": self._step,
            "initial_asked": self._initial_asked,
            "ids": [int(i) for i in backend.to_numpy(self._ids_array).tolist()],
            "genomes": self._genomes,
            "values": self._values,
            "violation": self._violation,
            "steps": self._steps,
            "pending": cast("list[Array]", blocks),
            **self._extra_state(),
        }

    def load_state_dict(self, state: StateDict) -> None:
        problem, ctx, box = self._bound()
        backend = ctx.backend
        expected = {"stream", "step", "initial_asked", "ids", "genomes", "values", "violation", "steps", "pending", *self._extra_keys}
        if set(state) != expected:
            raise ValueError(f"invalid GeneticAlgorithm state: expected keys {sorted(expected)}, got {sorted(state)}")
        ctx.rng.load_state_dict(cast("dict[str, object]", state["stream"]))
        self._step = cast("int", state["step"])
        self._initial_asked = cast("int", state["initial_asked"])
        self._ids_array = backend.asarray([int(i) for i in cast("list[int]", state["ids"])], dtype=backend.int_dtype)
        self._genomes = backend.asarray(state["genomes"])
        self._values = backend.asarray(state["values"])
        self._violation = backend.asarray(state["violation"])
        self._steps = None if state["steps"] is None else backend.asarray(state["steps"])
        self._load_extra_state(state)
        self._order = self._population_order()
        self._rank = backend.xp.argsort(self._order)
        self._ever_failed = bool(backend.xp.any(backend.xp.isinf(self._violation)))
        self._blocks, self._pending, self._next_block = {}, {}, 0
        for saved in cast("list[dict[str, object]]", state["pending"]):
            ids = [int(i) for i in cast("list[int]", saved["ids"])]
            steps = None if saved["steps"] is None else backend.asarray(saved["steps"])
            self._blocks[self._next_block] = _Block(ids, backend.asarray(saved["genomes"]), steps)
            self._pending.update({cid: (self._next_block, row) for row, cid in enumerate(ids)})
            self._next_block += 1
        _ = problem, box
