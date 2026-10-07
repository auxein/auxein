"""`GeneticAlgorithm`: a composable genetic algorithm with plus-survivor selection (design doc §3.3)."""

from dataclasses import dataclass
from typing import cast

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
from auxein.core.ids import CandidateId
from auxein.spaces import Box
from auxein.strategies.ga.base import BoundsRepair, Mutation, ParentSelection, Recombination
from auxein.strategies.ga.mutation import SelfAdaptiveMutation
from auxein.strategies.ga.ranking import rank_order, view_of
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

    It is single-objective, supports constraints, and needs a `Box` search space.
    """

    capabilities = StrategyCapabilities(max_objectives=1, supports_constraints=True, tell_mode="both")

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
        self._reset()

    def __repr__(self) -> str:
        return (
            f"GeneticAlgorithm(population_size={self.population_size}, offspring_size={self.offspring_size}, selection={self.selection!r}, "
            f"recombination={self.recombination!r}, mutation={self.mutation!r}, repair={self.repair!r}, "
            f"crossover_probability={self.crossover_probability})"
        )

    # --- state ---

    def _reset(self) -> None:
        self._problem: ProblemSpec[Array] | None = None
        self._ctx: StrategyContext | None = None
        self._box: Box | None = None
        self._step = 0
        self._initial_asked = 0
        self._blocks: dict[int, _Block] = {}
        self._pending: dict[int, tuple[int, int]] = {}  # candidate id -> (block, row)
        self._next_block = 0
        # the population, ranked or not: a member is a row of each array
        self._ids: list[int] = []
        self._ids_array: Array | None = None
        self._genomes: Array | None = None
        self._values: Array | None = None
        self._violation: Array | None = None
        self._steps: Array | None = None

    def _bound(self) -> tuple[ProblemSpec[Array], StrategyContext, Box]:
        if self._problem is None or self._ctx is None or self._box is None:
            raise RuntimeError("GeneticAlgorithm must be bound to a problem before use: the driver calls bind() first")
        return self._problem, self._ctx, self._box

    @property
    def size(self) -> int:
        """How many members the population holds now."""
        return len(self._ids)

    def ranked_ids(self) -> list[int]:
        """The ids of the population's members, best first (by the ranking in the class documentation)."""
        if self._ctx is None or self._ids_array is None or self._values is None or self._violation is None:
            return []
        order = rank_order(self._values, self._violation, self._ids_array, self._ctx.backend)
        return [self._ids[i] for i in self._ctx.backend.to_numpy(order).tolist()]

    def bind(self, problem: ProblemSpec[Array], ctx: StrategyContext) -> None:
        if len(problem.objectives) != 1:
            names = ", ".join(repr(o.name) for o in problem.objectives)
            raise ValueError(
                f"GeneticAlgorithm is single-objective but the problem has {len(problem.objectives)} objectives ({names}): use a single objective or a scalarisation"
            )
        if not isinstance(problem.space, Box):
            raise TypeError(f"GeneticAlgorithm needs a Box search space (a bounded real vector), got {type(problem.space).__name__}")
        self._reset()
        self._problem, self._ctx, self._box = problem, ctx, problem.space
        backend, dim = ctx.backend, problem.space.dim
        xp = backend.xp
        self._ids_array = xp.zeros((0,), dtype=backend.int_dtype, device=backend.device)
        self._genomes = xp.zeros((0, dim), dtype=backend.dtype, device=backend.device)
        self._values = xp.zeros((0,), dtype=backend.dtype, device=backend.device)
        self._violation = xp.zeros((0,), dtype=backend.dtype, device=backend.device)
        initial = self.mutation.initial_steps(0, dim, backend)
        self._steps = initial

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
            steps = self.mutation.initial_steps(count, box.dim, backend)
            origins = [origin] * count
        elif self.size < 2:
            # nothing to breed from yet (the initial candidates are still being evaluated): sample more at random
            count = n if self.offspring_size is None else self.offspring_size
            genomes, parents = self._sample(box, ctx, count), [()] * count
            steps = self.mutation.initial_steps(count, box.dim, backend)
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

    def _sample(self, box: Box, ctx: StrategyContext, count: int) -> Array:
        return box.sample_genomes(count, ctx.rng, ctx.backend)

    def _breed(self, box: Box, ctx: StrategyContext, count: int) -> tuple[Array, Array | None, list[tuple[int, ...]], list[str]]:
        backend, rng = ctx.backend, ctx.rng
        xp = backend.xp
        assert self._genomes is not None and self._values is not None and self._violation is not None and self._ids_array is not None
        view = view_of(self._values, self._violation, self._ids_array, backend)

        first, second = self.selection.select(view, count, rng)
        parents_a, parents_b = xp.take(self._genomes, first, axis=0), xp.take(self._genomes, second, axis=0)
        weights = self.recombination.weights(count, box.dim, rng, backend)
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
        if self.mutation.adaptive:
            assert self._steps is not None
            steps = mix_steps(weights, xp.take(self._steps, first, axis=0), xp.take(self._steps, second, axis=0), backend)
        width = backend.asarray(box.upper - box.lower)
        genomes, steps = self.mutation.mutate(genomes, steps, width, rng, backend)
        genomes = self.repair.repair(genomes, box)

        first_ids, second_ids = backend.to_numpy(first).tolist(), backend.to_numpy(second).tolist()
        parents: list[tuple[int, ...]] = [
            (self._ids[a],) if c else (self._ids[a], self._ids[b]) for a, b, c in zip(first_ids, second_ids, copied)
        ]
        base = f"{self.selection.name}+%s+{self.mutation.name}"
        origins = [base % ("copy" if c else self.recombination.name) for c in copied]
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

        genomes, steps, order_in_told = self._take_pending(told, backend)
        values = xp.take(results.minimisation_matrix(problem.objectives, backend)[:, 0], order_in_told, axis=0)
        violation = xp.take(results.total_violation(backend), order_in_told, axis=0)

        assert self._genomes is not None and self._values is not None and self._violation is not None and self._ids_array is not None
        new_ids = backend.asarray([told[i] for i in backend.to_numpy(order_in_told).tolist()], dtype=backend.int_dtype)
        all_ids = xp.concat([self._ids_array, new_ids], axis=0)
        all_values = xp.concat([self._values, values], axis=0)
        all_violation = xp.concat([self._violation, violation], axis=0)
        all_genomes = xp.concat([self._genomes, genomes], axis=0)
        all_steps = None if steps is None else xp.concat([cast("Array", self._steps), steps], axis=0)

        keep = rank_order(all_values, all_violation, all_ids, backend)[: self.population_size]
        host_ids = backend.to_numpy(all_ids).tolist()
        self._ids = [host_ids[i] for i in backend.to_numpy(keep).tolist()]
        self._ids_array = xp.take(all_ids, keep, axis=0)
        self._values = xp.take(all_values, keep, axis=0)
        self._violation = xp.take(all_violation, keep, axis=0)
        self._genomes = xp.take(all_genomes, keep, axis=0)
        self._steps = None if all_steps is None else xp.take(all_steps, keep, axis=0)

    def _take_pending(self, told: list[int], backend: Backend) -> tuple[Array, Array | None, Array]:
        """The genomes and step sizes of the told children, in the order of the blocks they came from, and the
        positions (in `told`) that this order corresponds to."""
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
        return genomes, steps, backend.asarray(positions_in_order, dtype=backend.int_dtype)

    # --- stopping ---

    def should_stop(self) -> bool:
        """False, unless `convergence_tolerance` is set: then true when the objective spread of a full population is below
        the tolerance and (for an adaptive mutation) every step size has reached its floor."""
        if self.convergence_tolerance is None or self._ctx is None or self.size < self.population_size:
            return False
        backend = self._ctx.backend
        xp = backend.xp
        assert self._values is not None
        if float(xp.max(self._values) - xp.min(self._values)) >= self.convergence_tolerance:
            return False
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
            "ids": list(self._ids),
            "genomes": self._genomes,
            "values": self._values,
            "violation": self._violation,
            "steps": self._steps,
            "pending": cast("list[Array]", blocks),
        }

    def load_state_dict(self, state: StateDict) -> None:
        problem, ctx, box = self._bound()
        backend = ctx.backend
        expected = {"stream", "step", "initial_asked", "ids", "genomes", "values", "violation", "steps", "pending"}
        if set(state) != expected:
            raise ValueError(f"invalid GeneticAlgorithm state: expected keys {sorted(expected)}, got {sorted(state)}")
        ctx.rng.load_state_dict(cast("dict[str, object]", state["stream"]))
        self._step = cast("int", state["step"])
        self._initial_asked = cast("int", state["initial_asked"])
        self._ids = [int(i) for i in cast("list[int]", state["ids"])]
        self._ids_array = backend.asarray(self._ids, dtype=backend.int_dtype)
        self._genomes = backend.asarray(state["genomes"])
        self._values = backend.asarray(state["values"])
        self._violation = backend.asarray(state["violation"])
        self._steps = None if state["steps"] is None else backend.asarray(state["steps"])
        self._blocks, self._pending, self._next_block = {}, {}, 0
        for saved in cast("list[dict[str, object]]", state["pending"]):
            ids = [int(i) for i in cast("list[int]", saved["ids"])]
            steps = None if saved["steps"] is None else backend.asarray(saved["steps"])
            self._blocks[self._next_block] = _Block(ids, backend.asarray(saved["genomes"]), steps)
            self._pending.update({cid: (self._next_block, row) for row, cid in enumerate(ids)})
            self._next_block += 1
        _ = problem, box
