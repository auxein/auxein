"""`StructuredGeneticAlgorithm`: a genetic algorithm for genomes that are not arrays (design doc §3.3, §4.4)."""

from collections.abc import Sequence
from typing import Any, Generic, cast

from auxein.backend import Array, Backend, is_array
from auxein.core import (
    Batch,
    Candidate,
    CandidateId,
    EvaluationBatch,
    ListBatch,
    ProblemSpec,
    StateDict,
    StrategyCapabilities,
    StrategyContext,
)
from auxein.core._typing import G
from auxein.core.evaluation_batch import INFEASIBLE
from auxein.spaces import GenomeCodec, SequenceSpace, codec_of
from auxein.strategies.ga.base import ParentSelection, PopulationView
from auxein.strategies.ga.ranking import rank_order
from auxein.strategies.ga.selection import TournamentSelection
from auxein.strategies.structured.sequence import SequenceCrossover, SequenceMutation
from auxein.strategies.structured.variation import (
    StructuredMutation,
    StructuredRecombination,
    VariationContext,
    warn_if_unrecorded,
)

DEFAULT_POPULATION_SIZE = 50


class StructuredGeneticAlgorithm(Generic[G]):
    """A genetic algorithm whose genomes are Python values (tuples, frozen dataclasses...), with per-genome variation.

    It follows the contract of the numeric `GeneticAlgorithm` and reuses its machinery that only looks at objective values:
    the ranking (feasible first, then violation, then objective, then id), "plus" survivor selection (the population always
    holds the best `population_size` of everything evaluated so far), `PopulationView` and the parent-selection operators,
    including their handling of failed members (never parents while any other member exists). Only variation differs: the
    operators work on one genome (or a pair) at a time. It is single-objective, supports constraints, accepts results in
    generations or one at a time, asks for exactly `offspring_size` children (or `n` when it is None), and never mates a member
    with itself. Every candidate is evaluated once.

    **Operators.** `mutation` and `recombination` are `StructuredMutation` and `StructuredRecombination` objects. For a
    `SequenceSpace` they default to `SequenceMutation()` and `SequenceCrossover()`; for any other space a mutation must be
    given (and recombination is optional: without it a child copies its first parent and records one parent). `mutation` may
    also be a list of `(operator, probability)` pairs, mixed by those probabilities, which is how an external operator
    (`ExternalMutation`, an LLM-driven rewrite) is used next to the built-in ones. A child is made by selecting two distinct
    parents, recombining them with probability `crossover_probability` (otherwise copying the first), and mutating the result.
    Origins name the operators used, e.g. `"tournament+one_point+sequence"`.

    The space needs a codec (design doc §4.1): the population goes into checkpoints through it, and genomes are recorded
    through it. Calls of external operators happen synchronously inside `ask`, which blocks the driver while they run; they are
    recorded so that a resumed or extended run does not repeat them (design doc §3.5).
    """

    capabilities = StrategyCapabilities(max_objectives=1, supports_constraints=True, tell_mode="both")

    def __init__(
        self,
        population_size: int = DEFAULT_POPULATION_SIZE,
        offspring_size: int | None = DEFAULT_POPULATION_SIZE,
        *,
        selection: ParentSelection | None = None,
        recombination: StructuredRecombination[G] | None = None,
        mutation: StructuredMutation[G] | Sequence[tuple[StructuredMutation[G], float]] | None = None,
        crossover_probability: float = 1.0,
        convergence_tolerance: float | None = None,
    ) -> None:
        if population_size < 2:
            raise ValueError(f"population_size must be at least 2, got {population_size}")
        if offspring_size is not None and offspring_size < 1:
            raise ValueError(f"offspring_size must be at least 1 (or None for the size the driver asks for), got {offspring_size}")
        if not 0.0 <= crossover_probability <= 1.0:
            raise ValueError(f"crossover_probability must be in [0, 1], got {crossover_probability}")
        self.population_size, self.offspring_size = population_size, offspring_size
        self.selection: ParentSelection = TournamentSelection() if selection is None else selection
        self.recombination = recombination
        self._mutations: list[tuple[StructuredMutation[G], float]] | None = None
        if mutation is not None:
            pairs: list[tuple[StructuredMutation[G], float]]
            if hasattr(mutation, "mutate"):
                pairs = [(cast("StructuredMutation[G]", mutation), 1.0)]
            else:
                pairs = list(cast("Sequence[tuple[StructuredMutation[G], float]]", mutation))
            if not pairs or any(weight < 0 for _, weight in pairs) or sum(weight for _, weight in pairs) <= 0:
                raise ValueError("mutation needs at least one operator, with non-negative probabilities that do not all vanish")
            self._mutations = pairs
        self.crossover_probability = crossover_probability
        self.convergence_tolerance = convergence_tolerance
        self._reset()

    def __repr__(self) -> str:
        mutation = "default" if self._mutations is None else [(m, w) for m, w in self._mutations]
        return (
            f"StructuredGeneticAlgorithm(population_size={self.population_size}, offspring_size={self.offspring_size}, "
            f"selection={self.selection!r}, recombination={self.recombination!r}, mutation={mutation!r}, "
            f"crossover_probability={self.crossover_probability})"
        )

    def _reset(self) -> None:
        self._problem: ProblemSpec[G] | None = None
        self._ctx: StrategyContext | None = None
        self._codec: GenomeCodec[G] | None = None
        self._step = 0
        self._initial_asked = 0
        self._ids: list[int] = []
        self._genomes: list[G] = []
        self._values: Array | None = None
        self._violation: Array | None = None
        self._pending: dict[int, G] = {}
        self._ever_failed = False
        self._operators_mutation: list[tuple[StructuredMutation[G], float]] = []
        self._operators_recombination: StructuredRecombination[G] | None = None

    # --- binding ---

    # --- the ranking policy: what `NSGA2` replaces (design doc §3.3); the defaults are the single-objective ones ---

    def _check_objectives(self, problem: ProblemSpec[G]) -> None:
        if len(problem.objectives) != 1:
            names = tuple(o.name for o in problem.objectives)
            raise ValueError(
                f"StructuredGeneticAlgorithm is single-objective but the problem has {len(problem.objectives)} objectives {names}"
            )

    def _told_values(self, results: EvaluationBatch[G], problem: ProblemSpec[G], backend: Backend) -> Array:
        return results.minimisation_matrix(problem.objectives, backend)[:, 0]

    def _pool_order(self, values: Array, violation: Array, ids: Array, backend: Backend) -> Array:
        """Member indices of a pool (the population followed by the told children), best first."""
        return rank_order(values, violation, ids, backend)

    def bind(self, problem: ProblemSpec[G], ctx: StrategyContext) -> None:
        self._check_objectives(problem)
        codec = cast("GenomeCodec[G] | None", codec_of(problem.space))
        if codec is None:
            raise TypeError(
                "StructuredGeneticAlgorithm needs a search space with a codec (design doc §4.1) to checkpoint its population; "
                "use GeneticAlgorithm for a Box"
            )
        mutations = self._mutations
        recombination = self.recombination
        if isinstance(problem.space, SequenceSpace):
            mutations = mutations or [(cast("StructuredMutation[G]", SequenceMutation()), 1.0)]
            recombination = recombination or cast("StructuredRecombination[G]", SequenceCrossover())
        if not mutations:
            raise ValueError(f"the space {problem.space!r} has no default operators: pass mutation= (and optionally recombination=)")
        self._problem, self._ctx, self._codec = problem, ctx, codec
        self._operators_mutation, self._operators_recombination = mutations, recombination
        used: list[object] = [m for m, _ in mutations]
        if recombination is not None:
            used.append(recombination)
        warn_if_unrecorded(ctx.operators, used)

    def _bound(self) -> tuple[ProblemSpec[G], StrategyContext, GenomeCodec[G]]:
        if self._problem is None or self._ctx is None or self._codec is None:
            raise RuntimeError("StructuredGeneticAlgorithm must be bound to a problem before use: the driver calls bind() first")
        return self._problem, self._ctx, self._codec

    @property
    def size(self) -> int:
        """How many members the population has now (up to `population_size`)."""
        return len(self._ids)

    def ranked_ids(self) -> list[int]:
        """The ids of the population, best first."""
        return list(self._ids)  # the population is stored in rank order

    # --- ask ---

    def ask(self, n: int) -> Batch[G]:
        problem, ctx, _ = self._bound()
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if self._initial_asked < self.population_size:
            count, origin = self.population_size - self._initial_asked, "init"
            self._initial_asked += count
            sampled = self._sample(count)
            ids = [int(ctx.new_id()) for _ in range(count)]
            genomes, parents, origins = sampled, [()] * count, [origin] * count
        elif self._breedable() < 2:
            count = n if self.offspring_size is None else self.offspring_size
            ids = [int(ctx.new_id()) for _ in range(count)]
            genomes, parents, origins = self._sample(count), [()] * count, ["init"] * count
        else:
            count = n if self.offspring_size is None else self.offspring_size
            ids = [int(ctx.new_id()) for _ in range(count)]
            genomes, parents, origins = self._breed(ids)
        step, self._step = self._step, self._step + 1
        for cid, genome in zip(ids, genomes, strict=True):
            self._pending[cid] = genome
        candidates = [
            Candidate(CandidateId(cid), genome, tuple(CandidateId(p) for p in pedigree), origin, step)
            for cid, genome, pedigree, origin in zip(ids, genomes, parents, origins, strict=True)
        ]
        _ = problem
        return ListBatch(candidates)

    def _sample(self, count: int) -> list[G]:
        problem, ctx, _ = self._bound()
        sampled = problem.space.sample_genomes(count, ctx.rng, ctx.backend)
        if is_array(sampled):
            raise TypeError("the space returned an array: StructuredGeneticAlgorithm is for structured genomes, use GeneticAlgorithm")
        genomes = list(cast("Sequence[G]", sampled))
        if len(genomes) != count:
            raise ValueError(f"the space returned {len(genomes)} genomes for a request of {count}")
        return genomes

    def _breedable(self) -> int:
        """How many members can be parents: those that did not fail. Failed members rank last and are never chosen while there
        is an alternative; this only costs anything once an evaluation has failed."""
        if not self._ever_failed:
            return self.size
        assert self._ctx is not None and self._violation is not None
        xp = self._ctx.backend.xp
        return int(xp.sum(xp.astype(xp.isfinite(self._violation), self._ctx.backend.int_dtype)))

    def _view(self) -> PopulationView:
        assert self._ctx is not None and self._values is not None and self._violation is not None
        backend = self._ctx.backend
        order = backend.asarray(list(range(self.size)), dtype=backend.int_dtype)  # stored in rank order: the order is the identity
        return PopulationView(self._values, self._violation, order, order, backend, self._breedable() if self._ever_failed else None)

    def _breed(self, ids: list[int]) -> tuple[list[G], list[tuple[int, ...]], list[str]]:
        problem, ctx, _ = self._bound()
        backend, rng = ctx.backend, ctx.rng
        first, second = self.selection.select(self._view(), len(ids), rng)
        first_index = cast("list[int]", backend.to_numpy(first).tolist())
        second_index = cast("list[int]", backend.to_numpy(second).tolist())
        mutations = self._operators_mutation
        recombination = self._operators_recombination
        genomes: list[G] = []
        parents: list[tuple[int, ...]] = []
        origins: list[str] = []
        for cid, i, j in zip(ids, first_index, second_index, strict=True):
            variation = VariationContext(problem.space, self._codec, ctx.operators, cid)
            a: G = self._genomes[i]
            b: G = self._genomes[j]
            crossed = recombination is not None and (
                self.crossover_probability >= 1.0 or float(backend.to_numpy(rng.uniform((1,)))[0]) < self.crossover_probability
            )
            child: G
            pedigree: tuple[int, ...]
            if crossed:
                assert recombination is not None
                child = recombination.recombine(a, b, rng, variation)
                pedigree, recombined = (self._ids[i], self._ids[j]), recombination.name
            else:
                child, pedigree, recombined = a, (self._ids[i],), "copy"
            mutation = self._pick_mutation(mutations, rng)
            child = mutation.mutate(child, rng, variation)
            genomes.append(child)
            parents.append(pedigree)
            origins.append(f"{self.selection.name}+{recombined}+{mutation.name}")
        return genomes, parents, origins

    @staticmethod
    def _pick_mutation(options: list[tuple[StructuredMutation[G], float]], rng: Any) -> StructuredMutation[G]:
        if len(options) == 1:
            return options[0][0]
        total = sum(weight for _, weight in options)
        # one scalar draw decides on the host which operator runs; `to_numpy` is the way off any device (`np.asarray` of a CUDA or
        # MPS tensor raises), and the same draw is made whichever way it is read
        point = float(rng.backend.to_numpy(rng.uniform((1,)))[0]) * total
        for operator, weight in options:
            point -= weight
            if point < 0:
                return operator
        return options[-1][0]

    # --- tell ---

    def tell(self, results: EvaluationBatch[G]) -> None:
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
        genomes = [self._pending.pop(cid) for cid in told]
        values = self._told_values(results, problem, backend)
        totals = results.violation_list()
        violation = backend.asarray(totals)
        if not self._ever_failed and INFEASIBLE in totals:
            self._ever_failed = True

        ids = self._ids + told
        pooled_genomes = self._genomes + genomes
        index = backend.asarray(ids, dtype=backend.int_dtype)
        if self._values is None or self._violation is None:
            pooled_values, pooled_violation = values, violation
        else:
            pooled_values, pooled_violation = xp.concat([self._values, values], axis=0), xp.concat([self._violation, violation], axis=0)
        keep = backend.to_numpy(self._pool_order(pooled_values, pooled_violation, index, backend)[: self.population_size])
        keep_index = backend.asarray(keep, dtype=backend.int_dtype)
        self._ids = [ids[i] for i in keep.tolist()]
        self._genomes = [pooled_genomes[i] for i in keep.tolist()]
        self._values = xp.take(pooled_values, keep_index, axis=0)
        self._violation = xp.take(pooled_violation, keep_index, axis=0)

    # --- stopping ---

    def should_stop(self) -> bool:
        """False, unless `convergence_tolerance` is set: then true when the objective spread of a full population is below it."""
        if self.convergence_tolerance is None or self._ctx is None or self.size < self.population_size:
            return False
        assert self._values is not None and self._violation is not None
        xp = self._ctx.backend.xp
        valid = xp.isfinite(self._violation)  # members that failed have no value: they say nothing about convergence
        if not bool(xp.any(valid)):
            return False
        spread = xp.max(xp.where(valid, self._values, -xp.inf)) - xp.min(xp.where(valid, self._values, xp.inf))
        return bool(float(spread) < self.convergence_tolerance)

    # --- state_dict ---

    def state_dict(self) -> StateDict:
        """Everything needed to continue identically: the population (genomes through the codec), the pending children, the
        counters and the stream state."""
        _, ctx, codec = self._bound()
        assert self._values is not None or not self._ids
        backend = ctx.backend
        empty = backend.asarray([])
        return {
            "stream": ctx.rng.state_dict(),
            "step": self._step,
            "initial_asked": self._initial_asked,
            "ids": list(self._ids),
            "genomes": [codec.encode(g) for g in self._genomes],  # pyright: ignore[reportAssignmentType]
            "values": self._values if self._values is not None else empty,
            "violation": self._violation if self._violation is not None else empty,
            "pending": [[cid, codec.encode(g)] for cid, g in self._pending.items()],  # pyright: ignore[reportAssignmentType]
        }

    def load_state_dict(self, state: StateDict) -> None:
        _, ctx, codec = self._bound()
        backend = ctx.backend
        expected = {"stream", "step", "initial_asked", "ids", "genomes", "values", "violation", "pending"}
        if set(state) != expected:
            raise ValueError(f"invalid StructuredGeneticAlgorithm state: expected keys {sorted(expected)}, got {sorted(state)}")
        ctx.rng.load_state_dict(cast("dict[str, object]", state["stream"]))
        self._step = cast("int", state["step"])
        self._initial_asked = cast("int", state["initial_asked"])
        self._ids = [int(i) for i in cast("list[int]", state["ids"])]
        self._genomes = [codec.decode(g) for g in cast("list[object]", state["genomes"])]
        self._values = backend.asarray(state["values"]) if self._ids else None
        self._violation = backend.asarray(state["violation"]) if self._ids else None
        self._ever_failed = bool(self._violation is not None and bool(backend.xp.any(backend.xp.isinf(self._violation))))
        self._pending = {int(cid): codec.decode(g) for cid, g in cast("list[list[Any]]", state["pending"])}


__all__ = ["StructuredGeneticAlgorithm"]
