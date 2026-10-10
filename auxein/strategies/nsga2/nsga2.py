"""`NSGA2`: the non-dominated sorting genetic algorithm (Deb et al., 2002) on arrays and on structured genomes."""

from collections.abc import Sequence
from typing import Any, Generic, cast

from auxein.backend import Array, Backend
from auxein.core import (
    Batch,
    EvaluationBatch,
    ProblemSpec,
    StateDict,
    StrategyCapabilities,
    StrategyContext,
)
from auxein.core._typing import G
from auxein.spaces import Box, MixedSpace, codec_of
from auxein.strategies.ga.base import BoundsRepair, Mutation, ParentSelection, Recombination
from auxein.strategies.ga.genetic_algorithm import GeneticAlgorithm
from auxein.strategies.ga.mixed import BitFlipMutation, CategoricalMutation, IntegerMutation
from auxein.strategies.ga.mutation import PolynomialMutation
from auxein.strategies.ga.recombination import SimulatedBinaryCrossover
from auxein.strategies.ga.repair import ClipRepair
from auxein.strategies.ga.selection import SigmaScalingSUS, TournamentSelection
from auxein.strategies.nsga2.sorting import crowded_order
from auxein.strategies.structured.genetic_algorithm import StructuredGeneticAlgorithm
from auxein.strategies.structured.variation import StructuredMutation, StructuredRecombination

DEFAULT_POPULATION_SIZE = 100
DEFAULT_CROSSOVER_PROBABILITY = 0.9


class _ArrayEngine(GeneticAlgorithm):
    """The numeric genetic algorithm with NSGA-II's ranking policy (see the hooks of `GeneticAlgorithm`).

    Everything else, breeding with the real, mixed or discrete operators, pending children, "plus" survivor selection with
    the survivors gathered once, is the GA's own code. The population is ranked by the crowded-comparison order of the *pool*
    it was selected from (that is Deb's algorithm: the fronts and crowding distances of the pooled population are what
    parents are chosen by), which depends on the pool and not only on the survivors, so the order is part of the state.
    """

    capabilities = StrategyCapabilities(max_objectives=None, supports_constraints=True, tell_mode="both")
    _extra_keys = ("order",)
    _skip_unchanged_tell = False  # the pool's fronts and crowding change even when every child lost

    _objective_count = 1

    def _check_objectives(self, problem: ProblemSpec[Array]) -> None:
        self._objective_count = len(problem.objectives)

    def _empty_values(self, backend: Backend) -> Array:
        return backend.xp.zeros((0, self._objective_count), dtype=backend.dtype, device=backend.device)

    def _told_values(self, results: EvaluationBatch[Array], problem: ProblemSpec[Array], backend: Backend) -> Array:
        return results.minimisation_matrix(problem.objectives, backend)

    def _pool_order(self, values: Array, violation: Array, ids: Array, backend: Backend) -> Array:
        return crowded_order(values, violation, ids, backend)[0]

    def _population_order(self) -> Array:
        assert self._order is not None
        return self._order

    def _extra_state(self) -> StateDict:
        return {"order": self._order}

    def _load_extra_state(self, state: StateDict) -> None:
        assert self._ctx is not None
        self._order = self._ctx.backend.asarray(state["order"], dtype=self._ctx.backend.int_dtype)

    def should_stop(self) -> bool:
        return False


class _StructuredEngine(StructuredGeneticAlgorithm[Any]):
    """The structured genetic algorithm with NSGA-II's ranking policy. It stores its population in rank order, so the
    crowded-comparison order of the pool is all it needs."""

    capabilities = StrategyCapabilities(max_objectives=None, supports_constraints=True, tell_mode="both")

    def _check_objectives(self, problem: ProblemSpec[Any]) -> None:
        return None

    def _told_values(self, results: EvaluationBatch[Any], problem: ProblemSpec[Any], backend: Backend) -> Array:
        return results.minimisation_matrix(problem.objectives, backend)

    def _pool_order(self, values: Array, violation: Array, ids: Array, backend: Backend) -> Array:
        return crowded_order(values, violation, ids, backend)[0]

    def should_stop(self) -> bool:
        return False


class NSGA2(Generic[G]):
    """NSGA-II, the standard multi-objective genetic algorithm (Deb, Pratap, Agarwal and Meyarivan, 2002).

    **Survivors.** "Plus" selection: the population (μ members) is pooled with the told children and the best μ are kept by
    **non-dominated sorting**, then by **crowding distance** within the last front that fits. Dominance is *constrained*
    (Deb's rule): a feasible member dominates every infeasible one, among infeasible members the lower total violation
    dominates, and among feasible ones ordinary Pareto domination on the objectives in **minimisation form** (directions are
    converted once, by the `EvaluationBatch`, never by hand). Failed evaluations (NaN objectives, infinite violation) end in
    the last front and are never parents while there is an alternative. Crowding distance is computed per front with every
    objective normalised by its range in the front, boundary members infinite and degenerate ranges counting for nothing.

    **Parents** are chosen by binary tournament on the *crowded-comparison order*: lower front, then larger crowding distance,
    then lower candidate id. That is a total order, kept as the `order` and `rank` of a `PopulationView`, so the tournament is
    the GA's own `TournamentSelection`, unchanged. Selections that need a scalar fitness (`SigmaScalingSUS`) are an error.

    **The same contract as the GAs:** exactly `offspring_size` children (or `n` when it is None), no self-mating, every
    candidate evaluated once, pending children kept, `state_dict` continuing identically, results told in generations or one
    at a time (`tell_mode="both"`). With steady-state delivery the result depends on how many children are told at once,
    because fronts and crowding are recomputed at every `tell`; in deterministic mode that is deterministic (design doc §8.1).
    It accepts any number of objectives (with one, it is a GA with plus selection) and constraints.

    **Spaces and operators.** On a `Box`, children are made by `SimulatedBinaryCrossover` (η = 15; probability 0.9 per pair)
    and `PolynomialMutation` (η = 20, probability 1/d per gene), then clipped to the bounds. On a `MixedSpace` the real genes
    get the same operators and the discrete genes the type-aware ones of the numeric GA (`integer_mutation`, `binary_mutation`,
    `categorical_mutation`, discrete recombination). On a `SequenceSpace` (or any space with a codec and operators of
    your own) the structured GA's variation is used: `recombination` and `mutation` are then structured operators.
    The defaults are the textbook ones, so the algorithm is comparable with other implementations: population 100, offspring 100.

    **Limit.** The domination relation is an `(n, n)` boolean matrix over the pool (population plus offspring), so memory
    grows with the square of the pool: fine to a few thousand members (about 50 MB for 4,000 members and 3 objectives).
    Sorting and crowding run on the backend; the only Python loop is over the fronts.
    """

    capabilities = StrategyCapabilities(max_objectives=None, supports_constraints=True, tell_mode="both")

    def __init__(
        self,
        population_size: int = DEFAULT_POPULATION_SIZE,
        offspring_size: int | None = DEFAULT_POPULATION_SIZE,
        *,
        selection: ParentSelection | None = None,
        recombination: Recombination | StructuredRecombination[G] | None = None,
        mutation: Mutation | StructuredMutation[G] | Sequence[tuple[StructuredMutation[G], float]] | None = None,
        repair: BoundsRepair | None = None,
        crossover_probability: float = DEFAULT_CROSSOVER_PROBABILITY,
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
        if isinstance(selection, SigmaScalingSUS):
            raise ValueError(
                "NSGA2 chooses parents by the crowded-comparison order, so it needs a rank-based selection such as "
                "TournamentSelection: SigmaScalingSUS weighs members by a scalar fitness, which a multi-objective ranking does not have"
            )
        self.population_size, self.offspring_size = population_size, offspring_size
        self.selection: ParentSelection = TournamentSelection() if selection is None else selection
        self.recombination, self.mutation, self.repair = recombination, mutation, repair
        self.crossover_probability = crossover_probability
        self._discrete = (integer_mutation, binary_mutation, categorical_mutation)
        self._engine: _ArrayEngine | _StructuredEngine | None = None

    def __repr__(self) -> str:
        names = ("integer_mutation", "binary_mutation", "categorical_mutation")
        discrete = "".join(f", {name}={op!r}" for name, op in zip(names, self._discrete, strict=True) if op is not None)
        return (
            f"NSGA2(population_size={self.population_size}, offspring_size={self.offspring_size}, selection={self.selection!r}, "
            f"recombination={self.recombination!r}, mutation={self.mutation!r}, repair={self.repair!r}, "
            f"crossover_probability={self.crossover_probability}{discrete})"
        )

    # --- binding: the space decides the engine ---

    def bind(self, problem: ProblemSpec[G], ctx: StrategyContext) -> None:
        space = problem.space
        engine: _ArrayEngine | _StructuredEngine
        if isinstance(space, (Box, MixedSpace)):
            engine = _ArrayEngine(
                self.population_size,
                self.offspring_size,
                selection=self.selection,
                recombination=self._array_operator(self.recombination, "recombination", SimulatedBinaryCrossover()),
                mutation=self._array_operator(self.mutation, "mutation", PolynomialMutation()),
                repair=self.repair if self.repair is not None else ClipRepair(),
                crossover_probability=self.crossover_probability,
                integer_mutation=self._discrete[0],
                binary_mutation=self._discrete[1],
                categorical_mutation=self._discrete[2],
            )
        elif codec_of(space) is not None:
            if self.repair is not None or any(op is not None for op in self._discrete):
                raise TypeError(
                    "repair and the integer, binary and categorical mutations are for Box and MixedSpace, not for " + type(space).__name__
                )
            engine = _StructuredEngine(
                self.population_size,
                self.offspring_size,
                selection=self.selection,
                recombination=cast("StructuredRecombination[Any] | None", self._structured_operator(self.recombination, "recombination")),
                mutation=cast("Any", self._structured_operator(self.mutation, "mutation")),
                crossover_probability=self.crossover_probability,
            )
        else:
            raise TypeError(f"NSGA2 needs a Box, a MixedSpace or a space with a codec such as SequenceSpace, got {type(space).__name__}")
        engine.bind(cast("Any", problem), ctx)
        self._engine = engine

    @staticmethod
    def _array_operator(given: object, what: str, default: Any) -> Any:
        if given is None:
            return default
        if what == "recombination" and not hasattr(given, "weights"):
            raise TypeError(f"{what} {given!r} is a structured operator, but the space is made of arrays")
        if what == "mutation" and not hasattr(given, "adaptive"):
            raise TypeError(f"{what} {given!r} is not a mutation of the numeric genetic algorithm, but the space is made of arrays")
        return given

    @staticmethod
    def _structured_operator(given: object, what: str) -> object:
        if given is None:
            return None
        if what == "recombination" and hasattr(given, "weights"):
            raise TypeError(f"{what} {given!r} works on arrays, but the space has structured genomes")
        if what == "mutation" and hasattr(given, "adaptive"):
            raise TypeError(f"{what} {given!r} works on arrays, but the space has structured genomes")
        return given

    def _bound(self) -> _ArrayEngine | _StructuredEngine:
        if self._engine is None:
            raise RuntimeError("NSGA2 must be bound to a problem before use: the driver calls bind() first")
        return self._engine

    # --- the contract ---

    @property
    def size(self) -> int:
        """How many members the population holds now."""
        return 0 if self._engine is None else self._engine.size

    def ranked_ids(self) -> list[int]:
        """The ids of the population in the crowded-comparison order, best first."""
        return self._bound().ranked_ids()

    def ask(self, n: int) -> Batch[G]:
        return cast("Batch[G]", self._bound().ask(n))

    def tell(self, results: EvaluationBatch[G]) -> None:
        self._bound().tell(cast("EvaluationBatch[Any]", results))

    def should_stop(self) -> bool:
        return False

    def state_dict(self) -> StateDict:
        return self._bound().state_dict()

    def load_state_dict(self, state: StateDict) -> None:
        self._bound().load_state_dict(state)
