"""Arrays stay on the backend's device through ask → evaluate → tell (design doc §7.2, rule 3).

On a CPU-only machine a torch tensor on "the device" and one that wandered to the host are the same thing, so these tests
can only catch a regression to numpy (an array that left the backend's namespace) or to the wrong dtype, and they run
everywhere. The same checks run on real devices in the GPU smoke suite (`tests/gpu/`), where a silent copy to the host
would show up as a failed check.

Only metadata is allowed to be on the host, and `docs/design/core.md` §7.2 lists it: ids and lineage, statuses, the
evaluations' Python floats (which become arrays again in the `EvaluationBatch` matrices), the aggregated per-candidate
columns, the bounds of a space, and everything that is recorded.
"""

import asyncio

import numpy as np

from auxein.aggregators import Aggregator, cvar_upper, maximum, mean
from auxein.backend import Backend
from auxein.core import EvaluationBatch, IdIssuer, Objective, ProblemSpec, StrategyContext, to_minimisation
from auxein.evaluators import EpisodeEvaluator, VectorisedEvaluator
from auxein.random import RunSeed
from auxein.spaces import Box
from auxein.strategies import GeneticAlgorithm, RandomSearch
from tests.strategies.ga_strategy_test import bind, evaluate, sphere_value
from tests.support import pointmass as pm
from tests.support.fixtures import assert_on_backend, assert_on_device
from tests.support.helpers import array_batch, eval_context


def test_random_streams_produce_arrays_on_the_backend(backend: Backend):
    rng = RunSeed(1).stream("strategy", backend=backend)
    assert_on_backend(rng.uniform((3, 2)), backend)
    assert_on_backend(rng.normal((3, 2)), backend)
    for ints in (rng.integers(0, 5, (4,)), rng.permutation(6), rng.choice(7, 3)):
        assert_on_device(ints, backend)
        assert backend.to_numpy(ints).dtype.kind == "i"


def test_box_samples_are_on_the_backend(backend: Backend):
    box = Box([0.0, 1e-3], [1.0, 1e3], dim=2, log_scale=[False, True])
    samples = box.sample_genomes(5, RunSeed(2).stream("strategy", backend=backend), backend)
    assert_on_backend(samples, backend)


def test_a_genetic_algorithm_keeps_its_population_and_its_batches_on_the_backend(backend: Backend):
    ga, _ = bind(GeneticAlgorithm(population_size=8, offspring_size=6), backend, dim=4)
    for _ in range(6):
        batch = ga.ask(1)
        array = batch.as_array()
        assert array is not None
        assert_on_backend(array, backend)
        ga.tell(EvaluationBatch(evaluate(batch, sphere_value(backend))))
    state = ga.state_dict()
    for name in ("genomes", "values", "violation", "steps"):
        assert_on_backend(state[name], backend)  # type: ignore[arg-type]


def test_random_search_batches_are_on_the_backend(backend: Backend):
    strategy = RandomSearch()
    ctx = StrategyContext(RunSeed(1).stream("strategy", backend=backend), backend, IdIssuer().next)
    strategy.bind(ProblemSpec(Box(-1.0, 1.0, dim=3), (Objective("value"),)), ctx)
    array = strategy.ask(5).as_array()
    assert array is not None
    assert_on_backend(array, backend)


def test_the_minimisation_form_and_the_matrices_of_an_evaluation_batch_are_on_the_backend(backend: Backend):
    ga, _ = bind(GeneticAlgorithm(population_size=4, offspring_size=4), backend)
    batch = ga.ask(1)
    results = EvaluationBatch(evaluate(batch, sphere_value(backend), lambda c: 0.0))
    objectives = (Objective("value", "maximise"),)
    assert_on_backend(results.objectives_matrix(objectives, backend), backend)
    assert_on_backend(results.minimisation_matrix(objectives, backend), backend)
    assert_on_backend(results.constraints_matrix(["cpa"], backend), backend)
    assert_on_backend(results.total_violation(backend), backend)
    assert_on_backend(to_minimisation(results.objectives_matrix(objectives, backend), objectives, backend), backend)


def test_a_vectorised_function_gets_the_genomes_on_the_backend_and_its_array_result_is_used_as_it_is(backend: Backend):
    seen: list[object] = []

    def fn(X):
        seen.append(X)
        return backend.xp.sum(X * X, axis=1)

    spec = ProblemSpec(Box(-5.0, 5.0, dim=3), (Objective("value"),))
    asyncio.run(VectorisedEvaluator(fn).evaluate(array_batch(backend, 4, 3), eval_context(spec, backend)))
    assert_on_backend(seen[0], backend)  # type: ignore[arg-type]


def test_reductions_and_the_aggregator_work_on_the_backend_and_hand_the_host_one_column_per_name(backend: Backend):
    rng = np.random.default_rng(0)
    measured = {"m": backend.asarray(rng.normal(size=(5, 8)))}
    for reduction in (mean("m"), maximum("m"), cvar_upper("m", 0.25)):
        assert_on_backend(reduction.reduce(measured, backend.xp), backend)
    aggregated = Aggregator({"value": mean("m")}, {"c": maximum("m")}).aggregate(measured, backend)
    # the aggregated columns are the evaluations' numbers: host float64, one per candidate, by design
    for column in (*aggregated.objectives.values(), *aggregated.constraints.values()):
        assert isinstance(column, np.ndarray) and column.dtype == np.float64 and column.shape == (5,)


def test_the_batched_episode_path_runs_on_the_backend(backend: Backend):
    seen: list[object] = []

    class Spy(pm.PointMassEnvironment):
        def run_batch(self, agents, scenarios, rng):
            seen.append((agents, rng.uniform((2,))))
            return super().run_batch(agents, scenarios, rng)

    gains = backend.asarray(np.array([[2.0, 3.0, 0.0], [5.0, 1.0, 0.1], [1.0, 1.0, -0.2]]))
    from auxein.core import ArrayBatch, CandidateId

    batch = ArrayBatch(gains, [CandidateId(i) for i in range(3)], 0, "random")
    evaluator = EpisodeEvaluator(pm.GainsDecoder(), Spy(), pm.scenario_set(5), pm.aggregator())
    results = asyncio.run(evaluator.evaluate(batch, eval_context(pm.problem(), backend)))
    assert len(results) == 3
    agents, draws = seen[0]  # type: ignore[misc]
    assert_on_backend(agents, backend)
    assert_on_backend(draws, backend)
