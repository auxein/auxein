"""`EpisodeEvaluator`: a decoder, an environment, a scenario set and an aggregator, as an evaluator (design doc §6.1)."""

import asyncio
import dataclasses
import inspect
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, cast

import numpy as np

from auxein.aggregators import Aggregator
from auxein.backend import Array, Backend
from auxein.core import (
    Batch,
    BatchResult,
    Candidate,
    Cost,
    EpisodeRecords,
    EvalContext,
    Evaluation,
    EvaluationBatch,
    RawRef,
    Status,
    describe_exception,
    evaluations_from_batch_return,
)
from auxein.core._typing import G
from auxein.environments import (
    Decoder,
    Environment,
    EpisodeBatchResult,
    EpisodeFailure,
    EpisodeResult,
    Scenario,
    ScenarioSet,
)
from auxein.evaluators.errors import EvaluationError
from auxein.evaluators.failures import await_with_timeout, failures_of
from auxein.execution import EvaluationTimeout, ExecutorError

_SHOWN = 3
"""How many failing scenarios of a candidate are described in full in its error; the rest are only counted."""


def describe_component(component: object) -> str:
    """A description of a component that is the same in every process, so that resume can compare them.

    The component's own `describe()` if it has one, its `repr` if it defines one, and otherwise its class: the default `repr`
    of an object holds its memory address, which changes from run to run.
    """
    custom = getattr(component, "describe", None)
    if callable(custom):
        return str(custom())
    if type(component).__repr__ is not object.__repr__:
        return repr(component)
    return f"{type(component).__module__}.{type(component).__qualname__}"


@dataclass
class _Outcome:
    """What the episodes of a batch produced, before the aggregator turns them into evaluations."""

    names: tuple[str, ...]
    arrays: Mapping[str, Array]
    """Measurement name to an `(n, s)` array on the backend. The entries of an episode that did not succeed are not meaningful."""
    failures: dict[tuple[int, int], EpisodeFailure]
    walls: list[float]
    """Seconds spent on each candidate."""
    undecodable: dict[int, str]
    """Rows whose genome could not be decoded: they ran no episode at all."""


class EpisodeEvaluator(Generic[G]):
    """Evaluates candidates by running their agents in every scenario of a scenario set and aggregating the measurements.

    `decoder.decode(genome)` makes the agent, `environment` runs an episode per scenario and returns raw measurements, and the
    `aggregator` turns each candidate's measurements into the objectives, constraints and descriptors of its `Evaluation`.
    This version evolves **one role** per episode (`role`, which can be omitted when the environment has a single role); the
    other participants belong to the scenario. Agents are passed to the environment by role, so several evolved roles will not
    change the interface.

    **Two paths, one aggregator.** If the environment has `run_batch` and the decoder `decode_batch`, and the batch is
    array-backed, the **batched path** decodes the whole batch, calls `run_batch` once with all scenarios and aggregates the
    `(n, s)` arrays it returns, on the driver's thread and on the backend's device. Otherwise the **per-episode path** decodes
    each candidate once and runs every scenario as one call of `environment.run_episode`, through `ctx.call` for a synchronous
    environment (so `executor=` applies) and natively for an `async def` one, with **at most `ctx.concurrency` episodes in
    progress across the whole batch** (candidates times scenarios) and results returned in ask order. Its results are stacked
    into the same `(n, s)` arrays. With `executor="process"` the environment, the decoded agents and the scenarios must be
    picklable.

    **Randomness.** The world's comes from the scenario's own seed (`scenario.rng()`), so every candidate faces the same
    realisation. The agent's comes from `rng`, a stream per (candidate, scenario) derived from the run seed and numpy-backed
    (on the batched path, one stream per batch). See design doc §8.

    **Failures** (design doc §6.6). A candidate fails if any of its episodes fails: it is then a `FAILED` (or, if every
    failing episode timed out, `TIMEOUT`) evaluation whose `error` lists the failing scenarios and their errors, and the run's
    `failure_policy` applies as for any evaluator. The measurements of its successful episodes are still recorded. An
    exception in the environment is a failed episode under `infeasible` and stops the run under `fail_fast`. **Timeouts** apply
    per episode on the per-episode path; the batched path cannot time out and rejects a timeout.

    Per-scenario measurements travel with the evaluations (`EvaluationBatch.episodes`) so that the recorder keeps them.
    """

    def __init__(
        self,
        decoder: Decoder[G],
        environment: Environment,
        scenarios: ScenarioSet,
        aggregator: Aggregator,
        *,
        role: str | None = None,
    ) -> None:
        roles = tuple(environment.roles)
        if role is None:
            if len(roles) != 1:
                raise ValueError(f"the environment has the roles {roles}: say which one the evolved agent plays with role=")
            role = roles[0]
        elif role not in roles:
            raise ValueError(f"the environment has the roles {roles}, not {role!r}")
        self._decoder, self._environment, self._scenarios, self._aggregator, self._role = decoder, environment, scenarios, aggregator, role
        self._scenario_list: tuple[Scenario, ...] = tuple(scenarios)
        self._is_async = inspect.iscoroutinefunction(getattr(environment, "run_episode", None))

    @property
    def batched(self) -> bool:
        """Whether this evaluator takes the batched path for an array-backed batch."""
        return hasattr(self._environment, "run_batch") and hasattr(self._decoder, "decode_batch")

    @property
    def batch_sensitive(self) -> bool:
        """The batched path draws randomness per batch, so a batch that is evaluated again is evaluated whole (design doc §10.4)."""
        return self.batched

    def __repr__(self) -> str:
        return (
            f"EpisodeEvaluator(environment={describe_component(self._environment)}, decoder={describe_component(self._decoder)}, "
            f"aggregator={self._aggregator.describe()}, role={self._role!r}, "
            f"scenarios={len(self._scenarios)}:{self._scenarios.fingerprint[:16]})"
        )

    # --- evaluate ---

    async def evaluate(self, batch: Batch[G], ctx: EvalContext[G]) -> EvaluationBatch[G]:
        if ctx.episode_rng_for is None or ctx.episode_batch_rng_for is None:
            raise ValueError("the EvalContext has no episode streams: the driver provides them, a hand-built context must too")
        self._aggregator.validate(ctx.problem)  # type: ignore[arg-type]
        candidates = list(batch.candidates)
        if not candidates:
            return EvaluationBatch([])
        genomes = batch.as_array()
        if self.batched and genomes is not None:
            return await self._batched(candidates, genomes, ctx)
        if self._is_async and ctx.executor.kind == "process":
            raise ValueError(
                f"{self!r} has an async environment, which always runs on the driver's event loop and cannot be sent to worker "
                "processes: use executor='thread' (or 'auto'), or make the environment synchronous"
            )
        if not hasattr(self._environment, "run_episode"):
            raise TypeError(f"{describe_component(self._environment)} has no run_episode, and this batch cannot take the batched path")
        return await self._per_episode(candidates, ctx)

    # --- the batched path ---

    async def _batched(self, candidates: list[Candidate[G]], genomes: Array, ctx: EvalContext[G]) -> EvaluationBatch[G]:
        if ctx.timeout is not None:
            raise ValueError(
                "a batched EpisodeEvaluator does not support a timeout: it makes one call per batch on the driver's thread, which "
                "nothing can interrupt. Use an environment without run_batch (or a decoder without decode_batch) to time "
                "episodes out"
            )
        backend = ctx.backend
        n, s = len(candidates), len(self._scenario_list)
        start = time.perf_counter()
        try:
            agents = cast("Any", self._decoder).decode_batch(genomes)
            returned = cast("Any", self._environment).run_batch(
                agents, self._scenario_list, cast("Any", ctx.episode_batch_rng_for)(candidates[0].id)
            )
            if inspect.isawaitable(returned):
                returned = await returned
        except Exception as error:
            return EvaluationBatch(failures_of(candidates, error, time.perf_counter() - start, ctx))
        wall = time.perf_counter() - start
        if not isinstance(returned, EpisodeBatchResult):
            raise TypeError(f"run_batch must return an EpisodeBatchResult, got {type(returned).__name__}")
        if returned.shape != (n, s):
            raise ValueError(f"run_batch returned measurements of shape {returned.shape} for {n} candidates and {s} scenarios")
        measurements = {name: backend.asarray(array) for name, array in returned.measurements.items()}
        outcome = _Outcome(tuple(sorted(measurements)), measurements, dict(returned.failures), [wall / n] * n, {})
        return self._evaluations(candidates, outcome, ctx)

    # --- the per-episode path ---

    async def _per_episode(self, candidates: list[Candidate[G]], ctx: EvalContext[G]) -> EvaluationBatch[G]:
        scenarios = self._scenario_list
        n, s = len(candidates), len(scenarios)
        agents: dict[int, Mapping[str, object]] = {}
        undecodable: dict[int, str] = {}
        for row, candidate in enumerate(candidates):
            try:
                agents[row] = {self._role: self._decoder.decode(candidate.genome)}
            except Exception as error:
                if ctx.failure_policy == "fail_fast":
                    raise EvaluationError([candidate.id], error) from error
                undecodable[row] = f"decoding the genome failed: {describe_exception(error)}"

        results: dict[tuple[int, int], EpisodeResult] = {}
        walls = [0.0] * n
        sequential = ctx.concurrency == 1 and ctx.executor.kind == "inline" and ctx.timeout is None
        pairs = [(row, scenario) for row in agents for scenario in range(s)]
        if sequential:
            for row, scenario in pairs:
                result, wall = await self._run_episode(candidates[row], agents[row], scenarios[scenario], ctx)
                results[(row, scenario)] = result
                walls[row] += wall
        else:
            slots = asyncio.Semaphore(ctx.concurrency)

            async def bounded(row: int, scenario: int) -> tuple[EpisodeResult, float]:
                async with slots:  # a slot is held while the run waits for this episode, and no longer
                    return await self._run_episode(candidates[row], agents[row], scenarios[scenario], ctx)

            tasks = [asyncio.ensure_future(bounded(row, scenario)) for row, scenario in pairs]
            try:
                returned = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)  # leave nothing running behind us
                raise
            for (row, scenario), (result, wall) in zip(pairs, returned, strict=True):
                results[(row, scenario)] = result
                walls[row] += wall

        outcome = self._stack(candidates, results, s, walls, undecodable, ctx.backend)
        return self._evaluations(candidates, outcome, ctx)

    async def _run_episode(
        self, candidate: Candidate[G], agents: Mapping[str, object], scenario: Scenario, ctx: EvalContext[G]
    ) -> tuple[EpisodeResult, float]:
        """One episode: the result (a failed one if the environment raised or timed out) and the seconds it took."""
        start = time.perf_counter()
        environment = self._environment
        try:
            rng = cast("Any", ctx.episode_rng_for)(candidate.id, scenario.index)
            returned: object
            if self._is_async:
                returned = await await_with_timeout(cast("Any", environment.run_episode(agents, scenario, rng)), ctx.timeout)
            else:
                returned = await ctx.call(environment.run_episode, agents, scenario, rng)
                if inspect.isawaitable(returned):  # a synchronous callable that hands back an awaitable
                    returned = await returned
        except Exception as error:
            if isinstance(error, ExecutorError) or ctx.failure_policy == "fail_fast":
                raise EvaluationError([candidate.id], error) from error
            wall = time.perf_counter() - start
            if isinstance(error, EvaluationTimeout):
                return EpisodeResult.failed(f"timed out: {error}", Status.TIMEOUT), error.timeout
            return EpisodeResult.failed(describe_exception(error)), wall
        if not isinstance(returned, EpisodeResult):
            raise TypeError(
                f"the environment's run_episode must return an EpisodeResult, got {type(returned).__name__} "
                f"(candidate {candidate.id}, scenario {scenario.id!r})"
            )
        return returned, time.perf_counter() - start

    @staticmethod
    def _stack(
        candidates: list[Candidate[G]],
        results: Mapping[tuple[int, int], EpisodeResult],
        s: int,
        walls: list[float],
        undecodable: dict[int, str],
        backend: Backend,
    ) -> _Outcome:
        """Stack the episode results into `(n, s)` arrays, one per measurement name."""
        n = len(candidates)
        failures: dict[tuple[int, int], EpisodeFailure] = {}
        names: tuple[str, ...] | None = None
        first_seen: tuple[int, int] | None = None
        for (row, scenario), result in results.items():
            if result.status is not Status.OK:
                failures[(row, scenario)] = EpisodeFailure(result.status, result.error or "")
                continue
            if names is None:
                names, first_seen = tuple(sorted(result.measurements)), (row, scenario)
            elif set(result.measurements) != set(names) and first_seen is not None:
                raise ValueError(
                    f"episodes of one batch must report the same measurements: candidate {candidates[row].id} in scenario "
                    f"{scenario} reported {sorted(result.measurements)} but candidate {candidates[first_seen[0]].id} in scenario "
                    f"{first_seen[1]} reported {list(names)}"
                )
        names = names or ()
        # host-side by design (design doc §7.2): per-episode results are Python floats produced one at a time by user code, so
        # they are collected on the host and moved to the backend once per measurement name, as the `(n, s)` arrays the
        # aggregator reduces on the device
        grids = {name: np.full((n, s), np.nan) for name in names}
        for (row, scenario), result in results.items():
            if result.status is Status.OK:
                for name in names:
                    grids[name][row, scenario] = result.measurements[name]
        arrays = {name: backend.asarray(grid) for name, grid in grids.items()}
        return _Outcome(names, arrays, failures, walls, undecodable)

    # --- from measurements to evaluations ---

    def _evaluations(self, candidates: list[Candidate[G]], outcome: _Outcome, ctx: EvalContext[G]) -> EvaluationBatch[G]:
        backend = ctx.backend
        n, s = len(candidates), len(self._scenario_list)
        by_row: dict[int, list[tuple[int, EpisodeFailure]]] = {}
        for (row, scenario), failure in sorted(outcome.failures.items()):
            by_row.setdefault(row, []).append((scenario, failure))
        broken = set(by_row) | set(outcome.undecodable)
        ok_rows = [row for row in range(n) if row not in broken]

        evaluated: dict[int, Evaluation[G]] = {}
        if ok_rows:
            xp = backend.xp
            index = backend.asarray(ok_rows, dtype=backend.int_dtype)
            subset = {name: (array if len(ok_rows) == n else xp.take(array, index, axis=0)) for name, array in outcome.arrays.items()}
            try:
                aggregated = self._aggregator.aggregate(subset, backend)
            except KeyError as error:
                raise ValueError(str(error.args[0])) from None
            invalid = {ok_rows[i]: text for i, text in aggregated.invalid.items()}
            batch_result = BatchResult(
                aggregated.objectives,
                _zeroed(aggregated.constraints, aggregated.invalid),
                aggregated.descriptors,
                _zeroed(aggregated.cost, aggregated.invalid),
            )
            ok_candidates = [candidates[row] for row in ok_rows]
            for row, evaluation in zip(ok_rows, evaluations_from_batch_return(batch_result, ok_candidates, ctx.problem, 0.0), strict=True):
                if row in invalid:
                    evaluated[row] = Evaluation.failed(candidates[row], Status.FAILED, f"non-finite value: {invalid[row]}")
                else:
                    evaluated[row] = evaluation

        final: list[Evaluation[G]] = []
        for row, candidate in enumerate(candidates):
            wall = outcome.walls[row]
            if row in outcome.undecodable:
                final.append(Evaluation.failed(candidate, Status.FAILED, outcome.undecodable[row], wall))
            elif row in by_row:
                status, error = _candidate_failure(by_row[row], self._scenario_list, s)
                final.append(Evaluation.failed(candidate, status, error, wall))
            else:
                evaluation = evaluated[row]
                raw = RawRef(f"episodes/{candidate.id}")
                final.append(dataclasses.replace(evaluation, cost=Cost(wall, evaluation.cost.units), raw=raw))
        return EvaluationBatch(final, self._records(candidates, outcome, backend))

    def _records(self, candidates: list[Candidate[G]], outcome: _Outcome, backend: Backend) -> EpisodeRecords | None:
        """The per-scenario measurements of the candidates that ran episodes, for the recorder."""
        rows = [row for row in range(len(candidates)) if row not in outcome.undecodable]
        if not rows:
            return None
        s = len(self._scenario_list)
        # host-side by design (design doc §7.2): the recorder writes the per-scenario measurements to SQLite, so this is the one
        # place where the measurements of a batched run leave the device, once per batch
        host = [backend.to_numpy(outcome.arrays[name]) for name in outcome.names]
        values = np.stack(host, axis=-1) if host else np.zeros((len(candidates), s, 0))  # pyright: ignore[reportUnknownMemberType]
        values = np.asarray(values, dtype=np.float64)[rows]
        position = {row: i for i, row in enumerate(rows)}
        failures = {(position[row], scenario): (f.status, f.error) for (row, scenario), f in outcome.failures.items() if row in position}
        for key in failures:  # a failed episode has nothing to record
            values[key] = np.nan
        return EpisodeRecords(
            tuple(candidates[row].id for row in rows), tuple(sc.id for sc in self._scenario_list), outcome.names, values, failures
        )


def _zeroed(columns: Mapping[str, Any], invalid: Mapping[int, str]) -> dict[str, Any]:
    """The columns with the invalid rows set to zero: those candidates fail, and a `BatchResult` insists on finite values."""
    if not invalid:
        return dict(columns)
    rows = list(invalid)  # the columns are the aggregator's host columns (see `Aggregator.aggregate`), so numpy is right here
    out: dict[str, Any] = {}
    for name, column in columns.items():
        copy = np.array(column, copy=True)
        copy[rows] = 0.0
        out[name] = copy
    return out


def _candidate_failure(failures: Sequence[tuple[int, EpisodeFailure]], scenarios: Sequence[Scenario], total: int) -> tuple[Status, str]:
    """The status and error of a candidate with failed episodes: TIMEOUT only if every failing episode timed out."""
    status = Status.TIMEOUT if all(f.status is Status.TIMEOUT for _, f in failures) else Status.FAILED
    lines = [f"{len(failures)} of {total} episodes did not succeed:"]
    for scenario, failure in failures[:_SHOWN]:
        lines.append(f"- scenario {scenarios[scenario].id!r} ({failure.status.value}): {failure.error}")
    if len(failures) > _SHOWN:
        rest = ", ".join(repr(scenarios[sc].id) for sc, _ in failures[_SHOWN:])
        lines.append(f"- and in scenario(s) {rest}")
    return status, "\n".join(lines)
