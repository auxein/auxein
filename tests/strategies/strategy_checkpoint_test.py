"""A strategy saved through the real checkpoint format and loaded into a fresh one continues byte for byte (design doc §10.4)."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import Candidate, CandidateId, Cost, Evaluation, EvaluationBatch, IdIssuer, Status, validate_state_dict
from auxein.random import RunSeed
from auxein.recording import checkpoints
from auxein.strategies import GeneticAlgorithm, RandomSearch
from auxein.strategies.ga import GaussianMutation, SigmaScalingSUS, UniformRecombination
from tests.strategies.ga_strategy_test import bind, evaluate, sphere_value


def through_a_checkpoint(state: Any, directory: Path, backend: Backend) -> Any:
    checkpoints.write(directory, 1, state)
    return checkpoints.read(directory / "ckpt-1", backend)[1]


def snapshot(batch: Any, backend: Backend) -> tuple[Any, ...]:
    return (tuple(batch.ids), backend.to_numpy(batch.as_array()).copy(), tuple(batch.parents), tuple(batch.origins), batch.step)


def run_on(strategy: Any, backend: Backend, rounds: int, per_round: int = 1) -> list[Any]:
    """Ask and tell for a few rounds; a candidate fails now and then. Returns what was asked."""
    value = sphere_value(backend)
    out: list[Any] = []
    for _ in range(rounds):
        batch = strategy.ask(per_round)
        out.append(snapshot(batch, backend))
        evaluations = evaluate(batch, value)
        evaluations = [Evaluation.failed(e.candidate, Status.FAILED, "boom") if e.candidate.id % 7 == 3 else e for e in evaluations]
        strategy.tell(EvaluationBatch(evaluations))
    return out


def same(a: list[Any], b: list[Any]) -> None:
    assert len(a) == len(b)
    for left, right in zip(a, b, strict=True):
        assert left[0] == right[0] and left[2] == right[2] and left[3] == right[3] and left[4] == right[4]
        np.testing.assert_array_equal(left[1], right[1])


GA_VARIANTS = [
    {},
    {"mutation": GaussianMutation(0.1), "selection": SigmaScalingSUS()},
    {"recombination": UniformRecombination(), "crossover_probability": 0.6},
]


@pytest.mark.parametrize("kwargs", GA_VARIANTS, ids=["default", "gaussian-sus", "uniform"])
def test_a_genetic_algorithm_restored_from_a_checkpoint_continues_byte_identically(
    tmp_path: Path, backend: Backend, kwargs: dict[str, Any]
):
    ga, issuer = bind(GeneticAlgorithm(population_size=8, offspring_size=5, **kwargs), backend, seed=21)
    run_on(ga, backend, 4)  # initial population, then children; some evaluations failed
    pending = ga.ask(1)  # asked, not told: part of the state
    half = EvaluationBatch(evaluate(pending, sphere_value(backend))[:2])
    ga.tell(half)  # and part of the block told, as in steady-state delivery
    state = through_a_checkpoint(ga.state_dict(), tmp_path / "c", backend)
    validate_state_dict(state)

    restored, _ = bind(GeneticAlgorithm(population_size=8, offspring_size=5, **kwargs), backend, seed=21, issuer=IdIssuer(issuer.issued))
    restored.load_state_dict(state)

    assert restored.ranked_ids() == ga.ranked_ids() and restored.size == ga.size
    rest = EvaluationBatch(evaluate(pending, sphere_value(backend))[2:])
    ga.tell(rest)
    restored.tell(rest)  # the block that was part told can still be completed after the restore
    same(run_on(ga, backend, 6), run_on(restored, backend, 6))
    assert restored.ranked_ids() == ga.ranked_ids()
    # and the states they end in are the same
    for name in ("ids", "stream", "step", "initial_asked"):
        assert restored.state_dict()[name] == ga.state_dict()[name]


def test_a_genetic_algorithm_restored_after_failures_keeps_ranking_failed_members_last(tmp_path: Path, backend: Backend):
    ga, issuer = bind(GeneticAlgorithm(population_size=6, offspring_size=6), backend, seed=2)
    batch = ga.ask(1)
    value = sphere_value(backend)
    evaluations = [Evaluation.failed(e.candidate, Status.FAILED, "boom") if i < 4 else e for i, e in enumerate(evaluate(batch, value))]
    ga.tell(EvaluationBatch(evaluations))  # four of six members have no value
    state = through_a_checkpoint(ga.state_dict(), tmp_path / "c", backend)
    restored, _ = bind(GeneticAlgorithm(population_size=6, offspring_size=6), backend, seed=2, issuer=IdIssuer(issuer.issued))
    restored.load_state_dict(state)
    assert restored.ranked_ids() == ga.ranked_ids() and set(restored.ranked_ids()[:2]) == {4, 5}
    same(run_on(ga, backend, 3), run_on(restored, backend, 3))  # both breed from the two valid members, not from the failed ones


def test_random_search_restored_from_a_checkpoint_continues_byte_identically(tmp_path: Path, backend: Backend):
    search, issuer = bind(RandomSearch(), backend, seed=5)  # type: ignore[arg-type]
    run_on(search, backend, 3, per_round=4)
    state = through_a_checkpoint(search.state_dict(), tmp_path / "c", backend)
    restored, _ = bind(RandomSearch(), backend, seed=5, issuer=IdIssuer(issuer.issued))  # type: ignore[arg-type]
    restored.load_state_dict(state)
    same(run_on(search, backend, 4, per_round=3), run_on(restored, backend, 4, per_round=3))
    assert restored.state_dict() == search.state_dict()


def test_random_search_after_a_restore_knows_what_it_asked_for(tmp_path: Path, backend: Backend):
    search, _ = bind(RandomSearch(), backend, seed=5)  # type: ignore[arg-type]
    batch = search.ask(3)
    state = through_a_checkpoint(search.state_dict(), tmp_path / "c", backend)
    restored, _ = bind(RandomSearch(), backend, seed=5, issuer=IdIssuer(3))  # type: ignore[arg-type]
    restored.load_state_dict(state)
    restored.tell(EvaluationBatch(evaluate(batch, lambda c: 1.0)))  # it asked for these before the checkpoint
    stranger = Candidate(CandidateId(99), np.zeros(3), (), "random", 0)
    with pytest.raises(ValueError, match="never asked for"):
        restored.tell(EvaluationBatch([Evaluation(stranger, Status.OK, {"value": 1.0}, cost=Cost())]))


def test_streams_of_every_backend_survive_a_checkpoint(tmp_path: Path, backend: Backend):
    seed = RunSeed(3)
    stream = seed.stream("strategy", backend=backend)
    stream.uniform((5,))
    state = through_a_checkpoint({"stream": stream.state_dict()}, tmp_path / "c", backend)
    other = seed.stream("strategy", backend=backend)
    other.load_state_dict(state["stream"])
    np.testing.assert_array_equal(backend.to_numpy(stream.uniform((7,))), backend.to_numpy(other.uniform((7,))))
