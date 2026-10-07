import json

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import (
    ArrayBatch,
    Candidate,
    CandidateId,
    Cost,
    Evaluation,
    EvaluationBatch,
    IdIssuer,
    ListBatch,
    Objective,
    ProblemSpec,
    Status,
    StrategyContext,
    validate_state_dict,
)
from auxein.random import RunSeed
from auxein.spaces import Box
from auxein.strategies import RandomSearch


def bound(backend: Backend, seed: int = 1, space=None, issuer: IdIssuer | None = None):
    strategy: RandomSearch = RandomSearch()
    issuer = issuer or IdIssuer()
    ctx = StrategyContext(RunSeed(seed).stream("strategy", backend=backend), backend, issuer.next)
    strategy.bind(ProblemSpec(space or Box(-5.0, 5.0, dim=4), (Objective("value"),)), ctx)
    return strategy, issuer


def told(batch) -> EvaluationBatch:
    return EvaluationBatch([Evaluation(c, Status.OK, {"value": 1.0}, cost=Cost()) for c in batch.candidates])


def test_capabilities():
    caps = RandomSearch.capabilities
    assert caps.max_objectives is None and caps.supports_constraints and caps.tell_mode == "both"
    assert repr(RandomSearch()) == "RandomSearch()"
    assert not RandomSearch().should_stop()


def test_ask_samples_an_array_batch_from_the_space(backend: Backend):
    strategy, _ = bound(backend)
    batch = strategy.ask(50)
    assert isinstance(batch, ArrayBatch) and len(batch.candidates) == 50
    x = backend.to_numpy(batch.as_array())
    assert x.shape == (50, 4) and (x >= -5).all() and (x <= 5).all()
    assert backend.matches(batch.as_array())


def test_ids_come_from_the_context_origin_and_parents_are_as_specified(backend: Backend):
    strategy, issuer = bound(backend)
    first, second = strategy.ask(3), strategy.ask(2)
    assert first.ids == (0, 1, 2) and second.ids == (3, 4) and issuer.issued == 5
    assert all(c.origin == "random" and c.parents == () for c in [*first.candidates, *second.candidates])
    assert first.step == 0 and second.step == 1  # the step counts the ask rounds


def test_the_same_seed_gives_the_same_proposals_and_different_seeds_differ(backend: Backend):
    a, b, c = bound(backend, 3)[0].ask(10), bound(backend, 3)[0].ask(10), bound(backend, 4)[0].ask(10)
    np.testing.assert_array_equal(backend.to_numpy(a.as_array()), backend.to_numpy(b.as_array()))
    assert not np.array_equal(backend.to_numpy(a.as_array()), backend.to_numpy(c.as_array()))


def test_successive_batches_differ(backend: Backend):
    strategy, _ = bound(backend)
    assert not np.array_equal(backend.to_numpy(strategy.ask(5).as_array()), backend.to_numpy(strategy.ask(5).as_array()))


class WordSpace:
    """A space that is not an array space: it samples words."""

    def sample_genomes(self, n, rng, backend):
        return [f"word{int(i)}" for i in backend.to_numpy(rng.integers(0, 100, (n,)))]

    def contains(self, genome):
        return str(genome).startswith("word")


def test_non_array_spaces_give_a_list_batch(backend: Backend):
    strategy, _ = bound(backend, space=WordSpace())
    batch = strategy.ask(6)
    assert isinstance(batch, ListBatch) and batch.as_array() is None
    assert [c.id for c in batch.candidates] == [0, 1, 2, 3, 4, 5]
    assert all(isinstance(c.genome, str) and c.genome.startswith("word") and c.origin == "random" and c.step == 0 for c in batch.candidates)


def test_a_space_returning_the_wrong_number_of_genomes_is_an_error(backend: Backend):
    class Bad(WordSpace):
        def sample_genomes(self, n, rng, backend):
            return ["a"] * (n + 1)

    with pytest.raises(ValueError, match="returned 4 genomes for a request of 3"):
        bound(backend, space=Bad())[0].ask(3)


def test_it_must_be_bound_first_and_n_must_be_positive(backend: Backend):
    fresh: RandomSearch = RandomSearch()
    for call in (lambda: fresh.ask(1), lambda: fresh.state_dict(), lambda: fresh.load_state_dict({})):
        with pytest.raises(RuntimeError, match="must be bound"):
            call()
    strategy, _ = bound(backend)
    for n in (0, -3):
        with pytest.raises(ValueError, match="n must be at least 1"):
            strategy.ask(n)


def test_tell_accepts_results_of_asked_candidates_in_any_grouping(backend: Backend):
    strategy, _ = bound(backend)
    batch = strategy.ask(6)
    strategy.tell(told(batch))
    evaluations = told(batch).evaluations
    for e in evaluations:  # steady-state: one at a time
        strategy.tell(EvaluationBatch([e]))


def test_tell_checks_basic_consistency(backend: Backend):
    strategy, _ = bound(backend)
    batch = strategy.ask(3)
    with pytest.raises(ValueError, match="no results"):
        strategy.tell(EvaluationBatch([]))
    evaluation = told(batch)[0]
    with pytest.raises(ValueError, match="more than once"):
        strategy.tell(EvaluationBatch([evaluation, evaluation]))
    stranger = Evaluation(Candidate(CandidateId(99), None, (), "random", 0), Status.OK, {"value": 1.0})
    with pytest.raises(ValueError, match=r"never asked for: \[99\]"):
        strategy.tell(EvaluationBatch([stranger]))


def test_state_dict_is_valid_and_json_serialisable(backend: Backend):
    strategy, _ = bound(backend)
    strategy.ask(5)
    state = strategy.state_dict()
    validate_state_dict(state)
    assert json.loads(json.dumps(state)) == state
    assert state["step"] == 1 and state["max_id"] == 4


def test_state_round_trip_continues_the_same_proposals(backend: Backend):
    strategy, issuer = bound(backend, seed=11)
    strategy.ask(7)
    strategy.ask(3)
    state = json.loads(json.dumps(strategy.state_dict()))
    expected = strategy.ask(8)

    restored, _ = bound(backend, seed=11, issuer=IdIssuer(issuer.issued - 8))  # a fresh run, brought to the saved state
    restored.load_state_dict(state)
    actual = restored.ask(8)
    np.testing.assert_array_equal(backend.to_numpy(actual.as_array()), backend.to_numpy(expected.as_array()))
    assert actual.step == expected.step == 2 and actual.ids == expected.ids


def test_load_state_dict_validates_its_keys(backend: Backend):
    strategy, _ = bound(backend)
    with pytest.raises(ValueError, match="invalid RandomSearch state"):
        strategy.load_state_dict({"step": 1})
