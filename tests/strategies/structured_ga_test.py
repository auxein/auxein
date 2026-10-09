import math
from pathlib import Path

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import (
    Cost,
    Evaluation,
    EvaluationBatch,
    IdIssuer,
    Objective,
    ProblemSpec,
    Status,
    StrategyContext,
    validate_state_dict,
)
from auxein.random import RunSeed
from auxein.recording import checkpoints
from auxein.spaces import Box, SequenceSpace
from auxein.strategies import StructuredGeneticAlgorithm
from auxein.strategies.ga import SigmaScalingSUS, TournamentSelection
from auxein.strategies.structured import SequenceCrossover, SequenceMutation

_BACKEND = [Backend()]


@pytest.fixture(autouse=True)
def current_backend(backend: Backend) -> None:
    """Every test of this module runs on each backend and precision: the genomes are Python tuples, but the ranking, the
    selection and the strategy's random stream are arrays on the backend."""
    _BACKEND[0] = backend


SPACE = SequenceSpace(tuple("abcdefgh"), 2, 8)
VALUE = (Objective("value"),)


def bind(ga: StructuredGeneticAlgorithm, space=SPACE, seed: int = 1, issuer: IdIssuer | None = None, constraints=("cpa",)):
    issuer = issuer or IdIssuer()
    ctx = StrategyContext(RunSeed(seed).stream("strategy", backend=_BACKEND[0]), _BACKEND[0], issuer.next)
    ga.bind(ProblemSpec(space, VALUE, tuple(constraints)), ctx)
    return ga, issuer


def value_of(genome) -> float:
    return float(sum(ord(x) - ord("a") for x in genome)) + 0.01 * len(genome)  # lower is better


def evaluations(batch, value=value_of, violation=lambda g: 0.0, failed=lambda c: False):
    out = []
    for c in batch.candidates:
        if failed(c):
            out.append(Evaluation.failed(c, Status.FAILED, "boom"))
        else:
            out.append(Evaluation(c, Status.OK, {"value": value(c.genome)}, {"cpa": violation(c.genome)}, cost=Cost()))
    return out


def tell(ga, batch, **kwargs):
    results = EvaluationBatch(evaluations(batch, **kwargs))
    ga.tell(results)
    return results


def started(**kwargs):
    ga, issuer = bind(StructuredGeneticAlgorithm(**kwargs))
    tell(ga, ga.ask(1))
    return ga, issuer


# --- construction and binding ---


def test_defaults_capabilities_and_repr():
    ga = StructuredGeneticAlgorithm()
    assert ga.population_size == 50 and ga.offspring_size == 50 and ga.crossover_probability == 1.0
    caps = StructuredGeneticAlgorithm.capabilities
    assert caps.max_objectives == 1 and caps.supports_constraints and caps.tell_mode == "both"
    assert "population_size=50" in repr(ga) and "TournamentSelection" in repr(ga)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"population_size": 1}, "at least 2"),
        ({"offspring_size": 0}, "offspring_size must be at least 1"),
        ({"crossover_probability": 1.5}, "crossover_probability"),
        ({"mutation": []}, "at least one operator"),
        ({"mutation": [(SequenceMutation(), -1.0)]}, "non-negative"),
        ({"mutation": [(SequenceMutation(), 0.0)]}, "do not all vanish"),
    ],
)
def test_constructor_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        StructuredGeneticAlgorithm(**kwargs)


def test_it_is_single_objective_and_needs_a_codec_and_operators():
    with pytest.raises(ValueError, match="single-objective"):
        ga = StructuredGeneticAlgorithm()
        ga.bind(
            ProblemSpec(SPACE, (Objective("a"), Objective("b"))),
            StrategyContext(RunSeed(1).stream("strategy", backend=_BACKEND[0]), _BACKEND[0], IdIssuer().next),
        )
    with pytest.raises(TypeError, match="needs a search space with a codec"):
        bind(StructuredGeneticAlgorithm(), space=Box(0.0, 1.0, dim=2))

    class Plain:
        codec = SPACE.codec

        def sample_genomes(self, n, rng, backend):
            return [("a",)] * n

        def contains(self, genome):
            return True

    with pytest.raises(ValueError, match="no default operators"):
        bind(StructuredGeneticAlgorithm(), space=Plain())
    bind(StructuredGeneticAlgorithm(mutation=SequenceMutation()), space=Plain())  # fine once a mutation is given


def test_it_must_be_bound_before_use():
    ga = StructuredGeneticAlgorithm()
    for call in (lambda: ga.ask(1), lambda: ga.state_dict(), lambda: ga.tell(EvaluationBatch([])), lambda: ga.load_state_dict({})):
        with pytest.raises(RuntimeError, match="must be bound"):
            call()
    assert ga.ranked_ids() == [] and ga.size == 0 and not ga.should_stop()


def test_n_must_be_positive():
    ga, _ = bind(StructuredGeneticAlgorithm())
    with pytest.raises(ValueError, match="n must be at least 1"):
        ga.ask(0)


# --- asking ---


def test_the_first_ask_is_the_whole_initial_population():
    ga, issuer = bind(StructuredGeneticAlgorithm(population_size=12, offspring_size=5))
    batch = ga.ask(1)
    assert len(batch.candidates) == 12 and batch.candidates[0].step == 0 and issuer.issued == 12
    assert all(c.origin == "init" and c.parents == () and SPACE.contains(c.genome) for c in batch.candidates)
    assert [c.id for c in batch.candidates] == list(range(12))


def test_then_exactly_lambda_children_or_n():
    ga, issuer = started(population_size=8, offspring_size=5)
    batch = ga.ask(99)
    assert len(batch.candidates) == 5 and batch.candidates[0].step == 1 and all(SPACE.contains(c.genome) for c in batch.candidates)
    assert [c.id for c in batch.candidates] == list(range(8, 13)) and issuer.issued == 13
    free, _ = started(population_size=8, offspring_size=None)
    assert len(free.ask(3).candidates) == 3 and len(free.ask(11).candidates) == 11


def test_children_record_their_parents_and_name_their_operators():
    ga, _ = started(population_size=8, offspring_size=40)
    batch = ga.ask(1)
    parents = {c.parents for c in batch.candidates}
    assert all(1 <= len(p) <= 2 and all(0 <= i < 8 for i in p) for p in parents)
    assert {c.origin for c in batch.candidates} == {"tournament+one_point+sequence"}
    asexual, _ = started(population_size=8, offspring_size=10, crossover_probability=0.0)
    children = asexual.ask(1).candidates
    assert all(len(c.parents) == 1 and c.origin == "tournament+copy+sequence" for c in children)
    two_point, _ = started(population_size=8, offspring_size=4, recombination=SequenceCrossover("two_point"), selection=SigmaScalingSUS())
    assert {c.origin for c in two_point.ask(1).candidates} == {"sus+two_point+sequence"}


def test_a_child_never_mates_with_itself():
    for size in (2, 3, 5):
        ga, _ = started(population_size=size, offspring_size=60)
        for _ in range(5):
            batch = ga.ask(1)
            assert all(len(set(c.parents)) == len(c.parents) for c in batch.candidates)
            assert any(len(c.parents) == 2 for c in batch.candidates)
            tell(ga, batch)


def test_mutations_can_be_mixed_by_probability_and_are_named():
    class Marker:
        name = "marker"

        def mutate(self, genome, rng, ctx):
            return ("a",) * len(genome)

    ga, _ = started(population_size=6, offspring_size=400, mutation=[(SequenceMutation(), 3.0), (Marker(), 1.0)], crossover_probability=0.0)
    origins = [c.origin for c in ga.ask(1).candidates]
    share = origins.count("tournament+copy+marker") / len(origins)
    assert 0.18 < share < 0.32 and set(origins) == {"tournament+copy+sequence", "tournament+copy+marker"}


# --- the population is the best of everything told ---


@pytest.mark.parametrize("seed", range(6))
def test_the_population_is_always_the_best_mu_of_everything_told(seed: int):
    generator = np.random.default_rng(seed)
    size = int(generator.integers(3, 9))
    ga, _ = bind(StructuredGeneticAlgorithm(population_size=size, offspring_size=int(generator.integers(2, 7))), seed=seed)
    told: list[tuple[float, float, int]] = []  # (violation, value, id)

    def feed(batch):
        evaluated = []
        for c in batch.candidates:
            value = float(generator.integers(0, 6))  # many ties, to exercise the id tie-break
            violation = float(generator.integers(0, 3)) if generator.random() < 0.4 else 0.0
            if generator.random() < 0.15:
                told.append((math.inf, math.inf, c.id))
                evaluated.append(Evaluation.failed(c, Status.FAILED, "boom"))
            else:
                told.append((violation, value, c.id))
                evaluated.append(Evaluation(c, Status.OK, {"value": value}, {"cpa": violation}, cost=Cost()))
        ga.tell(EvaluationBatch(evaluated))

    feed(ga.ask(1))
    for _ in range(12):
        batch = ga.ask(1)
        feed(batch)
        expected = [i for _, _, i in sorted(told)[:size]]
        assert ga.ranked_ids() == expected and ga.size == len(expected)


def test_results_can_be_told_in_any_grouping():
    ga, _ = bind(StructuredGeneticAlgorithm(population_size=4, offspring_size=6))
    init = ga.ask(1)
    ga.tell(EvaluationBatch(evaluations(init)[:1]))
    ga.tell(EvaluationBatch(evaluations(init)[1:]))  # one by one, as in steady-state delivery
    assert ga.size == 4
    one, two = ga.ask(1), ga.ask(1)  # asking again before results arrive is allowed
    assert len(one.candidates) == 6 and len(two.candidates) == 6 and {c.id for c in one.candidates}.isdisjoint(c.id for c in two.candidates)
    for evaluation in (*evaluations(two)[:3], *evaluations(one)[2:5], *evaluations(two)[3:], *evaluations(one)[:2], *evaluations(one)[5:]):
        ga.tell(EvaluationBatch([evaluation]))
    assert ga.size == 4


def test_tell_validates_its_input():
    ga, _ = bind(StructuredGeneticAlgorithm(population_size=3, offspring_size=3))
    batch = ga.ask(1)
    results = evaluations(batch)
    with pytest.raises(ValueError, match="no results"):
        ga.tell(EvaluationBatch([]))
    with pytest.raises(ValueError, match="more than once"):
        ga.tell(EvaluationBatch([results[0], results[0]]))
    ga.tell(EvaluationBatch(results))
    with pytest.raises(ValueError, match="not pending"):
        ga.tell(EvaluationBatch(results[:1]))


# --- failures ---


def test_failed_members_are_never_parents_while_there_is_an_alternative():
    ga, _ = bind(StructuredGeneticAlgorithm(population_size=6, offspring_size=60))
    init = ga.ask(1)
    tell(ga, init, failed=lambda c: c.id < 4)  # four of the six members failed
    assert set(ga.ranked_ids()[:2]) == {4, 5} and ga._breedable() == 2
    for _ in range(4):
        valid = {i for i, v in zip(ga._ids, np.asarray(ga._violation).tolist(), strict=True) if math.isfinite(v)}
        children = ga.opened = ga.ask(1)
        assert all(set(c.parents) <= valid for c in children.candidates)  # only members that did not fail are ever parents
        assert any(len(c.parents) == 2 for c in children.candidates)
        tell(ga, children, failed=lambda c: c.id % 2 == 0)


def test_with_fewer_than_two_valid_members_it_samples_at_random_instead_of_breeding():
    ga, _ = bind(StructuredGeneticAlgorithm(population_size=5, offspring_size=7))
    init = ga.ask(1)
    tell(ga, init, failed=lambda c: c.id != 2)
    children = ga.ask(1).candidates
    assert len(children) == 7 and all(c.origin == "init" and c.parents == () for c in children)


def test_failed_children_never_displace_feasible_members():
    ga, _ = started(population_size=4, offspring_size=8)
    before = ga.ranked_ids()
    tell(ga, ga.ask(1), failed=lambda c: True)
    assert ga.ranked_ids() == before


def test_sus_and_tournaments_are_unaffected_by_failed_members():
    for selection in (TournamentSelection(3), SigmaScalingSUS()):
        ga, _ = bind(StructuredGeneticAlgorithm(population_size=6, offspring_size=30, selection=selection))
        tell(ga, ga.ask(1), failed=lambda c: c.id in (0, 3))
        for child in ga.ask(1).candidates:
            assert not set(child.parents) & {0, 3}


# --- state ---


@pytest.mark.parametrize("variant", ["default", "mixed", "sus"])
def test_state_round_trips_through_a_checkpoint_and_continues_identically(tmp_path: Path, variant: str):
    options = {
        "default": {},
        "mixed": {"mutation": [(SequenceMutation(edits=2), 2.0), (SequenceMutation(insert=0, delete=0, swap=1, replace=0), 1.0)]},
        "sus": {"selection": SigmaScalingSUS(), "recombination": SequenceCrossover("two_point"), "crossover_probability": 0.7},
    }[variant]

    def fresh(issuer=None):
        return bind(StructuredGeneticAlgorithm(population_size=8, offspring_size=5, **options), seed=11, issuer=issuer)

    ga, issuer = fresh()
    for _ in range(3):
        tell(ga, ga.ask(1), failed=lambda c: c.id % 7 == 3)
    pending = ga.ask(1)
    tell(ga, type("B", (), {"candidates": pending.candidates[:2]})())  # part of a block told, part pending
    state = ga.state_dict()
    validate_state_dict(state)
    checkpoints.write(tmp_path / "c", 1, state)
    _, restored_state = checkpoints.read(tmp_path / "c" / "ckpt-1", _BACKEND[0])

    restored, _ = fresh(IdIssuer(issuer.issued))
    restored.load_state_dict(restored_state)
    assert restored.ranked_ids() == ga.ranked_ids() and restored._pending == ga._pending

    def go(strategy):
        out = []
        for _ in range(4):
            batch = strategy.ask(1)
            out.append([(c.id, c.genome, c.parents, c.origin) for c in batch.candidates])
            tell(strategy, batch)
        return out, strategy.ranked_ids()

    rest = type("B", (), {"candidates": pending.candidates[2:]})()
    tell(ga, rest)
    tell(restored, rest)  # the block that was part told can be completed after the restore
    assert go(ga) == go(restored)


def test_load_state_dict_validates_its_keys():
    ga, _ = bind(StructuredGeneticAlgorithm())
    with pytest.raises(ValueError, match="invalid StructuredGeneticAlgorithm state"):
        ga.load_state_dict({"step": 1})


def test_it_is_deterministic_per_seed():
    def run(seed: int):
        ga, _ = bind(StructuredGeneticAlgorithm(population_size=8, offspring_size=6), seed=seed)
        out = []
        for _ in range(6):
            batch = ga.ask(1)
            out.append([c.genome for c in batch.candidates])
            tell(ga, batch)
        return out

    assert run(3) == run(3) and run(3) != run(4)


def test_convergence_stops_a_full_population_with_no_spread():
    ga, _ = bind(StructuredGeneticAlgorithm(population_size=4, offspring_size=4, convergence_tolerance=0.5))
    batch = ga.ask(1)
    assert not ga.should_stop()
    tell(ga, batch, value=lambda g: 1.0)
    assert ga.should_stop()
    tell(ga, ga.ask(1), failed=lambda c: True)
    assert ga.should_stop()  # failed children change nothing
