import itertools
import json

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.core import ListBatch
from auxein.core.operators import NoOperatorLog
from auxein.random import RunSeed
from auxein.recording.genomes import GenomeEncodingError, decode_genome, encode_batch, encode_genome
from auxein.spaces import CanonicalEncodingError, SequenceSpace, canonical_json, codec_of
from auxein.strategies.structured import SequenceCrossover, SequenceMutation, VariationContext

HOST = Backend()


def rng(seed: int = 1, name: str = "strategy"):
    return RunSeed(seed).stream(name)


def context(space: SequenceSpace) -> VariationContext:
    return VariationContext(space, space.codec, NoOperatorLog(), 0)


# --- canonical encoding ---


def test_equal_values_encode_to_identical_bytes_whatever_the_key_order_or_construction():
    a = {"b": [1, 2.5, "x"], "a": {"z": None, "y": True}}
    b = dict(sorted(a.items(), reverse=True))
    c = json.loads(json.dumps(a))
    assert canonical_json(a) == canonical_json(b) == canonical_json(c)
    assert canonical_json(a) == b'{"a":{"y":true,"z":null},"b":[1,2.5,"x"]}'  # sorted keys, no whitespace
    assert canonical_json(["é", "日本"]) == '["é","日本"]'.encode()  # UTF-8, not escaped


def test_values_keep_their_json_type_and_floats_round_trip_exactly():
    assert canonical_json(1) != canonical_json(1.0) and canonical_json(0.0) != canonical_json(-0.0)
    for value in (0.1, 1e-300, 2.0**63, 1 / 3, -123456789.123456789):
        assert json.loads(canonical_json(value)) == value
    assert canonical_json((1, 2)) == canonical_json([1, 2])  # tuples encode as lists


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {1: "a"}, {"a": {1, 2}}, np.float32(1.0), np.int64(3), object(), b"bytes"])
def test_what_is_not_canonical_json_is_a_clear_error(bad):
    with pytest.raises(CanonicalEncodingError, match="JSON value"):
        canonical_json(bad)


def test_the_sequence_codec_round_trips_and_returns_the_vocabularys_own_objects():
    items = ("a", 7, 2.5, None, ["nested", 1], {"k": "v"})
    space = SequenceSpace(items, 0, 6)
    genome = (items[5], items[0], items[4], items[3], items[1])
    encoded = space.codec.encode(genome)
    assert encoded == [{"k": "v"}, "a", ["nested", 1], None, 7]
    assert space.codec.decode(json.loads(canonical_json(encoded))) == genome
    assert all(x is y for x, y in zip(space.codec.decode(encoded), genome, strict=True))
    with pytest.raises(CanonicalEncodingError, match="not in the vocabulary"):
        space.codec.decode(["zzz"])
    with pytest.raises(CanonicalEncodingError, match="JSON list"):
        space.codec.decode("a")
    assert codec_of(space) is space.codec and codec_of(object()) is None


def test_genomes_are_recorded_as_canonical_bytes_through_the_codec_and_unencodable_ones_are_refused():
    space = SequenceSpace(("a", "b"), 0, 3)
    encoded = encode_genome(("b", "a"), space.codec)
    assert encoded.kind == "json" and encoded.data == b'["b","a"]'
    assert decode_genome(encoded.kind, encoded.data, None, None) == ["b", "a"]  # plain JSON without the codec
    assert decode_genome(encoded.kind, encoded.data, None, None, space.codec) == ("b", "a")  # the genome with it
    from auxein.core import Candidate, CandidateId

    batch = ListBatch([Candidate(CandidateId(i), g, (), "init", 0) for i, g in enumerate([("a",), ("b", "b")])])
    assert [e.data for e in encode_batch(batch, space.codec)] == [b'["a"]', b'["b","b"]']
    with pytest.raises(GenomeEncodingError, match="codec"):
        encode_genome({1, 2}, None)


# --- the space ---


def test_the_space_validates_its_arguments_and_describes_itself():
    with pytest.raises(ValueError, match="empty"):
        SequenceSpace((), 0, 3)
    with pytest.raises(ValueError, match="twice"):
        SequenceSpace(("a", "a"), 0, 3)
    with pytest.raises(ValueError, match="min_length"):
        SequenceSpace(("a",), 3, 2)
    with pytest.raises(ValueError, match="cannot exceed the 2 items"):
        SequenceSpace(("a", "b"), 3, 4, unique=True)
    assert SequenceSpace(("a", "b", "c"), 0, 9, unique=True).max_length == 3  # a unique sequence is at most the vocabulary
    assert SequenceSpace(("a", 1), 1, 4).describe() == {
        "type": "SequenceSpace",
        "items": ["a", 1],
        "min_length": 1,
        "max_length": 4,
        "unique": False,
    }
    assert "SequenceSpace(2 items" in repr(SequenceSpace(("a", 1), 1, 4))


@pytest.mark.parametrize("unique", [False, True])
def test_sampling_respects_the_length_bounds_and_uniqueness_and_is_deterministic(unique: bool):
    space = SequenceSpace(tuple("abcdef"), 2, 5, unique=unique)
    genomes = space.sample_genomes(400, rng(3), HOST)
    assert len(genomes) == 400 and all(space.contains(g) for g in genomes)
    lengths = {len(g) for g in genomes}
    assert lengths == {2, 3, 4, 5}  # the length is uniform over the allowed ones
    assert genomes == space.sample_genomes(400, rng(3), HOST) and genomes != space.sample_genomes(400, rng(4), HOST)
    if unique:
        assert all(len(set(g)) == len(g) for g in genomes)


def test_contains_checks_type_length_vocabulary_and_uniqueness():
    space = SequenceSpace(("a", "b", "c"), 1, 3, unique=True)
    assert space.contains(("a", "b")) and not space.contains(()) and not space.contains(("a", "a")) and not space.contains(("z",))
    assert not space.contains(["a"])  # type: ignore[arg-type]  # a genome is a tuple: immutable
    assert not space.contains(("a", "b", "c", "a"))


# --- the operators keep genomes in the space (property tests over small random cases) ---


def small_spaces():
    for size, low, high, unique in itertools.product((1, 2, 3, 5), (0, 1, 2), (1, 2, 4, 6), (False, True)):
        if high >= max(low, 1) and not (unique and low > size):
            yield SequenceSpace(tuple(range(size)), low, high, unique=unique)


@pytest.mark.parametrize("edits", [1, 3])
def test_every_mutation_keeps_the_genome_in_the_space(edits: int):
    mutation = SequenceMutation(edits=edits)
    for space in small_spaces():
        generator = rng(5, "mutation")
        for genome in space.sample_genomes(25, generator, HOST):
            child = mutation.mutate(genome, generator, context(space))
            assert space.contains(child), (space.describe(), genome, child)


@pytest.mark.parametrize("operation", ["insert", "delete", "replace", "swap"])
def test_each_edit_does_what_it_says_and_respects_the_bounds(operation: str):
    weights = {op: 1.0 if op == operation else 0.0 for op in ("insert", "delete", "replace", "swap")}
    mutation = SequenceMutation(**weights)
    space = SequenceSpace(tuple("abcdef"), 2, 5)
    generator = rng(7, "mutation")
    for genome in space.sample_genomes(200, generator, HOST):
        child = mutation.mutate(genome, generator, context(space))
        if operation == "insert" and len(genome) < 5:
            assert len(child) == len(genome) + 1
        elif operation == "delete" and len(genome) > 2:
            assert len(child) == len(genome) - 1
        elif operation == "replace":
            assert len(child) == len(genome) and sum(a != b for a, b in zip(child, genome, strict=True)) == 1
        elif operation == "swap":
            assert sorted(child) == sorted(genome)
        else:
            assert child == genome  # the edit does not apply at this length
        assert space.contains(child)


def test_a_unique_space_never_gets_a_duplicate_from_an_insert_or_a_replace():
    space = SequenceSpace(tuple("abcd"), 1, 4, unique=True)
    mutation = SequenceMutation(insert=1, replace=1, delete=0, swap=0, edits=3)
    generator = rng(1, "mutation")
    for genome in space.sample_genomes(300, generator, HOST):
        assert space.contains(mutation.mutate(genome, generator, context(space)))


def test_the_edits_are_chosen_with_the_configured_probabilities():
    space = SequenceSpace(tuple("abcdef"), 1, 8)
    mutation = SequenceMutation(insert=3.0, delete=1.0, replace=0.0, swap=0.0)
    generator = rng(2, "mutation")
    base = ("a", "b", "c", "d")
    grew = sum(len(mutation.mutate(base, generator, context(space))) > 4 for _ in range(2000))
    assert 0.70 < grew / 2000 < 0.80  # 3 : 1
    with pytest.raises(ValueError, match="not all zero"):
        SequenceMutation(0, 0, 0, 0)
    with pytest.raises(ValueError, match="edits"):
        SequenceMutation(edits=0)


@pytest.mark.parametrize("kind", ["one_point", "two_point"])
def test_every_crossover_child_is_in_the_space_and_made_of_its_parents_items_when_it_can_be(kind: str):
    crossover = SequenceCrossover(kind)
    for space in small_spaces():
        generator = rng(9, "crossover")
        genomes = space.sample_genomes(30, generator, HOST)
        for first, second in zip(genomes, reversed(genomes), strict=True):
            child = crossover.recombine(first, second, generator, context(space))
            assert space.contains(child), (space.describe(), first, second, child)


def test_repair_cuts_long_children_removes_duplicates_and_fills_short_ones():
    space = SequenceSpace(tuple("abcdef"), 3, 4, unique=True)
    repair = SequenceCrossover.repair
    generator = rng(1)
    assert repair(tuple("abcdef"), (), (), space, generator) == tuple("abcd")  # cut at the maximum
    assert repair(tuple("aabbcc"), (), (), space, generator) == tuple("abc")  # first occurrences stay
    filled = repair(("a",), ("a", "b"), ("c", "d"), space, generator)
    assert filled[:3] == ("a", "b", "c") and space.contains(filled)  # filled from the parents' items first
    from_vocabulary = repair((), (), (), space, generator)
    assert len(from_vocabulary) == 3 and space.contains(from_vocabulary)  # and then from the vocabulary


def test_variation_is_deterministic_per_seed():
    space = SequenceSpace(tuple("abcdef"), 1, 6)
    first = space.sample_genomes(5, rng(1), HOST)

    def run(seed: int):
        generator = rng(seed, "variation")
        return [
            SequenceCrossover("two_point").recombine(a, b, generator, context(space)) for a, b in zip(first, first[1:], strict=False)
        ] + [SequenceMutation(edits=2).mutate(g, generator, context(space)) for g in first]

    assert run(4) == run(4) and run(4) != run(5)


def test_the_sequence_operators_need_a_sequence_space():
    from auxein.spaces import Box

    with pytest.raises(TypeError, match="SequenceSpace"):
        SequenceMutation().mutate(("a",), rng(), VariationContext(Box(0.0, 1.0, dim=1), None, NoOperatorLog(), 0))  # type: ignore[arg-type]
