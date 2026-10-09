"""`SequenceSpace`: variable-length sequences from a finite vocabulary (design doc §4.4)."""

from collections.abc import Sequence
from typing import cast

from auxein.backend import Backend
from auxein.random import RandomStream
from auxein.spaces.codec import CanonicalEncodingError, JsonValue, canonical_json

Genome = tuple[object, ...]


class SequenceCodec:
    """A sequence as the JSON list of its items; decoding gives back the vocabulary's own objects, in a tuple."""

    def __init__(self, vocabulary: Sequence[object]) -> None:
        self._by_key = {canonical_json(item): item for item in vocabulary}

    def encode(self, genome: Genome) -> JsonValue:
        return list(genome)

    def decode(self, value: JsonValue) -> Genome:
        if not isinstance(value, list):
            raise CanonicalEncodingError(f"a sequence genome is a JSON list, got {type(value).__name__}")
        items = cast("list[object]", value)
        try:
            return tuple(self._by_key[canonical_json(item)] for item in items)
        except KeyError as error:
            raise CanonicalEncodingError(f"{error.args[0].decode()} is not in the vocabulary of the space") from None


class SequenceSpace:
    """Genomes are tuples of items drawn from a finite vocabulary, with a length between `min_length` and `max_length`.

    Think of instructions, rule identifiers, tool names or tokens. Items are JSON-serialisable values; two items are the same
    when their canonical encodings are. With `unique=True` no item appears twice (a permutation or a subset, in order), which
    needs a vocabulary of at least `min_length` distinct items. Sampling draws a length uniformly, then the items. The genome
    is a tuple, so it is immutable and picklable; its codec is the JSON list of its items.

    The structured genetic algorithm has default operators for this space (insert, delete, replace and swap mutations, and
    cut-and-splice crossover that repair the length bounds and uniqueness).
    """

    def __init__(self, items: Sequence[object], min_length: int, max_length: int, unique: bool = False) -> None:
        keys = [canonical_json(item) for item in items]
        if len(set(keys)) != len(keys):
            raise ValueError("the vocabulary must not contain the same item twice")
        if not items:
            raise ValueError("the vocabulary must not be empty")
        if min_length < 0 or max_length < max(min_length, 1):
            raise ValueError(f"need 0 <= min_length <= max_length and max_length >= 1, got {min_length} and {max_length}")
        if unique and min_length > len(items):
            raise ValueError(f"with unique=True the minimum length {min_length} cannot exceed the {len(items)} items of the vocabulary")
        if unique and max_length > len(items):
            max_length = len(items)  # a unique sequence cannot be longer than the vocabulary
        self.items: tuple[object, ...] = tuple(items)
        self.min_length, self.max_length, self.unique = min_length, max_length, unique
        self.codec = SequenceCodec(self.items)
        self._index = {key: i for i, key in enumerate(keys)}

    def __repr__(self) -> str:
        return f"SequenceSpace({len(self.items)} items, length {self.min_length}..{self.max_length}, unique={self.unique})"

    def describe(self) -> dict[str, object]:
        """A description for metadata and resume validation: the vocabulary and the bounds."""
        return {
            "type": "SequenceSpace",
            "items": list(self.items),
            "min_length": self.min_length,
            "max_length": self.max_length,
            "unique": self.unique,
        }

    def sample_genomes(self, n: int, rng: RandomStream, backend: Backend) -> list[Genome]:
        """`n` sequences: a length uniform in [min_length, max_length], then that many items (distinct if `unique`)."""
        lengths = [int(x) for x in backend.to_numpy(rng.integers(self.min_length, self.max_length + 1, (n,))).tolist()]
        genomes: list[Genome] = []
        for length in lengths:
            if length == 0:
                genomes.append(())
                continue
            picks = rng.choice(len(self.items), length, replace=not self.unique)
            genomes.append(tuple(self.items[int(i)] for i in backend.to_numpy(picks).tolist()))
        return genomes

    def contains(self, genome: Genome) -> bool:
        if not isinstance(genome, tuple) or not self.min_length <= len(genome) <= self.max_length:  # pyright: ignore[reportUnnecessaryIsInstance]
            return False
        try:
            keys = [canonical_json(item) for item in genome]
        except CanonicalEncodingError:
            return False
        if any(key not in self._index for key in keys):
            return False
        return not self.unique or len(set(keys)) == len(keys)

    def key(self, item: object) -> bytes:
        """The canonical bytes that identify an item."""
        return canonical_json(item)
