"""The default operators for `SequenceSpace`: insert, delete, replace and swap mutations, and cut-and-splice crossover."""

from collections.abc import Sequence

from auxein.backend import Backend
from auxein.random import RandomStream
from auxein.spaces import SequenceSpace
from auxein.spaces.sequence import Genome
from auxein.strategies.structured.variation import VariationContext

_HOST = Backend()
_OPERATIONS = ("insert", "delete", "replace", "swap")


def _below(rng: RandomStream, high: int) -> int:
    """A uniform integer in [0, high)."""
    return int(_HOST.to_numpy(rng.integers(0, high, (1,)))[0])


def _upto(rng: RandomStream, high: int) -> int:
    """A uniform integer in [0, high] (both ends)."""
    return _below(rng, high + 1)


def _space(ctx: VariationContext[Genome]) -> SequenceSpace:
    space = ctx.space
    if not isinstance(space, SequenceSpace):
        raise TypeError(f"the sequence operators need a SequenceSpace, got {type(space).__name__}")
    return space


class SequenceMutation:
    """Edits a sequence: insert an item, delete one, replace one, or swap two, each respecting the length bounds and `unique`.

    Each edit picks one of the operations that applies to the current sequence (it cannot delete at the minimum length or
    insert at the maximum, and in a `unique` space inserts and replacements need an item that is not yet there), with
    probabilities proportional to `insert`, `delete`, `replace` and `swap`. A child gets `edits` edits in a row. A replacement
    always changes the item (when the vocabulary has another), and a swap exchanges two different positions.
    """

    name = "sequence"

    def __init__(self, insert: float = 1.0, delete: float = 1.0, replace: float = 1.0, swap: float = 1.0, edits: int = 1) -> None:
        weights = {"insert": insert, "delete": delete, "replace": replace, "swap": swap}
        if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError(f"the probabilities must not be negative and not all zero, got {weights}")
        if edits < 1:
            raise ValueError(f"edits must be at least 1, got {edits}")
        self.weights, self.edits = weights, edits

    def __repr__(self) -> str:
        return f"SequenceMutation({self.weights}, edits={self.edits})"

    def mutate(self, genome: Genome, rng: RandomStream, ctx: VariationContext[Genome]) -> Genome:
        space = _space(ctx)
        items = list(genome)
        for _ in range(self.edits):
            operations = [op for op in _OPERATIONS if self.weights[op] > 0 and self._possible(op, items, space)]
            if not operations:
                break
            operation = self._choose(operations, rng)
            if operation == "insert":
                items.insert(_upto(rng, len(items)), self._new_item(items, space, rng))
            elif operation == "delete":
                del items[_below(rng, len(items))]
            elif operation == "replace":
                position = _below(rng, len(items))
                items[position] = self._new_item(items, space, rng, replacing=position)
            else:
                first = _below(rng, len(items))
                second = _below(rng, len(items) - 1)
                second += second >= first  # a different position
                items[first], items[second] = items[second], items[first]
        return tuple(items)

    def _choose(self, operations: Sequence[str], rng: RandomStream) -> str:
        total = sum(self.weights[op] for op in operations)
        point = float(_HOST.to_numpy(rng.uniform((1,)))[0]) * total
        for op in operations:
            point -= self.weights[op]
            if point < 0:
                return op
        return operations[-1]

    @staticmethod
    def _possible(operation: str, items: list[object], space: SequenceSpace) -> bool:
        spare = not space.unique or len(items) < len(space.items)  # an item that is not yet in the sequence exists
        if operation == "insert":
            return len(items) < space.max_length and spare
        if operation == "delete":
            return len(items) > space.min_length
        if operation == "replace":
            return len(items) >= 1 and (spare if space.unique else len(space.items) > 1)
        return len(items) >= 2

    @staticmethod
    def _new_item(items: list[object], space: SequenceSpace, rng: RandomStream, replacing: int | None = None) -> object:
        """An item for an insertion or a replacement: unused in a `unique` space, different from the replaced one otherwise."""
        taken = {space.key(item) for item in items}
        pool = [item for item in space.items if space.key(item) not in taken] if space.unique else list(space.items)
        if replacing is not None and not space.unique:
            current = space.key(items[replacing])
            pool = [item for item in pool if space.key(item) != current]
        return pool[_below(rng, len(pool))]


class SequenceCrossover:
    """Cut-and-splice crossover for variable lengths.

    `kind="one_point"`: a cut point in each parent, and the child is the head of the first parent followed by the tail of the
    second. `kind="two_point"`: the child is the first parent with a segment of the second spliced in. Because the cut points
    are drawn separately, the child's length can be anything, so it is **repaired**: duplicates are removed in a `unique` space
    (the first occurrence stays), a child longer than `max_length` is cut at `max_length`, and one shorter than `min_length` is
    filled up with items of the two parents that are not in it yet, then with random items of the vocabulary.
    """

    def __init__(self, kind: str = "one_point") -> None:
        if kind not in ("one_point", "two_point"):
            raise ValueError(f"kind must be 'one_point' or 'two_point', got {kind!r}")
        self.kind = kind
        self.name = kind

    def __repr__(self) -> str:
        return f"SequenceCrossover({self.kind!r})"

    def recombine(self, first: Genome, second: Genome, rng: RandomStream, ctx: VariationContext[Genome]) -> Genome:
        space = _space(ctx)
        if self.kind == "one_point":
            child = first[: _upto(rng, len(first))] + second[_upto(rng, len(second)) :]
        else:
            a1, a2 = sorted((_upto(rng, len(first)), _upto(rng, len(first))))
            b1, b2 = sorted((_upto(rng, len(second)), _upto(rng, len(second))))
            child = first[:a1] + second[b1:b2] + first[a2:]
        return self.repair(child, first, second, space, rng)

    @staticmethod
    def repair(child: Genome, first: Genome, second: Genome, space: SequenceSpace, rng: RandomStream) -> Genome:
        """Bring a spliced sequence back into the space (see the class docstring)."""
        items = list(child)
        if space.unique:
            seen: set[bytes] = set()
            kept: list[object] = []
            for item in items:
                key = space.key(item)
                if key not in seen:
                    seen.add(key)
                    kept.append(item)
            items = kept
        items = items[: space.max_length]
        if len(items) < space.min_length:
            present = {space.key(item) for item in items}
            for item in (*first, *second):
                if len(items) >= space.min_length:
                    break
                if space.unique and space.key(item) in present:
                    continue
                items.append(item)
                present.add(space.key(item))
            while len(items) < space.min_length:
                pool = [item for item in space.items if space.key(item) not in present] if space.unique else list(space.items)
                item = pool[_below(rng, len(pool))]
                items.append(item)
                present.add(space.key(item))
        return tuple(items)
