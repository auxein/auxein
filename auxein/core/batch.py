"""Batches of candidates, and the array fast path (design doc §4.3)."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, cast, overload

from auxein.backend import Array, Backend, backend_of
from auxein.core._typing import G
from auxein.core.candidate import Candidate
from auxein.core.ids import CandidateId


class Batch(Protocol[G]):
    """A sequence of candidates, optionally backed by an array.

    Evaluators choose their view: a vectorised function takes `as_array()` in one call, an agent evaluator iterates
    over `candidates`.
    """

    @property
    def candidates(self) -> Sequence[Candidate[G]]: ...

    def as_array(self) -> Array | None:
        """An `(n, d)` array view of the genomes if the batch is array-backed, otherwise None."""
        ...


@dataclass(frozen=True)
class ListBatch(Generic[G]):
    """A batch of arbitrary genomes (structures, text, ...): the general path, with no array view."""

    candidates: Sequence[Candidate[G]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))

    def as_array(self) -> None:
        return None

    def __len__(self) -> int:
        return len(self.candidates)


class _LazyCandidates(Sequence[Candidate[Array]]):
    """The candidates of an `ArrayBatch`, built on access: a candidate's genome is a row view of the batch array."""

    def __init__(self, batch: "ArrayBatch") -> None:
        self._batch = batch

    def __len__(self) -> int:
        return len(self._batch)

    @overload
    def __getitem__(self, index: int) -> Candidate[Array]: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Candidate[Array]]: ...
    def __getitem__(self, index: int | slice) -> Candidate[Array] | Sequence[Candidate[Array]]:
        if isinstance(index, slice):
            return [self._batch.candidate(i) for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("candidate index out of range")
        return self._batch.candidate(index)

    def __iter__(self) -> Iterator[Candidate[Array]]:
        return (self._batch.candidate(i) for i in range(len(self)))


class ArrayBatch:
    """Candidates whose genomes are the rows of an `(n, d)` array on the backend: the fast path for numeric strategies.

    The `Candidate` objects are materialised lazily, and each genome is a row *view* of the array: no per-candidate
    copies. The array is exposed read-only where the backend allows (numpy; torch has no read-only flag, so there
    immutability is by convention), and the batch holds a view of the array it is given, so the caller's own array
    keeps its flags. Metadata is parallel to the rows: `ids[i]`, `parents[i]` and `origins[i]` describe row `i`, and
    `step` is the ask round of the whole batch.
    """

    def __init__(
        self,
        genomes: Array,
        ids: Sequence[CandidateId],
        step: int,
        origins: str | Sequence[str],
        parents: Sequence[tuple[CandidateId, ...]] | None = None,
    ) -> None:
        backend = backend_of(genomes)  # raises TypeError for anything that is not a numpy array or a torch tensor
        if genomes.ndim != 2:
            raise ValueError(f"genomes must be a 2-D (n, d) array, got shape {tuple(genomes.shape)}")
        n = int(genomes.shape[0])
        origin_list = [origins] * n if isinstance(origins, str) else list(origins)
        parent_list = [()] * n if parents is None else [tuple(p) for p in parents]
        for what, length in (("ids", len(ids)), ("origins", len(origin_list)), ("parents", len(parent_list))):
            if length != n:
                raise ValueError(f"{what} has {length} entries but there are {n} genomes")
        if len(set(ids)) != n:
            raise ValueError("candidate ids must be unique within a batch")
        if step < 0:
            raise ValueError(f"step must not be negative, got {step}")
        if not all(origin_list):
            raise ValueError("every candidate needs a non-empty origin")

        self._backend = backend
        self._genomes: Array = backend.readonly(genomes[...])
        self._ids = tuple(ids)
        self._parents = tuple(parent_list)
        self._origins = tuple(origin_list)
        self._step = step
        self._candidates = _LazyCandidates(self)

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def ids(self) -> tuple[CandidateId, ...]:
        return self._ids

    @property
    def parents(self) -> tuple[tuple[CandidateId, ...], ...]:
        return self._parents

    @property
    def origins(self) -> tuple[str, ...]:
        return self._origins

    @property
    def step(self) -> int:
        return self._step

    @property
    def dim(self) -> int:
        return int(self._genomes.shape[1])

    def __len__(self) -> int:
        return len(self._ids)

    def as_array(self) -> Array:
        """The `(n, d)` genome array, read-only where the backend allows."""
        return self._genomes

    @property
    def candidates(self) -> Sequence[Candidate[Array]]:
        """The candidates, built on access, with genomes that are row views of `as_array()`."""
        return self._candidates

    def candidate(self, index: int) -> Candidate[Array]:
        return Candidate(self._ids[index], self._genomes[index], self._parents[index], self._origins[index], self._step)

    def slice(self, start: int, stop: int) -> "ArrayBatch":
        """The candidates `start` to `stop`, as a batch sharing this batch's array (a view, not a copy)."""
        if not 0 <= start <= stop <= len(self):
            raise ValueError(f"cannot slice [{start}:{stop}] out of {len(self)} candidates")
        return ArrayBatch(
            self._genomes[start:stop], self._ids[start:stop], self._step, self._origins[start:stop], self._parents[start:stop]
        )

    def take(self, n: int) -> "ArrayBatch":
        """The first `n` candidates, as a batch sharing this batch's array (a slice, not a copy)."""
        if not 0 <= n <= len(self):
            raise ValueError(f"cannot take {n} candidates out of {len(self)}")
        return ArrayBatch(self._genomes[:n], self._ids[:n], self._step, self._origins[:n], self._parents[:n])


def single(batch: Batch[G], index: int) -> Batch[G]:
    """The one-candidate batch holding candidate `index`. Array-backed batches stay array-backed (a view, not a copy).

    Steady-state delivery evaluates and records candidates one at a time, through the same evaluators as batches.
    """
    if not 0 <= index < len(batch.candidates):
        raise ValueError(f"no candidate {index} in a batch of {len(batch.candidates)}")
    if isinstance(batch, ArrayBatch):
        return cast("Batch[G]", batch.slice(index, index + 1))
    return ListBatch((batch.candidates[index],))


def take(batch: Batch[G], n: int) -> Batch[G]:
    """The first `n` candidates of any batch, in order. Array-backed batches stay array-backed.

    The driver uses it to truncate an oversized final batch to the remaining evaluation budget.
    """
    if not 0 <= n <= len(batch.candidates):
        raise ValueError(f"cannot take {n} candidates out of {len(batch.candidates)}")
    if isinstance(batch, ArrayBatch):
        return cast("Batch[G]", batch.take(n))
    return ListBatch(tuple(batch.candidates[:n]))
