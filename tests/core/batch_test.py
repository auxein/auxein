import numpy as np
import pytest

from auxein.backend import Backend, backend_of
from auxein.core import ArrayBatch, Batch, Candidate, CandidateId, ListBatch
from tests.support.fixtures import assert_on_backend


def ids(n: int) -> list[CandidateId]:
    return [CandidateId(i) for i in range(n)]


def make(backend: Backend, n: int = 4, d: int = 3, **kwargs) -> tuple[ArrayBatch, object]:
    genomes = backend.asarray(np.arange(n * d, dtype=np.float64).reshape(n, d))
    return ArrayBatch(genomes, ids(n), step=2, origins="init", **kwargs), genomes


def test_list_batch_holds_candidates_and_has_no_array_view():
    candidates = [Candidate(CandidateId(i), f"g{i}", (), "init", 0) for i in range(3)]
    batch = ListBatch(candidates)
    assert list(batch.candidates) == candidates
    assert batch.as_array() is None
    assert len(batch) == 3
    assert isinstance(batch.candidates, tuple)


def test_list_batch_copies_its_sequence():
    candidates = [Candidate(CandidateId(0), "g", (), "init", 0)]
    batch = ListBatch(candidates)
    candidates.append(Candidate(CandidateId(1), "h", (), "init", 0))
    assert len(batch) == 1


def test_array_batch_exposes_the_array_and_metadata(backend: Backend):
    batch, genomes = make(backend, parents=[(CandidateId(9),), (), (CandidateId(1), CandidateId(2)), ()])
    assert_on_backend(batch.as_array(), backend)
    assert tuple(batch.as_array().shape) == (4, 3)
    np.testing.assert_array_equal(backend.to_numpy(batch.as_array()), backend.to_numpy(genomes))
    assert batch.ids == (0, 1, 2, 3)
    assert batch.parents == ((9,), (), (1, 2), ())
    assert batch.origins == ("init",) * 4
    assert batch.step == 2 and batch.dim == 3 and len(batch) == 4
    assert backend_of(batch.as_array()).name == backend.name
    assert batch.backend.name == backend.name


def test_origins_can_be_given_per_candidate(backend: Backend):
    genomes = backend.asarray(np.zeros((2, 2)))
    batch = ArrayBatch(genomes, ids(2), 0, ["init", "mutation:gaussian"])
    assert batch.origins == ("init", "mutation:gaussian")
    assert [c.origin for c in batch.candidates] == ["init", "mutation:gaussian"]


def test_parents_default_to_none(backend: Backend):
    batch, _ = make(backend)
    assert batch.parents == ((),) * 4


def test_candidates_are_materialised_lazily_with_the_right_content(backend: Backend):
    batch, _ = make(backend, parents=[(CandidateId(7),), (), (), ()])
    candidates = batch.candidates
    assert len(candidates) == 4
    first = candidates[0]
    assert isinstance(first, Candidate)
    assert (first.id, first.parents, first.origin, first.step) == (0, (7,), "init", 2)
    np.testing.assert_array_equal(backend.to_numpy(first.genome), [0.0, 1.0, 2.0])
    assert candidates[-1].id == 3
    assert [c.id for c in candidates] == [0, 1, 2, 3]
    assert [c.id for c in candidates[1:3]] == [1, 2]
    with pytest.raises(IndexError):
        candidates[4]
    with pytest.raises(IndexError):
        candidates[-5]


def test_candidates_are_not_built_until_asked(backend: Backend, monkeypatch: pytest.MonkeyPatch):
    batch, _ = make(backend)
    built = []
    original = ArrayBatch.candidate
    monkeypatch.setattr(ArrayBatch, "candidate", lambda self, i: built.append(i) or original(self, i))
    _ = batch.candidates
    assert built == []
    batch.candidates[2]
    assert built == [2]


def test_candidate_genomes_are_row_views_sharing_memory_with_the_array(backend: Backend):
    batch, _ = make(backend)
    array = batch.as_array()
    for i, candidate in enumerate(batch.candidates):
        if backend.name == "numpy":
            assert np.shares_memory(candidate.genome, array)
            assert candidate.genome.base is not None
        else:
            assert candidate.genome.data_ptr() == array[i].data_ptr()  # the same storage, no copy


def test_numpy_genomes_and_the_array_are_read_only():
    backend = Backend()
    batch, genomes = make(backend)
    with pytest.raises(ValueError, match="read-only"):
        batch.as_array()[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        batch.candidates[0].genome[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        batch.candidates[2].genome[:] = 0.0
    assert genomes.flags.writeable  # the caller's own array keeps its flags


def test_the_batch_sees_later_writes_to_the_callers_array_only_through_the_shared_buffer():
    genomes = np.zeros((2, 2))
    batch = ArrayBatch(genomes, ids(2), 0, "init")
    genomes[0, 0] = 5.0  # the caller still owns the buffer: a view, not a copy
    assert batch.as_array()[0, 0] == 5.0


def test_torch_batches_are_immutable_by_convention_only():
    torch = pytest.importorskip("torch")
    backend = Backend("torch", "cpu", "float64")
    batch = ArrayBatch(backend.asarray(np.zeros((2, 2))), ids(2), 0, "init")
    assert isinstance(batch.as_array(), torch.Tensor)  # no read-only flag exists in torch; see Backend.readonly


def test_array_batch_satisfies_the_batch_protocol(backend: Backend):
    batch, _ = make(backend)
    as_protocol: Batch = batch  # structural typing: this assignment is what the protocol promises
    assert as_protocol.as_array() is not None
    assert len(as_protocol.candidates) == 4


def test_an_empty_batch(backend: Backend):
    batch = ArrayBatch(backend.asarray(np.zeros((0, 3))), [], 0, "init")
    assert len(batch) == 0 and batch.dim == 3 and list(batch.candidates) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ids": ids(3)}, "ids has 3 entries but there are 4 genomes"),
        ({"origins": ["init"] * 3}, "origins has 3 entries"),
        ({"parents": [()] * 5}, "parents has 5 entries"),
        ({"ids": [CandidateId(0), CandidateId(1), CandidateId(1), CandidateId(2)]}, "unique"),
        ({"step": -1}, "step must not be negative"),
        ({"origins": ["init", "init", "", "init"]}, "non-empty origin"),
        ({"origins": ""}, "non-empty origin"),
    ],
)
def test_construction_validates_lengths_and_metadata(backend: Backend, kwargs: dict, message: str):
    args = {"ids": ids(4), "step": 0, "origins": "init", "parents": None, **kwargs}
    with pytest.raises(ValueError, match=message):
        ArrayBatch(backend.asarray(np.zeros((4, 3))), **args)


def test_construction_validates_the_array(backend: Backend):
    with pytest.raises(ValueError, match="2-D"):
        ArrayBatch(backend.asarray(np.zeros(3)), ids(3), 0, "init")
    with pytest.raises(ValueError, match="2-D"):
        ArrayBatch(backend.asarray(np.zeros((2, 2, 2))), ids(2), 0, "init")
    with pytest.raises(TypeError, match="cannot infer a backend"):
        ArrayBatch([[1.0, 2.0]], ids(1), 0, "init")  # type: ignore[arg-type]
