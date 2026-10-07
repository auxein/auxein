import numpy as np
import pytest

from auxein.backend import Backend
from auxein.random import RandomStream, RunSeed, stable_name_id


def test_stable_name_ids_are_pinned():
    # CRC-32 of the UTF-8 name: these values must never change, or old seeds would give new streams
    assert stable_name_id("strategy") == 340149741
    assert stable_name_id("scenarios") == 2469974053
    assert stable_name_id("evaluation") == 321103221
    assert stable_name_id("evaluation-batch") == 2065073135  # the one stream of a vectorised evaluator's batch (design doc §8)
    assert stable_name_id("é") == stable_name_id("é")


def test_the_names_auxein_uses_map_to_different_integers():
    names = ["strategy", "scenarios", "evaluation", "evaluation-batch", "init", "selection", "variation", "replacement"]
    assert len({stable_name_id(n) for n in names}) == len(names)


def test_derived_sequences_are_pinned():
    seed = RunSeed(42)
    assert seed.sequence("strategy").spawn_key == (340149741,)
    assert seed.sequence("evaluation", 7).spawn_key == (321103221, 7)
    assert seed.sequence("strategy").generate_state(2).tolist() == [2600430125, 710442397]
    assert seed.sequence("evaluation", 7).generate_state(2).tolist() == [2651669735, 1417912936]


def test_sequence_uses_the_run_seed_as_entropy():
    assert RunSeed(42).sequence("strategy").entropy == 42
    assert RunSeed(2**100).sequence("strategy").entropy == 2**100  # numpy accepts arbitrarily large seeds
    assert RunSeed(7).root.entropy == 7


def test_same_seed_name_and_keys_give_the_same_sequence():
    a, b = RunSeed(1).sequence("evaluation", 5), RunSeed(1).sequence("evaluation", 5)
    assert a.generate_state(4).tolist() == b.generate_state(4).tolist()


def test_different_seeds_names_or_keys_give_different_sequences():
    base = RunSeed(1).sequence("evaluation", 5).generate_state(4).tolist()
    assert RunSeed(2).sequence("evaluation", 5).generate_state(4).tolist() != base
    assert RunSeed(1).sequence("strategy", 5).generate_state(4).tolist() != base
    assert RunSeed(1).sequence("evaluation", 6).generate_state(4).tolist() != base
    assert RunSeed(1).sequence("evaluation").generate_state(4).tolist() != base
    assert RunSeed(1).sequence("evaluation", 5, 0).generate_state(4).tolist() != base
    assert (
        RunSeed(1).sequence("evaluation", 1, 2).generate_state(4).tolist()
        != RunSeed(1).sequence("evaluation", 2, 1).generate_state(4).tolist()
    )


def test_every_candidate_gets_its_own_evaluation_stream():
    seed = RunSeed(3)
    states = {tuple(seed.sequence("evaluation", i).generate_state(2)) for i in range(2000)}
    assert len(states) == 2000


def test_seed_validation():
    with pytest.raises(ValueError, match="negative"):
        RunSeed(-1)
    with pytest.raises(TypeError, match="must be an int"):
        RunSeed(1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be an int"):
        RunSeed(True)
    assert RunSeed(0).seed == 0


def test_name_and_key_validation():
    seed = RunSeed(1)
    with pytest.raises(ValueError, match="non-empty name"):
        seed.sequence("")
    with pytest.raises(ValueError, match="in \\[0, 2\\*\\*32\\)"):
        seed.sequence("evaluation", -1)
    with pytest.raises(ValueError, match="in \\[0, 2\\*\\*32\\)"):
        seed.sequence("evaluation", 2**32)
    assert seed.sequence("evaluation", 2**32 - 1).spawn_key == (321103221, 2**32 - 1)
    with pytest.raises(TypeError, match="must be ints"):
        seed.sequence("evaluation", 1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be ints"):
        seed.sequence("evaluation", True)


def test_run_seed_is_a_value():
    assert RunSeed(5) == RunSeed(5)
    assert RunSeed(5) != RunSeed(6)
    assert hash(RunSeed(5)) == hash(RunSeed(5))


def test_stream_returns_a_stream_on_the_requested_backend(backend: Backend):
    stream = RunSeed(1).stream("strategy", backend=backend)
    assert isinstance(stream, RandomStream)
    assert stream.backend == backend
    assert RunSeed(1).stream("strategy").backend == Backend()


def test_streams_do_not_depend_on_global_numpy_state():
    np.random.seed(0)
    a = RunSeed(1).stream("strategy").uniform(5)
    np.random.seed(999)
    np.random.uniform(size=100)
    b = RunSeed(1).stream("strategy").uniform(5)
    np.testing.assert_array_equal(a, b)
