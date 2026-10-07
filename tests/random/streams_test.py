import json
import os
import subprocess
import sys

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.random import RandomStream, RunSeed
from tests.support.fixtures import assert_on_backend

SEED = RunSeed(2024)


def stream(backend: Backend, name: str = "strategy", *keys: int) -> RandomStream:
    return SEED.stream(name, *keys, backend=backend)


def host(backend: Backend, x):
    return backend.to_numpy(x)


def test_every_method_returns_arrays_of_the_backends_namespace_device_and_dtype(backend: Backend):
    s = stream(backend)
    assert_on_backend(s.uniform((3, 4)), backend)
    assert_on_backend(s.normal((3, 4)), backend)
    assert_on_backend(s.integers(0, 10, (5,)), backend, backend.int_dtype)
    assert_on_backend(s.permutation(6), backend, backend.int_dtype)
    assert_on_backend(s.choice(10, 4), backend, backend.int_dtype)
    assert_on_backend(s.choice(10, 4, replace=False), backend, backend.int_dtype)
    assert_on_backend(s.choice(10, 4, p=np.arange(1.0, 11.0)), backend, backend.int_dtype)
    assert_on_backend(s.choice(10, 4, p=np.arange(1.0, 11.0), replace=False), backend, backend.int_dtype)
    assert_on_backend(s.uniform(3, backend.asarray([0.0, 1.0, 2.0]), 5.0), backend)


def test_shapes(backend: Backend):
    s = stream(backend)
    assert tuple(s.uniform((2, 3)).shape) == (2, 3)
    assert tuple(s.uniform(4).shape) == (4,)
    assert tuple(s.normal((0, 3)).shape) == (0, 3)
    assert tuple(s.integers(0, 5).shape) == ()
    assert tuple(s.integers(0, 5, (2, 2)).shape) == (2, 2)
    assert tuple(s.permutation(0).shape) == (0,)
    assert tuple(s.choice(5, 0).shape) == (0,)
    assert tuple(s.choice(5, 7).shape) == (7,)


def test_same_seed_name_and_keys_give_identical_draws(backend: Backend):
    a, b = stream(backend, "evaluation", 3), stream(backend, "evaluation", 3)
    np.testing.assert_array_equal(host(backend, a.uniform((50,))), host(backend, b.uniform((50,))))
    np.testing.assert_array_equal(host(backend, a.normal((50,))), host(backend, b.normal((50,))))
    np.testing.assert_array_equal(host(backend, a.integers(0, 100, (50,))), host(backend, b.integers(0, 100, (50,))))
    np.testing.assert_array_equal(host(backend, a.permutation(20)), host(backend, b.permutation(20)))
    np.testing.assert_array_equal(host(backend, a.choice(9, 5)), host(backend, b.choice(9, 5)))


def test_different_names_or_keys_give_different_draws(backend: Backend):
    draws = {
        name_keys: tuple(host(backend, stream(backend, name_keys[0], *name_keys[1:]).uniform((8,))).tolist())
        for name_keys in [("strategy",), ("scenarios",), ("evaluation",), ("evaluation", 0), ("evaluation", 1), ("evaluation", 0, 1)]
    }
    assert len(set(draws.values())) == len(draws)
    other_seed = RunSeed(2025).stream("strategy", backend=backend).uniform((8,))
    assert tuple(host(backend, other_seed).tolist()) != draws[("strategy",)]


def test_successive_draws_differ(backend: Backend):
    s = stream(backend)
    assert not np.array_equal(host(backend, s.uniform((10,))), host(backend, s.uniform((10,))))


def test_uniform_bounds_and_moments(backend: Backend):
    s = stream(backend)
    x = host(backend, s.uniform((20000,), -2.0, 3.0))
    assert x.min() >= -2.0 and x.max() <= 3.0
    assert abs(x.mean() - 0.5) < 0.1
    assert abs(x.std() - 5 / np.sqrt(12)) < 0.1


def test_uniform_with_array_bounds_broadcasts(backend: Backend):
    low, high = backend.asarray([0.0, 10.0, 100.0]), backend.asarray([1.0, 11.0, 101.0])
    x = host(backend, stream(backend).uniform((1000, 3), low, high))
    assert x.shape == (1000, 3)
    for column, (lo, hi) in enumerate([(0, 1), (10, 11), (100, 101)]):
        assert lo <= x[:, column].min() and x[:, column].max() <= hi


def test_normal_moments_and_scaling(backend: Backend):
    x = host(backend, stream(backend).normal((40000,), 5.0, 2.0))
    assert abs(x.mean() - 5.0) < 0.1
    assert abs(x.std() - 2.0) < 0.1
    mean, std = backend.asarray([0.0, 100.0]), backend.asarray([1.0, 0.001])
    y = host(backend, stream(backend).normal((5000, 2), mean, std))
    assert abs(y[:, 1].mean() - 100.0) < 0.01 and y[:, 1].std() < 0.01


def test_integers_are_in_range_and_cover_it(backend: Backend):
    x = host(backend, stream(backend).integers(3, 8, (2000,)))
    assert x.min() == 3 and x.max() == 7
    assert set(x.tolist()) == {3, 4, 5, 6, 7}


def test_permutation_is_a_permutation(backend: Backend):
    p = host(backend, stream(backend).permutation(50))
    assert sorted(p.tolist()) == list(range(50))
    assert p.tolist() != list(range(50))


def test_choice_without_replacement_is_distinct(backend: Backend):
    c = host(backend, stream(backend).choice(20, 20, replace=False))
    assert sorted(c.tolist()) == list(range(20))
    d = host(backend, stream(backend).choice(20, 5, replace=False))
    assert len(set(d.tolist())) == 5 and d.min() >= 0 and d.max() < 20


def test_choice_follows_the_probabilities(backend: Backend):
    p = [0.0, 0.1, 0.0, 0.6, 0.3]
    c = host(backend, stream(backend).choice(5, 20000, p=p))
    counts = np.bincount(c, minlength=5) / 20000
    assert counts[0] == 0 and counts[2] == 0
    np.testing.assert_allclose(counts, p, atol=0.02)


def test_choice_normalises_the_probabilities_and_accepts_arrays(backend: Backend):
    c = host(backend, stream(backend).choice(3, 3000, p=backend.asarray([1.0, 1.0, 2.0])))
    np.testing.assert_allclose(np.bincount(c, minlength=3) / 3000, [0.25, 0.25, 0.5], atol=0.04)


def test_choice_without_replacement_with_probabilities(backend: Backend):
    c = host(backend, stream(backend).choice(6, 3, p=[0, 0, 0, 1, 1, 1], replace=False))
    assert sorted(c.tolist()) == [3, 4, 5]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n": 0, "size": 1}, "n must be positive"),
        ({"n": 3, "size": -1}, "size must not be negative"),
        ({"n": 3, "size": 4, "replace": False}, "cannot draw 4 distinct"),
        ({"n": 3, "size": 1, "p": [1.0, 1.0]}, "p must have shape"),
        ({"n": 3, "size": 1, "p": [1.0, -1.0, 1.0]}, "finite, non-negative"),
        ({"n": 3, "size": 1, "p": [1.0, float("nan"), 1.0]}, "finite, non-negative"),
        ({"n": 3, "size": 1, "p": [0.0, 0.0, 0.0]}, "positive sum"),
        ({"n": 3, "size": 3, "p": [1.0, 1.0, 0.0], "replace": False}, "non-zero probability"),
    ],
)
def test_choice_validation(backend: Backend, kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        stream(backend).choice(**kwargs)


def test_argument_validation(backend: Backend):
    s = stream(backend)
    with pytest.raises(ValueError, match="low must not exceed high"):
        s.uniform(3, 2.0, 1.0)
    with pytest.raises(ValueError, match="std must not be negative"):
        s.normal(3, 0.0, -1.0)
    with pytest.raises(ValueError, match="low must be below high"):
        s.integers(3, 3)
    with pytest.raises(ValueError, match="negative dimensions"):
        s.uniform((-1,))
    with pytest.raises(ValueError, match="n must not be negative"):
        s.permutation(-1)


def test_state_dict_is_json_serialisable(backend: Backend):
    s = stream(backend)
    s.uniform((5,))
    state = s.state_dict()
    assert json.loads(json.dumps(state)) == state
    assert state["backend"] == backend.name


def test_state_dict_round_trip_continues_the_exact_sequence(backend: Backend):
    s = stream(backend)
    s.uniform((17,))
    s.normal((3,))
    saved = json.loads(json.dumps(s.state_dict()))  # through JSON, as a checkpoint would be
    expected = [
        host(backend, s.uniform((9,))),
        host(backend, s.normal((9,))),
        host(backend, s.integers(0, 50, (9,))),
        host(backend, s.choice(7, 4)),
    ]

    restored = stream(backend, "something-else", 99)  # a different stream, brought to the saved state
    restored.load_state_dict(saved)
    actual = [
        host(backend, restored.uniform((9,))),
        host(backend, restored.normal((9,))),
        host(backend, restored.integers(0, 50, (9,))),
        host(backend, restored.choice(7, 4)),
    ]
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)


def test_state_dict_is_a_snapshot(backend: Backend):
    s = stream(backend)
    state = s.state_dict()
    frozen = json.dumps(state)
    s.uniform((10,))
    assert json.dumps(state) == frozen
    assert json.dumps(s.state_dict()) != frozen


def test_state_dict_round_trip_without_any_draws(backend: Backend):
    s = stream(backend)
    first = host(backend, stream(backend).uniform((5,)))
    s.load_state_dict(stream(backend).state_dict())
    np.testing.assert_array_equal(host(backend, s.uniform((5,))), first)


def test_loading_a_state_of_another_backend_is_an_error():
    if Backend().name == "numpy":
        pytest.importorskip("torch")
    numpy_stream, torch_stream = stream(Backend("numpy")), stream(Backend("torch"))
    with pytest.raises(ValueError, match="cannot load a 'torch' state into a 'numpy' stream"):
        numpy_stream.load_state_dict(torch_stream.state_dict())
    with pytest.raises(ValueError, match="cannot load a 'numpy' state into a 'torch' stream"):
        torch_stream.load_state_dict(numpy_stream.state_dict())


def test_numpy_and_torch_are_different_streams():
    pytest.importorskip("torch")
    a = host(Backend("numpy"), stream(Backend("numpy")).uniform((5,)))
    b = host(Backend("torch"), stream(Backend("torch")).uniform((5,)))
    assert not np.array_equal(a, b)


_SCRIPT = """
import json, sys
from auxein.backend import Backend
from auxein.random import RunSeed
backend = Backend(sys.argv[1], "cpu", sys.argv[2])
s = RunSeed(2024).stream("evaluation", 12345, backend=backend)
out = [backend.to_numpy(x).tolist() for x in (s.uniform((6,)), s.normal((6,)), s.integers(0, 1000, (6,)), s.permutation(8), s.choice(10, 5, p=list(range(1, 11))))]
print(json.dumps(out))
"""


@pytest.mark.parametrize(
    ("name", "precision"),
    [
        ("numpy", "float64"),
        ("numpy", "float32"),
        ("torch", "float64"),
        ("torch", "float32"),
    ],
)
def test_identical_draws_in_separate_processes_whatever_the_hash_seed(name: str, precision: str):
    if name == "torch":
        pytest.importorskip("torch")
    here = [Backend(name, "cpu", precision).to_numpy(x).tolist() for x in _draws(Backend(name, "cpu", precision))]
    outputs = []
    for hash_seed in ("0", "1", "random"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run([sys.executable, "-c", _SCRIPT, name, precision], env=env, capture_output=True, text=True, check=True)
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1] == outputs[2] == here


def _draws(backend: Backend):
    s = RunSeed(2024).stream("evaluation", 12345, backend=backend)
    return (s.uniform((6,)), s.normal((6,)), s.integers(0, 1000, (6,)), s.permutation(8), s.choice(10, 5, p=list(range(1, 11))))
