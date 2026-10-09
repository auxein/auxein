"""Random streams on the device: generation where the arrays live, and `state_dict` round trips."""

import numpy as np

from auxein.backend import Backend
from auxein.random import RunSeed
from tests.support.fixtures import assert_on_backend, assert_on_device


def test_a_stream_generates_on_the_device_in_the_backends_precision(gpu_backend: Backend):
    rng = RunSeed(1).stream("strategy", backend=gpu_backend)
    assert_on_backend(rng.uniform((4, 3)), gpu_backend)
    assert_on_backend(rng.normal((4, 3)), gpu_backend)
    for ints in (rng.integers(0, 10, (5,)), rng.permutation(7), rng.choice(9, 4)):
        assert_on_device(ints, gpu_backend)


def test_uniform_and_normal_have_the_right_moments(gpu_backend: Backend):
    rng = RunSeed(2).stream("strategy", backend=gpu_backend)
    uniform = gpu_backend.to_numpy(rng.uniform((200_000,), -1.0, 3.0)).astype(np.float64)
    normal = gpu_backend.to_numpy(rng.normal((200_000,), 1.0, 2.0)).astype(np.float64)
    assert -1.0 <= uniform.min() and uniform.max() <= 3.0 and abs(uniform.mean() - 1.0) < 0.02
    assert abs(normal.mean() - 1.0) < 0.03 and abs(normal.std() - 2.0) < 0.03


def test_the_same_seed_gives_the_same_numbers_on_the_same_device(gpu_backend: Backend):
    one = gpu_backend.to_numpy(RunSeed(3).stream("strategy", backend=gpu_backend).normal((50,)))
    two = gpu_backend.to_numpy(RunSeed(3).stream("strategy", backend=gpu_backend).normal((50,)))
    other = gpu_backend.to_numpy(RunSeed(4).stream("strategy", backend=gpu_backend).normal((50,)))
    np.testing.assert_array_equal(one, two)
    assert not np.array_equal(one, other)


def test_a_state_dict_round_trip_continues_the_stream_exactly(gpu_backend: Backend):
    rng = RunSeed(5).stream("strategy", backend=gpu_backend)
    rng.normal((100,))  # advance
    saved = rng.state_dict()
    expected = gpu_backend.to_numpy(rng.normal((20,)))
    restored = RunSeed(5).stream("strategy", backend=gpu_backend)
    restored.load_state_dict(saved)
    np.testing.assert_array_equal(gpu_backend.to_numpy(restored.normal((20,))), expected)
    assert_on_backend(restored.uniform((2,)), gpu_backend)


def test_a_stream_is_picklable_for_the_device_too(gpu_backend: Backend):
    import pickle

    rng = RunSeed(6).stream("strategy", backend=gpu_backend)
    rng.uniform((10,))
    copy = pickle.loads(pickle.dumps(rng))
    np.testing.assert_array_equal(gpu_backend.to_numpy(copy.normal((8,))), gpu_backend.to_numpy(rng.normal((8,))))
