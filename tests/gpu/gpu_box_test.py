"""`Box` sampling on the device, including at the bounds and on a log scale."""

import numpy as np

from auxein.backend import Backend
from auxein.random import RunSeed
from auxein.spaces import Box
from tests.support.fixtures import assert_on_backend


def test_samples_are_on_the_device_and_inside_the_box(gpu_backend: Backend):
    box = Box([-5.0, 0.0, 1e-3], [5.0, 1.0, 1e3], dim=3, log_scale=[False, False, True])
    samples = box.sample_genomes(5000, RunSeed(1).stream("strategy", backend=gpu_backend), gpu_backend)
    assert_on_backend(samples, gpu_backend)
    host = gpu_backend.to_numpy(samples).astype(np.float64)
    assert (host >= box.lower).all() and (host <= box.upper).all()
    assert abs(host[:, 0].mean()) < 0.2 and 0.45 < host[:, 1].mean() < 0.55
    assert 1.6 < np.log10(host[:, 2]).std() < 1.85  # log-uniform over six decades: 6 / sqrt(12) = 1.73 decades of spread


def test_float32_samples_never_exceed_a_box_whose_bounds_float32_cannot_represent(float32_backend: Backend):
    box = Box(0.1, 0.3, dim=4)  # neither bound is a float32
    samples = float32_backend.to_numpy(box.sample_genomes(20_000, RunSeed(2).stream("strategy", backend=float32_backend), float32_backend))
    assert samples.dtype == np.float32
    assert (samples.astype(np.float64) >= 0.1).all() and (samples.astype(np.float64) <= 0.3).all()


def test_a_genome_from_the_device_is_checked_against_the_box(gpu_backend: Backend):
    box = Box(-1.0, 1.0, dim=3)
    inside = gpu_backend.asarray([0.5, -0.5, 0.0])
    outside = gpu_backend.asarray([0.5, -0.5, 2.0])
    assert box.contains(inside) and not box.contains(outside)
