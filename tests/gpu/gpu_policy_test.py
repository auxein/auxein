"""`MLPPolicy` on the device: decoding a batch and acting for whole teams, with the results staying there; and its probe."""

import statistics
import time
from typing import Any

import numpy as np

from auxein.backend import Backend
from auxein.environments import BatchPolicy, MLPPolicy, Tanh
from tests.gpu import report
from tests.gpu.conftest import device_name, synchronize
from tests.support.fixtures import assert_on_backend, assert_on_device

PROBE_NETWORKS = {"16 → 32 → 2": (32,), "16 → 64 → 64 → 2": (64, 64)}
CANDIDATES, SCENARIOS, AGENTS, OBSERVATION_SIZE, OUTPUTS = 100, 10, 40, 16, 2
CALLS, WARMUP = 20, 3


def test_decoding_and_acting_run_on_the_device_and_agree_with_numpy_float64(gpu_backend: Backend):
    policy = MLPPolicy(OBSERVATION_SIZE, (32,), OUTPUTS, output=Tanh(-2.0, 2.0))
    rng = np.random.default_rng(0)
    genomes = rng.uniform(-0.5, 0.5, size=(7, policy.dim))
    observations = rng.normal(size=(7, 3, AGENTS, OBSERVATION_SIZE))
    batch = policy.decode_batch(gpu_backend.asarray(genomes))
    for array in (*batch.weights, *batch.biases):
        assert_on_backend(array, gpu_backend)  # views of the genome array: never left the device
    actions = batch.act_batch(gpu_backend.asarray(observations))
    assert_on_backend(actions, gpu_backend)
    assert isinstance(batch, BatchPolicy) and tuple(actions.shape) == (7, 3, AGENTS, OUTPUTS)
    reference = policy.decode_batch(genomes).act_batch(observations)
    atol = 1e-4 if gpu_backend.precision == "float32" else 1e-9
    np.testing.assert_allclose(gpu_backend.to_numpy(actions), reference, atol=atol)


def test_argmax_indices_stay_on_the_device(gpu_backend: Backend):
    policy = MLPPolicy(OBSERVATION_SIZE, (8,), 4, output="argmax")
    genomes = gpu_backend.asarray(np.random.default_rng(1).uniform(-0.5, 0.5, size=(5, policy.dim)))
    indices = policy.decode_batch(genomes).act_batch(gpu_backend.asarray(np.random.default_rng(2).normal(size=(5, 6, OBSERVATION_SIZE))))
    assert_on_device(indices, gpu_backend)
    assert tuple(indices.shape) == (5, 6) and 0 <= int(indices.min()) and int(indices.max()) < 4


def median_milliseconds(call: Any, device: str) -> float:
    for _ in range(WARMUP):  # kernel compilation on Metal, cuBLAS initialisation on CUDA
        call()
    synchronize(device)
    times = []
    for _ in range(CALLS):
        start = time.perf_counter()
        call()
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def test_time_per_decode_and_act_on_the_cpu_and_on_the_device(device: str):
    import torch

    configurations = [
        ("numpy float64, CPU", Backend(), "cpu"),
        ("torch float32, CPU", Backend("torch", "cpu", "float32"), "cpu"),
        (f"torch float32, {device} ({device_name(device)})", Backend("torch", device, "float32"), device),
    ]
    rng = np.random.default_rng(0)
    for label, hidden in PROBE_NETWORKS.items():
        policy = MLPPolicy(OBSERVATION_SIZE, hidden, OUTPUTS, output="tanh")
        genomes = rng.uniform(-0.5, 0.5, size=(CANDIDATES, policy.dim))
        observations = rng.normal(size=(CANDIDATES, SCENARIOS, AGENTS, OBSERVATION_SIZE))
        for name, backend, where in configurations:
            g, o = backend.asarray(genomes), backend.asarray(observations)
            batch = policy.decode_batch(g)
            decode = median_milliseconds(lambda p=policy, g=g: p.decode_batch(g), where)  # type: ignore[misc]
            act = median_milliseconds(lambda b=batch, o=o: b.act_batch(o), where)  # type: ignore[misc]
            report.POLICY_ROWS.append(report.PolicyRow(label, name, decode, act))
    assert len(report.POLICY_ROWS) == len(PROBE_NETWORKS) * len(configurations)
    assert torch.__version__  # the probe needs torch
