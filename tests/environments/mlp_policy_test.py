"""`MLPPolicy`: the genome layout, the numbers, batched against single decoding, backends, the fan-in-scaled space, validation."""

import math
import pickle
from typing import Any

import numpy as np
import pytest

from auxein.backend import Backend
from auxein.environments import BatchPolicy, Clip, MLPAgent, MLPAgentBatch, MLPPolicy, Tanh
from auxein.random import RunSeed
from tests.support.fixtures import assert_on_backend, assert_on_device

ARCHITECTURES = [(3, (), 2), (3, (4,), 2), (16, (32,), 2), (5, (8, 8), 3), (1, (2, 3, 4), 1)]


def dimension(observation_size: int, hidden: tuple[int, ...], outputs: int) -> int:
    sizes = (observation_size, *hidden, outputs)
    return sum(out * inp + out for inp, out in zip(sizes[:-1], sizes[1:], strict=True))


# --- shapes and layout ---


@pytest.mark.parametrize(("observation_size", "hidden", "outputs"), ARCHITECTURES)
def test_the_genome_length_follows_the_documented_layout(observation_size: int, hidden: tuple[int, ...], outputs: int):
    policy = MLPPolicy(observation_size, hidden, outputs)
    assert policy.dim == dimension(observation_size, hidden, outputs)
    assert policy.space().dim == policy.dim
    assert policy.layers[0][0] == observation_size and policy.layers[-1][1] == outputs and len(policy.layers) == len(hidden) + 1


def test_the_documented_example_has_26_values_and_a_linear_policy_has_no_hidden_layer():
    assert MLPPolicy(3, (4,), 2).dim == 12 + 4 + 8 + 2
    linear = MLPPolicy(4, (), 2)
    assert linear.dim == 2 * (4 + 1)  # the same layout as LinearPolicy: the weight matrix row by row, then the bias
    agent = linear.decode(np.arange(10.0))
    np.testing.assert_array_equal(agent.weights[0], np.arange(8.0).reshape(2, 4))
    np.testing.assert_array_equal(agent.biases[0], [8.0, 9.0])


@pytest.mark.parametrize(("observation_size", "hidden", "outputs"), ARCHITECTURES)
def test_explain_returns_the_layers_by_name_and_they_rebuild_the_genome_exactly(
    observation_size: int, hidden: tuple[int, ...], outputs: int
):
    policy = MLPPolicy(observation_size, hidden, outputs)
    genome = np.random.default_rng(0).normal(size=policy.dim)
    layers = policy.explain(genome)
    assert list(layers) == [*(f"hidden{k + 1}" for k in range(len(hidden))), "output"]
    rebuilt = np.concatenate([np.concatenate([np.asarray(v["weights"]).reshape(-1), v["bias"]]) for v in layers.values()])  # type: ignore[index]
    np.testing.assert_array_equal(rebuilt, genome)
    first = layers["hidden1" if hidden else "output"]
    assert np.asarray(first["weights"]).shape == (hidden[0] if hidden else outputs, observation_size)  # type: ignore[index]


def test_a_decoded_batch_holds_each_layers_weights_and_biases_as_views_of_the_genomes(backend: Backend):
    policy = MLPPolicy(3, (4,), 2)
    genomes = backend.asarray(np.arange(3 * policy.dim, dtype=np.float64).reshape(3, policy.dim))
    batch = policy.decode_batch(genomes)
    assert isinstance(batch, MLPAgentBatch) and batch.n == 3 and batch.outputs == 2
    assert [tuple(w.shape) for w in batch.weights] == [(3, 4, 3), (3, 2, 4)]
    assert [tuple(b.shape) for b in batch.biases] == [(3, 4), (3, 2)]
    host = backend.to_numpy(genomes)
    np.testing.assert_array_equal(backend.to_numpy(batch.weights[0]), host[:, :12].reshape(3, 4, 3))
    np.testing.assert_array_equal(backend.to_numpy(batch.biases[0]), host[:, 12:16])
    np.testing.assert_array_equal(backend.to_numpy(batch.weights[1]), host[:, 16:24].reshape(3, 2, 4))
    np.testing.assert_array_equal(backend.to_numpy(batch.biases[1]), host[:, 24:26])


# --- the numbers, against a network computed by hand ---

W1 = [[1.0, -1.0], [0.5, 2.0]]
B1 = [0.1, -0.2]
W2 = [[1.0, 1.0], [-1.0, 2.0]]
B2 = [0.0, 0.5]
GENOME = np.array([*sum(W1, []), *B1, *sum(W2, []), *B2])
OBSERVATION = [0.3, -0.4]
ACTIVATIONS = {"tanh": math.tanh, "relu": lambda x: max(x, 0.0), "identity": lambda x: x}


def by_hand(activation: str) -> list[float]:
    """The raw outputs of the network above, in plain Python: independent of the implementation under test."""
    act = ACTIVATIONS[activation]
    hidden = [act(sum(w * o for w, o in zip(row, OBSERVATION, strict=True)) + b) for row, b in zip(W1, B1, strict=True)]
    return [sum(w * h for w, h in zip(row, hidden, strict=True)) + b for row, b in zip(W2, B2, strict=True)]


@pytest.mark.parametrize("activation", ["tanh", "relu", "identity"])
@pytest.mark.parametrize(
    "output",
    [
        "linear",
        "tanh",
        "argmax",
        Clip(-0.5, 0.25),
        Clip([-1.0, 0.0], [1.0, 0.3]),
        Tanh(),
        Tanh(0.0, 10.0),
        Tanh([0.0, -5.0], [1.0, 5.0]),
    ],
)
def test_act_and_act_batch_match_the_network_computed_by_hand(backend: Backend, activation: Any, output: Any):
    raw = np.array(by_hand(activation))
    low, high = (output.low, output.high) if isinstance(output, (Clip, Tanh)) else (None, None)
    if isinstance(output, Clip):
        expected = np.clip(raw, low, high)
    elif isinstance(output, Tanh):
        expected = np.asarray(low) + (np.tanh(raw) + 1.0) * 0.5 * (np.asarray(high) - np.asarray(low))
    elif output == "tanh":
        expected = np.tanh(raw)
    elif output == "argmax":
        expected = int(raw.argmax())
    else:
        expected = raw
    policy = MLPPolicy(2, (2,), 2, activation=activation, output=output)
    rtol, atol = (1e-12, 1e-12) if backend.precision == "float64" else (1e-5, 1e-6)
    single = policy.decode(GENOME).act(np.array(OBSERVATION))
    batched = policy.decode_batch(backend.asarray(np.stack([GENOME, GENOME]))).act_batch(
        backend.asarray(np.array([OBSERVATION, OBSERVATION]))
    )
    np.testing.assert_allclose(np.asarray(single), expected, rtol=1e-12)  # the single agent computes in float64 whatever the backend
    np.testing.assert_allclose(backend.to_numpy(batched)[0], expected, rtol=rtol, atol=atol)
    if output == "argmax":
        assert isinstance(single, int) and single == expected and int(backend.to_numpy(batched)[1]) == expected


def test_each_output_mode_keeps_its_contract_on_large_inputs():
    genome = np.random.default_rng(1).normal(size=MLPPolicy(2, (3,), 2).dim) * 20.0
    big = np.random.default_rng(2).normal(size=(50, 2)) * 10.0
    modes: list[tuple[Any, float, float]] = [("tanh", -1.0, 1.0), (Clip(-0.5, 0.25), -0.5, 0.25), (Tanh(2.0, 3.0), 2.0, 3.0)]
    for output, low, high in modes:
        actions = np.asarray(MLPPolicy(2, (3,), 2, output=output).decode(genome).act(big))
        assert actions.min() >= low and actions.max() <= high
    linear = np.asarray(MLPPolicy(2, (3,), 2).decode(genome).act(big))
    assert linear.max() > 5.0  # raw outputs are unbounded
    indices = np.asarray(MLPPolicy(2, (3,), 2, output="argmax").decode(genome).act(big))
    assert indices.shape == (50,) and set(indices.tolist()) <= {0, 1} and indices.dtype.kind == "i"


def test_a_clip_bound_may_be_infinite_on_one_side_and_a_tanh_one_may_not():
    policy = MLPPolicy(2, (), 2, output=Clip(-1.0, math.inf))
    assert np.asarray(policy.decode(np.array([0.0, 0.0, 0.0, 0.0, -5.0, 100.0])).act(np.zeros(2))).tolist() == [-1.0, 100.0]
    with pytest.raises(ValueError, match="finite bounds"):
        MLPPolicy(2, (), 2, output=Tanh(-math.inf, 1.0))


# --- batched equals single ---


@pytest.mark.parametrize(("observation_size", "hidden", "outputs"), ARCHITECTURES)
@pytest.mark.parametrize("output", ["linear", "tanh", "argmax"])
def test_act_batch_equals_decode_then_act_for_each_candidate_including_team_shaped_observations(
    backend: Backend, observation_size: int, hidden: tuple[int, ...], outputs: int, output: str
):
    policy = MLPPolicy(observation_size, hidden, outputs, output=output)  # type: ignore[arg-type]
    rng = np.random.default_rng(3)
    n, scenarios, agents = 4, 3, 5
    genomes = rng.normal(size=(n, policy.dim))
    observations = rng.normal(size=(n, scenarios, agents, observation_size))
    batch = policy.decode_batch(backend.asarray(genomes))
    batched = backend.to_numpy(batch.act_batch(backend.asarray(observations)))
    assert batched.shape == ((n, scenarios, agents) if output == "argmax" else (n, scenarios, agents, outputs))
    host_genomes = backend.to_numpy(backend.asarray(genomes))  # what this backend's precision made of them
    host_observations = backend.to_numpy(backend.asarray(observations))
    for i in range(n):
        agent = policy.decode(host_genomes[i])
        if backend.name == "numpy" and backend.precision == "float64":
            # the candidate's observations as one block reach the matrix product in the same grouping: bit for bit
            flat = np.asarray(agent.act(host_observations[i].reshape(-1, observation_size)))
            np.testing.assert_array_equal(flat.reshape(batched[i].shape), batched[i])
            per_team = np.stack([agent.act(host_observations[i, s]) for s in range(scenarios)])
            np.testing.assert_allclose(per_team, batched[i], rtol=1e-12, atol=1e-12)  # another grouping: a few ulp
        else:
            per_team = np.stack([agent.act(host_observations[i, s]) for s in range(scenarios)])
            if output == "argmax":
                assert (per_team == batched[i]).mean() > 0.98  # float32 rounding may flip a near tie
            else:
                np.testing.assert_allclose(per_team, batched[i], rtol=1e-4, atol=1e-4)


def test_the_batch_is_parameter_sharing_one_network_per_candidate_for_all_of_its_agents():
    policy = MLPPolicy(2, (3,), 1)
    rng = np.random.default_rng(4)
    genomes = rng.normal(size=(2, policy.dim))
    same = np.broadcast_to(rng.normal(size=(2,)), (2, 6, 2))  # every agent of both candidates sees the same observation
    actions = np.asarray(policy.decode_batch(genomes).act_batch(same))
    assert np.ptp(actions[0], axis=0).max() < 1e-12 and np.ptp(actions[1], axis=0).max() < 1e-12  # same network, same action
    assert abs(actions[0, 0, 0] - actions[1, 0, 0]) > 1e-3  # different candidates, different networks


def test_one_observation_a_team_and_a_flat_batch_have_matching_shapes():
    policy = MLPPolicy(3, (4,), 2)
    agent = policy.decode(np.random.default_rng(5).normal(size=policy.dim))
    assert isinstance(agent, MLPAgent)
    assert np.asarray(agent.act(np.zeros(3))).shape == (2,)
    assert np.asarray(agent.act(np.zeros((7, 3)))).shape == (7, 2)
    batch = policy.decode_batch(np.zeros((4, policy.dim)))
    assert batch.act_batch(np.zeros((4, 3))).shape == (4, 2)  # one observation per candidate
    assert batch.act_batch(np.zeros((4, 9, 3))).shape == (4, 9, 2)
    assert batch.act_batch(np.zeros((4, 2, 3, 5, 3))).shape == (4, 2, 3, 5, 2)


def test_the_agent_batch_implements_the_batch_policy_protocol_and_the_agent_pickles():
    policy = MLPPolicy(3, (4,), 2, output=Tanh(-2.0, 2.0))
    assert isinstance(policy.decode_batch(np.zeros((2, policy.dim))), BatchPolicy)
    agent = policy.decode(np.random.default_rng(6).normal(size=policy.dim))
    copy = pickle.loads(pickle.dumps(agent))
    observation = np.array([0.1, 0.2, 0.3])
    np.testing.assert_array_equal(np.asarray(copy.act(observation)), np.asarray(agent.act(observation)))
    assert pickle.loads(pickle.dumps(policy)).dim == policy.dim


# --- backends ---


def test_results_are_in_the_namespace_dtype_and_device_of_the_genome_array(backend: Backend):
    policy = MLPPolicy(4, (5,), 2, output="tanh")
    genomes = backend.asarray(np.random.default_rng(7).normal(size=(3, policy.dim)))
    batch = policy.decode_batch(genomes)
    for array in (*batch.weights, *batch.biases):
        assert_on_backend(array, backend)
    actions = batch.act_batch(backend.asarray(np.random.default_rng(8).normal(size=(3, 6, 4))))
    assert_on_backend(actions, backend)
    indices = MLPPolicy(4, (5,), 2, output="argmax").decode_batch(genomes).act_batch(backend.asarray(np.zeros((3, 6, 4))))
    assert_on_device(indices, backend)
    assert tuple(indices.shape) == (3, 6) and "int" in str(indices.dtype)
    bounded = MLPPolicy(4, (5,), 2, output=Tanh(-1.0, 1.0)).decode_batch(genomes).act_batch(backend.asarray(np.zeros((3, 4))))
    assert_on_backend(bounded, backend)  # the output bounds follow the genomes' backend, device and dtype too


def test_observations_from_another_backend_are_moved_to_the_genomes_backend(backend: Backend):
    policy = MLPPolicy(2, (3,), 1)
    batch = policy.decode_batch(backend.asarray(np.random.default_rng(9).normal(size=(2, policy.dim))))
    assert_on_backend(batch.act_batch(np.zeros((2, 4, 2), dtype=np.float32)), backend)


def test_decoding_and_acting_on_a_batch_never_go_through_the_host(backend: Backend, monkeypatch: pytest.MonkeyPatch):
    policy = MLPPolicy(4, (5, 5), 2, output=Tanh(-1.0, 1.0))
    genomes = backend.asarray(np.random.default_rng(10).normal(size=(3, policy.dim)))
    observations = backend.asarray(np.random.default_rng(11).normal(size=(3, 2, 4, 4)))

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a host round trip")

    monkeypatch.setattr(Backend, "to_numpy", refuse)
    if backend.name == "torch":
        import torch

        for name in ("numpy", "cpu", "tolist", "item", "__array__"):
            monkeypatch.setattr(torch.Tensor, name, refuse)
    result = policy.decode_batch(genomes).act_batch(observations)
    assert tuple(result.shape) == (3, 2, 4, 2)


# --- the genome space ---


def test_the_space_scales_each_weight_by_its_fan_in_and_each_bias_by_the_scale():
    policy = MLPPolicy(4, (9,), 2)
    space = policy.space(2.0)
    assert space.dim == policy.dim
    np.testing.assert_allclose(space.upper[:36], 2.0 / 2.0)  # first layer: fan-in 4
    np.testing.assert_allclose(space.upper[36:45], 2.0)  # its biases
    np.testing.assert_allclose(space.upper[45:63], 2.0 / 3.0)  # second layer: fan-in 9
    np.testing.assert_allclose(space.upper[63:], 2.0)
    np.testing.assert_array_equal(space.lower, -space.upper)
    with pytest.raises(ValueError, match="scale"):
        policy.space(0.0)


@pytest.mark.parametrize("hidden", [(32,), (64, 64)])
def test_random_networks_from_the_space_neither_saturate_nor_vanish(hidden: tuple[int, ...]):
    """Uniform sampling in `space()` is a sensible initialisation: unit-normal inputs keep hidden activations of order one."""
    policy = MLPPolicy(16, hidden, 4)
    backend = Backend()
    genomes = policy.space().sample_genomes(200, RunSeed(0).stream("init", backend=backend), backend)
    batch = policy.decode_batch(genomes)
    values = np.random.default_rng(0).normal(size=(200, 100, 16))
    for k, (weights, bias) in enumerate(zip(batch.weights[:-1], batch.biases[:-1], strict=True)):
        pre = np.einsum("nmi,noi->nmo", values, np.asarray(weights)) + np.asarray(bias)[:, None, :]
        values = np.tanh(pre)
        assert 0.3 < pre.std() < 1.2, f"layer {k}: pre-activations vanish or explode ({pre.std():.2f})"
        assert 0.25 < values.std() < 0.9 and np.mean(np.abs(values) > 0.99) < 0.05, f"layer {k}: saturated or vanished"
    # a 20 times larger scale does saturate, which is what the default avoids
    wide = policy.decode_batch(MLPPolicy(16, hidden, 4).space(20.0).sample_genomes(50, RunSeed(0).stream("init", backend=backend), backend))
    pre = np.einsum("nmi,noi->nmo", np.random.default_rng(0).normal(size=(50, 100, 16)), np.asarray(wide.weights[0]))
    assert np.mean(np.abs(np.tanh(pre + np.asarray(wide.biases[0])[:, None, :])) > 0.99) > 0.3


# --- validation ---


def test_sizes_and_widths_must_be_positive():
    for bad in (lambda: MLPPolicy(0, (), 1), lambda: MLPPolicy(3, (), 0), lambda: MLPPolicy(-1, (2,), 1)):
        with pytest.raises(ValueError, match="observation_size and outputs must be positive"):
            bad()
    with pytest.raises(ValueError, match=r"hidden layer widths must be positive, got \[4, 0\]"):
        MLPPolicy(3, (4, 0), 1)
    with pytest.raises(ValueError, match="unknown activation 'sigmoid'"):
        MLPPolicy(3, (4,), 1, activation="sigmoid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown output mode 'softmax'"):
        MLPPolicy(3, (4,), 1, output="softmax")  # type: ignore[arg-type]


def test_output_bounds_must_match_the_outputs():
    with pytest.raises(ValueError, match=r"low must be a number or one value per output \(2\), got shape \(3,\)"):
        MLPPolicy(3, (), 2, output=Clip([0.0, 0.0, 0.0], 1.0))
    with pytest.raises(ValueError, match=r"high must be a number or one value per output \(2\)"):
        MLPPolicy(3, (), 2, output=Tanh(0.0, [1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="at most its high bound"):
        MLPPolicy(3, (), 2, output=Clip([0.0, 2.0], [1.0, 1.0]))
    with pytest.raises(ValueError, match="contains NaN"):
        MLPPolicy(3, (), 1, output=Clip(math.nan, 1.0))


def test_a_genome_of_the_wrong_length_or_shape_is_refused_with_the_expected_length(backend: Backend):
    policy = MLPPolicy(3, (4,), 2)
    with pytest.raises(ValueError, match=r"a genome of this policy has 26 values, got shape \(5,\)"):
        policy.decode(backend.asarray(np.zeros(5)))
    with pytest.raises(ValueError, match=r"has 26 values"):
        policy.decode(backend.asarray(np.zeros((2, 26))))
    with pytest.raises(ValueError, match=r"expected shape \(n, 26\), got \(2, 5\)"):
        policy.decode_batch(backend.asarray(np.zeros((2, 5))))
    with pytest.raises(ValueError, match=r"expected shape \(n, 26\), got \(26,\)"):
        policy.decode_batch(backend.asarray(np.zeros(26)))


def test_observations_of_the_wrong_shape_are_refused_naming_the_expected_last_dimension(backend: Backend):
    policy = MLPPolicy(3, (4,), 2)
    batch = policy.decode_batch(backend.asarray(np.zeros((2, policy.dim))))
    for shape in ((2, 5), (2, 4, 4), (3, 4, 3), (3,), (2,)):
        with pytest.raises(ValueError, match=r"observations must have shape \(n, \.\.\., observation_size\) with n=2 .* dimension of 3"):
            batch.act_batch(backend.asarray(np.zeros(shape)))
    agent = policy.decode(np.zeros(policy.dim))
    for observation in (np.zeros(4), np.zeros((2, 4)), np.zeros((2, 3, 3)), np.zeros(())):
        with pytest.raises(ValueError, match=r"an observation must have shape \(3,\) or \(agents, 3\)"):
            agent.act(observation)


def test_the_policy_describes_itself_stably():
    assert repr(MLPPolicy(3, (4,), 2)) == "MLPPolicy(observation_size=3, hidden=(4,), outputs=2, activation='tanh', output='linear')"
    assert "output=Tanh(low=[-2.0, -2.0], high=[2.0, 2.0])" in repr(MLPPolicy(3, (), 2, output=Tanh(-2.0, 2.0)))
    assert "output=Clip(low=[0.0], high=[1.0])" in repr(MLPPolicy(3, (), 1, output=Clip(0.0, 1.0)))
    assert repr(MLPPolicy(3, (4,), 2)) != repr(MLPPolicy(3, (5,), 2)) != repr(MLPPolicy(3, (5,), 2, activation="relu"))
