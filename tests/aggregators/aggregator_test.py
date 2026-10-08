import numpy as np
import pytest

from auxein.aggregators import Aggregator, cvar_lower, cvar_upper, maximum, mean, minimum, quantile, total
from auxein.backend import Backend
from auxein.core import Objective, ProblemSpec
from auxein.spaces import Box

# three candidates on five scenarios: hand-checkable values
VALUES = np.array(
    [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [10.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, -2.0, -3.0, -4.0, -5.0],
    ]
)


def measurements(backend: Backend) -> dict:
    return {"m": backend.asarray(VALUES), "w": backend.asarray(VALUES[:, ::-1].copy())}


def reduce(reduction, backend: Backend) -> list[float]:
    return backend.to_numpy(reduction.reduce(measurements(backend), backend.xp)).astype(np.float64).tolist()


TOLERANCE = {"float64": 1e-12, "float32": 1e-6}


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        (mean("m"), [3.0, 2.0, -3.0]),
        (minimum("m"), [1.0, 0.0, -5.0]),
        (maximum("m"), [5.0, 10.0, -1.0]),
        (total("m"), [15.0, 10.0, -15.0]),
        (quantile("m", 0.5), [3.0, 0.0, -3.0]),
        (quantile("m", 0.0), [1.0, 0.0, -5.0]),
        (quantile("m", 1.0), [5.0, 10.0, -1.0]),
        (quantile("m", 0.25), [2.0, 0.0, -4.0]),  # position 1 of 0..4
        (quantile("m", 0.1), [1.4, 0.0, -4.6]),  # interpolated between the two smallest
        (cvar_upper("m", 0.4), [4.5, 5.0, -1.5]),  # the largest ceil(0.4 * 5) = 2 values
        (cvar_upper("m", 0.2), [5.0, 10.0, -1.0]),  # one value: the maximum
        (cvar_upper("m", 1.0), [3.0, 2.0, -3.0]),  # all of them: the mean
        (cvar_lower("m", 0.4), [1.5, 0.0, -4.5]),  # the smallest two
        (cvar_lower("m", 0.2), [1.0, 0.0, -5.0]),
        (cvar_lower("m", 1.0), [3.0, 2.0, -3.0]),
        (cvar_upper("m", 0.3), [4.5, 5.0, -1.5]),  # ceil(0.3 * 5) = 2
    ],
    ids=lambda x: repr(x) if not isinstance(x, list) else "",
)
def test_every_reducer_on_hand_checked_arrays(backend: Backend, reduction, expected):
    got = reduce(reduction, backend)
    np.testing.assert_allclose(got, expected, atol=TOLERANCE[backend.precision] * 10, rtol=1e-6)


def test_the_tails_of_cvar_are_the_worst_for_opposite_directions():
    """Higher is worse for a cost: the upper tail. Lower is worse for a reward: the lower tail."""
    backend = Backend()
    cost = {"c": backend.asarray(np.array([[1.0, 2.0, 9.0, 10.0]]))}
    reward = {"r": backend.asarray(np.array([[1.0, 2.0, 9.0, 10.0]]))}
    assert float(cvar_upper("c", 0.5).reduce(cost, backend.xp)[0]) == 9.5
    assert float(cvar_lower("r", 0.5).reduce(reward, backend.xp)[0]) == 1.5
    assert (
        float(cvar_lower("c", 0.5).reduce(cost, backend.xp)[0])
        < float(mean("c").reduce(cost, backend.xp)[0])
        < float(cvar_upper("c", 0.5).reduce(cost, backend.xp)[0])
    )


def test_function_sources_get_the_dict_of_arrays(backend: Backend):
    spread = mean(lambda m: m["m"] - m["w"])
    expected = (VALUES - VALUES[:, ::-1]).mean(axis=1)
    np.testing.assert_allclose(reduce(spread, backend), expected, atol=1e-5)
    worst = maximum(lambda m: (m["m"] + abs(m["m"])) / 2.0)  # a violation: the positive part, worst over scenarios
    np.testing.assert_allclose(reduce(worst, backend), [5.0, 10.0, 0.0], atol=1e-5)


def test_a_missing_measurement_says_what_the_environment_reported(backend: Backend):
    with pytest.raises(KeyError, match=r"reads the measurement 'nope'.*\['m', 'w'\]"):
        mean("nope").reduce(measurements(backend), backend.xp)


def test_parameters_are_validated():
    for bad in (lambda: quantile("m", 1.5), lambda: quantile("m", -0.1), lambda: cvar_upper("m", 0.0), lambda: cvar_lower("m", 1.2)):
        with pytest.raises(ValueError):
            bad()


def test_a_reduction_needs_a_candidates_by_scenarios_array(backend: Backend):
    with pytest.raises(ValueError, match=r"shape \(candidates, scenarios\)"):
        mean(lambda m: m["m"][:, 0]).reduce(measurements(backend), backend.xp)


def test_reductions_describe_themselves():
    assert repr(mean("fuel")) == "mean('fuel')" and repr(quantile("t", 0.9)) == "quantile('t', 0.9)"
    assert repr(cvar_upper("t", 0.1)) == "cvar_upper('t', 0.1)"
    assert repr(mean(lambda m: m["a"])).startswith("mean(<") and repr(mean(lambda m: m["a"])).endswith(">)")


def problem(constraints=(), descriptors=()) -> ProblemSpec:
    return ProblemSpec(Box(0.0, 1.0, dim=2), (Objective("fuel"), Objective("time")), constraints, descriptors)


def test_the_aggregator_computes_objectives_constraints_descriptors_and_costs(backend: Backend):
    aggregator = Aggregator(
        objectives={"fuel": mean("m"), "time": maximum("w")},
        constraints={"cpa": maximum(lambda m: (m["m"] + abs(m["m"])) / 2.0)},
        descriptors={"speed": minimum("m")},
        cost={"tokens": total(lambda m: abs(m["m"]))},
    )
    aggregated = aggregator.aggregate(measurements(backend), backend)
    np.testing.assert_allclose(aggregated.objectives["fuel"], [3.0, 2.0, -3.0], atol=1e-5)
    np.testing.assert_allclose(aggregated.objectives["time"], [5.0, 10.0, -1.0], atol=1e-5)
    np.testing.assert_allclose(aggregated.constraints["cpa"], [5.0, 10.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(aggregated.descriptors["speed"], [1.0, 0.0, -5.0], atol=1e-5)
    np.testing.assert_allclose(aggregated.cost["tokens"], [15.0, 10.0, 15.0], atol=1e-5)
    assert aggregated.invalid == {} and aggregated.objectives["fuel"].dtype == np.float64  # host float64 columns


def test_non_finite_constraints_and_costs_are_reported_per_row(backend: Backend):
    data = VALUES.copy()
    data[1, 2] = np.nan
    aggregator = Aggregator({"o": mean("m")}, {"c": maximum("m")}, cost={"t": total("m")})
    aggregated = aggregator.aggregate({"m": backend.asarray(data)}, backend)
    assert list(aggregated.invalid) == [1] and "constraint 'c'" in aggregated.invalid[1] and "cost unit 't'" in aggregated.invalid[1]


def test_names_are_checked_against_the_problem_like_a_result():
    good = Aggregator({"fuel": mean("m"), "time": mean("m")}, {"cpa": mean("m")}, {"speed": mean("m")})
    good.validate(problem(("cpa",), ("speed",)))
    with pytest.raises(ValueError, match=r"objectives the problem declares but the aggregator lacks: \['time'\]"):
        Aggregator({"fuel": mean("m")}).validate(problem())
    with pytest.raises(ValueError, match=r"constraints the aggregator computes but the problem does not declare: \['cpa'\]"):
        good.validate(problem((), ("speed",)))
    typo = Aggregator({"fuel": mean("m"), "tmie": mean("m")})
    with pytest.raises(ValueError) as raised:
        typo.validate(problem())
    assert "lacks: ['time']" in str(raised.value) and "does not declare: ['tmie']" in str(raised.value)


def test_construction_is_validated():
    with pytest.raises(ValueError, match="at least one objective"):
        Aggregator({})
    with pytest.raises(TypeError, match="must be a Reduction"):
        Aggregator({"o": "mean"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-empty"):
        Aggregator({"": mean("m")})


def test_the_description_names_every_part():
    text = Aggregator({"fuel": mean("f")}, {"cpa": maximum("c")}, {"d": minimum("d")}, {"tokens": total("t")}).describe()
    expected = (
        "Aggregator(objectives={'fuel': mean('f')}, constraints={'cpa': maximum('c')}, "
        "descriptors={'d': minimum('d')}, cost={'tokens': total('t')})"
    )
    assert text == expected
