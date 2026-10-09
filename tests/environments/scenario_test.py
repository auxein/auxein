import pickle
from pathlib import Path

import numpy as np
import pytest

from auxein.environments import Scenario, ScenarioSet


def params(index: int, rng) -> dict:
    return {"x": float(rng.uniform((1,))[0]), "k": index}


def test_a_scenario_validates_and_hashes_by_id():
    scenario = Scenario("a", 0, 5, {"x": 1.0, "nested": {"list": [1, 2]}})
    assert scenario.params["x"] == 1.0 and dict(scenario.params)["nested"] == {"list": [1, 2]}
    assert hash(scenario) == hash(Scenario("a", 3, 9, {"other": 1}))  # hashed by id, whatever the rest
    assert scenario != Scenario("a", 0, 5, {"x": 2.0})  # but equal only when everything is
    for bad in (lambda: Scenario("", 0, 0, {}), lambda: Scenario("a", -1, 0, {}), lambda: Scenario("a", 0, -1, {})):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(TypeError, match="JSON-serialisable"):
        Scenario("a", 0, 0, {"f": lambda: 0})


def test_params_are_read_only_copies_and_the_scenario_pickles():
    source = {"x": [1, 2]}
    scenario = Scenario("a", 0, 5, source)
    source["x"].append(3)
    assert scenario.params["x"] == [1, 2]
    with pytest.raises(TypeError):
        scenario.params["y"] = 1  # type: ignore[index]
    again = pickle.loads(pickle.dumps(scenario))
    assert again == scenario and again.params["x"] == [1, 2]  # worker processes receive scenarios


def test_the_world_stream_depends_on_the_scenario_seed_alone():
    a, b = Scenario("a", 0, 11, {}), Scenario("b", 7, 11, {"x": 1})
    np.testing.assert_array_equal(a.rng().normal((5,)), b.rng().normal((5,)))
    assert not np.array_equal(a.rng().normal((5,)), Scenario("c", 0, 12, {}).rng().normal((5,)))
    assert not np.array_equal(a.rng(0).normal((5,)), a.rng(1).normal((5,)))  # independent streams within a scenario


def test_generation_is_deterministic_per_seed():
    a, b, other = (
        ScenarioSet.generate(params, 12, seed=4),
        ScenarioSet.generate(params, 12, seed=4),
        ScenarioSet.generate(params, 12, seed=5),
    )
    assert a == b and a.fingerprint == b.fingerprint and [s.seed for s in a] == [s.seed for s in b]
    assert a.fingerprint != other.fingerprint
    assert [s.index for s in a] == list(range(12)) and a[3].id == "s0003" and len(set(s.seed for s in a)) == 12


def test_the_fingerprint_is_stable_and_changes_with_any_param_id_or_seed():
    base = ScenarioSet.from_params([{"a": 1}, {"a": 2}], seed=1)
    assert base.fingerprint == ScenarioSet.from_params([{"a": 1}, {"a": 2}], seed=1).fingerprint
    assert len(base.fingerprint) == 64
    assert base.fingerprint != ScenarioSet.from_params([{"a": 1}, {"a": 3}], seed=1).fingerprint  # a param
    assert base.fingerprint != ScenarioSet.from_params([{"a": 1}, {"a": 2}], seed=2).fingerprint  # the seeds
    assert base.fingerprint != ScenarioSet.from_params([{"a": 1}, {"a": 2}], seed=1, ids=["x", "y"]).fingerprint  # the ids
    assert base.fingerprint != ScenarioSet.from_params([{"a": 2}, {"a": 1}], seed=1).fingerprint  # the order
    assert "fingerprint" in repr(base)


def test_the_fingerprint_does_not_depend_on_dict_key_order():
    assert ScenarioSet.from_params([{"a": 1, "b": 2}]).fingerprint == ScenarioSet.from_params([{"b": 2, "a": 1}]).fingerprint


def test_a_set_checks_its_scenarios():
    with pytest.raises(ValueError, match="at least one"):
        ScenarioSet([])
    with pytest.raises(ValueError, match="index 1 but is at position 0"):
        ScenarioSet([Scenario("a", 1, 0, {})])
    with pytest.raises(ValueError, match="unique"):
        ScenarioSet([Scenario("a", 0, 0, {}), Scenario("a", 1, 0, {})])
    with pytest.raises(ValueError, match="ids"):
        ScenarioSet.from_params([{}], ids=["a", "b"])


def test_split_gives_disjoint_reindexed_sets():
    whole = ScenarioSet.generate(params, 10, seed=2)
    selection, held_out = whole.split(6, 3)
    assert len(selection) == 6 and len(held_out) == 3
    assert not set(selection.ids) & set(held_out.ids)
    assert not {s.seed for s in selection} & {s.seed for s in held_out}
    assert [s.index for s in held_out] == [0, 1, 2] and held_out[0].id == whole[6].id and held_out[0].seed == whole[6].seed
    with pytest.raises(ValueError, match="cannot split"):
        whole.split(8, 3)


def test_a_generated_split_does_not_overlap_and_is_the_same_whatever_the_sizes_are_split_into():
    selection, held_out = ScenarioSet.generate_split(params, 5, 4, seed=9)
    whole = ScenarioSet.generate(params, 9, seed=9)
    assert [dict(s.params) for s in selection] == [dict(s.params) for s in whole[:5]]
    assert [dict(s.params) for s in held_out] == [dict(s.params) for s in whole[5:]]
    assert not set(selection.ids) & set(held_out.ids)
    assert not [dict(s.params) for s in selection] == [dict(s.params) for s in held_out]
    with pytest.raises(ValueError):
        ScenarioSet.generate_split(params, 0, 3, seed=1)


def test_json_round_trip(tmp_path: Path):
    scenarios = ScenarioSet.generate(params, 6, seed=3)
    scenarios.save(tmp_path / "set.json")
    loaded = ScenarioSet.load(tmp_path / "set.json")
    assert loaded == scenarios and loaded.fingerprint == scenarios.fingerprint
    assert [(s.id, s.index, s.seed, dict(s.params)) for s in loaded] == [(s.id, s.index, s.seed, dict(s.params)) for s in scenarios]


def test_a_modified_or_unknown_file_is_refused(tmp_path: Path):
    scenarios = ScenarioSet.generate(params, 3, seed=3)
    document = scenarios.to_json()
    tampered = {**document, "scenarios": [{**document["scenarios"][0], "params": {"x": 99.0, "k": 0}}, *document["scenarios"][1:]]}  # type: ignore[index]
    with pytest.raises(ValueError, match="modified"):
        ScenarioSet.from_json(tampered)
    with pytest.raises(ValueError, match="format"):
        ScenarioSet.from_json({**document, "format": 7})
