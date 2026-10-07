import json

import pytest

from auxein.core import IdIssuer


def test_ids_are_consecutive_from_zero():
    issuer = IdIssuer()
    assert [issuer.next() for _ in range(5)] == [0, 1, 2, 3, 4]
    assert issuer.issued == 5


def test_two_issuers_are_deterministic_and_independent():
    a, b = IdIssuer(), IdIssuer()
    assert [a.next() for _ in range(100)] == [b.next() for _ in range(100)]
    a.next()
    assert b.next() == 100 and a.next() == 101


def test_ids_can_start_elsewhere():
    assert IdIssuer(10).next() == 10
    with pytest.raises(ValueError, match="start at 0"):
        IdIssuer(-1)


def test_the_id_counter_is_bound_to_the_stream_key_range():
    issuer = IdIssuer(2**32 - 1)
    assert issuer.next() == 2**32 - 1
    with pytest.raises(OverflowError, match="candidate ids"):
        issuer.next()


def test_state_round_trip_continues_the_sequence():
    issuer = IdIssuer()
    for _ in range(7):
        issuer.next()
    state = json.loads(json.dumps(issuer.state_dict()))
    expected = [issuer.next() for _ in range(3)]
    restored = IdIssuer()
    restored.load_state_dict(state)
    assert [restored.next() for _ in range(3)] == expected


@pytest.mark.parametrize("state", [{}, {"next": -1}, {"next": 1.5}, {"next": True}, {"next": 1, "extra": 2}, {"count": 3}])
def test_invalid_state_is_rejected(state: dict):
    with pytest.raises(ValueError, match="invalid IdIssuer state"):
        IdIssuer().load_state_dict(state)
