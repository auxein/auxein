"""Candidate ids: deterministic, run-scoped counters (design doc §4.2)."""

from typing import NewType

from auxein.random.seed import MAX_KEY

CandidateId = NewType("CandidateId", int)
"""The id of a candidate. A run-scoped counter, never a random UUID: evaluation randomness is derived from it (§8)."""


class IdIssuer:
    """Issues candidate ids 0, 1, 2, ... in order.

    Ids are deterministic because they seed the per-candidate evaluation streams, so that the same run always gives the
    same candidate the same id. They are limited to 32 bits (the range of a stream key), about four billion candidates.
    """

    def __init__(self, start: int = 0) -> None:
        if start < 0:
            raise ValueError(f"ids start at 0 or above, got {start}")
        self._next = start

    def next(self) -> CandidateId:
        """The next id."""
        if self._next > MAX_KEY:
            raise OverflowError(f"this run has issued all {MAX_KEY + 1} candidate ids that evaluation streams can be keyed by")
        issued = CandidateId(self._next)
        self._next += 1
        return issued

    @property
    def issued(self) -> int:
        """How many ids have been issued so far."""
        return self._next

    def state_dict(self) -> dict[str, int]:
        """JSON-serialisable state, for checkpoints."""
        return {"next": self._next}

    def load_state_dict(self, state: dict[str, int]) -> None:
        if set(state) != {"next"} or isinstance(state["next"], bool) or not isinstance(state["next"], int) or state["next"] < 0:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"invalid IdIssuer state: {state!r}")
        self._next = state["next"]
