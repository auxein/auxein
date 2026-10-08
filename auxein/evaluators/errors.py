"""Errors raised by evaluators."""

from collections.abc import Sequence

from auxein.core import CandidateId

_SHOWN = 5


class EvaluationError(Exception):
    """User code raised while evaluating one or more candidates, and the run is set to stop at a failure.

    That is the `fail_fast` failure policy (design doc §6.6); under the default `infeasible` policy a failure is recorded
    as a `FAILED` or `TIMEOUT` evaluation instead, and the run goes on. The error names the candidate id(s) and chains the
    original exception as `__cause__`, so both the candidate and the traceback of the user's code are visible.
    """

    def __init__(self, candidate_ids: Sequence[CandidateId], cause: Exception | None = None, *, detail: str | None = None) -> None:
        self.candidate_ids = tuple(candidate_ids)
        shown = ", ".join(str(i) for i in self.candidate_ids[:_SHOWN])
        more = f", ... ({len(self.candidate_ids)} candidates)" if len(self.candidate_ids) > _SHOWN else ""
        noun = "candidate" if len(self.candidate_ids) == 1 else "candidates"
        what = detail if cause is None else f"{type(cause).__name__}: {cause}"
        super().__init__(f"evaluating {noun} {shown}{more} failed: {what}")
