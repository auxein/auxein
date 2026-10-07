"""Errors raised by evaluators."""

from collections.abc import Sequence

from auxein.core import CandidateId

_SHOWN = 5


class EvaluationError(Exception):
    """User code raised while evaluating one or more candidates.

    In this version an exception in user code fails the run (failure policies, which turn it into a recorded `FAILED`
    result, come later). The error names the candidate id(s) and chains the original exception as `__cause__`, so
    both the candidate and the traceback of the user's code are visible.
    """

    def __init__(self, candidate_ids: Sequence[CandidateId], cause: Exception) -> None:
        self.candidate_ids = tuple(candidate_ids)
        shown = ", ".join(str(i) for i in self.candidate_ids[:_SHOWN])
        more = f", ... ({len(self.candidate_ids)} candidates)" if len(self.candidate_ids) > _SHOWN else ""
        noun = "candidate" if len(self.candidate_ids) == 1 else "candidates"
        super().__init__(f"evaluating {noun} {shown}{more} failed: {type(cause).__name__}: {cause}")
