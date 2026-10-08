"""What the built-in evaluators do when user code fails (design doc §6.6).

Two kinds of problem are kept apart. A *failure of an evaluation* (an exception in the user's function, a timeout, a worker
that died) says something about that candidate, so by default it is recorded as a `FAILED` or `TIMEOUT` evaluation and the
run goes on. *Misconfiguration* (a function that cannot be sent to a worker, a return value that breaks the contract)
means nothing will work, so it fails the run whatever the policy.
"""

from auxein.core import Candidate, EvalContext, Evaluation, Status, describe_exception
from auxein.core._typing import G
from auxein.evaluators.errors import EvaluationError
from auxein.execution import EvaluationTimeout, ExecutorError


def failure_of(candidate: Candidate[G], error: Exception, wall_time: float, ctx: EvalContext[G]) -> Evaluation[G]:
    """The evaluation to record for a candidate whose evaluation raised `error`, or an `EvaluationError` to stop the run.

    It stops the run for misconfiguration and under the `fail_fast` policy, chaining the original exception.
    """
    if isinstance(error, ExecutorError) or ctx.failure_policy == "fail_fast":
        raise EvaluationError([candidate.id], error) from error
    if isinstance(error, EvaluationTimeout):
        return Evaluation.failed(candidate, Status.TIMEOUT, f"timed out: {error}", error.timeout)
    return Evaluation.failed(candidate, Status.FAILED, describe_exception(error), wall_time)


def failures_of(candidates: list[Candidate[G]], error: Exception, wall_time: float, ctx: EvalContext[G]) -> list[Evaluation[G]]:
    """`failure_of` for a whole batch that failed together (a vectorised call): every candidate gets the same error."""
    if isinstance(error, ExecutorError) or ctx.failure_policy == "fail_fast":
        raise EvaluationError([c.id for c in candidates], error) from error
    text = describe_exception(error)
    each = wall_time / len(candidates) if candidates else 0.0
    return [Evaluation.failed(c, Status.FAILED, text, each) for c in candidates]
