"""Errors and warnings of the driver."""

from auxein.recording import ResumeError


class RecordingDisabledWarning(UserWarning):
    """Emitted once per run that has no `run_dir`: nothing is written to disk.

    Recording is opt-in because a run must not silently litter the current directory, but a run that was not recorded
    cannot be reconstructed afterwards (design doc §1.1, principle 6). To silence the warning for runs you do not want to
    record, add one line to your script:

        warnings.filterwarnings("ignore", category=RecordingDisabledWarning)
    """


class DriverError(RuntimeError):
    """The driver found a component breaking its contract."""


class StrategyError(DriverError):
    """A strategy returned an invalid batch from `ask`: empty, with ids that this run did not issue or has already seen, or
    with inconsistent steps."""


class EvaluatorError(DriverError):
    """An evaluator returned results that do not match the batch: not one evaluation per candidate, in ask order."""


class AllEvaluationsFailedError(DriverError):
    """The first evaluations of the run all failed, which almost certainly means the evaluation function is broken: a bug,
    not a result. The run stops (the recording is finalised as `failed`) instead of spending its whole budget on failures.
    Pass `initial_failure_guard=None` to `run` to turn this check off."""


class EvaluationFailureWarning(UserWarning):
    """The first evaluation of a run that did not succeed: raised once per run, with the candidate, the status and the full
    traceback, so that a bug in the evaluation function is seen at once and not found in the recording afterwards.
    Later failures are counted (`RunResult.status_counts`) and recorded, not warned about."""


class SteadyStateVectorisationWarning(UserWarning):
    """A `VectorisedEvaluator` is used with steady-state delivery, which calls it with one candidate at a time.

    That works but throws away the point of vectorising. Use generation delivery (`delivery="generation"`) instead.
    """


class ConfigurationMismatchError(ResumeError):
    """A run was resumed with settings that differ from the recorded ones. Only the budget may change when resuming: the
    message lists every setting that differs, with its recorded and its given value."""


class ReplayMismatchError(ResumeError):
    """Replaying a recorded run regenerated a candidate that differs from the recorded one, so the recording cannot be
    resumed: the strategy's configuration or code changed, or the seed is different. Nothing was recorded by the attempt."""


class ResumeWarning(UserWarning):
    """A resume that has nothing to do: the run is already complete with the given budget, so its result is returned
    without evaluating anything. Give a larger budget to extend the run."""
