"""Errors and warnings of the driver."""


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


class SteadyStateVectorisationWarning(UserWarning):
    """A `VectorisedEvaluator` is used with steady-state delivery, which calls it with one candidate at a time.

    That works but throws away the point of vectorising. Use generation delivery (`delivery="generation"`) instead.
    """
