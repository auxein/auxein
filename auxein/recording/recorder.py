"""The recorder interface: how a run is persisted (design doc §10.5)."""

from collections.abc import Mapping
from typing import Protocol, TypeVar

from auxein.core import Batch, EvaluationBatch, StateDict

G = TypeVar("G")


class Recorder(Protocol):
    """Receives the events of a run, from the driver, which is its only caller and the single writer.

    The run directory of `SQLiteRecorder` is the built-in implementation; other sinks (MLflow, Weights & Biases) can
    implement the same four hooks.
    """

    def on_start(self, metadata: Mapping[str, object]) -> None:
        """The run begins. `metadata` describes its configuration; the recorder adds the environment (versions, git)."""
        ...

    def on_batch(self, step: int, batch: Batch[G], results: EvaluationBatch[G], recorded: int = 0) -> None:
        """A batch was asked for and evaluated: its candidates, their lineage and their evaluations, in ask order.

        `recorded` is nonzero only when a resumed run replays: how many of the batch's first candidates came from the
        recording rather than from an evaluation (design doc §10.4)."""
        ...

    def on_tell(self, step: int, count: int, recorded: bool = False) -> None:
        """The strategy was told the results of `count` candidates of this step. `recorded`: the batch came whole from the recording."""
        ...

    def checkpoint(self, state: StateDict, evaluations_used: int, keep: int) -> None:
        """Save `state` as a checkpoint consistent with everything recorded so far, keeping the newest `keep`."""
        ...

    def on_end(self, status: str, stop_reason: str | None, summary: Mapping[str, object]) -> None:
        """The run is over. `status` is `completed`, `interrupted` or `failed`; `stop_reason` is None unless completed."""
        ...


class NoopRecorder:
    """What the driver calls when recording is disabled: it does nothing."""

    def on_start(self, metadata: Mapping[str, object]) -> None:
        pass

    def on_batch(self, step: int, batch: Batch[G], results: EvaluationBatch[G], recorded: int = 0) -> None:
        pass

    def on_tell(self, step: int, count: int, recorded: bool = False) -> None:
        pass

    def checkpoint(self, state: StateDict, evaluations_used: int, keep: int) -> None:
        pass

    def on_end(self, status: str, stop_reason: str | None, summary: Mapping[str, object]) -> None:
        pass
