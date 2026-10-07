"""The recorder interface: how a run is persisted (design doc §10.5)."""

from collections.abc import Mapping
from typing import Protocol, TypeVar

from auxein.core import Batch, EvaluationBatch

G = TypeVar("G")


class Recorder(Protocol):
    """Receives the events of a run, from the driver, which is its only caller and the single writer.

    The run directory of `SQLiteRecorder` is the built-in implementation; other sinks (MLflow, Weights & Biases) can
    implement the same four hooks.
    """

    def on_start(self, metadata: Mapping[str, object]) -> None:
        """The run begins. `metadata` describes its configuration; the recorder adds the environment (versions, git)."""
        ...

    def on_batch(self, step: int, batch: Batch[G], results: EvaluationBatch[G]) -> None:
        """A batch was asked for and evaluated: its candidates, their lineage and their evaluations, in ask order."""
        ...

    def on_tell(self, step: int, count: int) -> None:
        """The strategy was told the results of `count` candidates of this step."""
        ...

    def on_end(self, status: str, stop_reason: str | None, summary: Mapping[str, object]) -> None:
        """The run is over. `status` is `completed`, `interrupted` or `failed`; `stop_reason` is None unless completed."""
        ...


class NoopRecorder:
    """What the driver calls when recording is disabled: it does nothing."""

    def on_start(self, metadata: Mapping[str, object]) -> None:
        pass

    def on_batch(self, step: int, batch: Batch[G], results: EvaluationBatch[G]) -> None:
        pass

    def on_tell(self, step: int, count: int) -> None:
        pass

    def on_end(self, status: str, stop_reason: str | None, summary: Mapping[str, object]) -> None:
        pass
