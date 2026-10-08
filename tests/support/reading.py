"""Reads a recorded run and closes it again at once, so that tests leave no database connection to be finalised by the garbage collector."""

from pathlib import Path

from auxein.recording import CheckpointInfo, RecordedEvaluation, open_run


class Peek:
    def __init__(self, path: Path) -> None:
        self._path = path

    def checkpoints(self) -> list[CheckpointInfo]:
        with open_run(self._path) as run:
            return run.checkpoints()

    @property
    def metadata(self) -> dict[str, object]:
        with open_run(self._path) as run:
            return run.metadata

    @property
    def sessions(self) -> list[dict[str, object]]:
        with open_run(self._path) as run:
            return run.sessions

    def evaluations(self) -> list[RecordedEvaluation]:
        with open_run(self._path) as run:
            return list(run.evaluations())


def peek(path: Path) -> Peek:
    return Peek(path)
