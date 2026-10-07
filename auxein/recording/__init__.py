"""Recording runs (design doc §10): a recorder protocol, the SQLite run directory and a minimal reader."""

from auxein.recording.genomes import GenomeEncodingError
from auxein.recording.reader import RecordedEvaluation, RunReader, open_run
from auxein.recording.recorder import NoopRecorder, Recorder
from auxein.recording.sqlite import SCHEMA_VERSION, RunDirectoryError, SQLiteRecorder

__all__ = [
    "SCHEMA_VERSION",
    "GenomeEncodingError",
    "NoopRecorder",
    "RecordedEvaluation",
    "Recorder",
    "RunDirectoryError",
    "RunReader",
    "SQLiteRecorder",
    "open_run",
]
