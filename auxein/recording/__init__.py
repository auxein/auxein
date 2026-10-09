"""Recording runs (design doc §10): a recorder protocol, the SQLite run directory and a minimal reader."""

from auxein.recording.genomes import GenomeEncodingError
from auxein.recording.lock import RunLockedError
from auxein.recording.reader import (
    GenomeStoreStats,
    Reaggregated,
    RecordedEpisode,
    RecordedEvaluation,
    RecordedOperatorCall,
    RunReader,
    open_run,
)
from auxein.recording.recorder import NoopRecorder, Recorder
from auxein.recording.sqlite import SCHEMA_VERSION, CheckpointInfo, ExistingRun, ResumeError, RunDirectoryError, SQLiteRecorder

__all__ = [
    "SCHEMA_VERSION",
    "CheckpointInfo",
    "ExistingRun",
    "GenomeEncodingError",
    "GenomeStoreStats",
    "NoopRecorder",
    "Reaggregated",
    "RecordedEpisode",
    "RecordedEvaluation",
    "RecordedOperatorCall",
    "Recorder",
    "ResumeError",
    "RunDirectoryError",
    "RunLockedError",
    "RunReader",
    "SQLiteRecorder",
    "open_run",
]
