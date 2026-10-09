"""The per-scenario measurements of a batch of evaluations, as the recorder takes them (design doc §6.4, §10.2)."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from auxein.core.evaluation import Status
from auxein.core.ids import CandidateId


@dataclass(frozen=True)
class EpisodeRecords:
    """What the episodes of a batch of candidates measured, on the host, ready to be recorded.

    It travels next to the evaluations (`EvaluationBatch.episodes`) so that the driver and the recorder need not know what an
    episode is: the raw measurements are kept so that a run can be re-judged by another aggregator without re-simulating.
    A candidate that has no episodes (it failed before any ran) is simply absent from `candidate_ids`.
    """

    candidate_ids: tuple[CandidateId, ...]
    """One per row of `values`, in batch order."""
    scenario_ids: tuple[str, ...]
    """One per scenario, in scenario-set order; the position is the scenario's index."""
    names: tuple[str, ...]
    """The measurement names, the last axis of `values`."""
    values: npt.NDArray[np.float64]
    """Shape `(candidates, scenarios, names)`, float64. The row of an episode that did not succeed holds NaN and is not recorded."""
    failures: Mapping[tuple[int, int], tuple[Status, str]] = field(default_factory=dict[tuple[int, int], tuple[Status, str]])
    """Episodes that did not succeed, keyed by (row, scenario index): a sparse status and error grid, OK everywhere else."""

    def __post_init__(self) -> None:
        n, s, m = self.values.shape
        if (n, s, m) != (len(self.candidate_ids), len(self.scenario_ids), len(self.names)):
            raise ValueError(
                f"episode values have shape {self.values.shape} but there are {len(self.candidate_ids)} candidates, "
                f"{len(self.scenario_ids)} scenarios and {len(self.names)} measurement names"
            )
        for (row, scenario), (status, error) in self.failures.items():
            if not (0 <= row < n and 0 <= scenario < s) or status is Status.OK or not error:
                raise ValueError(f"invalid failed episode ({row}, {scenario}): {status}, {error!r}")
        object.__setattr__(self, "failures", MappingProxyType(dict(self.failures)))
