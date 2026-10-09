"""Core types of the Auxein design (design doc §3 to §5): candidates, batches, evaluations and the protocols."""

from auxein.core.batch import ArrayBatch, Batch, ListBatch, single, take
from auxein.core.candidate import Candidate
from auxein.core.episodes import EpisodeRecords
from auxein.core.evaluation import ArtifactRef, Cost, Direction, Evaluation, Objective, RawRef, Status, describe_exception
from auxein.core.evaluation_batch import EvaluationBatch, to_minimisation
from auxein.core.ids import CandidateId, IdIssuer
from auxein.core.normalise import evaluation_from_return, evaluations_from_batch_return
from auxein.core.problem import ProblemSpec
from auxein.core.protocols import EvalContext, Evaluator, FailurePolicy, Strategy, StrategyCapabilities, StrategyContext, TellMode
from auxein.core.results import BatchResult, Result
from auxein.core.state import StateDict, StateDictError, StateValue, validate_state_dict

__all__ = [
    "ArrayBatch",
    "ArtifactRef",
    "Batch",
    "BatchResult",
    "Candidate",
    "CandidateId",
    "Cost",
    "Direction",
    "EpisodeRecords",
    "EvalContext",
    "Evaluation",
    "EvaluationBatch",
    "Evaluator",
    "FailurePolicy",
    "IdIssuer",
    "ListBatch",
    "Objective",
    "ProblemSpec",
    "RawRef",
    "Result",
    "StateDict",
    "StateDictError",
    "StateValue",
    "Status",
    "Strategy",
    "StrategyCapabilities",
    "StrategyContext",
    "TellMode",
    "describe_exception",
    "single",
    "evaluation_from_return",
    "evaluations_from_batch_return",
    "take",
    "to_minimisation",
    "validate_state_dict",
]
