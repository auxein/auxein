"""Core types of the Auxein design (design doc §3 to §5): candidates, batches, evaluations and the protocols."""

from auxein.core.batch import ArrayBatch, Batch, ListBatch
from auxein.core.candidate import Candidate
from auxein.core.evaluation import ArtifactRef, Cost, Direction, Evaluation, Objective, RawRef, Status
from auxein.core.evaluation_batch import EvaluationBatch, to_minimisation
from auxein.core.ids import CandidateId, IdIssuer
from auxein.core.problem import ProblemSpec
from auxein.core.protocols import EvalContext, Evaluator, Strategy, StrategyCapabilities, StrategyContext, TellMode
from auxein.core.state import StateDict, StateDictError, StateValue, validate_state_dict

__all__ = [
    "ArrayBatch",
    "ArtifactRef",
    "Batch",
    "Candidate",
    "CandidateId",
    "Cost",
    "Direction",
    "EvalContext",
    "Evaluation",
    "EvaluationBatch",
    "Evaluator",
    "IdIssuer",
    "ListBatch",
    "Objective",
    "ProblemSpec",
    "RawRef",
    "StateDict",
    "StateDictError",
    "StateValue",
    "Status",
    "Strategy",
    "StrategyCapabilities",
    "StrategyContext",
    "TellMode",
    "to_minimisation",
    "validate_state_dict",
]
