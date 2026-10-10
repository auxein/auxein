"""Environments, scenarios and decoders: the agent layer's inputs (design doc §6)."""

from auxein.environments.decoders import IdentityDecoder
from auxein.environments.episode import EpisodeBatchResult, EpisodeFailure, EpisodeResult
from auxein.environments.gymnasium_adapter import GymnasiumEnvironment, LinearAgent, LinearPolicy
from auxein.environments.protocols import Agent, AgentBatch, BatchDecoder, BatchedEnvironment, Decoder, Environment
from auxein.environments.scenario import Params, Scenario, ScenarioSet
from auxein.environments.step import StepAgent, StepEnvironment, StepWorld

__all__ = [
    "Agent",
    "AgentBatch",
    "BatchDecoder",
    "BatchedEnvironment",
    "Decoder",
    "EpisodeBatchResult",
    "EpisodeFailure",
    "EpisodeResult",
    "GymnasiumEnvironment",
    "Environment",
    "IdentityDecoder",
    "LinearAgent",
    "LinearPolicy",
    "Params",
    "Scenario",
    "ScenarioSet",
    "StepAgent",
    "StepEnvironment",
    "StepWorld",
]
