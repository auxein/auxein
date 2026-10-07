"""The genetic algorithm and its operators (design doc §3.3). The operators are importable to compose your own."""

from auxein.strategies.ga.base import BoundsRepair, Mutation, ParentSelection, PopulationView, Recombination
from auxein.strategies.ga.genetic_algorithm import GeneticAlgorithm
from auxein.strategies.ga.mutation import GaussianMutation, SelfAdaptiveMutation
from auxein.strategies.ga.ranking import rank_order, view_of
from auxein.strategies.ga.recombination import IntermediateRecombination, NoRecombination, UniformRecombination, mix_genes, mix_steps
from auxein.strategies.ga.repair import ClipRepair, ReflectRepair
from auxein.strategies.ga.selection import SigmaScalingSUS, TournamentSelection

__all__ = [
    "BoundsRepair",
    "ClipRepair",
    "GaussianMutation",
    "GeneticAlgorithm",
    "IntermediateRecombination",
    "Mutation",
    "NoRecombination",
    "ParentSelection",
    "PopulationView",
    "Recombination",
    "ReflectRepair",
    "SelfAdaptiveMutation",
    "SigmaScalingSUS",
    "TournamentSelection",
    "UniformRecombination",
    "mix_genes",
    "mix_steps",
    "rank_order",
    "view_of",
]
