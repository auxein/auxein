"""Type variables shared by the core types."""

from typing import TypeVar

G = TypeVar("G")
"""The type of a genome. The core never inspects genomes (design doc §4.1)."""
