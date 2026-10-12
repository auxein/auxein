"""Ready-made progress callbacks (design doc §9.6): throttling, a progress line, stopping rules and a history for charts."""

from auxein.callbacks.helpers import ProgressHistory, every, every_evaluations, print_progress, stop_when

__all__ = ["ProgressHistory", "every", "every_evaluations", "print_progress", "stop_when"]
