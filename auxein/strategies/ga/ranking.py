"""The ranking used everywhere in the genetic algorithm: survivors, tournaments, elites (design doc §3.3)."""

from auxein.backend import Array, Backend
from auxein.strategies.ga.base import PopulationView


def rank_order(values: Array, violation: Array, ids: Array, backend: Backend) -> Array:
    """Member indices, best first: lower total violation, then lower objective (minimisation form), then lower id.

    Feasible members have violation 0, so they come before every infeasible one; failed members have infinite violation
    and NaN objectives and come last. Three stable sorts from the least to the most significant key give the
    lexicographic order, entirely on the backend.
    """
    xp = backend.xp
    values = xp.where(xp.isnan(values), float("inf"), values)  # NaN would sort unpredictably: a missing value is the worst
    order = xp.argsort(ids, stable=True)
    order = xp.take(order, xp.argsort(xp.take(values, order, axis=0), stable=True), axis=0)
    order = xp.take(order, xp.argsort(xp.take(violation, order, axis=0), stable=True), axis=0)
    return order


def view_of(values: Array, violation: Array, ids: Array, backend: Backend) -> PopulationView:
    """The ranked view of a population."""
    order = rank_order(values, violation, ids, backend)
    return PopulationView(values, violation, order, backend.xp.argsort(order), backend)
