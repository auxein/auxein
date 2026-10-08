"""Bounds repair: clip, and reflect."""

from auxein.backend import Array, backend_of
from auxein.spaces import Box


class ClipRepair:
    """Clips out-of-bounds genes to the nearest bound (`Box.clip`, which also rounds float32 bounds inward)."""

    name = "clip"

    def __repr__(self) -> str:
        return "ClipRepair()"

    def repair(self, genomes: Array, box: Box) -> Array:
        return box.clip(genomes)


class ReflectRepair:
    """Reflects out-of-bounds genes back into the box, as a ball bounces off a wall, however far out they are.

    Unlike clipping it doesn't pile mutated genes up on the bounds. A final clip guards against rounding.
    """

    name = "reflect"

    def __repr__(self) -> str:
        return "ReflectRepair()"

    def repair(self, genomes: Array, box: Box) -> Array:
        backend = backend_of(genomes)
        xp = backend.xp
        low, high = backend.asarray(box.lower), backend.asarray(box.upper)
        width = high - low
        folded = xp.remainder(genomes - low, 2.0 * width)  # position along a path that goes up and back down: 0 .. 2w
        reflected = xp.where(folded > width, 2.0 * width - folded, folded) + low
        return box.clip(reflected)
