"""The window of a steady-state run: its slots, and how they go into a checkpoint and come back (design doc §9.2, §10.4)."""

import asyncio
import json
from collections import deque
from dataclasses import dataclass
from typing import Generic, cast

import numpy as np

from auxein.backend import Array, Backend, is_array
from auxein.core import ArrayBatch, Batch, Candidate, CandidateId, EvaluationBatch, ListBatch, StateDict
from auxein.core._typing import G


@dataclass
class Slot(Generic[G]):
    """One candidate of a steady-state run, from the moment it is asked until it is told."""

    seq: int
    """Position in ask order across the whole run."""
    step: int
    batch: Batch[G]
    """The candidate as a one-candidate batch, so that every evaluator works unchanged."""
    task: "asyncio.Task[EvaluationBatch[G]] | None" = None
    result: EvaluationBatch[G] | None = None
    recorded: bool = False
    """The result came from the recording (a resumed run replaying), not from an evaluation."""


@dataclass
class Window(Generic[G]):
    """The state of a steady-state run."""

    queue: deque[Slot[G]]
    """Asked, not yet started, in ask order."""
    inflight: dict[int, Slot[G]]
    """Asked, not yet told, in ask order (a dict keeps insertion order). `len(inflight)` is what the window counts."""
    running: dict["asyncio.Task[EvaluationBatch[G]]", Slot[G]]
    finished: list[Slot[G]]
    """Throughput mode only: evaluated, not yet told, in completion order."""
    next_seq: int = 0
    stop: str | None = None
    """Why asking has stopped, once it has."""


def encode_window(window: Window[G]) -> StateDict:
    """The candidates asked but not told, in ask order, as a state dict: ids, lineage, genomes, and their positions.

    Results that finished but have not been told are not part of it: those candidates are evaluated again on resume.
    """
    slots = list(window.inflight.values())
    candidates = [slot.batch.candidates[0] for slot in slots]
    state: StateDict = {
        "next_seq": window.next_seq,
        "seq": [slot.seq for slot in slots],
        "step": [slot.step for slot in slots],
        "ids": [int(c.id) for c in candidates],
        "origins": [c.origin for c in candidates],
        "parents": [[int(p) for p in c.parents] for c in candidates],
        "genomes": None,
        "genome_list": None,
    }
    genomes = [c.genome for c in candidates]
    arrays = [g for g in genomes if is_array(g)]
    if arrays and len(arrays) == len(genomes):
        host = [Backend().to_numpy(cast("Array", g)) for g in genomes]
        if len({(a.shape, a.dtype) for a in host}) == 1:
            state["genomes"] = np.stack(host)  # pyright: ignore[reportUnknownMemberType]
            return state
    state["genome_list"] = [cast("Array", g) if is_array(g) else json.loads(json.dumps(g)) for g in genomes]
    return state


def decode_window(state: StateDict, backend: Backend) -> Window[object]:
    """The inverse of `encode_window`: every candidate asked but not told comes back queued, with its id and genome, ready to
    be evaluated again."""
    ids = cast("list[int]", state["ids"])
    seqs, steps = cast("list[int]", state["seq"]), cast("list[int]", state["step"])
    origins, parents = cast("list[str]", state["origins"]), cast("list[list[int]]", state["parents"])
    stacked = state["genomes"]
    listed = cast("list[object] | None", state["genome_list"])
    window: Window[object] = Window(deque(), {}, {}, [], next_seq=cast("int", state["next_seq"]))
    stack = None if stacked is None else backend.asarray(cast("Array", stacked))
    for index, candidate_id in enumerate(ids):
        pedigree = tuple(CandidateId(p) for p in parents[index])
        batch: Batch[object]
        if stack is not None:
            batch = cast(
                "Batch[object]",
                ArrayBatch(stack[index : index + 1], [CandidateId(candidate_id)], steps[index], [origins[index]], [pedigree]),
            )
        else:
            assert listed is not None
            genome = listed[index]
            if is_array(genome):
                genome = backend.asarray(cast("Array", genome))
            batch = ListBatch((Candidate(CandidateId(candidate_id), genome, pedigree, origins[index], steps[index]),))
        slot = Slot(seqs[index], steps[index], batch)
        window.queue.append(slot)
        window.inflight[slot.seq] = slot
    return window
