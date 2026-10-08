"""Reads the deterministic part of a recorded run: everything except timestamps and measured times."""

import json
import sqlite3
from pathlib import Path


def event_log(run_dir: Path) -> dict[str, list[object]]:
    """The events, candidates, lineage and evaluations of a run in the order they were written, without clock values."""
    db = sqlite3.connect(run_dir / "events.sqlite")
    try:
        events = [
            (kind, step, json.loads(payload)) for kind, step, payload in db.execute("SELECT kind, step, payload FROM events ORDER BY seq")
        ]
        candidates = db.execute(
            "SELECT id, step, origin, genome_kind, genome, genome_dtype, genome_shape FROM candidates ORDER BY rowid"
        ).fetchall()
        lineage = db.execute("SELECT parent_id, child_id FROM lineage ORDER BY rowid").fetchall()
        evaluations = db.execute(
            "SELECT candidate_id, status, objectives, constraints, descriptors, cost_units, error FROM evaluations ORDER BY rowid"
        ).fetchall()
    finally:
        db.close()
    return {"events": list(events), "candidates": candidates, "lineage": lineage, "evaluations": evaluations}


def told_ids(run_dir: Path) -> list[int]:
    """The ids of the evaluated candidates in the order they were told and recorded (steady-state: one 'ask' event each).

    The tables are keyed by candidate id, so only the events keep this order.
    """
    return [int(payload["first_id"]) for kind, _, payload in event_log(run_dir)["events"] if kind == "ask"]  # type: ignore[index, misc]
