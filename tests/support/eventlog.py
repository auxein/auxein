"""Reads the deterministic part of a recorded run: everything except timestamps, measured times and error texts.

The status of a failed evaluation is part of it, but its error text is not: a traceback shows different frames inline, in a
thread and in a worker process.
"""

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
            "SELECT id, step, origin, genome_kind, genome, genome_dtype, genome_shape, genome_hash FROM candidates ORDER BY rowid"
        ).fetchall()
        lineage = db.execute("SELECT parent_id, child_id FROM lineage ORDER BY rowid").fetchall()
        evaluations = db.execute(
            "SELECT candidate_id, status, objectives, constraints, descriptors, cost_units FROM evaluations ORDER BY rowid"
        ).fetchall()
        episodes = db.execute(
            "SELECT candidate_id, scenario_index, scenario_id, status, measurements FROM episodes ORDER BY candidate_id, scenario_index"
        ).fetchall()
    finally:
        db.close()
    return {"events": list(events), "candidates": candidates, "lineage": lineage, "evaluations": evaluations, "episodes": episodes}


def told_ids(run_dir: Path) -> list[int]:
    """The ids of the evaluated candidates in the order they were told and recorded (steady-state: one 'ask' event each).

    The tables are keyed by candidate id, so only the events keep this order.
    """
    return [int(payload["first_id"]) for kind, _, payload in event_log(run_dir)["events"] if kind == "ask"]  # type: ignore[index, misc]
