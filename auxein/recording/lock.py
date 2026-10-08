"""The single-writer lock of a run directory (design doc §10.4).

Two processes writing to one `events.sqlite` would interleave their events and corrupt the run, so a run holds a lock file
with its process id for as long as it writes. A lock left by a process that was killed is recognised (its process no
longer exists) and taken over; one held by a live process makes the other refuse, with a message that names it.
"""

import os
import time
from pathlib import Path

LOCK_FILE = "writer.lock"

_held: set[Path] = set()
"""Locks held by this process, so that a second writer in the same process is refused too."""


class RunLockedError(RuntimeError):
    """Another process is writing to this run directory."""


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # signal 0 only checks that the process exists
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists but belongs to someone else
    except OSError:
        return False
    return True


def _owner(path: Path) -> int | None:
    """The process id in a lock file, or None if it is empty or damaged (a writer that died while creating it)."""
    for _ in range(3):  # a writer may have created the file and not yet written its id
        try:
            text = path.read_text().strip()
        except OSError:
            return None
        if text.isdigit():
            return int(text)
        time.sleep(0.05)
    return None


class RunLock:
    """Holds the lock of one run directory between `acquire` and `release`."""

    def __init__(self, run_dir: Path) -> None:
        self._path = run_dir / LOCK_FILE
        self._locked = False

    def acquire(self) -> None:
        if self._locked:
            return
        for _ in range(5):
            try:
                descriptor = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                self._take_over_if_stale()
                continue
            with os.fdopen(descriptor, "w") as handle:
                handle.write(str(os.getpid()))
            _held.add(self._path)
            self._locked = True
            return
        raise RunLockedError(f"could not take the lock {self._path}")

    def _take_over_if_stale(self) -> None:
        owner = _owner(self._path)
        alive = owner is not None and _process_exists(owner) and (owner != os.getpid() or self._path in _held)
        if alive:
            who = "this process" if owner == os.getpid() else f"process {owner}"
            raise RunLockedError(
                f"the run in {self._path.parent} is being written by {who} (lock file {self._path}): "
                "two processes must never write to one run. Wait for it to finish, or stop it first"
            )
        try:
            self._path.unlink()  # a stale lock: its writer was killed
        except FileNotFoundError:
            pass

    def release(self) -> None:
        if not self._locked:
            return
        self._locked = False
        _held.discard(self._path)
        try:
            if _owner(self._path) == os.getpid():
                self._path.unlink()
        except OSError:
            pass
