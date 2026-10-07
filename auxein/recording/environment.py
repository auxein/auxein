"""Facts about the environment a run happened in, for its metadata. Best effort: it never fails the run."""

import platform
import subprocess
from importlib import metadata

from auxein.backend import devices


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def versions() -> dict[str, str | None]:
    """Versions of Auxein, Python, numpy and, if installed, torch."""
    result: dict[str, str | None] = {"auxein": _version("auxein"), "python": platform.python_version(), "numpy": _version("numpy")}
    if devices.torch_installed():
        result["torch"] = _version("torch")
    return result


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=5)
    except Exception:
        return None
    return completed.stdout.strip()


def git_state() -> dict[str, object] | None:
    """The git SHA and dirty flag of the working directory, or None outside a git repository (or without git).

    Untracked files do not make a tree dirty.
    """
    sha = _git("rev-parse", "HEAD")
    if not sha:
        return None
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"sha": sha, "dirty": None if status is None else bool(status)}
