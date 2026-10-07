"""What the results were produced with."""

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from benchmarks.config import Config

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_sha() -> str | None:
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool | None:
    status = _git("status", "--porcelain")
    return None if status is None else bool(status)


def cpu_info() -> dict[str, Any]:
    model = platform.processor() or None
    try:
        if sys.platform == "darwin":
            model = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
        elif sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"model": model, "machine": platform.machine(), "system": platform.platform(), "logical_cpus": os.cpu_count()}


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_metadata(config: Config, workers: int) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "versions": {
            "python": platform.python_version(),
            "auxein": package_version("auxein"),
            "numpy": package_version("numpy"),
            "cma": package_version("cma"),
            "scipy": package_version("scipy"),
            "matplotlib": package_version("matplotlib"),
        },
        "cpu": cpu_info(),
        "workers": workers,
        "config": config.raw,
    }
