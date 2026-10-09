"""Fixtures of the GPU smoke suite, and the report it writes (see README.md in this directory)."""

import datetime
import platform
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from auxein.backend import Backend, BackendError
from tests.gpu import report as probe

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest) -> str:
    chosen = request.config.getoption("--device")
    assert isinstance(chosen, str) and chosen, "the GPU smoke suite needs --device (the tests are skipped without it)"
    try:
        Backend("torch", chosen, "float32")
    except BackendError as error:
        # a device that was asked for and is not there is not a skipped test: whoever ran the suite believes it ran on that device
        pytest.exit(f"--device {chosen}: {error}", returncode=4)
    return chosen


def precisions(device: str) -> list[str]:
    """float32 everywhere; float64 too except on Metal, which has none (a configuration error, tested on its own)."""
    return ["float32"] if device.startswith("mps") else ["float32", "float64"]


@pytest.fixture(params=["float32", "float64"])
def gpu_backend(request: pytest.FixtureRequest, device: str) -> Backend:
    """The device's backend, in each precision it supports (float32; and float64 unless the device is Metal)."""
    precision = request.param
    if precision not in precisions(device):
        pytest.skip(f"{precision} is not supported on {device}")
    return Backend("torch", device, precision)


@pytest.fixture
def float32_backend(device: str) -> Backend:
    return Backend("torch", device, "float32")


def synchronize(device: str) -> None:
    """Wait for the device to finish what was queued, so that a timing measures the work and not just its launch."""
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device.startswith("mps"):
        torch.mps.synchronize()


def device_name(device: str) -> str:
    import torch

    if device.startswith("cuda"):
        return str(torch.cuda.get_device_name(device))
    if device.startswith("mps"):
        return "Apple Metal (MPS) on " + (platform.processor() or platform.machine())
    return "CPU (" + (platform.processor() or platform.machine()) + "), a dry run: no GPU involved"


# --- the report ---


_results: list[tuple[str, str, float]] = []


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if "tests/gpu/" not in report.nodeid.replace("\\", "/"):
        return
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        _results.append((report.nodeid, report.outcome, report.duration))


def _git_sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":!docs/gpu-smoke"], cwd=ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return sha + (" (with uncommitted changes)" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def pytest_sessionfinish(session: pytest.Session) -> None:
    chosen = session.config.getoption("--device")
    if not chosen or not _results:
        return
    import torch

    import auxein

    ran = [r for r in _results if r[1] != "skipped"]
    if not ran:
        return
    today = datetime.datetime.now(datetime.UTC).date().isoformat()
    slug = re.sub(r"[^a-z0-9]+", "-", str(chosen).lower()).strip("-")
    target = ROOT / "docs" / "gpu-smoke" / f"{slug}-{today}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for r in _results if r[1] == "passed")
    failed = sum(1 for r in _results if r[1] == "failed")
    skipped = sum(1 for r in _results if r[1] == "skipped")
    lines = [
        f"# GPU smoke suite: {chosen}, {today}",
        "",
        "Written by `tests/gpu` (see its README). "
        + ("**A dry run on the CPU: it proves the suite works, not the GPU.**" if chosen == "cpu" else ""),
        "",
        "| | |",
        "|---|---|",
        f"| Device | `{chosen}`: {device_name(str(chosen))} |",
        f"| Machine | {platform.machine()}, {platform.platform()} |",
        f"| Python | {platform.python_version()} |",
        f"| torch | {torch.__version__} |",
        f"| auxein | {auxein.__version__}, commit {_git_sha()} |",
        f"| Result | **{'FAILED' if failed else 'passed'}**: {passed} passed, {failed} failed, {skipped} skipped |",
        "",
        "## Tests",
        "",
        "| Test | Result | Seconds |",
        "|---|---|---|",
    ]
    for nodeid, outcome, duration in _results:
        lines.append(f"| `{nodeid.split('tests/gpu/', 1)[1]}` | {outcome} | {duration:.2f} |")
    lines += ["", *probe.probe_section(), ""]
    target.write_text("\n".join(lines))
    print(f"\nGPU smoke report written to {target.relative_to(ROOT)}")


@pytest.fixture(scope="session", autouse=True)
def _reset_report() -> Iterator[None]:
    probe.PROBE_ROWS.clear()
    yield
