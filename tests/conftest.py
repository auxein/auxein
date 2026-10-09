"""The backend fixtures for every test (see `tests/support/fixtures.py`), and the options of the GPU smoke suite (`tests/gpu/`)."""

import os

import pytest

from tests.support.fixtures import backend, corner_backend, use_corner_backend  # noqa: F401


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--device",
        default=os.environ.get("AUXEIN_GPU_DEVICE"),
        help="run the GPU smoke suite (tests/gpu, marker 'gpu') on this torch device: 'cuda', 'cuda:1', 'mps', or 'cpu' for a dry run. "
        "Also read from the environment variable AUXEIN_GPU_DEVICE. Without it the suite is skipped.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: the GPU smoke suite, run by hand on a device (see tests/gpu/README.md)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Everything under tests/gpu is marked `gpu`, and skipped (cleanly, with the reason) unless a device was named."""
    device = config.getoption("--device")
    skip = pytest.mark.skip(reason="the GPU smoke suite needs a device: pytest -m gpu --device mps (or cuda, or cpu for a dry run)")
    for item in items:
        if "tests/gpu/" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.gpu)
            if not device:
                item.add_marker(skip)
