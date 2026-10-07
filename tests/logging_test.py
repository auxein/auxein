import subprocess
import sys

import pytest


@pytest.mark.xfail(strict=True, reason="phase 3: importing auxein calls logging.basicConfig(level=DEBUG)")
def test_importing_auxein_does_not_configure_logging():
    code = "import logging; import auxein, auxein.playgrounds; root = logging.getLogger(); print(root.level, len(root.handlers))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.split() == [str(30), "0"]  # root logger untouched: WARNING and no handlers
