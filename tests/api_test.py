"""The public API of the `auxein` package, and the README quickstart (which must keep running)."""

import re
import subprocess
import sys
import tomllib
import warnings
from importlib import metadata
from pathlib import Path

import pytest

import auxein

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "Aggregator",
    "Backend",
    "BatchResult",
    "Box",
    "Budget",
    "EpisodeEvaluator",
    "EpisodeResult",
    "FunctionEvaluator",
    "GeneticAlgorithm",
    "Objective",
    "RandomSearch",
    "RecordingDisabledWarning",
    "Result",
    "RunResult",
    "Scenario",
    "ScenarioSet",
    "SequenceSpace",
    "Status",
    "StructuredGeneticAlgorithm",
    "VectorisedEvaluator",
    "__version__",
    "aresume",
    "arun",
    "open_run",
    "resume",
    "run",
}


def test_the_public_api_is_the_agreed_list():
    assert set(auxein.__all__) == EXPECTED
    assert len(auxein.__all__) == len(set(auxein.__all__))


def test_a_star_import_exposes_exactly_all():
    namespace: dict[str, object] = {}
    exec("from auxein import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(auxein.__all__)


def test_every_exported_name_is_the_object_of_its_subpackage():
    from auxein import backend, core, driver, evaluators, recording, spaces, strategies

    for module, names in (
        (backend, ["Backend"]),
        (core, ["BatchResult", "Objective", "Result", "Status"]),
        (driver, ["Budget", "RecordingDisabledWarning", "RunResult", "arun", "run"]),
        (evaluators, ["FunctionEvaluator", "VectorisedEvaluator"]),
        (recording, ["open_run"]),
        (spaces, ["Box"]),
        (strategies, ["GeneticAlgorithm", "RandomSearch"]),
    ):
        for name in names:
            assert getattr(auxein, name) is getattr(module, name)


def test_there_is_one_source_of_truth_for_the_version():
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert auxein.__version__ == metadata.version("auxein") == declared
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", auxein.__version__)


def test_importing_auxein_does_not_configure_logging():
    code = "import logging; import auxein; root = logging.getLogger(); print(root.level, len(root.handlers))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.split() == ["30", "0"]  # the root logger is untouched: WARNING and no handlers


def test_the_old_engine_is_gone():
    for name in ("fitness", "mutations", "parents", "playgrounds", "population", "recombinations", "replacements"):
        with pytest.raises(ModuleNotFoundError):
            __import__(f"auxein.{name}")
    assert not (ROOT / "notebooks").exists()


# --- the README quickstart ---


def quickstart() -> str:
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / "README.md").read_text(), flags=re.S)
    assert len(blocks) == 1, "the README has exactly one Python snippet: the quickstart"
    return blocks[0]


def run_snippet(code: str, capsys: pytest.CaptureFixture[str]) -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", auxein.RecordingDisabledWarning)
        exec(compile(code, "README.md", "exec"), {"__name__": "__main__"})
    return capsys.readouterr().out.split()


def test_the_readme_quickstart_runs(capsys: pytest.CaptureFixture[str]):
    out = run_snippet(quickstart(), capsys)
    best = float(out[0])
    assert 0 <= best < 20  # Rastrigin in 10 dimensions: the optimum is 0, a random point is around 100
    assert out[1].startswith("[")  # the genome of the best candidate, printed as an array


def test_the_readme_quickstart_warns_that_it_is_not_recorded():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec(compile(quickstart(), "README.md", "exec"), {"__name__": "__main__"})
    assert [w for w in caught if issubclass(w.category, auxein.RecordingDisabledWarning)]


def test_the_readme_quickstart_records_a_run_when_run_dir_is_uncommented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    code = quickstart()
    assert '# run_dir="runs/rastrigin"' in code
    monkeypatch.chdir(tmp_path)
    run_snippet(code.replace('# run_dir="runs/rastrigin"', 'run_dir="runs/rastrigin"'), capsys)
    with auxein.open_run(tmp_path / "runs" / "rastrigin") as recorded:
        assert recorded.metadata["status"] == "completed" and len(list(recorded.evaluations())) == 20_000
