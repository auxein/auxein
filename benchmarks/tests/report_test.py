import pytest

from benchmarks.report import Results, build_report, sanity_checks


def test_report_builds_from_a_results_directory(tiny_results):
    report = build_report(tiny_results)
    text = report.read_text()

    assert report == tiny_results / "report.md"
    for heading in (
        "## 1. Convergence",
        "## 2. Summary",
        "## 3. Statistical comparison",
        "## 4. Overhead",
        "## 5. Sanity checks",
        "## 6. Methodology",
    ):
        assert heading in text
    for algorithm in ("auxein-core-ga", "random-search", "cma-es"):
        assert algorithm in text


def test_report_has_a_plot_for_every_problem_and_dimension_and_for_the_overhead(tiny_results):
    build_report(tiny_results)
    for problem in ("sphere", "rastrigin", "noisy_sphere"):
        for dim in (2, 3):
            png = tiny_results / f"convergence-{problem}-d{dim}.png"
            assert png.stat().st_size > 1000
            assert f"({png.name})" in (tiny_results / "report.md").read_text()
    assert (tiny_results / "overhead-population.png").stat().st_size > 1000
    assert (tiny_results / "overhead-dimension.png").stat().st_size > 1000


def test_report_tables(tiny_results):
    text = build_report(tiny_results).read_text()
    assert "### sphere, d=2" in text
    assert (
        "| Algorithm | Final error, median [IQR] | Success 0.1 | Success 0.001 | Success 1e-06 | ERT 0.1 | ERT 0.001 | ERT 1e-06 |" in text
    )
    assert "| auxein-core-ga vs |" in text
    assert "| cma-es |" in text and "| random-search |" in text
    assert "Evaluations per generation" in text
    assert "| auxein-core-ga | 10 | 10 |" in text  # lambda = 10 children per generation, nothing re-scored
    assert "Holm" in text and "A12" in text and "ERT (expected running time)" in text


def test_every_comparison_row_has_a_plain_english_reading(tiny_results):
    text = build_report(tiny_results).read_text()
    rows = [line for line in text.splitlines() if line.startswith("| random-search |") and "better" in line or "no significant" in line]
    assert rows
    assert all(("better, " in row and " effect" in row) or "no significant difference" in row for row in rows)


def test_findings_are_included_at_the_top_when_present(tiny_results):
    (tiny_results / "findings.md").write_text("## Findings\n\n- a finding\n")
    try:
        text = build_report(tiny_results).read_text()
    finally:
        (tiny_results / "findings.md").unlink()
    assert text.index("## Findings") < text.index("## 1. Convergence")
    assert "- a finding" in text


def test_the_report_is_reproducible(tiny_results):
    first = build_report(tiny_results).read_text()
    assert build_report(tiny_results).read_text() == first


def test_sanity_checks_are_marked_not_applicable_below_the_full_budget(tiny_results):
    lines = sanity_checks(Results(tiny_results))
    assert len(lines) == 3
    assert all("not run in this benchmark" in line or "not applicable" in line for line in lines)


@pytest.mark.parametrize(
    ("hits", "expected"),
    [
        ({"cma-es": 25, "random-search": 0}, "PASS"),
        ({"cma-es": 20, "random-search": 0}, "FAIL"),
        ({"cma-es": 25, "random-search": 1}, "FAIL"),
    ],
)
def test_sanity_checks_pass_and_fail_on_success_counts(tiny_results, hits, expected):
    results = Results(tiny_results)
    results.groups = {}
    for algorithm, successes in hits.items():
        results.groups[("sphere", 10, algorithm)] = [
            {"budget": 20000, "evals": 20000, "hits": {"1e-06": 100 if i < successes else None, "0.001": 100 if i < successes else None}}
            for i in range(25)
        ]
    lines = sanity_checks(results)
    sphere_lines = [line for line in lines if "sphere" in line and "ellipsoid" not in line]
    assert [line.split(":")[0].strip("- ") for line in sphere_lines].count(expected) >= 1


def test_the_report_has_a_mean_rank_table(tiny_results):
    from benchmarks.report import mean_ranks

    text = build_report(tiny_results).read_text()
    assert "### Mean rank" in text and "| Algorithm | sphere d=2 |" in text and "Mean rank |" in text
    cells, ranks = mean_ranks(Results(tiny_results))
    assert len(cells) == 6 and set(ranks) == {"auxein-core-ga", "random-search", "cma-es"}
    for position in range(len(cells)):
        assert sorted(r[position] for r in ranks.values()) == [1, 2, 3]  # each cell ranks every algorithm exactly once


def test_a_config_can_name_several_reference_algorithms(tiny_results, tmp_path):
    import json
    import shutil

    copy = tmp_path / "results"
    shutil.copytree(tiny_results, copy)
    metadata = json.loads((copy / "metadata.json").read_text())
    metadata["config"]["report"] = {"references": ["auxein-core-ga", "cma-es", "not-in-this-run"]}
    (copy / "metadata.json").write_text(json.dumps(metadata))

    results = Results(copy)
    assert results.references == ["auxein-core-ga", "cma-es"]  # the unknown name is ignored
    text = build_report(copy).read_text()
    assert "### Reference: `auxein-core-ga`" in text and "### Reference: `cma-es`" in text
    assert "| auxein-core-ga vs |" in text and "| cma-es vs |" in text
    assert text.splitlines().count("### sphere, d=2") == 1  # one summary table, however many references
    assert "#### sphere, d=2" in text and "`auxein-core-ga`, `cma-es`" in text


def test_the_default_reference_is_the_first_algorithm_of_the_config(tiny_results):
    assert Results(tiny_results).references == ["auxein-core-ga"]
