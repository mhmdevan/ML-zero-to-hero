from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_foundation_report import (
    load_benchmark_rows,
    parse_coverage_percent,
    parse_junit_stats,
    render_summary_markdown,
)


def test_parse_junit_stats_handles_missing_file(tmp_path) -> None:
    stats = parse_junit_stats(tmp_path / "missing.xml")
    assert stats["tests"] == 0
    assert stats["passed"] == 0


def test_parse_junit_stats_reads_counts(tmp_path) -> None:
    xml_path = Path(tmp_path) / "junit.xml"
    xml_path.write_text(
        """
<testsuite tests=\"5\" failures=\"1\" errors=\"0\" skipped=\"2\">\n</testsuite>
""".strip(),
        encoding="utf-8",
    )

    stats = parse_junit_stats(xml_path)
    assert stats == {"tests": 5, "failures": 1, "errors": 0, "skipped": 2, "passed": 2}


def test_parse_coverage_percent_reads_json(tmp_path) -> None:
    path = Path(tmp_path) / "coverage.json"
    path.write_text(json.dumps({"totals": {"percent_covered": 87.5}}), encoding="utf-8")

    percent = parse_coverage_percent(path)
    assert percent == 87.5


def test_load_benchmark_rows_reads_list(tmp_path) -> None:
    path = Path(tmp_path) / "bench.json"
    rows = [{"backend": "numpy", "elapsed_seconds": 0.1, "avg_seconds_per_iter": 0.01, "peak_memory_mb": 12.0}]
    path.write_text(json.dumps(rows), encoding="utf-8")

    parsed = load_benchmark_rows(path)
    assert parsed == rows


def test_render_summary_markdown_contains_key_sections() -> None:
    summary = {
        "generated_at_utc": "2026-02-15T00:00:00+00:00",
        "python": "/usr/bin/python",
        "torch_available": False,
        "coverage_enabled": True,
        "tests": {"tests": 10, "passed": 8, "skipped": 2, "failures": 0, "errors": 0},
        "coverage_percent": 81.23,
        "artifacts": ["output/reports/foundation_summary.json"],
    }
    rows = [{"backend": "numpy", "elapsed_seconds": 0.1234, "avg_seconds_per_iter": 0.0123, "peak_memory_mb": 10.5}]

    md = render_summary_markdown(summary, rows)
    assert "# Foundation Report" in md
    assert "line_coverage_percent" in md
    assert "| backend | total_sec |" in md
