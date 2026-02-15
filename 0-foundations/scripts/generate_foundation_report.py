#!/usr/bin/env python
"""
Generate reproducible quality/perf artifacts for 0-foundations.

Artifacts:
- pytest junit xml + raw stdout/stderr logs
- coverage reports (if coverage command is available)
- numpy/torch CPU benchmark JSON+CSV
- numerical stability demo output log
- aggregated summary JSON + Markdown
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CommandResult:
    name: str
    command: list[str]
    return_code: int
    log_path: str


def run_command(name: str, command: list[str], cwd: Path, log_path: Path) -> CommandResult:
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    rendered = [f"$ {shlex.join(command)}", "", "[stdout]", proc.stdout, "", "[stderr]", proc.stderr]
    log_path.write_text("\n".join(rendered), encoding="utf-8")

    return CommandResult(
        name=name,
        command=command,
        return_code=proc.returncode,
        log_path=str(log_path),
    )


def parse_junit_stats(junit_path: Path) -> dict[str, int]:
    if not junit_path.exists():
        return {
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "passed": 0,
        }

    root = ET.parse(junit_path).getroot()
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]

    tests = failures = errors = skipped = 0
    for suite in suites:
        tests += int(suite.attrib.get("tests", 0))
        failures += int(suite.attrib.get("failures", 0))
        errors += int(suite.attrib.get("errors", 0))
        skipped += int(suite.attrib.get("skipped", 0))

    passed = max(tests - failures - errors - skipped, 0)
    return {
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "passed": passed,
    }


def parse_coverage_percent(coverage_json_path: Path) -> float | None:
    if not coverage_json_path.exists():
        return None

    payload = json.loads(coverage_json_path.read_text(encoding="utf-8"))
    totals = payload.get("totals", {})
    percent = totals.get("percent_covered")
    if percent is None:
        return None
    return float(percent)


def load_benchmark_rows(benchmark_json_path: Path) -> list[dict[str, Any]]:
    if not benchmark_json_path.exists():
        return []
    payload = json.loads(benchmark_json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def render_summary_markdown(summary: dict[str, Any], benchmark_rows: list[dict[str, Any]]) -> str:
    tests = summary["tests"]
    lines = [
        "# Foundation Report",
        "",
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- python: `{summary['python']}`",
        f"- torch_available: `{summary['torch_available']}`",
        f"- coverage_enabled: `{summary['coverage_enabled']}`",
        "",
        "## Test Results",
        "",
        f"- tests: `{tests['tests']}`",
        f"- passed: `{tests['passed']}`",
        f"- skipped: `{tests['skipped']}`",
        f"- failures: `{tests['failures']}`",
        f"- errors: `{tests['errors']}`",
        "",
    ]

    coverage_percent = summary.get("coverage_percent")
    if coverage_percent is not None:
        lines.extend(["## Coverage", "", f"- line_coverage_percent: `{coverage_percent:.2f}`", ""])

    lines.extend(["## Benchmark", ""])
    if benchmark_rows:
        lines.append("| backend | total_sec | sec_per_iter | peak_mem_mb |")
        lines.append("|---|---:|---:|---:|")
        for row in benchmark_rows:
            lines.append(
                f"| {row.get('backend', '')} | "
                f"{float(row.get('elapsed_seconds', 0.0)):.4f} | "
                f"{float(row.get('avg_seconds_per_iter', 0.0)):.6f} | "
                f"{float(row.get('peak_memory_mb', 0.0)):.2f} |"
            )
    else:
        lines.append("No benchmark rows produced.")

    lines.extend(["", "## Artifacts", ""])
    for artifact in summary.get("artifacts", []):
        lines.append(f"- `{artifact}`")

    return "\n".join(lines) + "\n"


def is_torch_available(python_executable: str) -> bool:
    probe = subprocess.run(
        [python_executable, "-c", "import torch"],
        text=True,
        capture_output=True,
    )
    return probe.returncode == 0


def has_coverage_module(python_executable: str) -> bool:
    probe = subprocess.run(
        [python_executable, "-m", "coverage", "--version"],
        text=True,
        capture_output=True,
    )
    return probe.returncode == 0


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output" / "reports"
    benchmark_dir = output_dir / "benchmarks"

    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    python_executable = sys.executable
    coverage_enabled = has_coverage_module(python_executable)

    junit_path = output_dir / "pytest-junit.xml"
    coverage_json_path = output_dir / "coverage.json"
    coverage_xml_path = output_dir / "coverage.xml"
    benchmark_json_path = benchmark_dir / "numpy_vs_torch_cpu_benchmark.json"

    commands: list[CommandResult] = []

    if coverage_enabled:
        pytest_cmd = [
            python_executable,
            "-m",
            "coverage",
            "run",
            "--source=src",
            "-m",
            "pytest",
            "-q",
            "tests",
            "--junit-xml",
            str(junit_path),
        ]
    else:
        pytest_cmd = [
            python_executable,
            "-m",
            "pytest",
            "-q",
            "tests",
            "--junit-xml",
            str(junit_path),
        ]

    commands.append(
        run_command(
            name="pytest",
            command=pytest_cmd,
            cwd=project_root,
            log_path=output_dir / "pytest-output.txt",
        )
    )

    if coverage_enabled and commands[-1].return_code == 0:
        commands.append(
            run_command(
                name="coverage-report",
                command=[python_executable, "-m", "coverage", "report", "-m"],
                cwd=project_root,
                log_path=output_dir / "coverage-report.txt",
            )
        )
        commands.append(
            run_command(
                name="coverage-json",
                command=[
                    python_executable,
                    "-m",
                    "coverage",
                    "json",
                    "-o",
                    str(coverage_json_path),
                ],
                cwd=project_root,
                log_path=output_dir / "coverage-json.txt",
            )
        )
        commands.append(
            run_command(
                name="coverage-xml",
                command=[python_executable, "-m", "coverage", "xml", "-o", str(coverage_xml_path)],
                cwd=project_root,
                log_path=output_dir / "coverage-xml.txt",
            )
        )

    torch_available = is_torch_available(python_executable)
    backend_mode = "both" if torch_available else "numpy"

    commands.append(
        run_command(
            name="benchmark",
            command=[
                python_executable,
                "-m",
                "src.benchmark_numpy_vs_torch_cpu",
                "--matrix-size",
                "256",
                "--repetitions",
                "8",
                "--backend",
                backend_mode,
                "--output-dir",
                str(benchmark_dir),
            ],
            cwd=project_root,
            log_path=output_dir / "benchmark-output.txt",
        )
    )

    commands.append(
        run_command(
            name="numerical-stability",
            command=[python_executable, "-m", "src.numerical_stability"],
            cwd=project_root,
            log_path=output_dir / "numerical-stability-output.txt",
        )
    )

    tests = parse_junit_stats(junit_path)
    coverage_percent = parse_coverage_percent(coverage_json_path)
    benchmark_rows = load_benchmark_rows(benchmark_json_path)

    artifacts = [
        str(junit_path),
        str(output_dir / "pytest-output.txt"),
        str(output_dir / "benchmark-output.txt"),
        str(output_dir / "numerical-stability-output.txt"),
        str(benchmark_json_path),
    ]
    if coverage_json_path.exists():
        artifacts.append(str(coverage_json_path))
    if coverage_xml_path.exists():
        artifacts.append(str(coverage_xml_path))

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "python": python_executable,
        "torch_available": torch_available,
        "coverage_enabled": coverage_enabled,
        "tests": tests,
        "coverage_percent": coverage_percent,
        "benchmark_rows": benchmark_rows,
        "artifacts": artifacts,
        "commands": [asdict(command) for command in commands],
    }

    summary_json_path = output_dir / "foundation_summary.json"
    summary_md_path = output_dir / "foundation_summary.md"

    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md_path.write_text(render_summary_markdown(summary, benchmark_rows), encoding="utf-8")

    print(f"[REPORT] Summary JSON: {summary_json_path}")
    print(f"[REPORT] Summary MD:   {summary_md_path}")

    return max(command.return_code for command in commands)


if __name__ == "__main__":
    raise SystemExit(main())
