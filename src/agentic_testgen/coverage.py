from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

from agentic_testgen.config import AppConfig
from agentic_testgen.models import CoverageRecord, FileWorkItem
from agentic_testgen.utils import run_command


class CoverageAnalyzer:
    def __init__(self, config: AppConfig):
        self.config = config

    def discover_modules(self, repo_root: Path) -> list[str]:
        modules: list[str] = []
        for pom in repo_root.rglob("pom.xml"):
            if ".git" in pom.parts or "target" in pom.parts:
                continue
            relative = pom.parent.relative_to(repo_root)
            modules.append(str(relative) if str(relative) != "." else ".")
        return sorted(set(modules))

    def detect_test_framework(self, repo_root: Path) -> str:
        for pom in repo_root.rglob("pom.xml"):
            text = pom.read_text(encoding="utf-8", errors="ignore").lower()
            if "junit-jupiter" in text or "<artifactid>junit</artifactid>" in text:
                return "junit"
            if "testng" in text:
                return "testng"
        return "unknown"

    def run_tests_with_coverage(self, repo_root: Path) -> tuple[object, list[CoverageRecord]]:
        env: dict[str, str] = {}
        if self.config.java_home:
            env["JAVA_HOME"] = self.config.java_home
        if self.config.maven_home:
            env["MAVEN_HOME"] = self.config.maven_home
        result = run_command(
            [
                self.config.maven_executable(),
                "-q",
                "-DskipTests=false",
                "test",
                "jacoco:report",
            ],
            cwd=repo_root,
            env=env,
        )
        return result, self.collect_reports(repo_root)

    def collect_reports(self, repo_root: Path) -> list[CoverageRecord]:
        records: list[CoverageRecord] = []
        for report in repo_root.rglob("target/site/jacoco/jacoco.xml"):
            records.extend(self.parse_jacoco_xml(report, repo_root))
        deduped: dict[str, CoverageRecord] = {}
        for record in records:
            deduped[record.file_path] = record
        return sorted(deduped.values(), key=lambda item: (item.coverage_percent, -item.missed_lines, item.file_path))

    def parse_jacoco_xml(self, report_path: Path, repo_root: Path) -> list[CoverageRecord]:
        if not report_path.exists():
            return []
        root = ET.fromstring(report_path.read_text(encoding="utf-8"))
        records: list[CoverageRecord] = []
        module_name = str(report_path.parent.parent.parent.relative_to(repo_root))
        for package in root.findall(".//package"):
            package_name = package.attrib.get("name", "")
            for source in package.findall("sourcefile"):
                covered = 0
                missed = 0
                missed_line_numbers: list[int] = []
                for line in source.findall("line"):
                    ci = int(line.attrib.get("ci", "0"))
                    mi = int(line.attrib.get("mi", "0"))
                    if ci > 0:
                        covered += 1
                    if mi > 0:
                        missed += 1
                        missed_line_numbers.append(int(line.attrib.get("nr", "0")))
                total = covered + missed
                coverage = round((covered / total) * 100, 2) if total else 100.0
                resolved = self._resolve_source_path(repo_root, package_name, source.attrib["name"])
                records.append(
                    CoverageRecord(
                        file_path=resolved,
                        module=module_name if module_name != "target" else ".",
                        covered_lines=covered,
                        missed_lines=missed,
                        coverage_percent=coverage,
                        missed_line_numbers=missed_line_numbers,
                        report_path=str(report_path),
                    )
                )
        return records

    def _resolve_source_path(self, repo_root: Path, package_name: str, file_name: str) -> str:
        package_path = Path(*[part for part in package_name.split("/") if part])
        direct_candidates = list(repo_root.rglob(str(Path("src/main/java") / package_path / file_name)))
        if direct_candidates:
            shortest = min(direct_candidates, key=lambda item: len(item.parts))
            return str(shortest.relative_to(repo_root))
        fallback_candidates = [path for path in repo_root.rglob(file_name) if "src" in path.parts and "main" in path.parts]
        if fallback_candidates:
            shortest = min(fallback_candidates, key=lambda item: len(item.parts))
            return str(shortest.relative_to(repo_root))
        guessed = Path("src/main/java") / package_path / file_name
        return str(guessed)

    def build_work_items(self, coverage_records: list[CoverageRecord]) -> list[FileWorkItem]:
        items: list[FileWorkItem] = []
        filtered = [
            record
            for record in coverage_records
            if record.missed_lines > 0 and "/src/test/" not in record.file_path.replace("\\", "/")
        ]
        for rank, record in enumerate(
            sorted(filtered, key=lambda item: (item.coverage_percent, -item.missed_lines, item.file_path)),
            start=1,
        ):
            items.append(
                FileWorkItem(
                    file_path=record.file_path,
                    module=record.module,
                    coverage_percent=record.coverage_percent,
                    covered_lines=record.covered_lines,
                    missed_lines=record.missed_lines,
                    missed_line_numbers=record.missed_line_numbers,
                    priority_rank=rank,
                    rationale=f"Coverage {record.coverage_percent}% with {record.missed_lines} missed lines",
                )
            )
        return items


def summarize_tree(root: Path, max_depth: int = 4) -> str:
    lines: list[str] = []
    for current_root, dirs, files in os.walk(root):
        path = Path(current_root)
        depth = len(path.relative_to(root).parts)
        if depth > max_depth:
            dirs[:] = []
            continue
        indent = "  " * depth
        rel = "." if path == root else str(path.relative_to(root))
        lines.append(f"{indent}{rel}/")
        for file_name in sorted(files)[:20]:
            lines.append(f"{indent}  {file_name}")
    return "\n".join(lines)
