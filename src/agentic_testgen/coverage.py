from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from agentic_testgen.config import AppConfig
from agentic_testgen.models import CoverageComparison, CoverageRecord, FileWorkItem, GlobalCoverageSummary
from agentic_testgen.utils import run_command, write_command_logs


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

    def detect_test_framework_info(self, repo_root: Path) -> tuple[str, str]:
        poms = [
            pom
            for pom in repo_root.rglob("pom.xml")
            if ".git" not in pom.parts and "target" not in pom.parts
        ]
        for pom in sorted(poms, key=lambda item: (len(item.relative_to(repo_root).parts), str(item))):
            framework, version = self._detect_test_framework_in_pom(pom)
            if framework != "unknown":
                return framework, version
        return "unknown", ""

    def run_tests_with_coverage(
        self,
        repo_root: Path,
        *,
        maven_logs_dir: Path | None = None,
        log_prefix: str = "project-coverage",
    ) -> tuple[object, list[CoverageRecord], dict[str, str]]:
        env: dict[str, str] = {}
        if self.config.java_home:
            env["JAVA_HOME"] = self.config.java_home
        if self.config.maven_home:
            env["MAVEN_HOME"] = self.config.maven_home
        result = run_command(
            self.config.maven_command(
                "-q",
                "-DskipTests=false",
                "test",
                "jacoco:report",
            ),
            cwd=repo_root,
            env=env,
        )
        log_paths = write_command_logs(maven_logs_dir, log_prefix, result) if maven_logs_dir else {}
        return result, self.collect_reports(repo_root), log_paths

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
        module_root = report_path.parents[3]
        module_name = str(module_root.relative_to(repo_root))
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
                resolved = self._resolve_source_path(repo_root, module_root, package_name, source.attrib["name"])
                records.append(
                    CoverageRecord(
                        file_path=resolved,
                        module=module_name,
                        covered_lines=covered,
                        missed_lines=missed,
                        coverage_percent=coverage,
                        missed_line_numbers=missed_line_numbers,
                        report_path=str(report_path),
                    )
                )
        return records

    def _resolve_source_path(self, repo_root: Path, module_root: Path, package_name: str, file_name: str) -> str:
        package_path = Path(*[part for part in package_name.split("/") if part])
        direct_candidates = list(module_root.rglob(str(Path("src/main/java") / package_path / file_name)))
        if direct_candidates:
            candidate = direct_candidates[0]
            return str(candidate.relative_to(repo_root))
        fallback_candidates = [
            path
            for path in module_root.rglob(file_name)
            if "src" in path.parts and "main" in path.parts
        ]
        if fallback_candidates:
            candidate = fallback_candidates[0]
            return str(candidate.relative_to(repo_root))
        guessed = module_root / "src" / "main" / "java" / package_path / file_name
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

    def summarize_global_coverage(self, coverage_records: list[CoverageRecord]) -> GlobalCoverageSummary:
        covered = sum(item.covered_lines for item in coverage_records)
        missed = sum(item.missed_lines for item in coverage_records)
        total = covered + missed
        percent = round((covered / total) * 100, 2) if total else 100.0
        return GlobalCoverageSummary(
            covered_lines=covered,
            missed_lines=missed,
            coverage_percent=percent,
            report_count=len(coverage_records),
        )

    def compare_global_coverage(
        self,
        before: GlobalCoverageSummary,
        after: GlobalCoverageSummary,
    ) -> CoverageComparison:
        return CoverageComparison(
            before=before,
            after=after,
            percentage_increase=round(after.coverage_percent - before.coverage_percent, 2),
            covered_line_increase=after.covered_lines - before.covered_lines,
            missed_line_reduction=before.missed_lines - after.missed_lines,
        )

    def _detect_test_framework_in_pom(self, pom_path: Path) -> tuple[str, str]:
        try:
            root = ET.fromstring(pom_path.read_text(encoding="utf-8", errors="ignore"))
        except ET.ParseError:
            return "unknown", ""
        namespace = self._pom_namespace(root)
        properties = self._pom_properties(root, namespace)
        dependencies = root.findall(f".//{self._pom_tag('dependency', namespace)}")
        for dependency in dependencies:
            group_id = self._dependency_text(dependency, "groupId", namespace).lower()
            artifact_id = self._dependency_text(dependency, "artifactId", namespace).lower()
            if not artifact_id:
                continue
            framework = self._classify_test_framework(group_id, artifact_id)
            if framework == "unknown":
                continue
            version = self._resolve_property_references(
                self._dependency_text(dependency, "version", namespace),
                properties,
            )
            return framework, version
        return "unknown", ""

    def _pom_namespace(self, root: ET.Element) -> str:
        if root.tag.startswith("{") and "}" in root.tag:
            return root.tag[1:].split("}", 1)[0]
        return ""

    def _pom_tag(self, name: str, namespace: str) -> str:
        return f"{{{namespace}}}{name}" if namespace else name

    def _pom_properties(self, root: ET.Element, namespace: str) -> dict[str, str]:
        properties: dict[str, str] = {}
        properties_node = root.find(self._pom_tag("properties", namespace))
        if properties_node is None:
            return properties
        for child in list(properties_node):
            tag_name = child.tag.split("}", 1)[-1]
            properties[tag_name] = (child.text or "").strip()
        return properties

    def _dependency_text(self, dependency: ET.Element, name: str, namespace: str) -> str:
        node = dependency.find(self._pom_tag(name, namespace))
        return (node.text or "").strip() if node is not None and node.text else ""

    def _resolve_property_references(self, value: str, properties: dict[str, str]) -> str:
        resolved = value.strip()
        for _ in range(5):
            match = re.fullmatch(r"\$\{([^}]+)\}", resolved)
            if not match:
                break
            key = match.group(1)
            next_value = properties.get(key, resolved)
            if next_value == resolved:
                break
            resolved = next_value.strip()
        return resolved

    def _classify_test_framework(self, group_id: str, artifact_id: str) -> str:
        if group_id == "org.junit.jupiter" or artifact_id.startswith("junit-jupiter"):
            return "junit5"
        if group_id == "junit" and artifact_id == "junit":
            return "junit4"
        if group_id == "org.testng" and artifact_id == "testng":
            return "testng"
        return "unknown"


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
