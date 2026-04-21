from __future__ import annotations

from typing import Any

from agentic_testgen.checkpointing import CheckpointStore
from agentic_testgen.coverage import CoverageAnalyzer
from agentic_testgen.logging import RunLogger
from agentic_testgen.models import CoverageComparison, CoverageRecord, GlobalCoverageSummary, RepoContext
from agentic_testgen.reporting import ReportWriter
from agentic_testgen.workspace import RunWorkspace


class CoverageComparator:
    def __init__(self, coverage: CoverageAnalyzer):
        self.coverage = coverage

    def compare_files(
        self,
        before: list[CoverageRecord],
        after: list[CoverageRecord],
    ) -> list[dict[str, Any]]:
        before_map = {item.file_path: item for item in before}
        after_map = {item.file_path: item for item in after}
        rows: list[dict[str, Any]] = []
        for file_path in sorted(set(before_map.keys()) | set(after_map.keys())):
            before_record = before_map.get(file_path)
            after_record = after_map.get(file_path)
            before_percent = before_record.coverage_percent if before_record else 0.0
            after_percent = after_record.coverage_percent if after_record else 0.0
            before_missed = before_record.missed_lines if before_record else 0
            after_missed = after_record.missed_lines if after_record else 0
            if before_record and after_record:
                status = "changed"
            elif before_record and not after_record:
                status = "removed"
            else:
                status = "new"
            rows.append(
                {
                    "file_path": file_path,
                    "module": (after_record.module if after_record else before_record.module if before_record else ""),
                    "before_coverage_percent": before_percent,
                    "after_coverage_percent": after_percent,
                    "coverage_delta": round(after_percent - before_percent, 2),
                    "before_missed_lines": before_missed,
                    "after_missed_lines": after_missed,
                    "missed_line_delta": before_missed - after_missed,
                    "status": status,
                }
            )
        return sorted(rows, key=lambda item: (item["coverage_delta"], item["missed_line_delta"], item["file_path"]))

    def rerun_post_merge(
        self,
        *,
        checkpoint_store: CheckpointStore,
        repo_context: RepoContext,
        workspace: RunWorkspace,
        logger: RunLogger,
        baseline_summary: GlobalCoverageSummary,
        baseline_records: list[CoverageRecord],
    ) -> CoverageComparison | None:
        comparison = self._finalize(
            repo_context=repo_context,
            workspace=workspace,
            logger=logger,
            baseline_summary=baseline_summary,
            baseline_records=baseline_records,
        )
        if comparison:
            self._persist_metadata(
                checkpoint_store,
                comparison,
                str(workspace.artifacts_dir / "coverage-comparison.md"),
                str(workspace.artifacts_dir / "file-coverage-comparison.json"),
                str(workspace.artifacts_dir / "file-coverage-comparison.md"),
                str(workspace.artifacts_dir / "file-coverage-comparison.csv"),
            )
        return comparison

    def _finalize(
        self,
        *,
        repo_context: RepoContext,
        workspace: RunWorkspace,
        logger: RunLogger,
        baseline_summary: GlobalCoverageSummary,
        baseline_records: list[CoverageRecord],
    ) -> CoverageComparison | None:
        report_writer = ReportWriter(workspace.artifacts_dir)
        with logger.step("coverage.after_merge", details={"repo_root": str(repo_context.clone_path)}) as step:
            result, records, maven_log_paths = self.coverage.run_tests_with_coverage(
                repo_context.clone_path,
                maven_logs_dir=workspace.logs_dir / "maven",
                log_prefix="after-merge-project-coverage",
            )
            after_summary = self.coverage.summarize_global_coverage(records)
            comparison = self.coverage.compare_global_coverage(baseline_summary, after_summary)
            comparison_path = report_writer.write_coverage_comparison(comparison)
            file_rows = self.compare_files(baseline_records, records)
            file_json_path, file_md_path, file_csv_path = report_writer.write_file_coverage_comparison(file_rows)
            step["summary"] = f"Coverage increased by {comparison.percentage_increase}%"
            step["exit_code"] = result.exit_code
            step["maven_log_paths"] = maven_log_paths
            step["comparison_path"] = str(comparison_path)
            step["file_coverage_comparison_json"] = str(file_json_path)
            step["file_coverage_comparison_md"] = str(file_md_path)
            step["file_coverage_comparison_csv"] = str(file_csv_path)
            return comparison

    def _persist_metadata(
        self,
        checkpoint_store: CheckpointStore,
        comparison: CoverageComparison,
        comparison_path: str,
        file_comparison_json_path: str,
        file_comparison_markdown_path: str,
        file_comparison_csv_path: str,
    ) -> None:
        checkpoint = checkpoint_store.load()
        if not checkpoint:
            return
        checkpoint.metadata["after_merge_coverage"] = comparison.after.to_json()
        checkpoint.metadata["coverage_percentage_increase"] = comparison.percentage_increase
        checkpoint.metadata["coverage_comparison_path"] = comparison_path
        checkpoint.metadata["file_coverage_comparison_json_path"] = file_comparison_json_path
        checkpoint.metadata["file_coverage_comparison_markdown_path"] = file_comparison_markdown_path
        checkpoint.metadata["file_coverage_comparison_csv_path"] = file_comparison_csv_path
        checkpoint_store.save(checkpoint)

    @staticmethod
    def from_metadata(metadata: dict[str, Any]) -> CoverageComparison | None:
        before_payload = metadata.get("baseline_coverage")
        after_payload = metadata.get("after_merge_coverage")
        if not before_payload or not after_payload:
            return None
        return CoverageComparison(
            before=GlobalCoverageSummary(**before_payload),
            after=GlobalCoverageSummary(**after_payload),
            percentage_increase=float(metadata.get("coverage_percentage_increase", 0.0)),
            covered_line_increase=after_payload.get("covered_lines", 0) - before_payload.get("covered_lines", 0),
            missed_line_reduction=before_payload.get("missed_lines", 0) - after_payload.get("missed_lines", 0),
        )
