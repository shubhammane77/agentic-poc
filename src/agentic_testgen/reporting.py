from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

from agentic_testgen.models import (
    AttemptRecord,
    CoverageComparison,
    FileWorkItem,
    ModelEvalResult,
    RepoContext,
    SubagentResult,
)
from agentic_testgen.utils import ensure_dir, write_json


def _column_name(index: int) -> str:
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _cell_xml(row_index: int, col_index: int, value: object) -> str:
    coord = f"{_column_name(col_index)}{row_index}"
    if value is None:
        return f'<c r="{coord}"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{coord}"><v>{value}</v></c>'
    return f'<c r="{coord}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def _sheet_xml(rows: list[list[object]]) -> str:
    row_nodes: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells = "".join(_cell_xml(row_index, col_index, value) for col_index, value in enumerate(row, start=1))
        row_nodes.append(f'<row r="{row_index}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(row_nodes)}</sheetData>"
        "</worksheet>"
    )


class ReportWriter:
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = ensure_dir(artifacts_dir)

    def write_overview(self, overview: str) -> Path:
        path = self.artifacts_dir / "overview.md"
        path.write_text(overview, encoding="utf-8")
        return path

    def write_json_summary(
        self,
        repo_context: RepoContext,
        work_items: list[FileWorkItem],
        results: list[SubagentResult],
        model_eval: list[ModelEvalResult],
        coverage_comparison: CoverageComparison | None = None,
    ) -> Path:
        path = self.artifacts_dir / "summary.json"
        write_json(
            path,
            {
                "repo": repo_context.to_json(),
                "files": [item.to_json() for item in work_items],
                "results": [item.to_json() for item in results],
                "model_eval": [item.to_json() for item in model_eval],
                "coverage_comparison": coverage_comparison.to_json() if coverage_comparison else None,
            },
        )
        return path

    def write_coverage_comparison(self, comparison: CoverageComparison) -> Path:
        path = self.artifacts_dir / "coverage-comparison.md"
        path.write_text(
            "\n".join(
                [
                    "# Coverage Comparison",
                    "",
                    f"- Before coverage: {comparison.before.coverage_percent}%",
                    f"- After coverage: {comparison.after.coverage_percent}%",
                    f"- Percentage increase: {comparison.percentage_increase}%",
                    f"- Covered line increase: {comparison.covered_line_increase}",
                    f"- Missed line reduction: {comparison.missed_line_reduction}",
                    "",
                    "## Before",
                    "",
                    f"- Covered lines: {comparison.before.covered_lines}",
                    f"- Missed lines: {comparison.before.missed_lines}",
                    f"- Coverage reports parsed: {comparison.before.report_count}",
                    "",
                    "## After",
                    "",
                    f"- Covered lines: {comparison.after.covered_lines}",
                    f"- Missed lines: {comparison.after.missed_lines}",
                    f"- Coverage reports parsed: {comparison.after.report_count}",
                ]
            ),
            encoding="utf-8",
        )
        return path

    def write_workbook(
        self,
        repo_context: RepoContext,
        work_items: list[FileWorkItem],
        attempts: list[AttemptRecord],
        model_eval: list[ModelEvalResult],
        coverage_comparison: CoverageComparison | None = None,
    ) -> Path:
        output = self.artifacts_dir / "results.xlsx"
        sheets = {
            "runs": [
                ["run_id", "repo_name", "repo_url", "clone_path", "source_type"],
                [
                    repo_context.run_id,
                    repo_context.repo_name,
                    repo_context.repo_url,
                    str(repo_context.clone_path),
                    repo_context.source_type,
                ],
            ],
            "files": [["file_path", "module", "coverage_before", "covered_lines", "missed_lines", "candidate_rank", "assigned_subagent", "status"]],
            "attempts": [[
                "run_id",
                "subagent_id",
                "file_path",
                "iteration",
                "prompt_version",
                "prompt_hash",
                "tool_calls",
                "generated_test_file",
                "single_test_command",
                "status",
                "failure_summary",
                "reflective_summary",
            ]],
            "model_eval": [[
                "case_id",
                "model_id",
                "fixture_name",
                "target_file",
                "status",
                "compile_success",
                "pass_rate",
                "coverage_delta",
                "missed_line_reduction",
                "forbidden_edit_rate",
                "flaky_rate",
                "latency_seconds",
                "tool_call_count",
                "iteration_count",
                "estimated_cost",
                "error_message",
            ]],
            "coverage_summary": [[
                "before_percent",
                "after_percent",
                "percentage_increase",
                "covered_line_increase",
                "missed_line_reduction",
            ]],
        }

        for item in work_items:
            sheets["files"].append(
                [
                    item.file_path,
                    item.module,
                    item.coverage_percent,
                    item.covered_lines,
                    item.missed_lines,
                    item.priority_rank,
                    item.assigned_subagent_id or "",
                    item.status,
                ]
            )

        for attempt in attempts:
            sheets["attempts"].append(
                [
                    attempt.run_id,
                    attempt.subagent_id,
                    attempt.file_path,
                    attempt.iteration,
                    attempt.prompt_version,
                    attempt.prompt_hash,
                    attempt.tool_call_summary,
                    attempt.generated_test_file or "",
                    attempt.single_test_command,
                    attempt.status,
                    attempt.failure_summary,
                    attempt.reflective_summary,
                ]
            )

        for row in model_eval:
            sheets["model_eval"].append(
                [
                    row.case_id,
                    row.model_id,
                    row.fixture_name,
                    row.target_file,
                    row.status,
                    row.compile_success,
                    row.pass_rate,
                    row.coverage_delta,
                    row.missed_line_reduction,
                    row.forbidden_edit_rate,
                    row.flaky_rate,
                    row.latency_seconds,
                    row.tool_call_count,
                    row.iteration_count,
                    row.estimated_cost,
                    row.error_message,
                ]
            )

        if coverage_comparison:
            sheets["coverage_summary"].append(
                [
                    coverage_comparison.before.coverage_percent,
                    coverage_comparison.after.coverage_percent,
                    coverage_comparison.percentage_increase,
                    coverage_comparison.covered_line_increase,
                    coverage_comparison.missed_line_reduction,
                ]
            )

        with ZipFile(output, "w", compression=ZIP_DEFLATED) as zf:
            zf.writestr(
                "[Content_Types].xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
                '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
                '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                '<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                '<Override PartName="/xl/worksheets/sheet3.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                '<Override PartName="/xl/worksheets/sheet4.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                '<Override PartName="/xl/worksheets/sheet5.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                "</Types>",
            )
            zf.writestr(
                "_rels/.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
                "</Relationships>",
            )
            zf.writestr(
                "docProps/core.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
                'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
                "<dc:title>agentic-testgen results</dc:title>"
                "</cp:coreProperties>",
            )
            zf.writestr(
                "docProps/app.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
                'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
                "<Application>agentic-testgen</Application>"
                "</Properties>",
            )
            zf.writestr(
                "xl/workbook.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                '<sheets>'
                '<sheet name="runs" sheetId="1" r:id="rId1"/>'
                '<sheet name="files" sheetId="2" r:id="rId2"/>'
                '<sheet name="attempts" sheetId="3" r:id="rId3"/>'
                '<sheet name="model_eval" sheetId="4" r:id="rId4"/>'
                '<sheet name="coverage_summary" sheetId="5" r:id="rId5"/>'
                "</sheets>"
                "</workbook>",
            )
            zf.writestr(
                "xl/_rels/workbook.xml.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
                '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>'
                '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet3.xml"/>'
                '<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet4.xml"/>'
                '<Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet5.xml"/>'
                '<Relationship Id="rId6" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                "</Relationships>",
            )
            zf.writestr(
                "xl/styles.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
                '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
                '<borders count="1"><border/></borders>'
                '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
                '<cellXfs count="1"><xf/></cellXfs>'
                "</styleSheet>",
            )
            for index, rows in enumerate(sheets.values(), start=1):
                zf.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(rows))
        return output
