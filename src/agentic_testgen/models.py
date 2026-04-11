from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


def _jsonify(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonify(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(item) for item in value]
    return value


@dataclass
class RepoContext:
    run_id: str
    repo_url: str
    repo_name: str
    clone_path: Path
    workspace_root: Path
    source_type: str = "gitlab"
    default_branch: str | None = None
    module_paths: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return _jsonify(self)


@dataclass
class CoverageRecord:
    file_path: str
    module: str
    covered_lines: int
    missed_lines: int
    coverage_percent: float
    missed_line_numbers: list[int] = field(default_factory=list)
    report_path: str | None = None

    def to_json(self) -> dict[str, Any]:
        return _jsonify(self)


@dataclass
class GlobalCoverageSummary:
    covered_lines: int
    missed_lines: int
    coverage_percent: float
    report_count: int = 0

    def to_json(self) -> dict[str, Any]:
        return _jsonify(self)


@dataclass
class CoverageComparison:
    before: GlobalCoverageSummary
    after: GlobalCoverageSummary
    percentage_increase: float
    covered_line_increase: int
    missed_line_reduction: int

    def to_json(self) -> dict[str, Any]:
        return _jsonify(self)


@dataclass
class FileWorkItem:
    file_path: str
    module: str
    coverage_percent: float
    covered_lines: int
    missed_lines: int
    missed_line_numbers: list[int]
    priority_rank: int = 0
    rationale: str = ""
    assigned_subagent_id: str | None = None
    status: str = "pending"

    def to_json(self) -> dict[str, Any]:
        return _jsonify(self)


@dataclass
class AttemptRecord:
    run_id: str
    subagent_id: str
    file_path: str
    iteration: int
    prompt_version: str
    prompt_hash: str
    tool_call_summary: str
    generated_test_file: str | None
    single_test_command: str
    status: str
    failure_summary: str
    reflective_summary: str

    def to_json(self) -> dict[str, Any]:
        return _jsonify(self)


@dataclass
class SubagentTask:
    subagent_id: str
    file_path: str
    module: str
    worktree_path: Path
    branch_name: str
    suggested_test_path: str
    max_iterations: int

    def to_json(self) -> dict[str, Any]:
        return _jsonify(self)


@dataclass
class SubagentResult:
    subagent_id: str
    file_path: str
    status: str
    worktree_path: Path
    branch_name: str
    commit_hash: str | None = None
    generated_test_files: list[str] = field(default_factory=list)
    attempts: list[AttemptRecord] = field(default_factory=list)
    final_summary: str = ""
    integration_status: str = "pending_review"
    error_message: str = ""
    coverage_after: float | None = None
    coverage_delta: float = 0.0
    missed_line_reduction: int = 0

    def to_json(self) -> dict[str, Any]:
        return _jsonify(self)


@dataclass
class IntegrationDecision:
    subagent_id: str
    branch_name: str
    commit_hash: str
    status: str
    file_path: str
    reason: str
    priority_rank: int = 0

    def to_json(self) -> dict[str, Any]:
        return _jsonify(self)


@dataclass
class LogEvent:
    run_id: str
    step_name: str
    status: str
    started_at: str
    finished_at: str | None = None
    duration_ms: int | None = None
    subagent_id: str | None = None
    file_path: str | None = None
    iteration: int | None = None
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return _jsonify(self)


@dataclass
class RunCheckpoint:
    run_id: str
    phase: str
    repo_url: str
    repo_name: str
    paused: bool
    created_at: str
    updated_at: str
    pending_work_items: list[FileWorkItem] = field(default_factory=list)
    completed_results: list[SubagentResult] = field(default_factory=list)
    pending_integrations: list[IntegrationDecision] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return _jsonify(self)


@dataclass
class ModelDefinition:
    model_id: str
    model_name: str
    api_key_env: str
    api_base: str | None = None

    def to_json(self) -> dict[str, Any]:
        return _jsonify(self)


@dataclass
class ModelEvalCase:
    case_id: str
    model_id: str
    fixture_name: str
    repo_source: Path
    target_file: str

    def to_json(self) -> dict[str, Any]:
        return _jsonify(self)


@dataclass
class ModelEvalResult:
    case_id: str
    model_id: str
    fixture_name: str
    target_file: str
    status: str
    compile_success: bool
    pass_rate: float
    coverage_delta: float
    missed_line_reduction: int
    forbidden_edit_rate: float
    flaky_rate: float
    latency_seconds: float
    tool_call_count: int
    iteration_count: int
    estimated_cost: float
    run_id: str | None = None
    error_message: str = ""

    def to_json(self) -> dict[str, Any]:
        return _jsonify(self)


@dataclass
class WorkflowRunResult:
    run_id: str
    repo_context: RepoContext
    work_items: list[FileWorkItem]
    subagent_results: list[SubagentResult]
    attempts: list[AttemptRecord]
    overview_path: str
    workbook_path: str
    summary_path: str
    coverage_comparison_path: str | None = None

    def to_json(self) -> dict[str, Any]:
        return _jsonify(self)
