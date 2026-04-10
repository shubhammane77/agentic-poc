from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from pathlib import Path

from agentic_testgen.models import AttemptRecord, FileWorkItem, IntegrationDecision, RunCheckpoint, SubagentResult
from agentic_testgen.utils import read_json, utc_timestamp, write_json


class CheckpointStore:
    def __init__(self, checkpoints_dir: Path):
        self.checkpoints_dir = checkpoints_dir
        self.latest_path = checkpoints_dir / "latest.json"

    def save(self, checkpoint: RunCheckpoint) -> None:
        checkpoint.updated_at = utc_timestamp()
        write_json(self.latest_path, checkpoint.to_json())

    def load(self) -> RunCheckpoint | None:
        payload = read_json(self.latest_path)
        if not payload:
            return None
        payload["pending_work_items"] = [FileWorkItem(**item) for item in payload.get("pending_work_items", [])]
        completed_results = []
        for item in payload.get("completed_results", []):
            completed_results.append(
                SubagentResult(
                    subagent_id=item["subagent_id"],
                    file_path=item["file_path"],
                    status=item["status"],
                    worktree_path=Path(item["worktree_path"]),
                    branch_name=item["branch_name"],
                    commit_hash=item.get("commit_hash"),
                    generated_test_files=item.get("generated_test_files", []),
                    attempts=[AttemptRecord(**attempt) for attempt in item.get("attempts", [])],
                    final_summary=item.get("final_summary", ""),
                    integration_status=item.get("integration_status", "pending_review"),
                    error_message=item.get("error_message", ""),
                    coverage_after=item.get("coverage_after"),
                    coverage_delta=item.get("coverage_delta", 0.0),
                    missed_line_reduction=item.get("missed_line_reduction", 0),
                )
            )
        payload["completed_results"] = completed_results
        payload["pending_integrations"] = [IntegrationDecision(**item) for item in payload.get("pending_integrations", [])]
        return RunCheckpoint(**payload)
