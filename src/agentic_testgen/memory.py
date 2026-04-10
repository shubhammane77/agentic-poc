from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_testgen.models import FileWorkItem, RepoContext, SubagentResult
from agentic_testgen.utils import read_json, utc_timestamp, write_json

MAX_MEMORY_ENTRIES = 50


@dataclass
class MemoryEntry:
    timestamp: str
    run_id: str
    subagent_id: str
    file_path: str
    module: str
    status: str
    testing_stack: str
    attempt_count: int
    lesson: str
    failure_summary: str
    generated_test_files: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "subagent_id": self.subagent_id,
            "file_path": self.file_path,
            "module": self.module,
            "status": self.status,
            "testing_stack": self.testing_stack,
            "attempt_count": self.attempt_count,
            "lesson": self.lesson,
            "failure_summary": self.failure_summary,
            "generated_test_files": self.generated_test_files,
        }


class MemoryManager:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root

    @property
    def global_memory_path(self) -> Path:
        return self.workspace_root / "global_memory.json"

    def initialize_run_memory(self, run_memory_path: Path, repo_context: RepoContext) -> None:
        payload = read_json(run_memory_path, default=None)
        if payload is None:
            payload = self._new_run_memory(repo_context)
        else:
            payload.update(
                {
                    "run_id": repo_context.run_id,
                    "repo_name": repo_context.repo_name,
                    "repo_url": repo_context.repo_url,
                    "repo_key": self._repo_key(repo_context),
                    "test_framework": repo_context.test_framework,
                    "test_framework_version": repo_context.test_framework_version,
                    "updated_at": utc_timestamp(),
                }
            )
            payload.setdefault("successes", [])
            payload.setdefault("failures", [])
        write_json(run_memory_path, payload)

    def record_result(self, run_memory_path: Path, repo_context: RepoContext, item: FileWorkItem, result: SubagentResult) -> None:
        entry = self._entry_from_result(repo_context, item, result)
        run_payload = read_json(run_memory_path, default=self._new_run_memory(repo_context))
        bucket_name = "successes" if result.status == "passed" else "failures"
        run_payload[bucket_name] = self._merge_entry(run_payload.get(bucket_name, []), entry)
        run_payload["updated_at"] = utc_timestamp()
        write_json(run_memory_path, run_payload)

        global_payload = read_json(self.global_memory_path, default={"repos": {}})
        repos = global_payload.setdefault("repos", {})
        repo_key = self._repo_key(repo_context)
        repo_memory = repos.setdefault(
            repo_key,
            {
                "repo_name": repo_context.repo_name,
                "repo_url": repo_context.repo_url,
                "test_framework": repo_context.test_framework,
                "test_framework_version": repo_context.test_framework_version,
                "successes": [],
                "failures": [],
                "updated_at": utc_timestamp(),
            },
        )
        repo_memory["repo_name"] = repo_context.repo_name
        repo_memory["repo_url"] = repo_context.repo_url
        repo_memory["test_framework"] = repo_context.test_framework
        repo_memory["test_framework_version"] = repo_context.test_framework_version
        repo_memory[bucket_name] = self._merge_entry(repo_memory.get(bucket_name, []), entry)
        repo_memory["updated_at"] = utc_timestamp()
        write_json(self.global_memory_path, global_payload)

    def lessons_for_item(self, run_memory_path: Path, repo_context: RepoContext, item: FileWorkItem, *, limit: int = 4) -> list[str]:
        run_payload = read_json(run_memory_path, default=self._new_run_memory(repo_context))
        global_payload = read_json(self.global_memory_path, default={"repos": {}})
        repo_payload = global_payload.get("repos", {}).get(self._repo_key(repo_context), {})

        successes = self._rank_entries(
            [*run_payload.get("successes", []), *repo_payload.get("successes", [])],
            repo_context,
            item,
        )
        failures = self._rank_entries(
            [*run_payload.get("failures", []), *repo_payload.get("failures", [])],
            repo_context,
            item,
        )

        lessons: list[str] = []
        for entry in successes[:2]:
            lessons.append(f"Success pattern from {entry['file_path']}: {entry['lesson']}")
        for entry in failures[:2]:
            lessons.append(f"Avoid failure from {entry['file_path']}: {entry['lesson']}")
        return lessons[:limit]

    def _new_run_memory(self, repo_context: RepoContext) -> dict[str, Any]:
        return {
            "run_id": repo_context.run_id,
            "repo_name": repo_context.repo_name,
            "repo_url": repo_context.repo_url,
            "repo_key": self._repo_key(repo_context),
            "test_framework": repo_context.test_framework,
            "test_framework_version": repo_context.test_framework_version,
            "updated_at": utc_timestamp(),
            "successes": [],
            "failures": [],
        }

    def _repo_key(self, repo_context: RepoContext) -> str:
        return f"{repo_context.repo_name}::{repo_context.repo_url}"

    def _entry_from_result(self, repo_context: RepoContext, item: FileWorkItem, result: SubagentResult) -> dict[str, Any]:
        return MemoryEntry(
            timestamp=utc_timestamp(),
            run_id=repo_context.run_id,
            subagent_id=result.subagent_id,
            file_path=result.file_path,
            module=item.module,
            status=result.status,
            testing_stack=repo_context.testing_stack_display,
            attempt_count=len(result.attempts),
            lesson=self._lesson_text(result),
            failure_summary=result.error_message[:4000],
            generated_test_files=result.generated_test_files[:],
        ).to_json()

    def _lesson_text(self, result: SubagentResult) -> str:
        if result.status == "passed":
            text = result.final_summary or "Generated tests passed validation."
        else:
            text = result.error_message or result.final_summary or "Generated tests failed validation."
        return text[:1000]

    def _merge_entry(self, entries: list[dict[str, Any]], entry: dict[str, Any]) -> list[dict[str, Any]]:
        filtered = [
            item
            for item in entries
            if not (
                item.get("run_id") == entry["run_id"]
                and item.get("subagent_id") == entry["subagent_id"]
                and item.get("status") == entry["status"]
            )
        ]
        filtered.insert(0, entry)
        return filtered[:MAX_MEMORY_ENTRIES]

    def _rank_entries(
        self,
        entries: list[dict[str, Any]],
        repo_context: RepoContext,
        item: FileWorkItem,
    ) -> list[dict[str, Any]]:
        deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
        for entry in entries:
            key = (
                entry.get("file_path", ""),
                entry.get("status", ""),
                entry.get("lesson", ""),
            )
            if key not in deduped:
                deduped[key] = entry
        return sorted(
            deduped.values(),
            key=lambda entry: (
                self._score_entry(entry, repo_context, item),
                entry.get("timestamp", ""),
            ),
            reverse=True,
        )

    def _score_entry(self, entry: dict[str, Any], repo_context: RepoContext, item: FileWorkItem) -> int:
        score = 0
        if entry.get("module") == item.module:
            score += 4
        if entry.get("testing_stack") == repo_context.testing_stack_display:
            score += 3
        if self._shared_parent(entry.get("file_path", ""), item.file_path):
            score += 2
        if entry.get("status") == "passed":
            score += 1
        return score

    def _shared_parent(self, left: str, right: str) -> bool:
        left_parent = str(Path(left).parent)
        right_parent = str(Path(right).parent)
        return left_parent == right_parent and left_parent not in {"", "."}
