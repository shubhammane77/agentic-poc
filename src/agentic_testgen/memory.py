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
    attempt_count: int
    lesson: str
    failure_cause: str
    failure_analysis: str
    failure_feedback: list[str]
    generated_test_files: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "subagent_id": self.subagent_id,
            "file_path": self.file_path,
            "module": self.module,
            "status": self.status,
            "attempt_count": self.attempt_count,
            "lesson": self.lesson,
            "failure_cause": self.failure_cause,
            "failure_analysis": self.failure_analysis,
            "failure_feedback": self.failure_feedback,
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
                    "updated_at": utc_timestamp(),
                }
            )
            payload["entries"] = self._combined_entries(payload)
        write_json(run_memory_path, payload)

    def record_result(self, run_memory_path: Path, repo_context: RepoContext, item: FileWorkItem, result: SubagentResult) -> None:
        entry = self._entry_from_result(repo_context, item, result)
        run_payload = read_json(run_memory_path, default=self._new_run_memory(repo_context))
        run_payload["entries"] = self._merge_entry(self._combined_entries(run_payload), entry)
        run_payload["updated_at"] = utc_timestamp()
        write_json(run_memory_path, run_payload)

        global_payload = read_json(self.global_memory_path, default={"repos": {}})
        repos = global_payload.setdefault("repos", {})
        repo_key = self._repo_key(repo_context)
        repo_memory = repos.setdefault(
            repo_key,
            {
                "lessons": [],
            },
        )
        repo_memory["lessons"] = self._merge_repo_lesson(repo_memory, result.status, entry["lesson"])
        write_json(self.global_memory_path, global_payload)

    def lessons_for_item(self, run_memory_path: Path, repo_context: RepoContext, item: FileWorkItem, *, limit: int = 4) -> list[str]:
        run_payload = read_json(run_memory_path, default=self._new_run_memory(repo_context))
        global_payload = read_json(self.global_memory_path, default={"repos": {}})
        repo_payload = global_payload.get("repos", {}).get(self._repo_key(repo_context), {})

        ranked_entries = self._rank_entries(self._combined_entries(run_payload), repo_context, item)
        successes = [entry for entry in ranked_entries if entry.get("status") == "passed"]
        failures = [entry for entry in ranked_entries if entry.get("status") != "passed"]
        repo_lessons = repo_payload.get("lessons", [])
        if not repo_lessons:
            # Backward compatibility for older global memory format.
            repo_lessons = [
                {"status": "passed", "lesson": lesson}
                for lesson in repo_payload.get("success_lessons", [])
            ] + [
                {"status": "failed", "lesson": lesson}
                for lesson in repo_payload.get("failure_lessons", [])
            ]
        success_lessons = [entry.get("lesson", "") for entry in repo_lessons if entry.get("status") == "passed"]
        failure_lessons = [entry.get("lesson", "") for entry in repo_lessons if entry.get("status") != "passed"]

        lessons: list[str] = []
        for entry in successes[:2]:
            lessons.append(f"Success pattern from {entry['file_path']}: {entry['lesson']}")
        for lesson in success_lessons[:2]:
            lessons.append(f"Success pattern: {lesson}")
        for entry in failures[:2]:
            lessons.append(f"Avoid failure from {entry['file_path']}: {entry['lesson']}")
        for lesson in failure_lessons[:2]:
            lessons.append(f"Avoid failure: {lesson}")
        return lessons[:limit]

    def _new_run_memory(self, repo_context: RepoContext) -> dict[str, Any]:
        return {
            "run_id": repo_context.run_id,
            "repo_name": repo_context.repo_name,
            "repo_url": repo_context.repo_url,
            "repo_key": self._repo_key(repo_context),
            "updated_at": utc_timestamp(),
            "entries": [],
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
            attempt_count=len(result.attempts),
            lesson=self._lesson_text(result),
            failure_cause=self._infer_failure_cause(result),
            failure_analysis=self._failure_analysis_text(result),
            failure_feedback=self._failure_feedback(result),
            generated_test_files=result.generated_test_files[:],
        ).to_json()

    def _lesson_text(self, result: SubagentResult) -> str:
        if result.status == "passed":
            text = result.final_summary or "Generated tests passed validation."
        else:
            reason = result.error_message or "Generated tests failed validation."
            summary = result.final_summary or ""
            text = f"Failure reason: {reason}" if not summary else f"Failure reason: {reason}\nAgent reflection: {summary}"
        return text[:1000]

    def _failure_feedback(self, result: SubagentResult) -> list[str]:
        if result.status == "passed":
            return []
        feedback: list[str] = []
        for attempt in result.attempts:
            analysis = (attempt.failure_analysis or "").strip()
            if analysis:
                feedback.append(f"Attempt {attempt.iteration} analysis: {analysis[:600]}")
        if not feedback:
            fallback = self._failure_analysis_text(result)
            if fallback:
                feedback.append(fallback[:600])
        return feedback[:8]

    def _failure_analysis_text(self, result: SubagentResult) -> str:
        if result.status == "passed":
            return ""
        analyses = [attempt.failure_analysis.strip() for attempt in result.attempts if attempt.failure_analysis.strip()]
        if analyses:
            return "\n".join(analyses)[:4000]
        return (result.final_summary or "").strip()[:4000]

    def _infer_failure_cause(self, result: SubagentResult) -> str:
        if result.status == "passed":
            return "none"
        latest = ""
        if result.attempts:
            latest = (result.attempts[-1].failure_summary or "").lower()
        analysis = self._failure_analysis_text(result).lower()
        text = f"{latest}\n{analysis}"
        if "compilation failure" in text or "cannot find symbol" in text or "package " in text and " does not exist" in text:
            return "compile_error"
        if "assertion" in text or "expected:" in text or "but was:" in text:
            return "assertion_mismatch"
        if "junit.jupiter" in text and "org.junit" in text:
            return "wrong_test_framework_usage"
        if "no tests were executed" in text or "no tests found" in text:
            return "test_discovery_error"
        if "timed out" in text or "timeout" in text:
            return "timeout"
        if "dependency" in text and "failed" in text:
            return "dependency_error"
        return "unknown"

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

    def _merge_repo_lesson(self, repo_memory: dict[str, Any], status: str, lesson: str) -> list[dict[str, str]]:
        lessons = repo_memory.get("lessons", [])
        if not lessons:
            # Backward compatibility for older global memory format.
            lessons = [
                {"status": "passed", "lesson": item}
                for item in repo_memory.get("success_lessons", [])
            ] + [
                {"status": "failed", "lesson": item}
                for item in repo_memory.get("failure_lessons", [])
            ]
        target_status = "passed" if status == "passed" else "failed"
        filtered = [
            item
            for item in lessons
            if not (item.get("status") == target_status and item.get("lesson") == lesson)
        ]
        filtered.insert(0, {"status": target_status, "lesson": lesson})
        return filtered[:MAX_MEMORY_ENTRIES]

    def _combined_entries(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        entries = payload.get("entries", [])
        if entries:
            return entries
        # Backward compatibility for older run memory format.
        return payload.get("successes", []) + payload.get("failures", [])

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
                self._score_entry(entry, item),
                entry.get("timestamp", ""),
            ),
            reverse=True,
        )

    def _score_entry(self, entry: dict[str, Any], item: FileWorkItem) -> int:
        score = 0
        if entry.get("module") == item.module:
            score += 4
        if self._shared_parent(entry.get("file_path", ""), item.file_path):
            score += 2
        if entry.get("status") == "passed":
            score += 1
        return score

    def _shared_parent(self, left: str, right: str) -> bool:
        left_parent = str(Path(left).parent)
        right_parent = str(Path(right).parent)
        return left_parent == right_parent and left_parent not in {"", "."}
