from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

from agentic_testgen.models import LogEvent
from agentic_testgen.utils import ensure_dir, utc_timestamp


class SecretRedactor:
    def __init__(self, secrets: list[str] | None = None):
        self._patterns: list[re.Pattern[str]] = []
        for secret in secrets or []:
            if secret:
                self._patterns.append(re.compile(re.escape(secret)))
        self._patterns.extend(
            [
                re.compile(r"(glpat-[A-Za-z0-9_\-]+)"),
                re.compile(r"(gsk_[A-Za-z0-9]+)"),
                re.compile(r"([A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})"),
            ]
        )

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            result = value
            for pattern in self._patterns:
                result = pattern.sub("[REDACTED]", result)
            return result
        if isinstance(value, dict):
            return {key: self.redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        return value


class RunLogger:
    def __init__(self, run_id: str, logs_dir: Path, secrets: list[str] | None = None):
        ensure_dir(logs_dir)
        self.run_id = run_id
        self.logs_dir = logs_dir
        self.events_path = logs_dir / "events.jsonl"
        self.run_log_path = logs_dir / "run.log"
        self.trace_path = logs_dir / "dspy_traces.jsonl"
        self.redactor = SecretRedactor(secrets)
        self._lock = Lock()

    def log_event(
        self,
        step_name: str,
        status: str,
        summary: str = "",
        *,
        subagent_id: str | None = None,
        file_path: str | None = None,
        iteration: int | None = None,
        details: dict[str, Any] | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        event = LogEvent(
            run_id=self.run_id,
            step_name=step_name,
            status=status,
            started_at=started_at or utc_timestamp(),
            finished_at=finished_at,
            duration_ms=duration_ms,
            subagent_id=subagent_id,
            file_path=file_path,
            iteration=iteration,
            summary=self.redactor.redact(summary),
            details=self.redactor.redact(details or {}),
        )
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_json()) + "\n")
        human = (
            f"[{event.started_at}] {event.step_name} {event.status}"
            f"{' subagent=' + subagent_id if subagent_id else ''}"
            f"{' file=' + file_path if file_path else ''}"
            f"{' iter=' + str(iteration) if iteration is not None else ''}"
            f" :: {event.summary}\n"
        )
        with self._lock:
            with self.run_log_path.open("a", encoding="utf-8") as handle:
                handle.write(human)

    def log_trace(self, payload: dict[str, Any]) -> None:
        redacted = self.redactor.redact(payload)
        line = json.dumps(redacted) + "\n"
        with self._lock:
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    @contextmanager
    def step(
        self,
        step_name: str,
        *,
        subagent_id: str | None = None,
        file_path: str | None = None,
        iteration: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        started_at = utc_timestamp()
        started = time.perf_counter()
        self.log_event(
            step_name,
            "started",
            details=details,
            subagent_id=subagent_id,
            file_path=file_path,
            iteration=iteration,
            started_at=started_at,
        )
        sink: dict[str, Any] = {}
        try:
            yield sink
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            self.log_event(
                step_name,
                "failed",
                summary=str(exc),
                details={**(details or {}), **sink},
                subagent_id=subagent_id,
                file_path=file_path,
                iteration=iteration,
                started_at=started_at,
                finished_at=utc_timestamp(),
                duration_ms=duration_ms,
            )
            raise
        duration_ms = int((time.perf_counter() - started) * 1000)
        self.log_event(
            step_name,
            "completed",
            summary=str(sink.get("summary", "")),
            details={**(details or {}), **sink},
            subagent_id=subagent_id,
            file_path=file_path,
            iteration=iteration,
            started_at=started_at,
            finished_at=utc_timestamp(),
            duration_ms=duration_ms,
        )
