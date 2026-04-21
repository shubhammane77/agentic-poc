from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.request import urlopen

from agentic_testgen.core.config import MlflowSettings
from agentic_testgen.core.logging import RunLogger


class MlflowTracer:
    def __init__(self, settings: MlflowSettings, logger: RunLogger):
        self.settings = settings
        self.logger = logger
        self.mlflow = None
        self._trace_ids: set[str] = set()
        if settings.enabled:
            try:
                import mlflow  # type: ignore
            except Exception as exc:  # pragma: no cover - depends on local environment
                logger.log_event("mlflow.import", "skipped", summary=f"MLflow unavailable: {exc}")
                if settings.strict:
                    raise
            else:
                self.mlflow = mlflow

    def validate(self) -> bool:
        if not self.settings.enabled:
            return False
        tracking_uri = self.settings.normalized_tracking_uri()
        try:
            urlopen(tracking_uri, timeout=2)
            self.logger.log_event(
                "mlflow.validate",
                "completed",
                summary="MLflow server reachable",
                details={"tracking_uri": tracking_uri},
            )
            return True
        except Exception as exc:
            self.logger.log_event(
                "mlflow.validate",
                "skipped",
                summary=f"MLflow unavailable: {exc}",
                details={"tracking_uri": tracking_uri},
            )
            if self.settings.strict:
                raise RuntimeError(f"MLflow server unavailable at {tracking_uri}") from exc
            return False

    def configure(self) -> None:
        if not (self.mlflow and self.settings.enabled):
            return
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
        tracking_uri = self.settings.normalized_tracking_uri()
        self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(self.settings.experiment_name)
        self.logger.log_event(
            "mlflow.configure",
            "completed",
            summary="Configured MLflow client",
            details={"tracking_uri": tracking_uri, "experiment_name": self.settings.experiment_name},
        )
        if hasattr(self.mlflow, "dspy"):
            try:
                self.mlflow.dspy.autolog()
            except Exception as exc:  # pragma: no cover - optional runtime integration
                self.logger.log_event("mlflow.autolog", "skipped", summary=f"DSPy autolog unavailable: {exc}")

    @property
    def active(self) -> bool:
        return bool(self.mlflow and self.settings.enabled)

    @contextmanager
    def run(self, name: str, tags: dict[str, str] | None = None) -> Iterator[Any]:
        if not (self.mlflow and self.settings.enabled):
            yield None
            return
        try:
            with self.mlflow.start_run(run_name=name):
                if tags:
                    self.mlflow.set_tags(tags)
                yield self.mlflow
        except Exception as exc:
            self.logger.log_event(
                "mlflow.run",
                "skipped",
                summary=f"MLflow run unavailable: {exc}",
                details={"run_name": name},
            )
            yield None

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.active:
            return
        try:
            sanitized = {key: str(value)[:500] for key, value in params.items()}
            self.mlflow.log_params(sanitized)
        except Exception as exc:  # pragma: no cover
            self.logger.log_event("mlflow.log_params", "skipped", summary=str(exc))

    def log_metrics(self, metrics: dict[str, float | int]) -> None:
        if not self.active:
            return
        try:
            self.mlflow.log_metrics(metrics)
        except Exception as exc:  # pragma: no cover
            self.logger.log_event("mlflow.log_metrics", "skipped", summary=str(exc))

    def log_text(self, text: str, artifact_file: str) -> None:
        if not self.active:
            return
        try:
            self.mlflow.log_text(text, artifact_file)
        except Exception as exc:  # pragma: no cover
            self.logger.log_event("mlflow.log_text", "skipped", summary=str(exc))

    def log_artifact(self, path: str | Path) -> None:
        if not self.active:
            return
        try:
            self.mlflow.log_artifact(str(path))
        except Exception as exc:  # pragma: no cover
            self.logger.log_event("mlflow.log_artifact", "skipped", summary=str(exc))

    def tag_last_trace(self, tags: dict[str, str]) -> None:
        if not self.active:
            return
        try:
            trace_id = self.mlflow.get_last_active_trace_id()
            if not trace_id:
                return
            self._trace_ids.add(trace_id)
            for key, value in tags.items():
                self.mlflow.set_trace_tag(trace_id, key, str(value))
        except Exception as exc:  # pragma: no cover
            self.logger.log_event("mlflow.tag_trace", "skipped", summary=str(exc))

    def token_usage_summary(self) -> dict[str, int]:
        summary = {
            "trace_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        if not self.active:
            return summary
        for trace_id in sorted(self._trace_ids):
            try:
                trace = self.mlflow.get_trace(trace_id)
            except Exception as exc:  # pragma: no cover
                self.logger.log_event("mlflow.get_trace", "skipped", summary=str(exc), details={"trace_id": trace_id})
                continue
            if not trace:
                continue
            usage = getattr(getattr(trace, "info", None), "token_usage", None) or {}
            input_tokens = self._safe_int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
            output_tokens = self._safe_int(usage.get("output_tokens", usage.get("completion_tokens", 0)))
            total_tokens = self._safe_int(usage.get("total_tokens", input_tokens + output_tokens))
            summary["trace_count"] += 1
            summary["input_tokens"] += input_tokens
            summary["output_tokens"] += output_tokens
            summary["total_tokens"] += total_tokens
        return summary

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value or 0)
        except Exception:
            return 0
