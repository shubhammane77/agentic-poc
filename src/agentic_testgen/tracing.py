from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator
from urllib.error import URLError
from urllib.request import urlopen
from pathlib import Path

from agentic_testgen.config import MlflowSettings
from agentic_testgen.logging import RunLogger


class MlflowTracer:
    def __init__(self, settings: MlflowSettings, logger: RunLogger):
        self.settings = settings
        self.logger = logger
        self.mlflow = None
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
        with self.mlflow.start_run(run_name=name):
            if tags:
                self.mlflow.set_tags(tags)
            yield self.mlflow

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
