from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator
from urllib.error import URLError
from urllib.request import urlopen

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
        try:
            urlopen(self.settings.tracking_uri, timeout=2)
            self.logger.log_event("mlflow.validate", "completed", summary="MLflow server reachable")
            return True
        except URLError as exc:
            self.logger.log_event("mlflow.validate", "skipped", summary=f"MLflow unavailable: {exc}")
            if self.settings.strict:
                raise RuntimeError(f"MLflow server unavailable at {self.settings.tracking_uri}") from exc
            return False

    def configure(self) -> None:
        if not (self.mlflow and self.settings.enabled):
            return
        self.mlflow.set_tracking_uri(self.settings.tracking_uri)
        self.mlflow.set_experiment(self.settings.experiment_name)
        if hasattr(self.mlflow, "dspy"):
            try:
                self.mlflow.dspy.autolog()
            except Exception as exc:  # pragma: no cover - optional runtime integration
                self.logger.log_event("mlflow.autolog", "skipped", summary=f"DSPy autolog unavailable: {exc}")

    @contextmanager
    def run(self, name: str, tags: dict[str, str] | None = None) -> Iterator[Any]:
        if not (self.mlflow and self.settings.enabled):
            yield None
            return
        with self.mlflow.start_run(run_name=name):
            if tags:
                self.mlflow.set_tags(tags)
            yield self.mlflow

