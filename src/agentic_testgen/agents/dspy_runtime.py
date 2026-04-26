from __future__ import annotations

from typing import Any

try:
    import dspy
except ImportError:  # pragma: no cover - optional runtime dependency
    dspy = None  # type: ignore[assignment]

from agentic_testgen.core.config import AppConfig
from agentic_testgen.core.logging import RunLogger
from agentic_testgen.core.models import AnalysisSummary, ModelDefinition


class DSPyRuntime:
    def __init__(self, config: AppConfig, logger: RunLogger, model_override: ModelDefinition | None = None):
        self.config = config
        self.logger = logger
        self.model_override = model_override
        self.enabled = False
        self.model_id = "unconfigured"
        self._configure()

    def _configure(self) -> None:
        if dspy is None:
            self.logger.log_event("dspy.configure", "skipped", summary="DSPy not installed")
            return
        model_name = self.model_override.model_name if self.model_override else self.config.model.model_name
        if not model_name:
            self.logger.log_event("dspy.configure", "skipped", summary="No model configured")
            return
        api_key = ""
        api_base = ""
        if self.model_override:
            import os

            api_key = os.getenv(self.model_override.api_key_env, "")
            api_base = self.model_override.api_base or ""
            self.model_id = self.model_override.model_id
        else:
            api_key = self.config.model.api_key
            api_base = self.config.model.api_base
            self.model_id = model_name
        temperature: float = self.config.model.temperature
        top_p: float = self.config.model.top_p
        max_tokens: int | None = self.config.model.max_tokens

        final_model = model_name
        if "/" not in final_model and self.config.model.provider:
            final_model = f"{self.config.model.provider}/{final_model}"
        try:
            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if api_base:
                kwargs["api_base"] = api_base
            kwargs["temperature"] = float(temperature)
            kwargs["top_p"] = float(top_p)
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)
            lm = dspy.LM(final_model, **kwargs)
            dspy.configure(lm=lm)
            self.enabled = True
            self.logger.log_event("dspy.configure", "completed", summary=f"Configured {self.model_id}")
        except Exception as exc:
            self.logger.log_event("dspy.configure", "failed", summary=str(exc))

    def overview(self, repo_tree: str, module_paths: list[str]) -> str:
        if not self.enabled:
            return (
                "# Repository Overview\n\n"
                f"- Modules: {', '.join(module_paths) if module_paths else '(none detected)'}\n\n"
                "## Tree\n\n```text\n"
                f"{repo_tree}\n```"
            )
        try:
            program = dspy.ChainOfThought("repo_tree, module_paths -> overview_markdown")
            result = program(
                repo_tree=repo_tree,
                module_paths=", ".join(module_paths),
            )
            return getattr(result, "overview_markdown", str(result))
        except Exception as exc:
            self.logger.log_event("dspy.overview", "failed", summary=str(exc))
            return (
                "# Repository Overview\n\n"
                f"- Modules: {', '.join(module_paths) if module_paths else '(none detected)'}\n"
            )

    def reflect(self, objective: str, latest_output: str, prior_failures: str) -> str:
        if not self.enabled:
            return latest_output or prior_failures or "No reflection available."
        try:
            program = dspy.Predict("objective, latest_output, prior_failures -> summary")
            result = program(
                objective=objective,
                latest_output=latest_output[:8000],
                prior_failures=prior_failures[:8000],
            )
            return getattr(result, "summary", str(result))
        except Exception as exc:
            self.logger.log_event("dspy.reflect", "failed", summary=str(exc))
            return latest_output or prior_failures or str(exc)

    def analyze_failure(self, file_path: str, iteration: int, failure_output: str) -> str:
        if not self.enabled:
            output = failure_output or "No failure output provided."
            return self._limit_words(output, 500)
        try:
            program = dspy.Predict(
                "file_path, iteration, failure_output, answer_style -> concise_failure_analysis"
            )
            result = program(
                file_path=file_path,
                iteration=str(iteration),
                failure_output=failure_output[:8000],
                answer_style=(
                    "Summarize only the root cause context from this stack/output. "
                    "Do not include full failure history, repeated logs, or step-by-step timeline. "
                    "Maximum 500 words."
                ),
            )
            analysis = getattr(result, "concise_failure_analysis", str(result))
            return self._limit_words(analysis, 500)
        except Exception as exc:
            self.logger.log_event("dspy.failure_analysis", "failed", summary=str(exc))
            output = failure_output or str(exc)
            return self._limit_words(output, 500)

    def _limit_words(self, text: str, limit: int) -> str:
        words = text.split()
        if len(words) <= limit:
            return text
        return " ".join(words[:limit])
