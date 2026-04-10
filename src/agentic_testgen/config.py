from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from agentic_testgen.utils import bool_from_env, int_from_env, workspace_default_root


@dataclass
class ModelSettings:
    provider: str = "openai"
    model_name: str = ""
    api_key: str = ""
    api_base: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.model_name and (self.api_key or self.api_base))


@dataclass
class MlflowSettings:
    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "agentic-testgen"
    enabled: bool = True
    strict: bool = False

    def normalized_tracking_uri(self) -> str:
        value = self.tracking_uri.strip()
        if not value:
            return "http://127.0.0.1:5000"
        if "://" not in value:
            return f"http://{value}"
        return value


@dataclass
class AppConfig:
    gitlab_base_url: str = ""
    gitlab_token: str = ""
    gitlab_username: str = "oauth2"
    java_home: str = ""
    maven_home: str = ""
    mvn_bin: str = ""
    max_parallel_subagents: int = 2
    max_subagent_iterations: int = 3
    auto_integrate_successful_worktrees: bool = False
    workspace_root: Path = field(default_factory=workspace_default_root)
    model: ModelSettings = field(default_factory=ModelSettings)
    mlflow: MlflowSettings = field(default_factory=MlflowSettings)

    @classmethod
    def load(cls) -> "AppConfig":
        load_dotenv()
        return cls(
            gitlab_base_url=os.getenv("GITLAB_BASE_URL", "").strip(),
            gitlab_token=os.getenv("GITLAB_TOKEN", "").strip(),
            gitlab_username=os.getenv("GITLAB_USERNAME", "oauth2").strip() or "oauth2",
            java_home=os.getenv("JAVA_HOME", "").strip(),
            maven_home=os.getenv("MAVEN_HOME", "").strip(),
            mvn_bin=os.getenv("MVN_BIN", "").strip(),
            max_parallel_subagents=int_from_env(os.getenv("MAX_PARALLEL_SUBAGENTS"), 2),
            max_subagent_iterations=int_from_env(os.getenv("MAX_SUBAGENT_ITERATIONS"), 3),
            auto_integrate_successful_worktrees=bool_from_env(
                os.getenv("AUTO_INTEGRATE_SUCCESSFUL_WORKTREES"),
                default=False,
            ),
            workspace_root=Path(os.getenv("WORKSPACE_ROOT", str(workspace_default_root()))).expanduser(),
            model=ModelSettings(
                provider=os.getenv("MODEL_PROVIDER", "openai").strip(),
                model_name=os.getenv("MODEL_NAME", "").strip(),
                api_key=os.getenv("MODEL_API_KEY", "").strip(),
                api_base=os.getenv("MODEL_API_BASE", "").strip(),
            ),
            mlflow=MlflowSettings(
                tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000").strip(),
                experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "agentic-testgen").strip(),
                enabled=bool_from_env(os.getenv("ENABLE_MLFLOW_TRACING"), default=True),
                strict=bool_from_env(os.getenv("STRICT_MLFLOW_TRACING"), default=False),
            ),
        )

    def validate_for_run(self) -> None:
        if not self.gitlab_token:
            raise ValueError("GITLAB_TOKEN is required.")

    def maven_executable(self) -> str:
        if self.mvn_bin:
            return self.mvn_bin
        suffix = ".cmd" if platform.system().lower().startswith("win") else ""
        if self.maven_home:
            return str(Path(self.maven_home) / "bin" / f"mvn{suffix}")
        return f"mvn{suffix}"

    def java_executable(self) -> str:
        suffix = ".exe" if platform.system().lower().startswith("win") else ""
        if self.java_home:
            return str(Path(self.java_home) / "bin" / f"java{suffix}")
        return f"java{suffix}"
