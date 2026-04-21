from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from agentic_testgen.core.config import AppConfig
from agentic_testgen.core.logging import RunLogger
from agentic_testgen.core.utils import CommandResult, run_command


def sanitize_repo_url(repo_url: str) -> str:
    parts = urlsplit(repo_url)
    return urlunsplit((parts.scheme, parts.hostname or "", parts.path, parts.query, parts.fragment))


def authenticated_repo_url(repo_url: str, username: str, token: str) -> str:
    parts = urlsplit(repo_url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, f"{username}:{token}@{host}", parts.path, parts.query, parts.fragment))


class GitLabRepositoryManager:
    def __init__(self, config: AppConfig, logger: RunLogger):
        self.config = config
        self.logger = logger

    def clone(self, repo_url: str, destination: Path) -> CommandResult:
        auth_url = authenticated_repo_url(repo_url, self.config.gitlab_username, self.config.gitlab_token)
        result = run_command(
            [
                "git",
                "-c",
                "http.sslVerify=false",
                "-c",
                "core.longpaths=true",
                "clone",
                auth_url,
                str(destination),
            ]
        )
        self.logger.log_event(
            "git.clone",
            "completed" if result.ok else "failed",
            summary=f"Cloned {sanitize_repo_url(repo_url)}",
            details={"exit_code": result.exit_code, "stderr": result.stderr, "stdout": result.stdout},
        )
        return result
