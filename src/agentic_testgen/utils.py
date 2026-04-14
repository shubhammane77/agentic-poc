from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "item"


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def workspace_default_root() -> Path:
    return Path(tempfile.gettempdir()) / "agt"


def read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_command_logs(base_dir: Path, prefix: str, result: "CommandResult") -> dict[str, str]:
    ensure_dir(base_dir)
    stem = f"{slugify(prefix)}-{uuid.uuid4().hex[:8]}"
    stdout_path = base_dir / f"{stem}.stdout.log"
    stderr_path = base_dir / f"{stem}.stderr.log"
    combined_path = base_dir / f"{stem}.combined.log"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    combined_path.write_text(
        f"COMMAND: {sanitize_command(result.args)}\n"
        f"EXIT_CODE: {result.exit_code}\n"
        f"DURATION_SECONDS: {result.duration_seconds:.3f}\n\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
        encoding="utf-8",
    )
    return {
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "combined": str(combined_path),
    }


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def bool_from_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def int_from_env(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    return int(value)


def float_from_env(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    return float(value)


def tail_lines(path: Path, limit: int = 50) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[-limit:]


def sanitize_command(args: list[str]) -> str:
    return " ".join(args)


@dataclass
class CommandResult:
    args: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def run_command(
    args: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> CommandResult:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env={**os.environ, **(env or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            args=args,
            exit_code=127,
            stdout="",
            stderr=str(exc),
            duration_seconds=time.perf_counter() - start,
        )
    return CommandResult(
        args=args,
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        duration_seconds=time.perf_counter() - start,
    )
