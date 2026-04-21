# Coding Conventions

**Analysis Date:** 2026-04-21

## Naming Patterns

**Files:**
- `snake_case.py` for all modules: `agents.py`, `coverage.py`, `memory.py`, `checkpointing.py`
- No suffix conventions (no `_service`, `_manager` uniformity — mix of descriptive nouns: `WorkspaceManager`, `CoverageAnalyzer`, `ReportWriter`)

**Classes:**
- `PascalCase` throughout: `DaddySubagentsReflectiveWorkflow`, `SafeToolset`, `RunLogger`, `SecretRedactor`
- Dataclasses named as nouns: `RepoContext`, `FileWorkItem`, `SubagentResult`, `AttemptRecord`

**Functions and Methods:**
- `snake_case` for all: `run_tests_with_coverage`, `build_work_items`, `parse_jacoco_xml`
- Private methods prefixed with `_`: `_configure`, `_resolve_active_path`, `_is_within_test_tree`, `_jsonify`
- Helper module-level functions also `snake_case`: `utc_timestamp`, `new_run_id`, `prompt_hash`, `bool_from_env`

**Variables and Parameters:**
- `snake_case` everywhere: `run_id`, `repo_root`, `maven_logs_dir`, `api_key_env`
- Constants: `UPPER_SNAKE_CASE` — `PROMPT_VERSION = "daddy_subagents_reflective_v1"`, `MAX_MEMORY_ENTRIES = 50`

**Type Annotations:**
- All public functions and methods carry full type annotations
- Return types always declared: `def run_command(...) -> CommandResult:`
- `list[str]` not `List[str]` (lowercase generics, Python 3.10+ style enabled via `from __future__ import annotations`)

## Code Style

**Formatting:**
- No formatter config detected (no `.prettierrc`, `ruff.toml`, `.flake8`, or `[tool.ruff]` section)
- Indentation: 4 spaces (PEP 8)
- Line length: approximately 110-120 chars observed in practice (no enforced limit found)
- Trailing comma on multi-line structures used consistently

**Linting:**
- No linter config file present in the project root
- Inline suppressions used sparingly with targeted codes:
  - `# type: ignore[assignment]` for optional-import `dspy = None` pattern
  - `# noqa: F401` for intentional path-setup imports in tests
  - `# pragma: no cover` on optional-dependency branches

## Future Annotations

Every source file begins with `from __future__ import annotations` — all 16 modules. This enables PEP 563 deferred evaluation, allowing forward references and lowercase generic types (`list[str]`, `dict[str, Any]`, `Path | None`) without runtime cost.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
```

## Import Organization

**Order observed:**
1. `from __future__ import annotations` (always first)
2. Standard library imports (alphabetical within group)
3. Third-party imports (e.g., `dspy`, `typer`, `dotenv`)
4. Local `agentic_testgen.*` imports (alphabetical)

**Optional dependency guard pattern:**
```python
try:
    import dspy
except ImportError:  # pragma: no cover - optional runtime dependency
    dspy = None  # type: ignore[assignment]
```
Used in `src/agentic_testgen/agents.py` and `src/agentic_testgen/tools.py` for `dspy`, and `src/agentic_testgen/tracing.py` for `mlflow`.

**Path aliases:** None — imports use full `agentic_testgen.*` package paths.

## Data Modeling

**Pattern:** All domain data types are `@dataclass` (not Pydantic, despite `pydantic` being a dependency).

```python
@dataclass
class FileWorkItem:
    file_path: str
    module: str
    coverage_percent: float
    covered_lines: int
    missed_lines: int
    missed_line_numbers: list[int]
    priority_rank: int = 0
    rationale: str = ""
    assigned_subagent_id: str | None = None
    status: str = "pending"

    def to_json(self) -> dict[str, Any]:
        return _jsonify(self)
```

Every dataclass implements `to_json() -> dict[str, Any]` for JSON serialization. The shared `_jsonify()` helper in `src/agentic_testgen/models.py` recursively serializes dataclasses, `Path` objects, dicts, and lists.

**Mutable defaults:** Use `field(default_factory=list)` and `field(default_factory=dict)` — never bare mutable defaults.

## Error Handling

**Strategy:** Raise built-in exceptions with descriptive messages rather than custom exception classes.

```python
# Path guard in SafeToolset
try:
    resolved.relative_to(self.active_root_resolved)
except ValueError:
    raise ValueError(f"Path outside allowed root: {path_value}")

# Config validation
def validate_for_run(self) -> None:
    if not self.gitlab_token:
        raise ValueError("GITLAB_TOKEN is required.")
```

**Subprocess errors:** `run_command()` in `src/agentic_testgen/utils.py` catches `FileNotFoundError` and returns a `CommandResult` with `exit_code=127` rather than propagating the exception — callers check `result.ok`.

**Optional dependencies:** Failures silently degrade — DSPy and MLflow being unavailable returns stub behavior, logged via `RunLogger.log_event()`.

**CLI errors:** `typer.BadParameter` used for user-facing validation in `src/agentic_testgen/cli.py`.

## Logging

**Framework:** Custom `RunLogger` class in `src/agentic_testgen/logging.py` — not Python's `logging` module.

**Patterns:**
- Structured events via `logger.log_event(step_name, status, summary=..., details={...})`
- Events written as JSONL to `events.jsonl`; human-readable lines to `run.log`
- Context manager `logger.step(...)` wraps operations for timing
- Thread-safe via `threading.Lock`
- All output is secret-redacted via `SecretRedactor` before writing

**Secret redaction:** `SecretRedactor` in `src/agentic_testgen/logging.py` strips GitLab tokens (`glpat-*`), Groq API keys (`gsk_*`), and JWT-like tokens automatically.

## Comments

**When to comment:**
- Inline on non-obvious suppressions: `# pragma: no cover - optional runtime dependency`
- Module-level docstring in `__init__.py`: `"""Agentic test generation platform for GitLab Maven repositories."""`
- Typer CLI command docstrings double as `--help` text

**No JSDoc-style block comments** in source. Functions are self-documenting via type annotations and descriptive naming.

## Function Design

**Size:** Utilities kept small (5-15 lines). Service/workflow methods longer but single-purpose.

**Parameters:**
- Keyword-only parameters enforced with `*` for boolean flags: `console_enabled: bool = True` in `RunLogger.__init__`
- `Path | None = None` used consistently for optional path params

**Return Values:**
- Complex returns use named dataclasses (`CommandResult`, `SubagentResult`)
- Tuples used for multi-value returns in `run_tests_with_coverage` → `tuple[object, list[CoverageRecord], dict[str, str]]`

## Module Design

**Exports:**
- `__init__.py` exports only `__version__` via `__all__`; no re-exporting of internals
- Each module is cohesive: `models.py` for data types only, `utils.py` for stateless helpers, `config.py` for settings

**Barrel Files:** No barrel-file pattern — consumers import directly from submodules:
```python
from agentic_testgen.models import FileWorkItem, SubagentResult
from agentic_testgen.utils import run_command, write_json
```

---

*Convention analysis: 2026-04-21*
