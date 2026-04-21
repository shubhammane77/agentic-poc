# Codebase Structure

**Analysis Date:** 2026-04-21

## Directory Layout

```
agentic-poc/                        # Project root
├── main.py                         # Thin entry point: imports cli.main and calls it
├── pyproject.toml                  # Package metadata, dependencies, CLI entry point
├── uv.lock                         # Dependency lockfile (uv)
├── .env.example                    # Template for required environment variables
├── .python-version                 # Pins Python version (3.13)
├── .gitignore                      # Standard ignores
├── README.md                       # Project documentation
│
├── src/                            # Installable Python package source
│   └── agentic_testgen/            # Main package (installed as `agentic-testgen`)
│       ├── __init__.py             # Version declaration + SSL cert bootstrap
│       ├── __main__.py             # Enables `python -m agentic_testgen`
│       ├── agents.py               # Core orchestrator + DSPy runtime (largest file ~65 KB)
│       ├── checkpointing.py        # RunCheckpoint serialization/deserialization
│       ├── cli.py                  # Typer CLI commands (run, status, pause, resume, etc.)
│       ├── config.py               # AppConfig, ModelSettings, MlflowSettings dataclasses
│       ├── coverage.py             # Maven/JaCoCo test runner + XML report parser
│       ├── evaluation.py           # Model matrix evaluator (eval harness)
│       ├── gitlab.py               # GitLab clone with token authentication
│       ├── logging.py              # RunLogger, SecretRedactor (thread-safe structured logging)
│       ├── memory.py               # MemoryManager (per-run + global lessons persistence)
│       ├── models.py               # All data classes (no business logic)
│       ├── reporting.py            # ReportWriter: Markdown, JSON, CSV, Excel output
│       ├── tools.py                # SafeToolset + ToolContext for DSPy ReAct loops
│       ├── tracing.py              # MlflowTracer wrapper
│       ├── utils.py                # Shared utilities: run_command, JSON I/O, slugify, etc.
│       └── workspace.py            # WorkspaceManager + RunWorkspace directory structure
│
├── core/                           # Leftover skeleton directory (empty, no source files)
│   └── models/                     # Empty — no files present
│
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── conftest.py                 # Shared pytest fixtures
│   ├── _path_setup.py              # Adds src/ to sys.path for imports
│   ├── test_batching.py            # Tests for parallel subagent dispatching
│   ├── test_coverage.py            # Tests for CoverageAnalyzer
│   ├── test_evaluation.py          # Tests for ModelMatrixEvaluator
│   ├── test_memory.py              # Tests for MemoryManager
│   ├── test_redaction.py           # Tests for SecretRedactor
│   ├── test_reporting.py           # Tests for ReportWriter
│   ├── test_tools.py               # Tests for SafeToolset
│   ├── test_workflow.py            # Integration tests for DaddySubagentsReflectiveWorkflow
│   └── test_workspace.py           # Tests for WorkspaceManager
│   └── fixtures/
│       └── repos/
│           ├── simple-service/     # Single-module Maven fixture repo
│           └── multi-module-root/  # Multi-module Maven fixture repo
│
├── docs/
│   ├── architecture.md             # Narrative architecture documentation
│   ├── agentic-testgen-plan.md     # Original planning document
│   └── todo.md                     # Outstanding work items
│
├── examples/                       # Usage examples (minimal, currently sparse)
│
├── mlartifacts/                    # Local MLflow artifact store (git-ignored data)
│
├── .planning/                      # GSD planning documents
│   └── codebase/                   # Codebase analysis docs written by map-codebase
│
├── .venv/                          # Virtual environment (uv-managed, not committed)
└── .pytest_cache/                  # pytest cache (not committed)
```

## Directory Purposes

**`src/agentic_testgen/`:**
- Purpose: The entire application codebase as an installable Python package
- Contains: All business logic modules (see layout above)
- Key files: `agents.py` (orchestrator), `models.py` (data contracts), `cli.py` (entry point)

**`tests/`:**
- Purpose: pytest test suite, co-located at project root (not inside `src/`)
- Contains: One test file per source module, plus shared fixtures
- Key files: `test_workflow.py` (end-to-end integration), `conftest.py` (shared fixtures)

**`tests/fixtures/repos/`:**
- Purpose: Minimal Maven project trees used as test inputs
- Contains: `simple-service/` (single pom.xml) and `multi-module-root/` (multi-module Maven)
- Generated: No — committed as static test data
- Committed: Yes

**`docs/`:**
- Purpose: Human-authored documentation and planning artifacts
- Contains: Architecture narrative, planning notes
- Key files: `docs/architecture.md`

**`core/`:**
- Purpose: Originally planned as a separate module; currently empty skeleton with no source files
- Contains: Empty `models/` subdirectory only
- Note: Not imported anywhere; safe to ignore

**`mlartifacts/`:**
- Purpose: Local MLflow artifact storage when running against local tracking server
- Generated: Yes (by MLflow at runtime)
- Committed: No

## Key File Locations

**Entry Points:**
- `main.py`: Top-level script entry; calls `cli.main()`
- `src/agentic_testgen/__main__.py`: Module entry; same delegation
- `src/agentic_testgen/cli.py`: All CLI commands and their implementations

**Configuration:**
- `pyproject.toml`: Package name, version, dependencies, `testgen` script entry point
- `.env.example`: Documents all required and optional environment variables
- `src/agentic_testgen/config.py`: `AppConfig.load()` — reads env vars via `python-dotenv`

**Core Logic:**
- `src/agentic_testgen/agents.py`: `DaddySubagentsReflectiveWorkflow` and `DSPyRuntime` — the heart of the system
- `src/agentic_testgen/tools.py`: `SafeToolset` — all file/Maven operations available to LLM agents
- `src/agentic_testgen/coverage.py`: `CoverageAnalyzer` — Maven test execution and JaCoCo parsing
- `src/agentic_testgen/memory.py`: `MemoryManager` — cross-run lesson persistence

**Data Contracts:**
- `src/agentic_testgen/models.py`: All dataclasses used across layers; the single source of truth for data shapes

**Testing:**
- `tests/conftest.py`: Shared fixtures
- `tests/fixtures/repos/`: Static Maven project fixtures for integration tests

**Runtime Workspace (generated at runtime, not in repo):**
- `{WORKSPACE_ROOT}/runs/{run_id}/clone/` — cloned/copied repo
- `{WORKSPACE_ROOT}/runs/{run_id}/worktrees/{subagent_id}/` — git worktrees per subagent
- `{WORKSPACE_ROOT}/runs/{run_id}/artifacts/` — reports, coverage context, workbook
- `{WORKSPACE_ROOT}/runs/{run_id}/logs/` — `run.log`, `events.jsonl`, `dspy_traces.jsonl`
- `{WORKSPACE_ROOT}/runs/{run_id}/checkpoints/latest.json` — resumable state
- `{WORKSPACE_ROOT}/runs/{run_id}/control/pause.requested` — pause signal flag
- `{WORKSPACE_ROOT}/global_memory.json` — cross-run global lessons

## Naming Conventions

**Files:**
- All lowercase with underscores (snake_case): `agents.py`, `checkpointing.py`, `coverage.py`
- Test files prefixed with `test_`: `test_workflow.py`, `test_coverage.py`

**Directories:**
- Lowercase with underscores: `agentic_testgen/`, `worktrees/`, `artifacts/`
- Fixture repos use hyphenated names: `simple-service/`, `multi-module-root/`

**Classes:**
- PascalCase: `DaddySubagentsReflectiveWorkflow`, `SafeToolset`, `RunLogger`, `CoverageAnalyzer`
- Dataclasses follow same PascalCase: `RunCheckpoint`, `FileWorkItem`, `SubagentResult`

**Functions/Methods:**
- snake_case: `run_from_gitlab()`, `build_work_items()`, `initialize_run_memory()`
- Private helpers prefixed with `_`: `_dispatch_subagents()`, `_run_subagent()`, `_make_repo_context()`

**Environment Variables:**
- SCREAMING_SNAKE_CASE: `REPO_URL`, `GITLAB_TOKEN`, `MODEL_API_KEY`, `MAX_PARALLEL_SUBAGENTS`

## Where to Add New Code

**New CLI command:**
- Add `@app.command()` function in `src/agentic_testgen/cli.py`
- Keep CLI thin: parse args, call workflow/utility methods, print output

**New workflow phase:**
- Add method to `DaddySubagentsReflectiveWorkflow` in `src/agentic_testgen/agents.py`
- Call `_save_checkpoint()` at the end of the new phase
- Add a new `phase` string value (e.g., `"my_phase_completed"`) as a checkpoint marker

**New tool for LLM subagents:**
- Add a method to `SafeToolset` in `src/agentic_testgen/tools.py`
- Validate all paths via `_resolve_active_path()` before any I/O
- Register the new tool in `build_dspy_tools()` at the bottom of the class

**New data model:**
- Add a dataclass to `src/agentic_testgen/models.py`
- Include `to_json()` method following the existing pattern using `_jsonify()`

**New configuration option:**
- Add field to `AppConfig` (or `ModelSettings` / `MlflowSettings`) in `src/agentic_testgen/config.py`
- Read the env var in `AppConfig.load()` with a safe default

**New report type:**
- Add a `write_*` method to `ReportWriter` in `src/agentic_testgen/reporting.py`
- Call it from `_write_reports()` in `agents.py` and log artifact via `MlflowTracer`

**New utility function:**
- Add to `src/agentic_testgen/utils.py` if it has no module-specific imports
- Otherwise add as a private method in the relevant module

**Tests for new code:**
- Create `tests/test_{module_name}.py` or add to the existing test file for that module
- Place Maven fixture repos in `tests/fixtures/repos/` if needed for integration tests
- Use `tests/conftest.py` for shared fixtures

## Special Directories

**`.planning/codebase/`:**
- Purpose: GSD codebase analysis documents (this file and siblings)
- Generated: Yes (by `/gsd:map-codebase`)
- Committed: Yes

**`mlartifacts/`:**
- Purpose: MLflow artifact store for local experiment tracking
- Generated: Yes (at runtime by MLflow)
- Committed: No (add to `.gitignore` if not already)

**`.venv/`:**
- Purpose: uv-managed virtual environment
- Generated: Yes (`uv sync`)
- Committed: No

**`tests/fixtures/repos/`:**
- Purpose: Static Maven project trees used as deterministic test inputs
- Generated: No (manually authored)
- Committed: Yes

---

*Structure analysis: 2026-04-21*
