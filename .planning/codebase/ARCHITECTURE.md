# Architecture

**Analysis Date:** 2026-04-21

## Pattern Overview

**Overall:** Orchestrator + Parallel Subagent Workflow with DSPy-driven LLM calls

**Key Characteristics:**
- Single orchestrator (`DaddySubagentsReflectiveWorkflow`) coordinates all phases
- Parallel subagents run concurrently via `ThreadPoolExecutor`, each isolated in a git worktree
- Each subagent iterates using DSPy `ReAct` (reason + act) loops to generate and validate tests
- After each subagent iteration, a `reflect` call summarizes results and informs retries
- All long-running runs are checkpointed to disk so they can be paused and resumed
- MLflow tracing records every LLM call and workflow metric

## Layers

**CLI Layer:**
- Purpose: User-facing commands (run, status, pause, resume, review, integrate, eval)
- Location: `src/agentic_testgen/cli.py`
- Contains: Typer app with one command per operation
- Depends on: `DaddySubagentsReflectiveWorkflow`, `AppConfig`, `CheckpointStore`, `WorkspaceManager`
- Used by: End users via `testgen` CLI entry point

**Workflow Orchestration Layer:**
- Purpose: Top-level coordination of all workflow phases
- Location: `src/agentic_testgen/agents.py` — class `DaddySubagentsReflectiveWorkflow`
- Contains: `run_from_gitlab()`, `run_from_local_path()`, `resume()`, `_execute()`, `_dispatch_subagents()`, `_run_subagent()`
- Depends on: All lower layers
- Used by: CLI layer, evaluation harness

**DSPy Runtime Layer:**
- Purpose: Wraps DSPy LLM configuration and provides typed LLM call helpers
- Location: `src/agentic_testgen/agents.py` — class `DSPyRuntime`
- Contains: `overview()`, `reflect()`, `analyze_failure()`
- Depends on: `dspy`, `AppConfig`, `RunLogger`
- Used by: `DaddySubagentsReflectiveWorkflow`, `_run_subagent()`

**Tools Layer:**
- Purpose: Safe sandboxed toolset exposed to DSPy ReAct loops (filesystem + Maven operations)
- Location: `src/agentic_testgen/tools.py` — class `SafeToolset`, `ToolContext`
- Contains: `read_file()`, `search_file()`, `write_file()`, `run_single_test()`, `create_worktree()`, `build_dspy_tools()`
- Depends on: `CoverageAnalyzer`, `RunLogger`, `AppConfig`
- Used by: `_run_subagent()` inside each subagent loop

**Coverage Layer:**
- Purpose: Run Maven+JaCoCo tests, parse XML reports, build prioritized work item lists
- Location: `src/agentic_testgen/coverage.py` — class `CoverageAnalyzer`
- Contains: `discover_modules()`, `run_tests_with_coverage()`, `collect_reports()`, `parse_jacoco_xml()`, `build_work_items()`
- Depends on: `AppConfig`, `utils.run_command`
- Used by: `DaddySubagentsReflectiveWorkflow`, `SafeToolset`

**Memory Layer:**
- Purpose: Persist lessons learned across subagent runs, both per-run and globally across repos
- Location: `src/agentic_testgen/memory.py` — class `MemoryManager`
- Contains: `initialize_run_memory()`, `record_result()`, `lessons_for_item()`
- Depends on: `models`, `utils`
- Used by: `DaddySubagentsReflectiveWorkflow` to seed subagent objectives with prior lessons

**Checkpointing Layer:**
- Purpose: Serialize and restore `RunCheckpoint` state to disk as `latest.json`
- Location: `src/agentic_testgen/checkpointing.py` — class `CheckpointStore`
- Contains: `save()`, `load()`
- Depends on: `models`, `utils`
- Used by: `DaddySubagentsReflectiveWorkflow` after every phase transition and after each subagent completes

**Workspace Layer:**
- Purpose: Manage per-run directory layout on disk
- Location: `src/agentic_testgen/workspace.py` — classes `WorkspaceManager`, `RunWorkspace`
- Contains: `create()`, `copy_local_repo()`
- Depends on: `utils.ensure_dir`
- Used by: `DaddySubagentsReflectiveWorkflow`, CLI

**Reporting Layer:**
- Purpose: Write Markdown, JSON, CSV, and Excel workbook artifacts
- Location: `src/agentic_testgen/reporting.py` — class `ReportWriter`
- Contains: `write_overview()`, `write_json_summary()`, `write_workbook()`, `write_coverage_comparison()`
- Depends on: `models`, `utils`
- Used by: `DaddySubagentsReflectiveWorkflow`

**Tracing Layer:**
- Purpose: Wrap MLflow for experiment tracking, metric/artifact logging, and token budget recording
- Location: `src/agentic_testgen/tracing.py` — class `MlflowTracer`
- Contains: `validate()`, `configure()`, `run()`, `log_params()`, `log_metrics()`, `log_artifact()`, `tag_last_trace()`, `token_usage_summary()`
- Depends on: `mlflow` (optional), `RunLogger`
- Used by: `DaddySubagentsReflectiveWorkflow`

**Logging Layer:**
- Purpose: Structured JSON event log + human-readable run log with secret redaction
- Location: `src/agentic_testgen/logging.py` — classes `RunLogger`, `SecretRedactor`
- Contains: `log_event()`, `step()` (context manager), `log_trace()`
- Depends on: `models.LogEvent`, `utils`
- Used by: every other layer

**Models Layer:**
- Purpose: Pure data classes (no business logic) shared across all layers
- Location: `src/agentic_testgen/models.py`
- Contains: `RepoContext`, `FileWorkItem`, `SubagentResult`, `AttemptRecord`, `RunCheckpoint`, `IntegrationDecision`, `WorkflowRunResult`, `ModelEvalResult`, etc.
- Depends on: Python stdlib only
- Used by: all layers

**Config Layer:**
- Purpose: Load and expose typed application configuration from environment variables
- Location: `src/agentic_testgen/config.py` — dataclasses `AppConfig`, `ModelSettings`, `MlflowSettings`
- Contains: `AppConfig.load()`, `maven_executable()`, `java_executable()`, `maven_command()`
- Depends on: `python-dotenv`, stdlib
- Used by: all layers that need configuration

**GitLab Integration Layer:**
- Purpose: Clone repositories from GitLab with token authentication
- Location: `src/agentic_testgen/gitlab.py` — class `GitLabRepositoryManager`
- Contains: `clone()`, `authenticated_repo_url()`, `sanitize_repo_url()`
- Depends on: `AppConfig`, `RunLogger`, `utils.run_command`
- Used by: `DaddySubagentsReflectiveWorkflow.run_from_gitlab()`

**Evaluation Layer:**
- Purpose: Run a model comparison matrix over fixture repos to benchmark LLMs
- Location: `src/agentic_testgen/evaluation.py` — classes `ModelMatrixEvaluator`, `EvaluationConfig`
- Contains: `run()`, `EvaluationConfig.load()`
- Depends on: `DaddySubagentsReflectiveWorkflow`, `ReportWriter`, `models`
- Used by: `cli.evaluate` command

## Data Flow

**Primary Workflow (run command):**

1. CLI parses args and calls `DaddySubagentsReflectiveWorkflow.run_from_gitlab()` or `run_from_local_path()`
2. Repository is cloned or copied into `{workspace_root}/runs/{run_id}/clone/`
3. `CoverageAnalyzer.run_tests_with_coverage()` runs Maven + JaCoCo against the cloned repo
4. JaCoCo XML reports are parsed into `CoverageRecord` objects; `build_work_items()` produces a priority-ranked `FileWorkItem` list
5. `_dispatch_subagents()` submits each `FileWorkItem` to a `ThreadPoolExecutor` (up to `max_parallel_subagents` concurrently)
6. Each subagent (`_run_subagent()`) creates a git worktree in `worktrees/{subagent_id}/`, then runs up to `max_subagent_iterations` DSPy ReAct loops
7. Within each ReAct loop, the LLM calls tools from `SafeToolset` (read files, write test files, run Maven single test)
8. After each iteration, `DSPyRuntime.reflect()` summarizes; on failure, `analyze_failure()` provides a concise root cause
9. If a test passes, the generated file is committed to the worktree branch
10. After all subagents finish, `ReportWriter` writes Markdown/JSON/Excel artifacts to `artifacts/`
11. `CheckpointStore` persists final state; `MlflowTracer` logs metrics and artifacts

**Pause/Resume Flow:**
1. CLI `pause` command writes `control/pause.requested` flag file
2. `_dispatch_subagents()` checks this flag before dispatching new subagents; current in-flight agents finish normally
3. `CheckpointStore.save()` captures pending work items and completed results
4. CLI `resume` command loads checkpoint and calls `_dispatch_subagents()` with remaining work items

**Integration Flow:**
1. CLI `review` command reads `pending_integrations.json` to list candidates
2. CLI `integrate` command cherry-picks each approved commit hash into the cloned repo
3. After integration, `rerun_after_merge_coverage()` re-runs Maven to measure the coverage improvement

**State Management:**
- No in-memory global state; all state is serialized to JSON files in the run workspace
- `RunCheckpoint` in `checkpoints/latest.json` is the source of truth for resumable state
- `MemoryManager` maintains `global_memory.json` at the workspace root and per-run `memory.json` to persist lessons

## Key Abstractions

**`DaddySubagentsReflectiveWorkflow`:**
- Purpose: Top-level orchestrator for the entire test generation workflow
- File: `src/agentic_testgen/agents.py`
- Pattern: Facade that sequences phases (clone → coverage → subagents → report → integrate)

**`SafeToolset` + `ToolContext`:**
- Purpose: Sandboxed tool interface passed to DSPy ReAct; enforces path restrictions
- File: `src/agentic_testgen/tools.py`
- Pattern: Each tool method validates paths against `active_root` before any I/O; builds DSPy-compatible tool descriptors via `build_dspy_tools()`

**`DSPyRuntime`:**
- Purpose: Thin adapter between the workflow and DSPy LLM programs
- File: `src/agentic_testgen/agents.py`
- Pattern: Wraps `dspy.ChainOfThought`, `dspy.Predict`, and `dspy.ReAct` with graceful fallbacks when DSPy is unconfigured

**`RunWorkspace` + `WorkspaceManager`:**
- Purpose: Typed directory structure for a single run
- File: `src/agentic_testgen/workspace.py`
- Pattern: Factory creates all subdirs on first access; paths are `Path` objects exposed as dataclass fields

**`CheckpointStore`:**
- Purpose: Serialize/deserialize `RunCheckpoint` to `checkpoints/latest.json`
- File: `src/agentic_testgen/checkpointing.py`
- Pattern: JSON round-trip; load deserializes nested model objects by hand

## Entry Points

**CLI Entry Point:**
- Location: `src/agentic_testgen/cli.py` — `main()` function, registered as `testgen` in `pyproject.toml`
- Triggers: Shell command `testgen <subcommand>` or `python main.py`
- Responsibilities: Parse arguments, load `AppConfig`, delegate to workflow or utility classes

**Python Module Entry Point:**
- Location: `src/agentic_testgen/__main__.py` → `main.py`
- Triggers: `python -m agentic_testgen` or `python main.py`
- Responsibilities: Delegates immediately to `cli.main()`

**Evaluation Entry Point:**
- Location: `src/agentic_testgen/evaluation.py` — `ModelMatrixEvaluator.run()`
- Triggers: `testgen eval <config_path>`
- Responsibilities: Load TOML evaluation config, iterate model × fixture combinations, produce `ModelEvalResult` list

## Error Handling

**Strategy:** Exceptions propagate upward with contextual logging at each layer boundary; non-fatal failures are caught and recorded as `status="failed"` on the relevant model object

**Patterns:**
- `run_command()` in `utils.py` never raises; returns `CommandResult` with `ok: bool` and captures `FileNotFoundError` (exit code 127)
- DSPy calls in `DSPyRuntime` are wrapped in `try/except Exception` — on failure the method returns a safe default string and logs via `RunLogger`
- `MlflowTracer` treats MLflow unavailability as non-fatal unless `strict=True` in config
- Subagent failures set `SubagentResult.status = "failed"` and record `error_message`; they do not abort other running subagents
- Secret values are stripped from all log output by `SecretRedactor` in `logging.py`

## Cross-Cutting Concerns

**Logging:** `RunLogger` (thread-safe via `Lock`) writes structured JSONL to `logs/events.jsonl` and human-readable text to `logs/run.log`; every significant operation uses the `step()` context manager which records start/end timestamps
**Validation:** `AppConfig.validate_for_run()` checks required env vars before network operations; `SafeToolset` validates all file paths against the worktree root
**Authentication:** GitLab token injected into repo URL in `gitlab.py`; model API keys read from env vars named in `ModelDefinition.api_key_env`
**Concurrency:** `ThreadPoolExecutor` with `wait(FIRST_COMPLETED)` loop in `_dispatch_subagents()`; `RunLogger` uses a `threading.Lock` for safe concurrent writes

---

*Architecture analysis: 2026-04-21*
