# Technology Stack

**Analysis Date:** 2026-04-21

## Languages

**Primary:**
- Python 3.13+ - All application logic, agents, CLI, tests

**Target Language (Generated Output):**
- Java - The system generates JUnit test files for Maven-based Java repositories

## Runtime

**Environment:**
- CPython 3.13 (required minimum per `pyproject.toml`: `requires-python = ">=3.13"`)

**Package Manager:**
- `uv` - Lockfile present at `uv.lock`
- Lockfile: Present (`uv.lock`)

**Virtual Environment:**
- `.venv/` at project root

## Frameworks

**Core AI / LLM Orchestration:**
- `dspy` 3.1.3 - Primary LLM programming framework; drives all agent logic, `dspy.ReAct`, `dspy.ChainOfThought`, `dspy.Predict`, and tool-calling via `dspy.Tool`
- `dspy-ai` 3.1.3 - Alias/companion package to `dspy`

**CLI:**
- `typer` 0.12.0 (installed: 0.24.1) - CLI entry point exposed as `testgen` script defined in `pyproject.toml`

**Experiment Tracking:**
- `mlflow` 3.11.1 - Run tracking, artifact logging, metric recording, DSPy autolog integration; wrapped by `MlflowTracer` in `src/agentic_testgen/tracing.py`

**Data Validation:**
- `pydantic` 2.0.0+ (installed: 2.12.5) - Used transitively; all application models use plain Python `dataclasses` (not Pydantic models)

**Configuration:**
- `python-dotenv` 1.0.0 - Loads `.env` file via `dotenv.load_dotenv()` in `src/agentic_testgen/config.py`

**Build:**
- `setuptools` 68+ - Build backend declared in `pyproject.toml`

## Key Dependencies

**Critical:**
- `dspy` 3.1.3 - The entire agent workflow (ReAct loops, chain-of-thought, reflection, failure analysis) is DSPy-driven. Without it, agents degrade to no-op stubs.
- `mlflow` 3.11.1 - Experiment tracking for all workflow runs, DSPy trace capture, artifact storage, and token budget reporting.
- `groq` 1.1.2 - Groq LLM API client; listed as a direct dependency suggesting Groq models are a supported provider alongside OpenAI-compatible endpoints.
- `pip-system-certs` 5.0 - Injects system CA certificates into pip/requests; needed for corporate/self-signed SSL environments (relevant to the GitLab clone flow which uses `http.sslVerify=false`).

**Infrastructure:**
- Standard library only for file I/O, subprocess, concurrency (`concurrent.futures.ThreadPoolExecutor`), and XML parsing (`xml.etree.ElementTree`)

**Notable Transitive Dependencies (from lockfile):**
- `fastapi` 0.135.3 - Pulled in transitively (likely via mlflow server); not used directly in application code
- `alembic` 1.18.4 - Pulled in transitively (likely via mlflow)
- `databricks-sdk` 0.102.0 - Pulled in transitively via `dspy`

## Configuration

**Environment:**
- All configuration loaded from `.env` file via `src/agentic_testgen/config.py` → `AppConfig.load()`
- Key variables (see `.env.example`):
  - `GITLAB_BASE_URL`, `GITLAB_TOKEN`, `GITLAB_USERNAME` - GitLab repo access
  - `MODEL_PROVIDER`, `MODEL_NAME`, `MODEL_API_KEY`, `MODEL_API_BASE`, `TEMPERATURE` - LLM selection
  - `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `ENABLE_MLFLOW_TRACING` - MLflow
  - `JAVA_HOME`, `MAVEN_HOME`, `MVN_BIN`, `MAVEN_SETTINGS_XML` - Java/Maven toolchain
  - `MAX_FILES_PER_RUN`, `MAX_PARALLEL_SUBAGENTS`, `MAX_SUBAGENT_ITERATIONS` - Concurrency tuning
  - `WORKSPACE_ROOT` - Where run artifacts and worktrees are written
  - `AUTO_INTEGRATE_SUCCESSFUL_WORKTREES` - Auto-merge toggle

**Build:**
- `pyproject.toml` at project root
- Package source: `src/` layout (`package-dir = {"" = "src"}`)
- Installed CLI entry point: `testgen = "agentic_testgen.cli:main"`

## Platform Requirements

**Development:**
- Python 3.13+
- Java JDK (configurable via `JAVA_HOME`)
- Apache Maven (configurable via `MAVEN_HOME` / `MVN_BIN`)
- `git` CLI on PATH (used directly via subprocess for clone, worktree, cherry-pick operations)
- Optional: `rg` (ripgrep) on PATH for the `search_occurrences` tool (falls back to `grep`)
- Optional: MLflow server running at `MLFLOW_TRACKING_URI` (defaults to `http://127.0.0.1:5000`)

**Production:**
- Same requirements as development; no containerization or cloud-specific deployment config detected in the repository

---

*Stack analysis: 2026-04-21*
