# External Integrations

**Analysis Date:** 2026-04-21

## APIs & External Services

**LLM Providers (via DSPy):**
- Any OpenAI-compatible API - Primary supported target; model configured via `MODEL_PROVIDER` + `MODEL_NAME` (e.g. `openai/gpt-4o-mini`)
  - SDK/Client: `dspy.LM(f"{provider}/{model_name}", api_key=..., api_base=...)`
  - Auth: `MODEL_API_KEY` env var
  - Base URL override: `MODEL_API_BASE` env var
- Groq - Explicitly listed as a direct dependency (`groq>=1.1.2`); supported by setting `MODEL_PROVIDER=groq`
  - SDK/Client: `groq` 1.1.2 (via DSPy provider routing)
  - Auth: `MODEL_API_KEY` env var
- Any LiteLLM-compatible endpoint - DSPy uses LiteLLM under the hood; any provider string DSPy/LiteLLM supports works (Anthropic, Azure, Ollama, etc.)
  - Auth: Env var name defined per `ModelDefinition.api_key_env` in evaluation mode

**Source Control:**
- GitLab (self-hosted or cloud) - Clone private Maven repos for test generation
  - Client: Raw `git` CLI subprocess (`git clone`, `git worktree`, `git cherry-pick`, `git commit`)
  - Auth: `GITLAB_TOKEN` (OAuth token or PAT), `GITLAB_USERNAME` (default: `oauth2`)
  - Base URL: `GITLAB_BASE_URL` env var
  - Implementation: `src/agentic_testgen/gitlab.py` → `GitLabRepositoryManager.clone()`
  - Note: SSL verification disabled during clone (`http.sslVerify=false`) to support self-signed certificates

## Data Storage

**Databases:**
- None - No database is used. All state is stored as JSON files on the local filesystem.

**File Storage:**
- Local filesystem only
  - Run artifacts: `{WORKSPACE_ROOT}/{run_id}/artifacts/` - Markdown reports, JSON summaries, coverage context, token budget
  - Checkpoints: `{WORKSPACE_ROOT}/{run_id}/checkpoints/` - Serialized `RunCheckpoint` JSON for resume support
  - Logs: `{WORKSPACE_ROOT}/{run_id}/logs/` - `run.log`, `events.jsonl`, `dspy_traces.jsonl`, Maven logs
  - Git worktrees: `{WORKSPACE_ROOT}/{run_id}/worktrees/` - Isolated working trees per subagent
  - Cloned repos: `{WORKSPACE_ROOT}/{run_id}/clone/` - Git clone of target repo
  - Global memory: `{WORKSPACE_ROOT}/global_memory.json` - Persistent cross-run lessons per repo
  - Integration decisions: `{WORKSPACE_ROOT}/{run_id}/integrations.json`
  - MLflow artifacts: Uploaded to MLflow tracking server (local or remote)

**Caching:**
- None at application layer; DSPy may cache LLM calls internally via `diskcache` (present as transitive dependency)

## Authentication & Identity

**Auth Provider:**
- No user auth system - This is a CLI tool, not a web service
- GitLab token-based auth for repo cloning (injected into git URL as `username:token@host`)
  - Implementation: `src/agentic_testgen/gitlab.py` → `authenticated_repo_url()`

## Monitoring & Observability

**Experiment Tracking / Tracing:**
- MLflow 3.11.1 - Full run tracking including params, metrics, artifacts, and DSPy traces
  - Server URL: `MLFLOW_TRACKING_URI` (default: `http://127.0.0.1:5000`)
  - Experiment: `MLFLOW_EXPERIMENT_NAME` (default: `agentic-testgen`)
  - Enable/disable: `ENABLE_MLFLOW_TRACING` (default: `true`)
  - Strict mode: `STRICT_MLFLOW_TRACING` - if `true`, raises if server unreachable
  - Implementation: `src/agentic_testgen/tracing.py` → `MlflowTracer`
  - DSPy autolog: `mlflow.dspy.autolog()` called automatically if available, captures all LLM call traces
  - Token usage tracked per DSPy trace and summarized into `token-budget.json` artifact

**Logs:**
- Structured JSON events: `{run_id}/logs/events.jsonl` - machine-readable event stream
- Human-readable log: `{run_id}/logs/run.log` - plain text
- DSPy traces: `{run_id}/logs/dspy_traces.jsonl` - trajectory dumps from ReAct loops
- Maven build logs: `{run_id}/logs/maven/` - per-subagent stdout/stderr from Maven commands
- Implementation: `src/agentic_testgen/logging.py` → `RunLogger`

**Error Tracking:**
- None - Errors are caught, logged to run log and MLflow events, and surfaced in checkpoint metadata. No external error tracking service.

## CI/CD & Deployment

**Hosting:**
- Not applicable - This is a local CLI tool (`testgen` command)

**CI Pipeline:**
- None detected in the repository

## Build Tool Integration

**Apache Maven:**
- The target repositories being analyzed are Maven projects
- Maven is invoked via subprocess for running tests with JaCoCo coverage: `mvn -q -DskipTests=false test jacoco:report`
- Single test validation: `mvn -q -Dtest={ClassName} test`
- Executable resolved from: `MVN_BIN` env var → `MAVEN_HOME/bin/mvn` → system `mvn` on PATH
- Settings file: `MAVEN_SETTINGS_XML` env var (passed as `-s` flag)
- JaCoCo XML reports parsed from `target/site/jacoco/jacoco.xml` in each Maven module
- Implementation: `src/agentic_testgen/coverage.py` → `CoverageAnalyzer`

## Webhooks & Callbacks

**Incoming:**
- None - No web server or webhook listener

**Outgoing:**
- None - Only outbound calls are to LLM APIs (via DSPy/LiteLLM) and to MLflow tracking server

## Environment Configuration

**Required env vars for a GitLab run:**
- `GITLAB_TOKEN` - Mandatory (validated in `AppConfig.validate_for_run()`)
- `MODEL_NAME` - Required for LLM to function; without it DSPy is disabled and agents produce no-op stubs
- `MODEL_API_KEY` or `MODEL_API_BASE` - Required alongside `MODEL_NAME` for DSPy to be enabled

**Optional but important:**
- `GITLAB_BASE_URL` - Needed for GitLab clone
- `MLFLOW_TRACKING_URI` - Defaults to `http://127.0.0.1:5000`
- `JAVA_HOME`, `MAVEN_HOME` / `MVN_BIN` - Needed if Java/Maven not on system PATH
- `WORKSPACE_ROOT` - Defaults to OS-appropriate user directory

**Secrets location:**
- `.env` file at project root (never committed; `.env.example` committed as template)
- Model API keys also supported via per-model env var names in evaluation matrix TOML (`ModelDefinition.api_key_env`)

---

*Integration audit: 2026-04-21*
