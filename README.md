# Agentic Test Generation Platform

DSPy-based tooling for cloning GitLab-hosted Maven repositories, analyzing line coverage, generating new tests in isolated Git worktrees, and comparing LLM model behavior on synthetic benchmark repos.

## Highlights

- `uv`/Python 3.11+ project with a Typer CLI
- one production workflow: `daddy_subagents_reflective`
- GitLab token auth with secret redaction
- JaCoCo XML coverage parsing and worklist generation
- structured JSONL logs plus human-readable logs
- optional MLflow tracing at `http://127.0.0.1:5000`
- model-only evaluation harness over synthetic Maven fixtures

## Prerequisites

- Python 3.11+
- `uv`
- Git
- Java and Maven installed locally
- access token for your GitLab instance
- optional MLflow server at `127.0.0.1:5000`

## Setup

1. Sync the project environment:

```bash
uv sync
```

2. Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

3. Set the required variables in `.env`:

- `GITLAB_BASE_URL`
- `GITLAB_TOKEN`
- `GITLAB_USERNAME`
- `JAVA_HOME`
- `MAVEN_HOME`
- optional `MVN_BIN`
- optional `MAVEN_SETTINGS_XML`
- optional `MAX_FILES_PER_RUN`
- optional `REPO_URL` (used when `--repo-url` is omitted)
- optional `REPO_PATH` (used when `--repo-path` is omitted)
- `MODEL_PROVIDER`
- `MODEL_NAME`
- optional `TEMPERATURE`
- optional `TOP_P`
- optional `MAX_TOKENS`
- `MODEL_API_KEY`
- optional `MODEL_API_BASE`
- optional `WORKSPACE_ROOT`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME`
- `ENABLE_MLFLOW_TRACING`

## Run The Agent On A Git Repo

For local repos (no clone), copy the local project into the run sandbox and execute there:

```bash
uv run testgen run --repo-path /absolute/path/to/local-maven-repo
```

Local repo copy excludes build artifacts by default (for example `target/`, `build/`, `out/`).

You can also set `REPO_PATH` in `.env` and run:

```bash
uv run testgen run
```

The agent clones the target repo internally using:

```bash
git -c http.sslVerify=false clone <repo-url>
```

That means HTTPS certificate verification is disabled for the clone step, which is useful for internal GitLab instances with self-signed certs.

The clone also enables `core.longpaths=true` to reduce Windows path-length issues.

Run the workflow with `uv`:

```bash
uv run python -m agentic_testgen run --repo-url https://gitlab.example.com/group/project.git
```

For a controlled trial run, the workflow limits itself to the top 5 files by default. You can override that from the CLI:

```bash
uv run testgen run --repo-url https://gitlab.example.com/group/project.git --max-files 5
```

You can also use the generated script entrypoint:

```bash
uv run testgen run --repo-url https://gitlab.example.com/group/project.git
```

Or run without `--repo-url` if `REPO_URL` is set in `.env`:

```bash
uv run testgen run
```

The command prints:

- `run_id`
- `overview` artifact path
- `workbook` artifact path
- number of subagent results

## Monitor A Run

Check current status:

```bash
uv run testgen status --run-id <run_id>
```

Tail logs:

```bash
uv run testgen logs --run-id <run_id>
```

Pause a run:

```bash
uv run testgen pause --run-id <run_id>
```

Resume a run:

```bash
uv run testgen resume --run-id <run_id>
```

Review pending integrations:

```bash
uv run testgen review --run-id <run_id>
```

Integrate approved commits:

```bash
uv run testgen integrate --run-id <run_id>
```

## Run Evaluation

The evaluation harness compares models only and uses the same `daddy_subagents_reflective` workflow for every case.

```bash
uv run testgen eval --config examples/model_matrix.toml
```

## CLI Reference

```bash
uv run testgen run --repo-url https://gitlab.example.com/group/project.git
uv run testgen run --repo-path /absolute/path/to/local-maven-repo
uv run testgen status --run-id <run_id>
uv run testgen logs --run-id <run_id>
uv run testgen pause --run-id <run_id>
uv run testgen resume --run-id <run_id>
uv run testgen review --run-id <run_id>
uv run testgen integrate --run-id <run_id>
uv run testgen eval --config examples/model_matrix.toml
```

## Environment

Copy the values you need into `.env` or export them directly:

- `GITLAB_BASE_URL`
- `GITLAB_TOKEN`
- `JAVA_HOME`
- `MAVEN_HOME`
- `MVN_BIN`
- `MAVEN_SETTINGS_XML`
- `MAX_FILES_PER_RUN=5`
- `MODEL_PROVIDER`
- `MODEL_NAME`
- `TEMPERATURE`
- `TOP_P`
- `MAX_TOKENS`
- `MODEL_API_KEY`
- `MODEL_API_BASE`
- `MAX_PARALLEL_SUBAGENTS`
- `MAX_SUBAGENT_ITERATIONS`
- `MAX_REACT_ITERS_SUBAGENT`
- `MAX_REACT_ITERS_DADDY`
- `AUTO_INTEGRATE_SUCCESSFUL_WORKTREES`
- `WORKSPACE_ROOT`
- `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`
- `MLFLOW_EXPERIMENT_NAME=agentic-testgen`
- `ENABLE_MLFLOW_TRACING=true`

## Troubleshooting

If Git fails with `filename too long`, use a very short workspace root so the cloned path is shorter:

```bash
WORKSPACE_ROOT=/tmp/agt
```

On Windows, prefer something short like:

```bash
WORKSPACE_ROOT=C:\\agt
```

## Plan

The implementation plan is stored in [docs/agentic-testgen-plan.md](/Users/shubh/Repos/Github_Shubham/agentic-poc/docs/agentic-testgen-plan.md).

## Architecture

The high-level architecture map and visual workflow diagrams are stored in [docs/architecture.md](/Users/shubh/Repos/Github_Shubham/agentic-poc/docs/architecture.md).
