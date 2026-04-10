# Plan: DSPy-Based Test Generation Platform for GitLab Maven Repos

## Summary

- Build a Python 3.11+ `uv` CLI application that clones a custom GitLab Maven repo using token-based HTTPS auth, analyzes the repo, generates new unit tests in isolated Git worktrees, and reports results.
- Implement one production workflow now: `daddy_subagents_reflective`.
- Support macOS and Windows through config-driven command resolution for Java and Maven.
- Keep run artifacts outside the cloned repo, except generated test files created in subagent worktrees.
- Configure MLflow tracing at `http://127.0.0.1:5000` for local observability of DSPy runs and evaluation experiments.

## Core Decisions

- Git input is a custom self-hosted GitLab HTTPS URL with token-based access.
- Cloud-hosted LLMs are used for now.
- Maven plus JaCoCo XML is the only supported execution path in v1.
- Generated tests are unit tests only.
- Subagents may create only new test files.
- Review queue is the default integration mode.
- Auto-integration is opt-in.
- The evaluation framework compares models only, not systems.

## Workflow

1. `testgen run --repo-url ...` creates a managed run workspace and clones the target repo.
2. The Daddy agent validates the environment, analyzes repo structure, runs full-project tests and coverage, and writes `overview.md`.
3. The Daddy agent ranks target files by missed-line coverage and spawns one subagent per file in bounded parallelism.
4. Each subagent works inside a dedicated Git worktree, may read/search code, writes only new test files, runs validation, and reflects up to the configured iteration cap.
5. Passing worktree changes are either queued for review or auto-integrated if enabled.
6. The evaluation harness reuses the same workflow across synthetic Maven fixtures and multiple configured models.

## Observability

- Persist `logs/events.jsonl` for structured event logs.
- Persist `logs/run.log` for human-readable logs.
- Persist `logs/dspy_traces.jsonl` for DSPy prompt/tool trajectories.
- Configure MLflow with `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`.
- Redact GitLab tokens, API keys, auth headers, and secret-like values from logs, checkpoints, reports, and MLflow metadata.

## Reporting

- Write `overview.md` for repo analysis.
- Write an XLSX workbook with `runs`, `files`, `attempts`, and `model_eval` sheets.
- Write JSON summaries for checkpoints and evaluation outputs.

## Evaluation

- Use only the `daddy_subagents_reflective` workflow.
- Benchmark on synthetic Maven fixture repos.
- Compare `(model, fixture_repo, target_file)` cases.
- Report compile success, pass rate, coverage delta, missed-line reduction, forbidden-edit rate, flaky rate, latency, tool-call count, iteration count, and estimated token/cost usage.
