# HLAM and Architecture

This document captures the high-level architecture map for the DSPy-based test generation platform and shows how the main runtime components fit together.

## HLAM

HLAM here means High-Level Architecture Map:

- `User / Operator`
  - starts runs, pauses/resumes work, reviews integrations, and triggers evaluation
- `CLI Layer`
  - entrypoint for `run`, `status`, `logs`, `pause`, `resume`, `review`, `integrate`, and `eval`
- `DaddySubagentsReflectiveWorkflow`
  - central orchestrator for repo analysis, coverage collection, work-item ranking, subagent dispatch, checkpointing, and post-merge coverage refresh
- `Subagents`
  - one subagent per target source file, each isolated in its own Git worktree
- `Safe Tool Layer`
  - bounded file, Git, and Maven test operations exposed to DSPy
- `Run Workspace`
  - isolated run directories for clone, worktrees, artifacts, logs, and checkpoints
- `Observability Layer`
  - structured logs, human-readable logs, DSPy traces, and MLflow tracking
- `Reporting Layer`
  - overview, workbook, JSON summary, and coverage comparison outputs
- `Evaluation Layer`
  - model-only benchmark execution over synthetic Maven fixtures

## System Architecture

```mermaid
flowchart LR
    U["User / Operator"] --> CLI["Typer CLI"]
    CLI --> WF["DaddySubagentsReflectiveWorkflow"]

    WF --> CFG["AppConfig"]
    WF --> WS["WorkspaceManager / RunWorkspace"]
    WF --> GL["GitLabRepositoryManager"]
    WF --> COV["CoverageAnalyzer"]
    WF --> CK["CheckpointStore"]
    WF --> REP["ReportWriter"]
    WF --> TR["MlflowTracer + RunLogger"]

    WF --> DRT["DSPyRuntime"]
    DRT --> DAD["Daddy Agent (ReAct / CoT)"]
    WF --> SUB["File Subagents"]

    SUB --> TOOLS["SafeToolset"]
    TOOLS --> GIT["Git / Worktrees"]
    TOOLS --> MVN["Maven / JaCoCo"]
    TOOLS --> FS["Repo File Operations"]

    WS --> CLONE["clone/"]
    WS --> WTS["worktrees/"]
    WS --> ART["artifacts/"]
    WS --> LOGS["logs/"]
    WS --> CHK["checkpoints/"]

    REP --> ART
    TR --> LOGS
    CK --> CHK
```

## Workflow Architecture

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Daddy as Daddy Agent
    participant Coverage as CoverageAnalyzer
    participant Sub as File Subagent
    participant Git as Git Worktree
    participant Reports as ReportWriter

    User->>CLI: testgen run --repo-url ...
    CLI->>Daddy: start workflow
    Daddy->>Daddy: validate config and tracing
    Daddy->>Daddy: clone/copy repo into run workspace
    Daddy->>Coverage: detect modules and testing stack
    Daddy->>Reports: write overview.md
    Daddy->>Coverage: run full tests + JaCoCo
    Coverage-->>Daddy: ranked file work items

    loop per selected file
        Daddy->>Sub: spawn subagent with file + coverage + testing stack
        Sub->>Git: create isolated worktree
        Sub->>Sub: analyze source, tests, and missed lines
        Sub->>Sub: write only new test file
        Sub->>Sub: run single-test validation
        Sub->>Sub: reflect on failure/success
        Sub-->>Daddy: attempts + summary + commit/pending review
    end

    Daddy->>Reports: write workbook + summary.json
    alt integrations merged
        Daddy->>Coverage: rerun full repo coverage
        Daddy->>Reports: write coverage-comparison.md
    end
```

## Module View

```mermaid
flowchart TB
    CLI["cli.py"] --> AG["agents.py"]
    CLI --> EV["evaluation.py"]

    AG --> CFG["config.py"]
    AG --> COV["coverage.py"]
    AG --> TOOLS["tools.py"]
    AG --> GL["gitlab.py"]
    AG --> LOG["logging.py"]
    AG --> TRACE["tracing.py"]
    AG --> CK["checkpointing.py"]
    AG --> REP["reporting.py"]
    AG --> MODELS["models.py"]
    AG --> WS["workspace.py"]
    AG --> UT["utils.py"]

    EV --> AG
    EV --> REP
    EV --> MODELS
```

## Parent and Child Agent Responsibilities

- `Daddy agent`
  - understands repo structure
  - detects test framework and version
  - runs global coverage
  - ranks candidate files
  - spawns subagents
  - persists checkpoints and reports
  - reruns global coverage after merged changes
- `Child subagent`
  - owns exactly one file
  - receives repo-level testing-stack context
  - works inside one worktree
  - creates only new test files
  - validates through Maven single-test execution
  - returns attempts, reflections, and a final summary

## Data and Artifact Layout

```mermaid
flowchart TB
    RUN["runs/<run_id>/"] --> CLONE["clone/<repo>/"]
    RUN --> WTS["worktrees/<subagent_id>/"]
    RUN --> ART["artifacts/"]
    RUN --> LOGS["logs/"]
    RUN --> CHK["checkpoints/"]
    RUN --> CTRL["control/"]
    RUN --> PEND["pending_integrations.json"]

    ART --> OV["overview.md"]
    ART --> XLSX["results.xlsx"]
    ART --> SUM["summary.json"]
    ART --> COMP["coverage-comparison.md"]

    LOGS --> RUNLOG["run.log"]
    LOGS --> EVENTS["events.jsonl"]
    LOGS --> DSPY["dspy_traces.jsonl"]
    LOGS --> MVN["maven/*.log"]
```

## Key Runtime Decisions

- Only one production workflow exists right now: `daddy_subagents_reflective`
- Test generation is limited to Maven/JaCoCo Java repos in v1
- The detected testing framework and version are passed from repo analysis into child-agent prompts
- Subagents may create only new files under `src/test/java`
- Review-first integration is the default
- Post-merge full-repo coverage refresh is supported
- Evaluation varies the model only, not the workflow

## Future Evolution

- Add shared memory so later subagents can reuse prior successes and avoid prior failures
- Add project-level memory across runs
- Extend project adapters for UI stacks
- Add prompt-optimization loops from successful DSPy traces
