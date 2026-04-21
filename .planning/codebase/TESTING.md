# Testing Patterns

**Analysis Date:** 2026-04-21

## Test Framework

**Runner:**
- pytest 9.0.2
- Config: discovered via `pyproject.toml` (no `[tool.pytest.ini_options]` section — pytest uses default discovery)
- Underlying test classes are `unittest.TestCase` subclasses, run by pytest's unittest compatibility layer

**Plugins installed:**
- `pytest-mock` 3.15.1
- `pytest-cov` 7.0.0
- `anyio` 4.9.0

**Assertion Library:**
- `unittest.TestCase` assertion methods: `assertEqual`, `assertGreater`, `assertTrue`, `assertFalse`, `assertIn`, `assertNotIn`, `assertIsNotNone`, `assertGreaterEqual`, `assertRaisesRegex`

**Run Commands:**
```bash
python -m pytest              # Run all 22 tests
python -m pytest tests/test_coverage.py   # Run specific module
python -m pytest -v           # Verbose output with test names
python -m pytest --cov=src    # Run with coverage (pytest-cov)
```

## Test File Organization

**Location:** Separate `tests/` directory at project root — not co-located with source.

**Naming:**
- Test files: `test_<module_name>.py` matching the source module being tested
- Test classes: `<Subject>Tests` (e.g., `CoverageAnalyzerTests`, `WorkflowTests`, `BatchingTests`)
- Test methods: `test_<descriptive_behavior_statement>` in `snake_case`

**Structure:**
```
tests/
├── __init__.py               # Empty package marker
├── _path_setup.py            # Adds src/ to sys.path; imported by all test files
├── conftest.py               # Minimal: sets ENABLE_MLFLOW_TRACING=false env var
├── fixtures/
│   └── repos/
│       ├── simple-service/   # Java Maven fixture repo with jacoco.xml report
│       └── multi-module-root/ # Multi-module Maven fixture
├── test_batching.py
├── test_coverage.py
├── test_evaluation.py
├── test_memory.py
├── test_redaction.py
├── test_reporting.py
├── test_tools.py
├── test_workflow.py
└── test_workspace.py
```

**Total tests:** 22 collected across 9 test modules.

## Test Structure

**Suite Organization:**

Every test file follows this exact pattern:
```python
import unittest
from pathlib import Path

import tests._path_setup  # noqa: F401  ← mandatory first, adds src/ to sys.path

from agentic_testgen.<module> import <Class>
from agentic_testgen.config import AppConfig


class <Subject>Tests(unittest.TestCase):
    def test_<behavior>(self) -> None:
        # arrange
        # act
        # assert

if __name__ == "__main__":
    unittest.main()
```

**Patterns:**
- Arrange-Act-Assert structure (no explicit comments, but clear separation)
- No `setUp`/`tearDown` used — each test is self-contained
- No test fixtures via pytest `@fixture` — uses manual construction or `tempfile.TemporaryDirectory`

## Temporary Directories

All tests that write files use `tempfile.TemporaryDirectory` as a context manager so cleanup is automatic:

```python
def test_writes_xlsx_workbook(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ReportWriter(Path(tmpdir))
        # ... test body
        self.assertTrue(workbook.exists())
```

This pattern appears in `test_reporting.py`, `test_tools.py`, `test_workflow.py`, `test_memory.py`, `test_batching.py`, `test_workspace.py`.

## Mocking

**Framework:** No `unittest.mock` or `pytest-mock` detected in current tests. The pattern used instead is **direct attribute monkey-patching** (lambda replacement):

```python
# From test_workflow.py and test_memory.py
workflow.coverage.run_tests_with_coverage = lambda *args, **kwargs: (
    CommandResult(args=["mvn"], exit_code=0, stdout="", stderr="", duration_seconds=0.01),
    [
        CoverageRecord(
            file_path="src/main/java/com/example/Calculator.java",
            module=".",
            covered_lines=3,
            missed_lines=2,
            coverage_percent=60.0,
            missed_line_numbers=[10, 11],
        )
    ],
    {},
)
```

This replaces the `run_tests_with_coverage` method on the instance directly, avoiding any real Maven invocations.

**What to mock:**
- External subprocess calls (Maven/Java) via the lambda pattern on `CoverageAnalyzer.run_tests_with_coverage`
- MLflow: disabled via `MlflowSettings(enabled=False)` in config rather than mocking

**What NOT to mock:**
- File I/O — tests use real temporary directories
- `AppConfig` construction — constructed directly with test values
- `RunLogger` — constructed with `console_enabled=False` to suppress output

## Configuration in Tests

Tests construct `AppConfig` directly with minimal required values:

```python
config = AppConfig(
    gitlab_token="dummy-token",
    workspace_root=Path(tmpdir),
    mlflow=MlflowSettings(enabled=False),
)
```

MLflow is always disabled in tests via `MlflowSettings(enabled=False)`. The `conftest.py` also sets `os.environ["ENABLE_MLFLOW_TRACING"] = "false"` as a belt-and-suspenders guard.

## Fixtures and Factories

**Test Data:**
- Domain objects constructed inline using dataclass constructors — no factory library
- `CoverageRecord`, `FileWorkItem`, `IntegrationDecision`, `RepoContext` all constructed directly in test bodies
- No shared fixture factories or builder helpers

**Fixture Repos:**
- `tests/fixtures/repos/simple-service/` — Maven project with pre-generated `jacoco.xml` at `target/site/jacoco/jacoco.xml`
- `tests/fixtures/repos/multi-module-root/` — Multi-module Maven layout for module discovery tests
- Used via relative path: `fixture = Path("tests/fixtures/repos/simple-service")`

## Path Setup

**Problem solved:** The `src/` layout means `agentic_testgen` is not on `sys.path` by default.

**Solution:** `tests/_path_setup.py` inserts `src/` into `sys.path`:
```python
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

Every test file imports this module as its first import:
```python
import tests._path_setup  # noqa: F401
```

## Coverage

**Requirements:** No enforced coverage target found in config.

**View Coverage:**
```bash
python -m pytest --cov=src --cov-report=term-missing
python -m pytest --cov=src --cov-report=html   # generates htmlcov/
```

## Test Types

**Unit Tests:**
- Dominant pattern — all 22 tests are unit or integration-level
- Small-scope: `test_redaction.py` (1 test), `test_workspace.py` (2 tests)

**Integration Tests:**
- `test_workflow.py` — runs the full `DaddySubagentsReflectiveWorkflow` against fixture repos with mocked Maven
- `test_memory.py` — exercises full workflow + memory persistence to filesystem

**E2E Tests:** Not present — no real GitLab or Maven calls in the test suite.

## Common Patterns

**Testing exception raising:**
```python
def test_rejects_write_outside_test_tree(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        toolset = self._toolset(root)
        with self.assertRaisesRegex(ValueError, "src/test/java"):
            toolset.write_new_test_file(
                "module-b/src/main/java/com/example/GeneratedTest.java",
                "class GeneratedTest {}",
            )
```

**Testing file existence:**
```python
self.assertTrue(Path(result.overview_path).exists())
self.assertTrue(Path(result.workbook_path).exists())
self.assertFalse((destination / "target").exists())
```

**Testing content:**
```python
self.assertIn("Coverage context artifact:", objective)
self.assertNotIn("Testing stack:", objective)
content = (root / relative_path).read_text(encoding="utf-8")
self.assertIn("update", content)
```

**Environment manipulation:**
```python
os.environ.pop("MODEL_API_KEY", None)   # in test_evaluation.py
```
Use `pop` with default to avoid `KeyError` when the variable is absent.

**Helper factory method pattern:**
```python
class ToolWriteGuardTests(unittest.TestCase):
    def _toolset(self, root: Path) -> SafeToolset:
        logger = RunLogger("test-run", root / "logs", console_enabled=False)
        context = ToolContext(run_id="test-run", repo_root=root, ...)
        return SafeToolset(context)
```
Private `_helper` methods on the test class avoid repeating construction code.

---

*Testing analysis: 2026-04-21*
