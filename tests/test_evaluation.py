import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tests._path_setup  # noqa: F401

from agentic_testgen.core.config import AppConfig, MlflowSettings
from agentic_testgen.analysis.evaluation import ModelMatrixEvaluator
from agentic_testgen.core.models import AttemptRecord


class EvaluationTests(unittest.TestCase):
    def test_model_matrix_reports_missing_credentials(self) -> None:
        os.environ.pop("MODEL_API_KEY", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(
                gitlab_token="dummy-token",
                workspace_root=Path(tmpdir),
                mlflow=MlflowSettings(enabled=False),
            )
            results = ModelMatrixEvaluator(config).run(Path("examples/model_matrix.toml"))
            self.assertGreaterEqual(len(results), 1)
            self.assertEqual("failed", results[0].status)
            self.assertIn("Missing credential env", results[0].error_message)

    def test_model_matrix_computes_test_success_ratio_from_attempt_counts(self) -> None:
        os.environ["MODEL_API_KEY"] = "dummy-key"
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "matrix.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[[models]]",
                        'model_id = "example-model"',
                        'model_name = "openai/gpt-4o-mini"',
                        'api_key_env = "MODEL_API_KEY"',
                        "",
                        "[[fixtures]]",
                        'name = "simple-service"',
                        'repo_path = "tests/fixtures/repos/simple-service"',
                        'target_files = ["src/main/java/com/example/Calculator.java"]',
                    ]
                ),
                encoding="utf-8",
            )
            config = AppConfig(
                gitlab_token="dummy-token",
                workspace_root=Path(tmpdir),
                mlflow=MlflowSettings(enabled=False),
            )
            attempts = [
                AttemptRecord(
                    run_id="run_1",
                    subagent_id="subagent_001",
                    file_path="src/main/java/com/example/Calculator.java",
                    iteration=1,
                    prompt_version="v1",
                    prompt_hash="abc",
                    tool_call_summary="{}",
                    generated_test_file="src/test/java/com/example/CalculatorGeneratedTestIter1.java",
                    single_test_command="mvn test",
                    status="passed",
                    failure_summary="",
                    reflective_summary="ok",
                    created_test_count=3,
                    successful_test_count=2,
                    candidate_count=1,
                ),
                AttemptRecord(
                    run_id="run_1",
                    subagent_id="subagent_001",
                    file_path="src/main/java/com/example/Calculator.java",
                    iteration=2,
                    prompt_version="v1",
                    prompt_hash="def",
                    tool_call_summary="{}",
                    generated_test_file="src/test/java/com/example/CalculatorGeneratedTestIter2.java",
                    single_test_command="mvn test",
                    status="failed",
                    failure_summary="failed",
                    reflective_summary="retry",
                    created_test_count=2,
                    successful_test_count=1,
                    candidate_count=2,
                ),
            ]
            run_result = SimpleNamespace(
                attempts=attempts,
                subagent_results=[
                    SimpleNamespace(status="passed", coverage_delta=1.5, missed_line_reduction=2),
                ],
                run_id="run_1",
            )

            class FakeWorkflow:
                def __init__(self, _config: AppConfig):
                    pass

                def run_from_local_path(self, *args, **kwargs):
                    return run_result

            with patch("agentic_testgen.analysis.evaluation.DaddySubagentsReflectiveWorkflow", FakeWorkflow):
                results = ModelMatrixEvaluator(config).run(config_path)

            self.assertEqual(1, len(results))
            self.assertEqual(5, results[0].created_test_count)
            self.assertEqual(3, results[0].successful_test_count)
            self.assertAlmostEqual(0.6, results[0].test_success_ratio)
            self.assertAlmostEqual(0.5, results[0].pass_rate)

    def test_model_matrix_logs_single_mlflow_summary_run(self) -> None:
        os.environ["MODEL_API_KEY"] = "dummy-key"
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "matrix.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[[models]]",
                        'model_id = "example-model"',
                        'model_name = "openai/gpt-4o-mini"',
                        'api_key_env = "MODEL_API_KEY"',
                        "",
                        "[[fixtures]]",
                        'name = "simple-service"',
                        'repo_path = "tests/fixtures/repos/simple-service"',
                        'target_files = ["src/main/java/com/example/Calculator.java"]',
                    ]
                ),
                encoding="utf-8",
            )
            config = AppConfig(
                gitlab_token="dummy-token",
                workspace_root=Path(tmpdir),
                mlflow=MlflowSettings(enabled=True, strict=False),
            )
            run_result = SimpleNamespace(
                attempts=[
                    AttemptRecord(
                        run_id="run_2",
                        subagent_id="subagent_001",
                        file_path="src/main/java/com/example/Calculator.java",
                        iteration=1,
                        prompt_version="v1",
                        prompt_hash="abc",
                        tool_call_summary="{}",
                        generated_test_file="src/test/java/com/example/CalculatorGeneratedTestIter1.java",
                        single_test_command="mvn test",
                        status="passed",
                        failure_summary="",
                        reflective_summary="ok",
                        created_test_count=2,
                        successful_test_count=2,
                        candidate_count=1,
                    )
                ],
                subagent_results=[
                    SimpleNamespace(status="passed", coverage_delta=1.0, missed_line_reduction=1),
                ],
                run_id="run_2",
            )

            class FakeWorkflow:
                def __init__(self, _config: AppConfig):
                    pass

                def run_from_local_path(self, *args, **kwargs):
                    return run_result

            class FakeTracer:
                instances: list["FakeTracer"] = []

                def __init__(self, settings, logger):
                    self.run_calls: list[tuple[str, dict[str, str] | None]] = []
                    self.params: list[dict] = []
                    self.metrics: list[dict] = []
                    self.artifacts: list[str] = []
                    FakeTracer.instances.append(self)

                def validate(self) -> bool:
                    return True

                def configure(self) -> None:
                    return None

                @contextmanager
                def run(self, name: str, tags: dict[str, str] | None = None):
                    self.run_calls.append((name, tags))
                    yield None

                def log_params(self, params: dict) -> None:
                    self.params.append(params)

                def log_metrics(self, metrics: dict) -> None:
                    self.metrics.append(metrics)

                def log_artifact(self, path: str | Path) -> None:
                    self.artifacts.append(str(path))

            FakeTracer.instances = []
            with patch("agentic_testgen.analysis.evaluation.DaddySubagentsReflectiveWorkflow", FakeWorkflow), patch(
                "agentic_testgen.analysis.evaluation.MlflowTracer", FakeTracer
            ):
                ModelMatrixEvaluator(config).run(config_path)

            self.assertEqual(1, len(FakeTracer.instances))
            tracer = FakeTracer.instances[0]
            self.assertEqual(1, len(tracer.run_calls))
            self.assertEqual(1, len(tracer.params))
            self.assertEqual(1, len(tracer.metrics))
            self.assertEqual(4, len(tracer.artifacts))
            self.assertTrue(any(path.endswith("model_eval.json") for path in tracer.artifacts))
            self.assertTrue(any(path.endswith("model_eval.csv") for path in tracer.artifacts))
            self.assertTrue(any(path.endswith("results.xlsx") for path in tracer.artifacts))
            self.assertTrue(any(path.endswith("summary.json") for path in tracer.artifacts))


if __name__ == "__main__":
    unittest.main()
