from __future__ import annotations

import os
import tomllib
import csv
from statistics import mean, median
from dataclasses import dataclass
from pathlib import Path

from agentic_testgen.agents.agents import DaddySubagentsReflectiveWorkflow
from agentic_testgen.core.config import AppConfig
from agentic_testgen.core.logging import RunLogger
from agentic_testgen.core.models import ModelDefinition, ModelEvalCase, ModelEvalResult, RepoContext
from agentic_testgen.analysis.reporting import ReportWriter
from agentic_testgen.core.utils import new_run_id, write_json
from agentic_testgen.integrations.tracing import MlflowTracer


@dataclass
class FixtureDefinition:
    name: str
    repo_path: str
    target_files: list[str]


@dataclass
class EvaluationConfig:
    models: list[ModelDefinition]
    fixtures: list[FixtureDefinition]

    @classmethod
    def load(cls, path: Path) -> "EvaluationConfig":
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        base_dir = path.parent
        models = [ModelDefinition(**model) for model in payload.get("models", [])]
        fixtures = []
        for fixture in payload.get("fixtures", []):
            repo_path = Path(fixture["repo_path"])
            if not repo_path.is_absolute():
                repo_path = (base_dir / repo_path).resolve()
            fixtures.append(
                FixtureDefinition(
                    name=fixture["name"],
                    repo_path=str(repo_path),
                    target_files=fixture["target_files"],
                )
            )
        return cls(models=models, fixtures=fixtures)


class ModelMatrixEvaluator:
    def __init__(self, config: AppConfig):
        self.config = config

    def run(self, config_path: Path) -> list[ModelEvalResult]:
        evaluation_config = EvaluationConfig.load(config_path)
        workflow = DaddySubagentsReflectiveWorkflow(self.config)
        results: list[ModelEvalResult] = []
        for model in evaluation_config.models:
            if not os.getenv(model.api_key_env):
                results.append(
                    ModelEvalResult(
                        case_id=f"{model.model_id}-missing-creds",
                        model_id=model.model_id,
                        fixture_name="*",
                        target_file="*",
                        status="failed",
                        compile_success=False,
                        pass_rate=0.0,
                        test_success_ratio=0.0,
                        created_test_count=0,
                        successful_test_count=0,
                        coverage_delta=0.0,
                        missed_line_reduction=0,
                        forbidden_edit_rate=0.0,
                        flaky_rate=0.0,
                        latency_seconds=0.0,
                        tool_call_count=0,
                        iteration_count=0,
                        estimated_cost=0.0,
                        error_message=f"Missing credential env: {model.api_key_env}",
                    )
                )
                continue
            for fixture in evaluation_config.fixtures:
                for target_file in fixture.target_files:
                    case = ModelEvalCase(
                        case_id=f"{model.model_id}-{fixture.name}-{Path(target_file).stem}",
                        model_id=model.model_id,
                        fixture_name=fixture.name,
                        repo_source=Path(fixture.repo_path),
                        target_file=target_file,
                    )
                    try:
                        run_result = workflow.run_from_local_path(
                            case.repo_source,
                            run_id=new_run_id("eval"),
                            source_name=fixture.name,
                            selected_files=[target_file],
                            model_override=model,
                        )
                        passed = [attempt for attempt in run_result.attempts if attempt.status == "passed"]
                        created_test_count = sum(attempt.created_test_count for attempt in run_result.attempts)
                        successful_test_count = sum(attempt.successful_test_count for attempt in run_result.attempts)
                        test_success_ratio = (
                            successful_test_count / created_test_count if created_test_count else 0.0
                        )
                        compile_success = any(result.status == "passed" for result in run_result.subagent_results)
                        subagent = run_result.subagent_results[0] if run_result.subagent_results else None
                        results.append(
                            ModelEvalResult(
                                case_id=case.case_id,
                                model_id=model.model_id,
                                fixture_name=fixture.name,
                                target_file=target_file,
                                status="completed",
                                compile_success=compile_success,
                                pass_rate=(len(passed) / len(run_result.attempts)) if run_result.attempts else 0.0,
                                test_success_ratio=test_success_ratio,
                                created_test_count=created_test_count,
                                successful_test_count=successful_test_count,
                                coverage_delta=subagent.coverage_delta if subagent else 0.0,
                                missed_line_reduction=subagent.missed_line_reduction if subagent else 0,
                                forbidden_edit_rate=0.0,
                                flaky_rate=0.0,
                                latency_seconds=0.0,
                                tool_call_count=sum(1 for attempt in run_result.attempts if attempt.tool_call_summary),
                                iteration_count=len(run_result.attempts),
                                estimated_cost=0.0,
                                run_id=run_result.run_id,
                            )
                        )
                    except Exception as exc:
                        results.append(
                            ModelEvalResult(
                                case_id=case.case_id,
                                model_id=model.model_id,
                                fixture_name=fixture.name,
                                target_file=target_file,
                                status="failed",
                                compile_success=False,
                                pass_rate=0.0,
                                test_success_ratio=0.0,
                                created_test_count=0,
                                successful_test_count=0,
                                coverage_delta=0.0,
                                missed_line_reduction=0,
                                forbidden_edit_rate=0.0,
                                flaky_rate=0.0,
                                latency_seconds=0.0,
                                tool_call_count=0,
                                iteration_count=0,
                                estimated_cost=0.0,
                                error_message=str(exc),
                            )
                        )
        artifacts_dir = self.config.workspace_root / "evaluation-artifacts"
        report_writer = ReportWriter(artifacts_dir)
        repo_context = RepoContext(
            run_id="evaluation",
            repo_url=str(config_path),
            repo_name="model-evaluation",
            clone_path=artifacts_dir,
            workspace_root=artifacts_dir,
            source_type="evaluation",
        )
        workbook_path = report_writer.write_workbook(
            repo_context=repo_context,
            work_items=[],
            attempts=[],
            model_eval=results,
        )
        summary_path = report_writer.write_json_summary(
            repo_context=repo_context,
            work_items=[],
            results=[],
            model_eval=results,
        )
        model_eval_json_path = artifacts_dir / "model_eval.json"
        write_json(model_eval_json_path, [result.to_json() for result in results])
        model_eval_csv_path = artifacts_dir / "model_eval.csv"
        with model_eval_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "case_id",
                    "model_id",
                    "fixture_name",
                    "target_file",
                    "status",
                    "compile_success",
                    "pass_rate",
                    "test_success_ratio",
                    "created_test_count",
                    "successful_test_count",
                    "coverage_delta",
                    "missed_line_reduction",
                    "forbidden_edit_rate",
                    "flaky_rate",
                    "latency_seconds",
                    "tool_call_count",
                    "iteration_count",
                    "estimated_cost",
                    "run_id",
                    "error_message",
                ]
            )
            for result in results:
                writer.writerow(
                    [
                        result.case_id,
                        result.model_id,
                        result.fixture_name,
                        result.target_file,
                        result.status,
                        result.compile_success,
                        result.pass_rate,
                        result.test_success_ratio,
                        result.created_test_count,
                        result.successful_test_count,
                        result.coverage_delta,
                        result.missed_line_reduction,
                        result.forbidden_edit_rate,
                        result.flaky_rate,
                        result.latency_seconds,
                        result.tool_call_count,
                        result.iteration_count,
                        result.estimated_cost,
                        result.run_id or "",
                        result.error_message,
                    ]
                )
        self._log_evaluation_summary_to_mlflow(
            evaluation_config=evaluation_config,
            config_path=config_path,
            results=results,
            artifacts={
                "model_eval.json": model_eval_json_path,
                "model_eval.csv": model_eval_csv_path,
                "results.xlsx": workbook_path,
                "summary.json": summary_path,
            },
        )
        return results

    def _log_evaluation_summary_to_mlflow(
        self,
        *,
        evaluation_config: EvaluationConfig,
        config_path: Path,
        results: list[ModelEvalResult],
        artifacts: dict[str, Path],
    ) -> None:
        logs_dir = self.config.workspace_root / "evaluation-artifacts" / "logs"
        logger = RunLogger("evaluation-summary", logs_dir, console_enabled=False)
        tracer = MlflowTracer(self.config.mlflow, logger)
        tracer.validate()
        tracer.configure()
        total_cases = sum(len(fixture.target_files) for fixture in evaluation_config.fixtures) * len(
            evaluation_config.models
        )
        completed_cases = sum(1 for result in results if result.status == "completed")
        failed_cases = sum(1 for result in results if result.status == "failed")
        ratios = [result.test_success_ratio for result in results]
        with tracer.run(
            f"evaluation-summary-{new_run_id('eval')}",
            tags={"workflow": "model_matrix_evaluation", "source_type": "evaluation"},
        ):
            tracer.log_params(
                {
                    "evaluation_config_path": str(config_path.resolve()),
                    "model_count": len(evaluation_config.models),
                    "fixture_count": len(evaluation_config.fixtures),
                }
            )
            tracer.log_metrics(
                {
                    "total_cases": total_cases,
                    "completed_cases": completed_cases,
                    "failed_cases": failed_cases,
                    "mean_test_success_ratio": mean(ratios) if ratios else 0.0,
                    "median_test_success_ratio": median(ratios) if ratios else 0.0,
                    "created_test_count_total": sum(result.created_test_count for result in results),
                    "successful_test_count_total": sum(result.successful_test_count for result in results),
                }
            )
            for path in artifacts.values():
                if path.exists():
                    tracer.log_artifact(path)
