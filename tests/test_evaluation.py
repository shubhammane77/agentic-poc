import os
import tempfile
import unittest
from pathlib import Path

import tests._path_setup  # noqa: F401

from agentic_testgen.config import AppConfig, MlflowSettings
from agentic_testgen.evaluation import ModelMatrixEvaluator


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


if __name__ == "__main__":
    unittest.main()
