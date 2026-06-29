from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = PROJECT_ROOT / "run_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_pipeline_module", PIPELINE_PATH)
assert SPEC and SPEC.loader
run_pipeline = importlib.util.module_from_spec(SPEC)
sys.modules["run_pipeline_module"] = run_pipeline
SPEC.loader.exec_module(run_pipeline)


class PipelineContractTests(unittest.TestCase):
    def test_pipeline_module_exists_and_uses_relative_project_paths(self) -> None:
        self.assertTrue(PIPELINE_PATH.exists())
        checked_files = [
            PIPELINE_PATH,
            PROJECT_ROOT / "src" / "04_fcm_model.py",
            PROJECT_ROOT / "src" / "05_validation_robustness_analysis.py",
            PROJECT_ROOT / "src" / "06_spatial_mapping.py",
            PROJECT_ROOT / "src" / "07_visualization.py",
        ]
        for path in checked_files:
            with self.subTest(path=path.name):
                self.assertNotIn("/Users/ghaniandawafiqarifah", path.read_text(encoding="utf-8"))

    def test_output_schema_contracts_are_backward_compatible(self) -> None:
        membership = pd.read_csv(PROJECT_ROOT / "outputs" / "model" / "cluster_membership.csv", nrows=1)
        for column in [
            "province_name",
            "membership_cluster_1",
            "membership_cluster_2",
            "maximum_membership",
            "second_highest_membership",
            "membership_margin",
            "crisp_cluster",
        ]:
            self.assertIn(column, membership.columns)

        with (PROJECT_ROOT / "outputs" / "model" / "best_fcm_parameters.json").open(encoding="utf-8") as f:
            best = json.load(f)
        self.assertIn("best_c", best)
        self.assertIn("best_m", best)

    def test_validate_input_file_fails_without_dummy_results(self) -> None:
        with self.assertRaises(FileNotFoundError):
            run_pipeline.validate_input_file(Path("data/processed/does_not_exist.csv"))


if __name__ == "__main__":
    unittest.main()
