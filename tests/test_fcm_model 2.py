from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "04_fcm_model.py"
SPEC = importlib.util.spec_from_file_location("fcm_model", MODULE_PATH)
assert SPEC and SPEC.loader
fcm_model = importlib.util.module_from_spec(SPEC)
sys.modules["fcm_model"] = fcm_model
SPEC.loader.exec_module(fcm_model)


class FCMModelTests(unittest.TestCase):
    def test_partition_metrics_are_valid(self) -> None:
        membership = np.array([[0.8, 0.2], [0.4, 0.6], [1.0, 0.0]])
        pc = fcm_model.calculate_partition_coefficient(membership)
        mpc = fcm_model.calculate_modified_partition_coefficient(pc, c=2)
        pe = fcm_model.calculate_partition_entropy(membership)

        self.assertGreaterEqual(pc, 0.5)
        self.assertLessEqual(pc, 1.0)
        self.assertTrue(np.isclose(mpc, (pc - 0.5) / 0.5))
        self.assertTrue(np.isfinite(pe))

    def test_xie_beni_handles_identical_centroids(self) -> None:
        x = np.array([[0.0, 0.0], [1.0, 1.0]])
        centroids = np.array([[0.5, 0.5], [0.5, 0.5]])
        membership = np.array([[0.5, 0.5], [0.5, 0.5]])

        self.assertTrue(np.isinf(fcm_model.calculate_xie_beni(x, centroids, membership, m=2.0)))

    def test_membership_rows_sum_and_crisp_argmax(self) -> None:
        membership = np.array([[0.1, 0.9], [0.7, 0.3]])
        crisp = np.argmax(membership, axis=1) + 1

        self.assertTrue(np.allclose(membership.sum(axis=1), 1.0))
        self.assertEqual(crisp.tolist(), [2, 1])

    def test_label_alignment_preserves_membership_content(self) -> None:
        reference = np.array([[0.0, 0.0], [10.0, 10.0]])
        run = fcm_model.FCMRunResult(
            c=2,
            m=2.0,
            seed=1,
            converged=True,
            iterations=5,
            final_objective=1.0,
            objective_history=[2.0, 1.0],
            centroids=np.array([[10.0, 10.0], [0.0, 0.0]]),
            membership=np.array([[0.2, 0.8], [0.9, 0.1]]),
            crisp_cluster=np.array([1, 0]),
            maximum_membership=np.array([0.8, 0.9]),
            xie_beni=0.1,
            partition_coefficient=0.75,
            modified_partition_coefficient=0.5,
            partition_entropy=0.3,
            minimum_centroid_distance=200.0,
        )

        aligned = fcm_model.align_cluster_labels(reference, run)

        self.assertTrue(np.allclose(aligned.centroids, reference))
        self.assertTrue(np.allclose(np.sort(aligned.membership, axis=1), np.sort(run.membership, axis=1)))
        self.assertTrue(np.allclose(aligned.membership.sum(axis=1), 1.0))

    def test_incomplete_input_raises_informative_error(self) -> None:
        bad_path = Path(__file__).parent / "bad_input_for_fcm_test.csv"
        try:
            pd.DataFrame({"province_name": ["A"], "maternal_age_risk_z": [0.0]}).to_csv(
                bad_path, index=False
            )
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                fcm_model.load_and_validate_data(bad_path)
        finally:
            if bad_path.exists():
                bad_path.unlink()


if __name__ == "__main__":
    unittest.main()
