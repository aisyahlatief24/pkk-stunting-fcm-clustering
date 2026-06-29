from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "05_validation_robustness_analysis.py"
SPEC = importlib.util.spec_from_file_location("validation_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
validation = importlib.util.module_from_spec(SPEC)
sys.modules["validation_analysis"] = validation
SPEC.loader.exec_module(validation)


def membership_frame(cluster_count: int) -> pd.DataFrame:
    rows = []
    for row_index in range(cluster_count):
        values = np.full(cluster_count, 0.1 / (cluster_count - 1))
        values[row_index] = 0.9
        row = {"province_name": f"P{row_index + 1}"}
        for cluster_number, value in enumerate(values, start=1):
            row[f"membership_cluster_{cluster_number}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def centroid_frame(cluster_count: int) -> pd.DataFrame:
    records = []
    for cluster_number in range(1, cluster_count + 1):
        base = -1.0 + cluster_number * 0.5
        records.append(
            {
                "cluster": f"cluster_{cluster_number}",
                "maternal_age_risk_z": base,
                "low_knowledge_z": base - 0.1,
                "water_no_or_unimproved_z": base + 0.2,
                "water_limited_z": base + 0.1,
                "sanitation_babs_z": base - 0.2,
                "sanitation_unimproved_z": base,
            }
        )
    return pd.DataFrame(records)


class DynamicMembershipTests(unittest.TestCase):
    def test_detect_and_recompute_membership_for_two_to_five_clusters(self) -> None:
        for cluster_count in range(2, 6):
            with self.subTest(cluster_count=cluster_count):
                df = membership_frame(cluster_count)
                columns = validation.detect_membership_columns(df)
                result = validation.validate_and_recompute_membership(df, columns)

                self.assertEqual(columns, [f"membership_cluster_{i}" for i in range(1, cluster_count + 1)])
                self.assertTrue(np.allclose(result[columns].sum(axis=1), 1.0))
                self.assertTrue(np.allclose(result["maximum_membership"], 0.9))
                self.assertTrue(np.allclose(result["second_highest_membership"], 0.1 / (cluster_count - 1)))
                self.assertTrue(np.allclose(result["membership_margin"], 0.9 - 0.1 / (cluster_count - 1)))
                self.assertEqual(result["crisp_cluster"].tolist(), list(range(1, cluster_count + 1)))

    def test_detect_membership_columns_requires_consecutive_numeric_suffix(self) -> None:
        df = pd.DataFrame({"membership_cluster_1": [0.5], "membership_cluster_3": [0.5]})
        with self.assertRaisesRegex(ValueError, "consecutive"):
            validation.detect_membership_columns(df)

    def test_risk_labels_follow_scores_not_cluster_numbers(self) -> None:
        centroids = centroid_frame(3)
        centroids.loc[0, validation.FEATURE_COLUMNS] = 1.0
        centroids.loc[1, validation.FEATURE_COLUMNS] = -1.0
        centroids.loc[2, validation.FEATURE_COLUMNS] = 0.0
        membership = validation.validate_and_recompute_membership(membership_frame(3))

        profiles = validation.build_cluster_profiles(centroids, membership)
        rank_by_cluster = profiles.set_index("cluster")["risk_rank"].to_dict()

        self.assertEqual(rank_by_cluster[2], 1)
        self.assertEqual(rank_by_cluster[3], 2)
        self.assertEqual(rank_by_cluster[1], 3)


class DominantIndicatorTests(unittest.TestCase):
    def build_profiles(self, values: list[float]) -> pd.DataFrame:
        df = pd.DataFrame([{column: value for column, value in zip(validation.FEATURE_COLUMNS, values)}])
        df["cluster"] = 1
        return validation.add_indicator_interpretation(df)

    def test_mixed_positive_negative_centroid(self) -> None:
        profiles = self.build_profiles([-0.4, 0.2, 0.7, -0.8, 0.1, -0.2])
        row = profiles.iloc[0]
        self.assertEqual(row["highest_centroid_indicator"], "water_no_or_unimproved_z")
        self.assertEqual(row["most_elevated_risk_indicator"], "water_no_or_unimproved_z")
        self.assertEqual(row["most_distinguishing_indicator"], "water_limited_z")

    def test_all_negative_centroid_has_no_elevated_risk(self) -> None:
        profiles = self.build_profiles([-0.4, -0.2, -0.7, -0.8, -0.1, -0.2])
        row = profiles.iloc[0]
        self.assertEqual(row["highest_centroid_indicator"], "sanitation_babs_z")
        self.assertEqual(row["most_elevated_risk_indicator"], "Tidak ada indikator di atas rata-rata")
        self.assertTrue(np.isnan(row["most_elevated_risk_value"]))
        self.assertEqual(row["most_distinguishing_indicator"], "water_limited_z")

    def test_all_positive_and_tied_centroid(self) -> None:
        profiles = self.build_profiles([0.5, 0.5, 0.1, 0.2, 0.5, 0.3])
        row = profiles.iloc[0]
        self.assertIn("maternal_age_risk_z", row["highest_centroid_indicator"])
        self.assertIn("low_knowledge_z", row["highest_centroid_indicator"])
        self.assertIn("sanitation_babs_z", row["most_distinguishing_indicator"])


class ExternalValidationTests(unittest.TestCase):
    def test_external_validation_dynamic_and_nonmonotonic_note(self) -> None:
        province_analysis = pd.DataFrame(
            {
                "province_name": ["A", "B", "C", "D"],
                "crisp_cluster": [1, 1, 2, 3],
                "risk_rank": [1, 1, 2, 3],
                "cluster_label": ["low", "low", "mid", "high"],
                "stunting_prevalence_pct": [30.0, 31.0, 20.0, 25.0],
            }
        )
        result = validation.build_external_validation(province_analysis)

        self.assertEqual(result["cluster"].tolist(), [1, 2, 3])
        self.assertTrue(np.isnan(result.loc[result["cluster"] == 2, "standard_deviation"].iloc[0]))
        self.assertIn("tidak sepenuhnya", result["prevalence_trend_note"].iloc[0])

    def test_missing_prevalence_raises_error(self) -> None:
        membership = validation.validate_and_recompute_membership(membership_frame(2))
        risk_profile = pd.DataFrame({"province_name": ["P1"], "stunting_prevalence_pct": [10.0]})
        profiles = validation.build_cluster_profiles(centroid_frame(2), membership)

        with self.assertRaisesRegex(ValueError, "Missing stunting prevalence"):
            validation.build_province_membership_analysis(membership, risk_profile, profiles)


if __name__ == "__main__":
    unittest.main()
