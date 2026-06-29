from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "04_fcm_model.py"
SPEC = importlib.util.spec_from_file_location("fcm_model_ranking", MODULE_PATH)
assert SPEC and SPEC.loader
fcm_model = importlib.util.module_from_spec(SPEC)
sys.modules["fcm_model_ranking"] = fcm_model
SPEC.loader.exec_module(fcm_model)


def summary(
    c: int,
    m: float,
    xb: float,
    mpc: float,
    pe: float,
    ari: float,
    membership_change: float,
    centroid_variation: float,
    convergence: float = 1.0,
    min_distance: float = 1.0,
    min_cluster_size: int = 2,
):
    pc = (mpc * (1 - 1 / c)) + (1 / c)
    return fcm_model.ConfigurationSummary(
        c=c,
        m=m,
        runs=[],
        representative_seed=0,
        representative_index=0,
        convergence_rate=convergence,
        mean_xie_beni=xb,
        std_xie_beni=0.0,
        mean_partition_coefficient=pc,
        std_partition_coefficient=0.0,
        mean_modified_partition_coefficient=mpc,
        std_modified_partition_coefficient=0.0,
        mean_partition_entropy=pe,
        std_partition_entropy=0.0,
        mean_final_objective=1.0,
        std_final_objective=0.0,
        mean_minimum_centroid_distance=min_distance,
        centroid_variation=centroid_variation,
        mean_pairwise_ari=ari,
        mean_membership_change=membership_change,
        minimum_crisp_cluster_size=min_cluster_size,
        empty_crisp_clusters=int(min_cluster_size == 0),
    )


class FCMRankingTests(unittest.TestCase):
    def test_grouped_ranking_is_deterministic_and_directional(self) -> None:
        summaries = [
            summary(2, 1.5, xb=0.3, mpc=0.6, pe=0.2, ari=1.0, membership_change=0.01, centroid_variation=0.01, min_distance=3),
            summary(3, 2.0, xb=0.8, mpc=0.3, pe=0.7, ari=0.6, membership_change=0.30, centroid_variation=0.40, min_distance=1),
            summary(4, 2.5, xb=0.5, mpc=0.4, pe=0.5, ari=0.8, membership_change=0.20, centroid_variation=0.20, min_distance=2),
        ]

        first = fcm_model.rank_fcm_configurations(summaries)
        second = fcm_model.rank_fcm_configurations(summaries)

        self.assertEqual(first[["c", "m"]].values.tolist(), second[["c", "m"]].values.tolist())
        self.assertEqual((int(first.iloc[0]["c"]), float(first.iloc[0]["m"])), (2, 1.5))

    def test_pc_and_mpc_are_not_two_full_independent_votes(self) -> None:
        summaries = [
            summary(2, 1.5, xb=0.3, mpc=0.6, pe=0.2, ari=1.0, membership_change=0.01, centroid_variation=0.01, min_distance=3),
            summary(3, 2.0, xb=0.4, mpc=0.5, pe=0.3, ari=0.9, membership_change=0.02, centroid_variation=0.02, min_distance=2),
        ]
        ranked = fcm_model.rank_fcm_configurations(summaries)

        self.assertIn("fuzzy_quality_score", ranked.columns)
        self.assertNotIn("score_pc", ranked.columns)
        self.assertIn("mean_partition_coefficient", ranked.columns)

    def test_sensitivity_ranking_and_selection_reason_are_available(self) -> None:
        summaries = [
            summary(2, 1.5, xb=0.3, mpc=0.6, pe=0.2, ari=1.0, membership_change=0.01, centroid_variation=0.01, min_distance=3),
            summary(3, 2.0, xb=0.4, mpc=0.5, pe=0.3, ari=0.9, membership_change=0.02, centroid_variation=0.02, min_distance=2),
        ]
        sensitivity = fcm_model.build_ranking_sensitivity(summaries)
        best = fcm_model.select_best_configuration(summaries)

        for scheme in fcm_model.RANKING_SCHEMES:
            self.assertIn(f"rank_{scheme}", sensitivity.columns)
        self.assertIsNotNone(best.selection_consistency)
        self.assertIsNotNone(best.composite_rank_score)

    def test_nonconverged_empty_or_colliding_configuration_is_penalized(self) -> None:
        summaries = [
            summary(2, 1.5, xb=0.3, mpc=0.6, pe=0.2, ari=1.0, membership_change=0.01, centroid_variation=0.01, min_distance=3),
            summary(
                3,
                2.0,
                xb=0.2,
                mpc=0.7,
                pe=0.1,
                ari=1.0,
                membership_change=0.01,
                centroid_variation=0.01,
                convergence=0.0,
                min_distance=0.0,
                min_cluster_size=0,
            ),
        ]
        ranked = fcm_model.rank_fcm_configurations(summaries)
        bad_row = ranked[ranked["c"] == 3].iloc[0]

        self.assertGreater(bad_row["diagnostic_score"], 0.0)
        self.assertEqual(int(ranked.iloc[0]["c"]), 2)


if __name__ == "__main__":
    unittest.main()
