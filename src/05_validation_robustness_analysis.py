"""Validation and robustness analysis for the selected FCM clustering output.

This script does not rerun Fuzzy C-Means. It only consumes the saved model
artifacts and processed risk profile data, then writes analysis-ready CSV files
for centroid interpretation, membership uncertainty, and external validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "outputs" / "model"
DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs" / "analysis"

CENTROIDS_PATH = MODEL_DIR / "cluster_centroids_standardized.csv"
MEMBERSHIP_PATH = MODEL_DIR / "cluster_membership.csv"
RISK_PROFILE_PATH = DATA_DIR / "fcm_risk_profile_2024.csv"

FEATURE_COLUMNS = [
    "maternal_age_risk_z",
    "low_knowledge_z",
    "water_no_or_unimproved_z",
    "water_limited_z",
    "sanitation_babs_z",
    "sanitation_unimproved_z",
]

DIMENSION_COLUMNS = {
    "family_score": ["maternal_age_risk_z", "low_knowledge_z"],
    "water_score": ["water_no_or_unimproved_z", "water_limited_z"],
    "sanitation_score": ["sanitation_babs_z", "sanitation_unimproved_z"],
}


def normalize_province_name(value: str) -> str:
    """Return a robust key for joining province names from different files."""
    return " ".join(str(value).strip().upper().split())


def cluster_number(cluster_value: str) -> int:
    """Convert labels such as 'cluster_1' to integer cluster IDs."""
    return int(str(cluster_value).replace("cluster_", ""))


def membership_status(row: pd.Series) -> str:
    """Classify fuzzy membership certainty using the specified thresholds."""
    if row["maximum_membership"] < 0.60 or row["membership_margin"] < 0.20:
        return "Ambigu tinggi"
    if 0.60 <= row["maximum_membership"] <= 0.75:
        return "Transisi moderat"
    return "Keanggotaan kuat"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load centroid, membership, and external validation data."""
    centroids = pd.read_csv(CENTROIDS_PATH)
    membership = pd.read_csv(MEMBERSHIP_PATH)
    risk_profile = pd.read_csv(RISK_PROFILE_PATH)

    missing_centroid_columns = {"cluster", *FEATURE_COLUMNS} - set(centroids.columns)
    missing_membership_columns = {
        "province_name",
        "membership_cluster_1",
        "membership_cluster_2",
        "maximum_membership",
        "second_highest_membership",
        "membership_margin",
        "crisp_cluster",
    } - set(membership.columns)
    missing_risk_columns = {"province_name", "stunting_prevalence_pct"} - set(
        risk_profile.columns
    )

    if missing_centroid_columns:
        raise ValueError(f"Missing centroid columns: {sorted(missing_centroid_columns)}")
    if missing_membership_columns:
        raise ValueError(
            f"Missing membership columns: {sorted(missing_membership_columns)}"
        )
    if missing_risk_columns:
        raise ValueError(f"Missing risk profile columns: {sorted(missing_risk_columns)}")

    return centroids, membership, risk_profile


def build_cluster_profiles(
    centroids: pd.DataFrame, membership: pd.DataFrame
) -> pd.DataFrame:
    """Compute dimension scores, labels, and membership summaries per cluster."""
    profiles = centroids.copy()
    profiles["cluster"] = profiles["cluster"].map(cluster_number)

    for score_column, source_columns in DIMENSION_COLUMNS.items():
        profiles[score_column] = profiles[source_columns].mean(axis=1)

    profiles["overall_risk_score"] = profiles[
        ["family_score", "water_score", "sanitation_score"]
    ].mean(axis=1)

    dimension_score_columns = ["family_score", "water_score", "sanitation_score"]
    profiles["dominant_dimension"] = (
        profiles[dimension_score_columns].idxmax(axis=1).str.replace("_score", "")
    )
    profiles["dominant_indicator"] = profiles[FEATURE_COLUMNS].idxmax(axis=1)

    ordered_clusters = profiles.sort_values("overall_risk_score")["cluster"].tolist()
    if len(ordered_clusters) != 2:
        raise ValueError("This analysis expects the selected two-cluster FCM solution.")

    label_by_cluster = {
        ordered_clusters[0]: "Profil faktor risiko relatif lebih rendah",
        ordered_clusters[1]: "Profil faktor risiko relatif lebih tinggi",
    }
    profiles["cluster_label"] = profiles["cluster"].map(label_by_cluster)

    membership_summary = (
        membership.groupby("crisp_cluster", as_index=False)
        .agg(
            number_of_provinces=("province_name", "count"),
            mean_maximum_membership=("maximum_membership", "mean"),
        )
        .rename(columns={"crisp_cluster": "cluster"})
    )

    output_columns = [
        "cluster",
        "family_score",
        "water_score",
        "sanitation_score",
        "overall_risk_score",
        "dominant_dimension",
        "dominant_indicator",
        "cluster_label",
        "number_of_provinces",
        "mean_maximum_membership",
    ]

    return (
        profiles.merge(membership_summary, on="cluster", how="left")
        .sort_values("overall_risk_score", ascending=False)
        .loc[:, output_columns]
    )


def build_dominant_factors(centroids: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Create a compact table for centroid and dominant-factor interpretation."""
    centroid_table = centroids.copy()
    centroid_table["cluster"] = centroid_table["cluster"].map(cluster_number)

    factor_table = centroid_table.merge(
        profiles[
            [
                "cluster",
                "family_score",
                "water_score",
                "sanitation_score",
                "overall_risk_score",
                "dominant_dimension",
                "dominant_indicator",
                "cluster_label",
            ]
        ],
        on="cluster",
        how="left",
    )

    ordered_columns = [
        "cluster",
        *FEATURE_COLUMNS,
        "family_score",
        "water_score",
        "sanitation_score",
        "overall_risk_score",
        "dominant_dimension",
        "dominant_indicator",
        "cluster_label",
    ]
    return factor_table.loc[:, ordered_columns].sort_values(
        "overall_risk_score", ascending=False
    )


def build_province_membership_analysis(
    membership: pd.DataFrame, risk_profile: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    """Combine fuzzy membership, cluster labels, and stunting prevalence."""
    membership_analysis = membership.copy()
    membership_analysis["membership_status"] = membership_analysis.apply(
        membership_status, axis=1
    )
    membership_analysis["_province_key"] = membership_analysis["province_name"].map(
        normalize_province_name
    )

    risk_lookup = risk_profile[["province_name", "stunting_prevalence_pct"]].copy()
    risk_lookup["_province_key"] = risk_lookup["province_name"].map(normalize_province_name)
    risk_lookup = risk_lookup.drop(columns=["province_name"])

    label_lookup = profiles[["cluster", "cluster_label"]].copy()

    membership_analysis = (
        membership_analysis.merge(risk_lookup, on="_province_key", how="left")
        .drop(columns=["_province_key"])
        .merge(
            label_lookup,
            left_on="crisp_cluster",
            right_on="cluster",
            how="left",
        )
        .drop(columns=["cluster"])
    )

    if membership_analysis["stunting_prevalence_pct"].isna().any():
        missing = membership_analysis.loc[
            membership_analysis["stunting_prevalence_pct"].isna(), "province_name"
        ].tolist()
        raise ValueError(f"Missing stunting prevalence for provinces: {missing}")

    ordered_columns = [
        "province_name",
        "membership_cluster_1",
        "membership_cluster_2",
        "maximum_membership",
        "second_highest_membership",
        "membership_margin",
        "crisp_cluster",
        "cluster_label",
        "membership_status",
        "stunting_prevalence_pct",
    ]

    return membership_analysis.loc[:, ordered_columns].sort_values(
        ["crisp_cluster", "maximum_membership"], ascending=[True, True]
    )


def build_external_validation(province_analysis: pd.DataFrame) -> pd.DataFrame:
    """Summarize stunting prevalence by crisp FCM cluster."""
    validation = (
        province_analysis.groupby(["crisp_cluster", "cluster_label"], as_index=False)
        .agg(
            number_of_provinces=("province_name", "count"),
            mean_stunting_prevalence=("stunting_prevalence_pct", "mean"),
            median_stunting_prevalence=("stunting_prevalence_pct", "median"),
            minimum_stunting_prevalence=("stunting_prevalence_pct", "min"),
            maximum_stunting_prevalence=("stunting_prevalence_pct", "max"),
            standard_deviation=("stunting_prevalence_pct", "std"),
        )
        .rename(columns={"crisp_cluster": "cluster"})
        .sort_values("cluster")
    )

    return validation


def build_ambiguous_provinces(province_analysis: pd.DataFrame) -> pd.DataFrame:
    """Extract provinces whose membership is ambiguous or transitional."""
    statuses = ["Ambigu tinggi", "Transisi moderat"]
    return province_analysis.loc[
        province_analysis["membership_status"].isin(statuses)
    ].sort_values(["membership_status", "maximum_membership"])


def write_outputs(
    province_analysis: pd.DataFrame,
    ambiguous_provinces: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    dominant_factors: pd.DataFrame,
    external_validation: pd.DataFrame,
) -> None:
    """Persist all requested analysis CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    province_analysis.to_csv(
        OUTPUT_DIR / "province_membership_analysis.csv", index=False
    )
    ambiguous_provinces.to_csv(OUTPUT_DIR / "ambiguous_provinces.csv", index=False)
    cluster_profiles.to_csv(OUTPUT_DIR / "cluster_profiles.csv", index=False)
    dominant_factors.to_csv(OUTPUT_DIR / "dominant_factors.csv", index=False)
    external_validation.to_csv(OUTPUT_DIR / "external_validation.csv", index=False)


def main() -> None:
    centroids, membership, risk_profile = load_inputs()
    cluster_profiles = build_cluster_profiles(centroids, membership)
    dominant_factors = build_dominant_factors(centroids, cluster_profiles)
    province_analysis = build_province_membership_analysis(
        membership, risk_profile, cluster_profiles
    )
    ambiguous_provinces = build_ambiguous_provinces(province_analysis)
    external_validation = build_external_validation(province_analysis)

    write_outputs(
        province_analysis,
        ambiguous_provinces,
        cluster_profiles,
        dominant_factors,
        external_validation,
    )

    print(f"Analysis files written to: {OUTPUT_DIR}")
    print(
        cluster_profiles[
            [
                "cluster",
                "cluster_label",
                "overall_risk_score",
                "dominant_dimension",
                "dominant_indicator",
            ]
        ].to_string(index=False)
    )
    print(external_validation.to_string(index=False))


if __name__ == "__main__":
    main()
