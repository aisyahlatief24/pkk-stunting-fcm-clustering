from __future__ import annotations

import re
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

CENTROID_TIE_TOLERANCE = 1e-9
RISK_LABEL_TIE_TOLERANCE = 0.05
MEMBERSHIP_SUM_TOLERANCE = 1e-5

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

DIMENSION_LABELS = {
    "family": "keluarga",
    "water": "air minum",
    "sanitation": "sanitasi",
}

RISK_LABELS_BY_CLUSTER_COUNT = {
    2: [
        "Profil faktor risiko relatif lebih rendah",
        "Profil faktor risiko relatif lebih tinggi",
    ],
    3: [
        "Profil faktor risiko relatif rendah",
        "Profil faktor risiko relatif sedang",
        "Profil faktor risiko relatif tinggi",
    ],
    4: [
        "Profil faktor risiko relatif rendah",
        "Profil faktor risiko relatif menengah bawah",
        "Profil faktor risiko relatif menengah atas",
        "Profil faktor risiko relatif tinggi",
    ],
    5: [
        "Profil faktor risiko relatif sangat rendah",
        "Profil faktor risiko relatif rendah",
        "Profil faktor risiko relatif sedang",
        "Profil faktor risiko relatif tinggi",
        "Profil faktor risiko relatif sangat tinggi",
    ],
}


def normalize_province_name(value: str) -> str:
    """Return a robust key for joining province names from different files."""
    return " ".join(str(value).strip().upper().split())


def cluster_number(cluster_value: str | int) -> int:
    """Convert labels such as 'cluster_1' to integer cluster IDs."""
    if isinstance(cluster_value, (int, np.integer)):
        return int(cluster_value)
    match = re.fullmatch(r"cluster_(\d+)", str(cluster_value).strip())
    if not match:
        raise ValueError(f"Invalid cluster label: {cluster_value!r}")
    return int(match.group(1))


def detect_membership_columns(df: pd.DataFrame) -> list[str]:
    """Detect membership_cluster_<number> columns ordered by numeric cluster ID."""
    detected: list[tuple[int, str]] = []
    invalid = []
    pattern = re.compile(r"^membership_cluster_(\d+)$")

    for column in df.columns:
        if not column.startswith("membership_cluster_"):
            continue
        match = pattern.fullmatch(column)
        if not match:
            invalid.append(column)
            continue
        detected.append((int(match.group(1)), column))

    if invalid:
        raise ValueError(f"Invalid membership column format: {invalid}")
    if len(detected) < 2:
        raise ValueError("At least two membership_cluster_<number> columns are required.")

    numbers = [number for number, _ in detected]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"Duplicate membership cluster numbers detected: {numbers}")

    detected = sorted(detected, key=lambda item: item[0])
    sorted_numbers = [number for number, _ in detected]
    expected = list(range(sorted_numbers[0], sorted_numbers[0] + len(sorted_numbers)))
    if sorted_numbers != expected:
        raise ValueError(
            "Membership cluster numbers must be consecutive and consistently ordered; "
            f"found {sorted_numbers}, expected {expected}."
        )

    return [column for _, column in detected]


def membership_cluster_numbers(membership_columns: list[str]) -> list[int]:
    return [int(column.rsplit("_", 1)[1]) for column in membership_columns]


def validate_and_recompute_membership(
    membership: pd.DataFrame,
    membership_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Validate fuzzy membership columns and recompute summary columns dynamically."""
    df = membership.copy()
    membership_columns = membership_columns or detect_membership_columns(df)
    cluster_numbers = membership_cluster_numbers(membership_columns)

    values = df[membership_columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        bad_columns = values.columns[values.isna().any()].tolist()
        raise ValueError(f"Membership columns contain NaN or non-numeric values: {bad_columns}")

    array = values.to_numpy(dtype=float)
    if ((array < -1e-8) | (array > 1 + 1e-8)).any():
        raise ValueError("Membership values must be between 0 and 1.")

    row_sums = array.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=MEMBERSHIP_SUM_TOLERANCE):
        bad_rows = df.loc[~np.isclose(row_sums, 1.0, atol=MEMBERSHIP_SUM_TOLERANCE), "province_name"].tolist()
        raise ValueError(f"Membership rows do not sum to 1 within tolerance: {bad_rows}")

    sorted_values = np.sort(array, axis=1)
    argmax_positions = np.argmax(array, axis=1)
    crisp_clusters = np.array(cluster_numbers, dtype=int)[argmax_positions]

    df.loc[:, membership_columns] = values
    df["maximum_membership"] = sorted_values[:, -1]
    df["second_highest_membership"] = sorted_values[:, -2]
    df["membership_margin"] = df["maximum_membership"] - df["second_highest_membership"]
    df["crisp_cluster"] = crisp_clusters
    return df


def membership_status(row: pd.Series) -> str:
    """Classify fuzzy membership certainty using transparent thresholds."""
    if row["maximum_membership"] < 0.60 or row["membership_margin"] < 0.20:
        return "Ambigu tinggi"
    if 0.60 <= row["maximum_membership"] <= 0.75:
        return "Transisi moderat"
    return "Keanggotaan kuat"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load centroid, membership, and external validation data."""
    centroids = pd.read_csv(CENTROIDS_PATH)
    membership = pd.read_csv(MEMBERSHIP_PATH)
    risk_profile = pd.read_csv(RISK_PROFILE_PATH)

    missing_centroid_columns = {"cluster", *FEATURE_COLUMNS} - set(centroids.columns)
    missing_membership_columns = {"province_name"} - set(membership.columns)
    missing_risk_columns = {"province_name", "stunting_prevalence_pct"} - set(risk_profile.columns)

    if missing_centroid_columns:
        raise ValueError(f"Missing centroid columns: {sorted(missing_centroid_columns)}")
    if missing_membership_columns:
        raise ValueError(f"Missing membership columns: {sorted(missing_membership_columns)}")
    if missing_risk_columns:
        raise ValueError(f"Missing risk profile columns: {sorted(missing_risk_columns)}")

    membership_columns = detect_membership_columns(membership)
    membership = validate_and_recompute_membership(membership, membership_columns)
    return centroids, membership, risk_profile, membership_columns


def _tied_indicator_label(values: pd.Series, target_value: float, absolute: bool = False) -> str:
    compare = values.abs() if absolute else values
    tied = compare[np.isclose(compare, target_value, atol=CENTROID_TIE_TOLERANCE)].index.tolist()
    return "; ".join(tied)


def add_indicator_interpretation(profiles: pd.DataFrame) -> pd.DataFrame:
    """Add highest, elevated, and distinguishing indicator columns."""
    profiles = profiles.copy()
    highest_indicators = []
    highest_values = []
    elevated_indicators = []
    elevated_values = []
    distinguishing_indicators = []
    distinguishing_values = []

    for _, row in profiles.iterrows():
        feature_values = row[FEATURE_COLUMNS].astype(float)
        highest_value = float(feature_values.max())
        highest_indicators.append(_tied_indicator_label(feature_values, highest_value))
        highest_values.append(highest_value)

        positive_values = feature_values[feature_values > 0]
        if positive_values.empty:
            elevated_indicators.append("Tidak ada indikator di atas rata-rata")
            elevated_values.append(np.nan)
        else:
            elevated_value = float(positive_values.max())
            elevated_indicators.append(_tied_indicator_label(feature_values, elevated_value))
            elevated_values.append(elevated_value)

        absolute_values = feature_values.abs()
        distinguishing_value = float(absolute_values.max())
        distinguishing_indicators.append(
            _tied_indicator_label(feature_values, distinguishing_value, absolute=True)
        )
        distinguishing_values.append(distinguishing_value)

    profiles["highest_centroid_indicator"] = highest_indicators
    profiles["highest_centroid_value"] = highest_values
    profiles["most_elevated_risk_indicator"] = elevated_indicators
    profiles["most_elevated_risk_value"] = elevated_values
    profiles["most_distinguishing_indicator"] = distinguishing_indicators
    profiles["most_distinguishing_value"] = distinguishing_values
    profiles["dominant_indicator"] = profiles["highest_centroid_indicator"]
    return profiles


def _dominance_interpretation(row: pd.Series) -> str:
    dimension = DIMENSION_LABELS.get(str(row["dominant_dimension"]), str(row["dominant_dimension"]))
    if row["dominant_dimension_score"] > 0:
        return f"{dimension.capitalize()} merupakan dimensi risiko paling menonjol pada klaster ini."
    return (
        "Seluruh dimensi berada di bawah rata-rata; "
        f"{dimension} merupakan dimensi yang relatif paling mendekati rata-rata."
    )


def add_dimension_interpretation(profiles: pd.DataFrame) -> pd.DataFrame:
    profiles = profiles.copy()
    dimension_score_columns = ["family_score", "water_score", "sanitation_score"]
    profiles["dominant_dimension"] = (
        profiles[dimension_score_columns].idxmax(axis=1).str.replace("_score", "", regex=False)
    )
    profiles["dominant_dimension_score"] = profiles[dimension_score_columns].max(axis=1)
    profiles["dominance_interpretation"] = profiles.apply(_dominance_interpretation, axis=1)
    return profiles


def add_risk_labels(profiles: pd.DataFrame, tie_tolerance: float = RISK_LABEL_TIE_TOLERANCE) -> pd.DataFrame:
    """Assign risk_rank and cluster_label from overall risk score, preserving cluster IDs."""
    profiles = profiles.copy()
    cluster_count = len(profiles)
    base_labels = RISK_LABELS_BY_CLUSTER_COUNT.get(cluster_count)
    if base_labels is None:
        base_labels = [f"Profil faktor risiko relatif peringkat {rank}" for rank in range(1, cluster_count + 1)]

    ordered = profiles.sort_values(["overall_risk_score", "cluster"], ascending=[True, True]).copy()
    ordered["risk_rank"] = np.arange(1, cluster_count + 1)
    ordered["cluster_label"] = base_labels

    previous_score: float | None = None
    for idx, row in ordered.iterrows():
        score = float(row["overall_risk_score"])
        if previous_score is not None and abs(score - previous_score) <= tie_tolerance:
            dimension = DIMENSION_LABELS.get(str(row["dominant_dimension"]), str(row["dominant_dimension"]))
            ordered.at[idx, "cluster_label"] = f"{row['cluster_label']} ({dimension} relatif menonjol)"
        previous_score = score

    return profiles.drop(columns=[col for col in ["risk_rank", "cluster_label"] if col in profiles.columns]).merge(
        ordered[["cluster", "risk_rank", "cluster_label"]],
        on="cluster",
        how="left",
    )


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

    profiles = add_dimension_interpretation(profiles)
    profiles = add_indicator_interpretation(profiles)
    profiles = add_risk_labels(profiles)

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
        "risk_rank",
        "family_score",
        "water_score",
        "sanitation_score",
        "overall_risk_score",
        "dominant_dimension",
        "dominant_dimension_score",
        "dominance_interpretation",
        "dominant_indicator",
        "highest_centroid_indicator",
        "highest_centroid_value",
        "most_elevated_risk_indicator",
        "most_elevated_risk_value",
        "most_distinguishing_indicator",
        "most_distinguishing_value",
        "cluster_label",
        "number_of_provinces",
        "mean_maximum_membership",
    ]

    return (
        profiles.merge(membership_summary, on="cluster", how="left")
        .sort_values("risk_rank")
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
                "risk_rank",
                "family_score",
                "water_score",
                "sanitation_score",
                "overall_risk_score",
                "dominant_dimension",
                "dominant_dimension_score",
                "dominance_interpretation",
                "dominant_indicator",
                "highest_centroid_indicator",
                "highest_centroid_value",
                "most_elevated_risk_indicator",
                "most_elevated_risk_value",
                "most_distinguishing_indicator",
                "most_distinguishing_value",
                "cluster_label",
            ]
        ],
        on="cluster",
        how="left",
    )

    ordered_columns = [
        "cluster",
        "risk_rank",
        *FEATURE_COLUMNS,
        "family_score",
        "water_score",
        "sanitation_score",
        "overall_risk_score",
        "dominant_dimension",
        "dominant_dimension_score",
        "dominance_interpretation",
        "dominant_indicator",
        "highest_centroid_indicator",
        "highest_centroid_value",
        "most_elevated_risk_indicator",
        "most_elevated_risk_value",
        "most_distinguishing_indicator",
        "most_distinguishing_value",
        "cluster_label",
    ]
    return factor_table.loc[:, ordered_columns].sort_values("risk_rank")


def build_province_membership_analysis(
    membership: pd.DataFrame,
    risk_profile: pd.DataFrame,
    profiles: pd.DataFrame,
    membership_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Combine fuzzy membership, cluster labels, risk rank, and stunting prevalence."""
    membership_columns = membership_columns or detect_membership_columns(membership)
    membership_analysis = validate_and_recompute_membership(membership, membership_columns)
    membership_analysis["membership_status"] = membership_analysis.apply(membership_status, axis=1)
    membership_analysis["_province_key"] = membership_analysis["province_name"].map(normalize_province_name)

    risk_lookup = risk_profile[["province_name", "stunting_prevalence_pct"]].copy()
    risk_lookup["_province_key"] = risk_lookup["province_name"].map(normalize_province_name)
    risk_lookup = risk_lookup.drop(columns=["province_name"])

    label_lookup = profiles[["cluster", "risk_rank", "cluster_label"]].copy()

    membership_analysis = (
        membership_analysis.merge(risk_lookup, on="_province_key", how="left")
        .drop(columns=["_province_key"])
        .merge(label_lookup, left_on="crisp_cluster", right_on="cluster", how="left")
        .drop(columns=["cluster"])
    )

    if membership_analysis["stunting_prevalence_pct"].isna().any():
        missing = membership_analysis.loc[
            membership_analysis["stunting_prevalence_pct"].isna(), "province_name"
        ].tolist()
        raise ValueError(f"Missing stunting prevalence for provinces: {missing}")

    ordered_columns = [
        "province_name",
        *membership_columns,
        "maximum_membership",
        "second_highest_membership",
        "membership_margin",
        "crisp_cluster",
        "risk_rank",
        "cluster_label",
        "membership_status",
        "stunting_prevalence_pct",
    ]

    return membership_analysis.loc[:, ordered_columns].sort_values(
        ["risk_rank", "maximum_membership"], ascending=[True, True]
    )


def _prevalence_trend_note(validation: pd.DataFrame) -> str:
    means = validation.sort_values("risk_rank")["mean_stunting_prevalence"].to_numpy(dtype=float)
    if len(means) < 2:
        return "Tidak cukup klaster untuk mengevaluasi pola prevalensi."
    diffs = np.diff(means)
    if np.all(diffs >= -1e-9):
        return "Rata-rata prevalensi stunting meningkat atau tetap searah dengan urutan skor risiko."
    return (
        "Rata-rata prevalensi stunting tidak sepenuhnya meningkat searah dengan urutan skor risiko; "
        "prevalensi digunakan sebagai validasi eksternal, bukan ground truth klaster."
    )


def build_external_validation(province_analysis: pd.DataFrame) -> pd.DataFrame:
    """Summarize stunting prevalence by detected FCM cluster."""
    validation = (
        province_analysis.groupby(["crisp_cluster", "risk_rank", "cluster_label"], as_index=False)
        .agg(
            number_of_provinces=("province_name", "count"),
            mean_stunting_prevalence=("stunting_prevalence_pct", "mean"),
            median_stunting_prevalence=("stunting_prevalence_pct", "median"),
            minimum_stunting_prevalence=("stunting_prevalence_pct", "min"),
            maximum_stunting_prevalence=("stunting_prevalence_pct", "max"),
            standard_deviation=("stunting_prevalence_pct", "std"),
        )
        .rename(columns={"crisp_cluster": "cluster"})
        .sort_values("risk_rank")
    )
    validation["prevalence_trend_note"] = _prevalence_trend_note(validation)
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

    province_analysis.to_csv(OUTPUT_DIR / "province_membership_analysis.csv", index=False)
    ambiguous_provinces.to_csv(OUTPUT_DIR / "ambiguous_provinces.csv", index=False)
    cluster_profiles.to_csv(OUTPUT_DIR / "cluster_profiles.csv", index=False)
    dominant_factors.to_csv(OUTPUT_DIR / "dominant_factors.csv", index=False)
    external_validation.to_csv(OUTPUT_DIR / "external_validation.csv", index=False)


def run_validation_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run post-FCM cluster interpretation and external validation."""
    centroids, membership, risk_profile, membership_columns = load_inputs()
    cluster_profiles = build_cluster_profiles(centroids, membership)
    dominant_factors = build_dominant_factors(centroids, cluster_profiles)
    province_analysis = build_province_membership_analysis(
        membership, risk_profile, cluster_profiles, membership_columns
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
    return province_analysis, ambiguous_provinces, cluster_profiles, dominant_factors, external_validation


def main() -> None:
    _, _, cluster_profiles, _, external_validation = run_validation_analysis()
    print(f"Analysis files written to: {OUTPUT_DIR}")
    print(
        cluster_profiles[
            [
                "cluster",
                "risk_rank",
                "cluster_label",
                "overall_risk_score",
                "dominant_dimension",
                "highest_centroid_indicator",
                "most_elevated_risk_indicator",
            ]
        ].to_string(index=False)
    )
    print(external_validation.to_string(index=False))


if __name__ == "__main__":
    main()
