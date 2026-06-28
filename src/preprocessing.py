from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterable

TEMP_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "pkk-stunting-fcm-cache"
os.environ.setdefault("MPLCONFIGDIR", str(TEMP_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TEMP_CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# definisi path dan kolom

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

FAMILY_INDICATORS_PATH = DATA_INTERIM / "family_stunting_indicators.csv"
MATERNAL_AGE_PATH = DATA_RAW / "maternal_age.csv"
STUNTING_KNOWLEDGE_PATH = DATA_RAW / "stunting_knowledge.csv"
STUNTING_PREVALENCE_PATH = DATA_RAW / "stunting_prevalence.csv"
WATER_ACCESS_PATH = DATA_RAW / "water_access.csv"
SANITATION_ACCESS_PATH = DATA_RAW / "sanitation_access.csv"

OUTPUT_RISK_PROFILE = DATA_PROCESSED / "fcm_risk_profile_2024.csv"
OUTPUT_ZSCORE_MATRIX = DATA_PROCESSED / "fcm_model_matrix_zscore.csv"
OUTPUT_CORRELATION_PLOT = DATA_PROCESSED / "correlation_matrix.png"

FCM_FEATURES: list[str] = [
    "maternal_age_risk_pct",
    "low_knowledge_pct",
    "water_no_or_unimproved_pct",
    "water_limited_pct",
    "sanitation_babs_pct",
    "sanitation_unimproved_pct",
]

RISK_PROFILE_COLUMNS: list[str] = [
    "province_name",
    *FCM_FEATURES,
    "stunting_prevalence_pct",
]

ZSCORE_COLUMNS: list[str] = [
    "province_name",
    "maternal_age_risk_z",
    "low_knowledge_z",
    "water_no_or_unimproved_z",
    "water_limited_z",
    "sanitation_babs_z",
    "sanitation_unimproved_z",
]

FAMILY_CORE_COLUMNS = [
    "province_name",
    "maternal_age_risk_pct",
    "low_knowledge_pct",
    "stunting_prevalence_pct",
]

FAMILY_OUTPUT_COLUMNS = [
    *FAMILY_CORE_COLUMNS,
    "maternal_age_under21_pct",
    "maternal_age_21_39_pct",
    "maternal_age_over40_pct",
    "knowledge_attitude_high_pct",
    "severely_stunting_pct",
    "stunting_pct",
    "normal_height_for_age_pct",
]

NATIONAL_AGGREGATE_LABEL = "INDONESIA"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# loading & harmonisasi data

def load_csv(path: Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    """Load a CSV file with error handling."""
    try:
        df = pd.read_csv(path, usecols=usecols)
        logger.info("Loaded %s (%d rows, %d columns)", path.name, len(df), df.shape[1])
        return df
    except FileNotFoundError:
        logger.error("File not found: %s", path)
        raise
    except pd.errors.EmptyDataError:
        logger.error("File is empty: %s", path)
        raise
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        raise

def harmonize_province_names(df: pd.DataFrame, column: str = "province_name") -> pd.DataFrame:
    """Strip whitespace and uppercase province names for consistent merging."""
    df = df.copy()
    df[column] = df[column].astype(str).str.strip().str.upper()
    df[column] = df[column].replace(
        {
            "KEP.BANGKA BELITUNG": "BANGKA BELITUNG",
            "KEP. BANGKA BELITUNG": "BANGKA BELITUNG",
        }
    )
    return df

def normalize_decimal_format(df: pd.DataFrame, exclude: Iterable[str] = ("province_name",)) -> pd.DataFrame:
    """Ensure numeric columns use dot decimal notation (handles comma strings)."""
    df = df.copy()
    exclude_set = set(exclude)
    for col in df.columns:
        if col in exclude_set:
            continue
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def filter_national_aggregate(df: pd.DataFrame, column: str = "province_name") -> pd.DataFrame:
    """Remove the national-level INDONESIA aggregate row."""
    mask = df[column].str.upper() != NATIONAL_AGGREGATE_LABEL
    removed = (~mask).sum()
    if removed:
        logger.info("Removed %d national aggregate row(s) ('%s')", removed, NATIONAL_AGGREGATE_LABEL)
    return df.loc[mask].reset_index(drop=True)

def load_maternal_age() -> pd.DataFrame:
    """Load maternal-age table and derive the age-risk feature."""
    df = load_csv(MATERNAL_AGE_PATH)
    df = harmonize_province_names(df)
    df = normalize_decimal_format(df)
    df["maternal_age_risk_pct"] = (
        df["maternal_age_under21_pct"] + df["maternal_age_over40_pct"]
    ).round(1)
    return df

def load_stunting_knowledge() -> pd.DataFrame:
    """Load knowledge table and invert the protective high-knowledge measure."""
    df = load_csv(STUNTING_KNOWLEDGE_PATH)
    df = harmonize_province_names(df)
    df = normalize_decimal_format(df)
    df["low_knowledge_pct"] = (100 - df["knowledge_attitude_high_pct"]).round(1)
    return df

def load_stunting_prevalence() -> pd.DataFrame:
    """Load TB/U status table and derive total stunting prevalence."""
    df = load_csv(STUNTING_PREVALENCE_PATH)
    df = harmonize_province_names(df)
    df = normalize_decimal_format(df)
    df["stunting_prevalence_pct"] = (
        df["severely_stunting_pct"] + df["stunting_pct"]
    ).round(1)
    return df

def build_family_indicators() -> pd.DataFrame:
    """Build family and external-validation indicators from raw SSGI tables."""
    maternal_df = load_maternal_age()
    knowledge_df = load_stunting_knowledge()
    prevalence_df = load_stunting_prevalence()

    merged = maternal_df.merge(
        knowledge_df[["province_name", "knowledge_attitude_high_pct", "low_knowledge_pct"]],
        on="province_name",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        prevalence_df[
            [
                "province_name",
                "severely_stunting_pct",
                "stunting_pct",
                "normal_height_for_age_pct",
                "stunting_prevalence_pct",
            ]
        ],
        on="province_name",
        how="inner",
        validate="one_to_one",
    )

    merged = merged[FAMILY_OUTPUT_COLUMNS].sort_values("province_name").reset_index(drop=True)
    export_csv(merged, FAMILY_INDICATORS_PATH)
    return filter_national_aggregate(merged)

def load_water_access() -> pd.DataFrame:
    """Load water access data and derive combined no/unimproved feature."""
    df = load_csv(WATER_ACCESS_PATH)
    df = harmonize_province_names(df)
    df = normalize_decimal_format(df)
    df = filter_national_aggregate(df)

    df["water_no_or_unimproved_pct"] = (
        df["water_no_access_pct"] + df["water_unimproved_pct"]
    )
    return df[
        ["province_name", "water_no_or_unimproved_pct", "water_limited_pct"]
    ]

def load_sanitation_access() -> pd.DataFrame:
    """Load sanitation data and derive open+closed BABS feature."""
    df = load_csv(SANITATION_ACCESS_PATH)
    df = harmonize_province_names(df)
    df = normalize_decimal_format(df)
    df = filter_national_aggregate(df)

    df["sanitation_babs_pct"] = (
        df["sanitation_open_babs_pct"] + df["sanitation_closed_babs_pct"]
    )
    return df[["province_name", "sanitation_babs_pct", "sanitation_unimproved_pct"]]

def merge_datasets(
    family_df: pd.DataFrame,
    water_df: pd.DataFrame,
    sanitation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-merge all sources on harmonized province_name."""
    merged = family_df.merge(water_df, on="province_name", how="inner", validate="one_to_one")
    merged = merged.merge(sanitation_df, on="province_name", how="inner", validate="one_to_one")

    family_provinces = set(family_df["province_name"])
    merged_provinces = set(merged["province_name"])
    dropped = family_provinces - merged_provinces
    if dropped:
        logger.warning(
            "Provinces excluded after merge (missing water/sanitation match): %s",
            sorted(dropped),
        )

    merged = merged[RISK_PROFILE_COLUMNS].sort_values("province_name").reset_index(drop=True)
    logger.info("Merged dataset: %d provinces", len(merged))
    return merged


# cek kualitas data

def report_missing_values(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Log and return a summary of missing values per column."""
    missing = df[list(columns)].isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        logger.info("Missing values: none detected in target columns.")
    else:
        logger.warning("Missing values detected:\n%s", missing.to_string())
        for province in df.loc[df[list(columns)].isna().any(axis=1), "province_name"]:
            logger.warning("  -> province with missing data: %s", province)
    return missing.to_frame("missing_count")

def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: Iterable[str],
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Flag outliers using the interquartile range (IQR) rule."""
    records: list[dict] = []
    for col in columns:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        for idx in df.index[mask & df[col].notna()]:
            records.append(
                {
                    "province_name": df.at[idx, "province_name"],
                    "feature": col,
                    "value": df.at[idx, col],
                    "method": "IQR",
                    "lower_bound": lower,
                    "upper_bound": upper,
                }
            )
    result = pd.DataFrame(records)
    if result.empty:
        logger.info("IQR outlier check: no outliers flagged.")
    else:
        logger.warning("IQR outliers flagged (%d):\n%s", len(result), result.to_string(index=False))
    return result

def detect_outliers_zscore(
    df: pd.DataFrame,
    columns: Iterable[str],
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Flag outliers where absolute sample Z-score exceeds threshold."""
    records: list[dict] = []
    for col in columns:
        series = df[col].dropna()
        if len(series) < 2:
            continue
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0:
            continue
        z_scores = (df[col] - mean) / std
        mask = z_scores.abs() > threshold
        for idx in df.index[mask & df[col].notna()]:
            records.append(
                {
                    "province_name": df.at[idx, "province_name"],
                    "feature": col,
                    "value": df.at[idx, col],
                    "method": "Z-score",
                    "z_score": z_scores.at[idx],
                    "threshold": threshold,
                }
            )
    result = pd.DataFrame(records)
    if result.empty:
        logger.info("Z-score outlier check: no outliers flagged (|z| > %.1f).", threshold)
    else:
        logger.warning(
            "Z-score outliers flagged (%d):\n%s", len(result), result.to_string(index=False)
        )
    return result

def run_quality_checks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run missing-value and outlier diagnostics on FCM input features."""
    logger.info("--- Data quality checks ---")
    quality_report = {
        "missing": report_missing_values(df, FCM_FEATURES),
        "outliers_iqr": detect_outliers_iqr(df, FCM_FEATURES),
        "outliers_zscore": detect_outliers_zscore(df, FCM_FEATURES),
    }
    return quality_report


# visualisasi & standardisasi

def save_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Compute Pearson correlation among FCM features and save heatmap."""
    corr = df[FCM_FEATURES].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("FCM Input Feature Correlation Matrix (SSGI 2024)", fontsize=13, pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Correlation heatmap saved to %s", output_path)
    except OSError as exc:
        logger.error("Failed to save correlation plot: %s", exc)
        raise
    finally:
        plt.close(fig)

def standardize_fcm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sklearn StandardScaler (Z-score) to the six FCM input features.

    Provinces with any missing FCM feature receive NaN z-scores; the scaler is
    fit only on complete cases.
    """
    complete_mask = df[FCM_FEATURES].notna().all(axis=1)
    n_incomplete = (~complete_mask).sum()
    if n_incomplete:
        incomplete = df.loc[~complete_mask, "province_name"].tolist()
        logger.warning(
            "Standardization skipped for %d province(s) with incomplete FCM features: %s",
            n_incomplete,
            incomplete,
        )

    zscore_df = pd.DataFrame({"province_name": df["province_name"]})
    z_array = np.full((len(df), len(FCM_FEATURES)), np.nan)

    if complete_mask.any():
        scaler = StandardScaler()
        z_array[complete_mask.to_numpy()] = scaler.fit_transform(
            df.loc[complete_mask, FCM_FEATURES]
        )

    zscore_names = [col.replace("_pct", "_z") for col in FCM_FEATURES]
    for i, name in enumerate(zscore_names):
        zscore_df[name] = z_array[:, i]

    return zscore_df[ZSCORE_COLUMNS]


# export data

def export_csv(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to CSV with error handling."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, float_format="%.4f")
        logger.info("Exported %s (%d rows)", path.name, len(df))
    except OSError as exc:
        logger.error("Failed to write %s: %s", path, exc)
        raise


# pipeline orchestration

def run_preprocessing_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full preprocessing workflow and return both output tables."""
    logger.info("Starting preprocessing pipeline")

    family_df = build_family_indicators()
    water_df = load_water_access()
    sanitation_df = load_sanitation_access()
    merged_df = merge_datasets(family_df, water_df, sanitation_df)

    run_quality_checks(merged_df)
    save_correlation_heatmap(merged_df, OUTPUT_CORRELATION_PLOT)

    zscore_df = standardize_fcm_features(merged_df)

    export_csv(merged_df, OUTPUT_RISK_PROFILE)
    export_csv(zscore_df, OUTPUT_ZSCORE_MATRIX)

    logger.info("Preprocessing pipeline completed successfully")
    return merged_df, zscore_df

def main() -> int:
    """CLI entry point."""
    try:
        run_preprocessing_pipeline()
        return 0
    except Exception:
        logger.exception("Preprocessing pipeline failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
