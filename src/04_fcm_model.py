from __future__ import annotations

import itertools
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "fcm_model_matrix_zscore.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model"

EXPERIMENT_RESULTS_PATH = OUTPUT_DIR / "fcm_experiment_results.csv"
CONFIGURATION_SUMMARY_PATH = OUTPUT_DIR / "fcm_configuration_summary.csv"
RANKING_SENSITIVITY_PATH = OUTPUT_DIR / "fcm_ranking_sensitivity.csv"
BEST_PARAMETERS_PATH = OUTPUT_DIR / "best_fcm_parameters.json"
CENTROIDS_PATH = OUTPUT_DIR / "cluster_centroids_standardized.csv"
MEMBERSHIP_PATH = OUTPUT_DIR / "cluster_membership.csv"

FEATURE_COLUMNS: list[str] = [
    "maternal_age_risk_z",
    "low_knowledge_z",
    "water_no_or_unimproved_z",
    "water_limited_z",
    "sanitation_babs_z",
    "sanitation_unimproved_z",
]
REQUIRED_COLUMNS = ["province_name", *FEATURE_COLUMNS]

CLUSTER_COUNTS = [2, 3, 4, 5]
FUZZINESS_EXPONENTS = [1.5, 1.75, 2.0, 2.25, 2.5]
RANDOM_SEEDS = list(range(20))

# FCM stopping rule: stop when membership changes less than this tolerance,
# or when the maximum number of iterations is reached.
ERROR_TOLERANCE = 1e-5
MAX_ITERATIONS = 1000
EPSILON = 1e-12
MIN_CENTROID_DISTANCE_EPS = 1e-10

RANKING_SCHEMES: dict[str, dict[str, float]] = {
    "balanced": {
        "structure": 0.40,
        "stability": 0.30,
        "fuzzy_quality": 0.20,
        "diagnostic": 0.10,
    },
    "validity_focused": {
        "structure": 0.55,
        "stability": 0.20,
        "fuzzy_quality": 0.15,
        "diagnostic": 0.10,
    },
    "stability_focused": {
        "structure": 0.25,
        "stability": 0.50,
        "fuzzy_quality": 0.15,
        "diagnostic": 0.10,
    },
    "fuzzy_quality_focused": {
        "structure": 0.30,
        "stability": 0.20,
        "fuzzy_quality": 0.40,
        "diagnostic": 0.10,
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class FCMRunResult:
    """Container for one FCM run."""

    c: int
    m: float
    seed: int
    converged: bool
    iterations: int
    final_objective: float
    objective_history: list[float]
    centroids: np.ndarray
    membership: np.ndarray
    crisp_cluster: np.ndarray
    maximum_membership: np.ndarray
    xie_beni: float
    partition_coefficient: float
    modified_partition_coefficient: float
    partition_entropy: float
    minimum_centroid_distance: float


@dataclass
class ConfigurationSummary:
    """Stability and validity summary for one c/m configuration."""

    c: int
    m: float
    runs: list[FCMRunResult]
    representative_seed: int
    representative_index: int
    convergence_rate: float
    mean_xie_beni: float
    std_xie_beni: float
    mean_partition_coefficient: float
    std_partition_coefficient: float
    mean_modified_partition_coefficient: float
    std_modified_partition_coefficient: float
    mean_partition_entropy: float
    std_partition_entropy: float
    mean_final_objective: float
    std_final_objective: float
    mean_minimum_centroid_distance: float
    centroid_variation: float
    mean_pairwise_ari: float
    mean_membership_change: float
    minimum_crisp_cluster_size: int
    empty_crisp_clusters: int
    aligned_runs: list[FCMRunResult] = field(default_factory=list)
    composite_rank_score: float | None = None
    selection_consistency: str | None = None


def load_and_validate_data(
    input_path: Path = INPUT_PATH,
    feature_columns: list[str] = FEATURE_COLUMNS,
    cluster_counts: list[int] = CLUSTER_COUNTS,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load standardized FCM matrix and fail loudly on invalid input."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Input file is missing required columns: {missing_columns}")

    if df["province_name"].isna().any() or df["province_name"].astype(str).str.strip().eq("").any():
        raise ValueError("province_name contains missing or empty values.")
    if df["province_name"].duplicated().any():
        duplicated = df.loc[df["province_name"].duplicated(), "province_name"].tolist()
        raise ValueError(f"province_name contains duplicate values: {duplicated}")

    features = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        bad_columns = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Feature columns contain NaN or non-numeric values: {bad_columns}")

    x = features.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")

    max_c = max(cluster_counts)
    if len(df) <= max_c:
        raise ValueError(
            f"Number of observations ({len(df)}) must be greater than maximum cluster count ({max_c})."
        )

    return df[REQUIRED_COLUMNS].copy(), x


def calculate_partition_coefficient(membership: np.ndarray) -> float:
    """Calculate Bezdek's Partition Coefficient."""
    return float(np.mean(np.sum(membership**2, axis=1)))


def calculate_modified_partition_coefficient(partition_coefficient: float, c: int) -> float:
    """Calculate Modified Partition Coefficient."""
    if c <= 1:
        raise ValueError("MPC requires c > 1.")
    return float((partition_coefficient - (1 / c)) / (1 - (1 / c)))


def calculate_partition_entropy(membership: np.ndarray, epsilon: float = EPSILON) -> float:
    """Calculate Partition Entropy with numerical protection around log(0)."""
    clipped = np.clip(membership, epsilon, 1.0)
    return float(-np.mean(np.sum(clipped * np.log(clipped), axis=1)))


def minimum_centroid_distance(centroids: np.ndarray) -> float:
    """Return the minimum squared Euclidean distance between different centroids."""
    if len(centroids) < 2:
        return 0.0
    distances = [
        float(np.sum((centroids[i] - centroids[j]) ** 2))
        for i in range(len(centroids))
        for j in range(i + 1, len(centroids))
    ]
    return min(distances)


def calculate_xie_beni(
    x: np.ndarray,
    centroids: np.ndarray,
    membership: np.ndarray,
    m: float,
    min_distance_epsilon: float = MIN_CENTROID_DISTANCE_EPS,
) -> float:
    """Calculate the Xie-Beni index, returning inf for near-identical centroids."""
    n = x.shape[0]
    distances = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    numerator = float(np.sum((membership**m) * distances))
    min_distance = minimum_centroid_distance(centroids)
    if min_distance <= min_distance_epsilon:
        return float("inf")
    return float(numerator / (n * min_distance))


def run_single_fcm(x: np.ndarray, c: int, m: float, seed: int) -> FCMRunResult:
    """Run scikit-fuzzy FCM once for one c/m/seed setting."""
    centroids, u, _, _, objective_history, iterations, _ = fuzz.cluster.cmeans(
        data=x.T,
        c=c,
        m=m,
        error=ERROR_TOLERANCE,
        maxiter=MAX_ITERATIONS,
        init=None,
        seed=seed,
    )
    membership = u.T
    crisp_cluster = np.argmax(membership, axis=1)
    maximum_membership = np.max(membership, axis=1)
    pc = calculate_partition_coefficient(membership)
    mpc = calculate_modified_partition_coefficient(pc, c)
    pe = calculate_partition_entropy(membership)
    xb = calculate_xie_beni(x, centroids, membership, m)
    min_dist = minimum_centroid_distance(centroids)

    return FCMRunResult(
        c=c,
        m=m,
        seed=seed,
        converged=iterations < MAX_ITERATIONS,
        iterations=int(iterations),
        final_objective=float(objective_history[-1]),
        objective_history=[float(value) for value in objective_history],
        centroids=centroids,
        membership=membership,
        crisp_cluster=crisp_cluster,
        maximum_membership=maximum_membership,
        xie_beni=xb,
        partition_coefficient=pc,
        modified_partition_coefficient=mpc,
        partition_entropy=pe,
        minimum_centroid_distance=min_dist,
    )


def _copy_run_with_alignment(
    run: FCMRunResult,
    centroids: np.ndarray,
    membership: np.ndarray,
) -> FCMRunResult:
    crisp_cluster = np.argmax(membership, axis=1)
    return FCMRunResult(
        c=run.c,
        m=run.m,
        seed=run.seed,
        converged=run.converged,
        iterations=run.iterations,
        final_objective=run.final_objective,
        objective_history=run.objective_history,
        centroids=centroids,
        membership=membership,
        crisp_cluster=crisp_cluster,
        maximum_membership=np.max(membership, axis=1),
        xie_beni=run.xie_beni,
        partition_coefficient=run.partition_coefficient,
        modified_partition_coefficient=run.modified_partition_coefficient,
        partition_entropy=run.partition_entropy,
        minimum_centroid_distance=run.minimum_centroid_distance,
    )


def align_cluster_labels(reference_centroids: np.ndarray, run: FCMRunResult) -> FCMRunResult:
    """Align one run's cluster labels to a reference centroid ordering."""
    cost = np.linalg.norm(run.centroids[:, None, :] - reference_centroids[None, :, :], axis=2)
    current_indices, reference_indices = linear_sum_assignment(cost)

    aligned_centroids = np.zeros_like(run.centroids)
    aligned_membership = np.zeros_like(run.membership)
    for current_idx, reference_idx in zip(current_indices, reference_indices):
        aligned_centroids[reference_idx] = run.centroids[current_idx]
        aligned_membership[:, reference_idx] = run.membership[:, current_idx]

    return _copy_run_with_alignment(run, aligned_centroids, aligned_membership)


def run_fcm_experiments(x: np.ndarray) -> dict[tuple[int, float], list[FCMRunResult]]:
    """Run all FCM experiments over the configured c, m, and seed grid."""
    results: dict[tuple[int, float], list[FCMRunResult]] = {}
    for c, m in itertools.product(CLUSTER_COUNTS, FUZZINESS_EXPONENTS):
        logger.info("Running FCM experiments for c=%s, m=%s", c, m)
        runs = [run_single_fcm(x, c=c, m=m, seed=seed) for seed in RANDOM_SEEDS]
        results[(c, m)] = runs
    return results


def select_representative_run(runs: list[FCMRunResult]) -> tuple[int, list[FCMRunResult]]:
    """Select the converged run closest to the aligned mean centroid."""
    candidates = [run for run in runs if run.converged and np.isfinite(run.xie_beni)]
    if not candidates:
        candidates = runs

    reference = min(
        candidates,
        key=lambda run: (run.xie_beni, -run.partition_coefficient, run.seed),
    )
    aligned_runs = [align_cluster_labels(reference.centroids, run) for run in runs]
    mean_centroids = np.mean(np.stack([run.centroids for run in aligned_runs]), axis=0)

    finite_xb = np.array([run.xie_beni for run in aligned_runs if np.isfinite(run.xie_beni)])
    best_xb = float(np.min(finite_xb)) if len(finite_xb) else float("inf")

    def score(run: FCMRunResult) -> tuple[float, float, float, int]:
        centroid_distance = float(np.linalg.norm(run.centroids - mean_centroids))
        validity_gap = abs(run.xie_beni - best_xb) if np.isfinite(run.xie_beni) else float("inf")
        non_convergence_penalty = 0.0 if run.converged else 1.0
        return (non_convergence_penalty, centroid_distance, validity_gap, run.seed)

    representative = min(aligned_runs, key=score)
    representative_index = next(
        idx for idx, run in enumerate(aligned_runs) if run.seed == representative.seed
    )
    return representative_index, aligned_runs


def _mean_pairwise_ari(aligned_runs: list[FCMRunResult]) -> float:
    pairs = list(itertools.combinations(aligned_runs, 2))
    if not pairs:
        return 1.0
    scores = [
        adjusted_rand_score(left.crisp_cluster, right.crisp_cluster)
        for left, right in pairs
    ]
    return float(np.mean(scores))


def _mean_membership_change(aligned_runs: list[FCMRunResult]) -> float:
    pairs = list(itertools.combinations(aligned_runs, 2))
    if not pairs:
        return 0.0
    changes = [
        float(np.mean(np.abs(left.membership - right.membership)))
        for left, right in pairs
    ]
    return float(np.mean(changes))


def _minimum_crisp_cluster_size(aligned_runs: list[FCMRunResult]) -> int:
    if not aligned_runs:
        return 0
    minimum_sizes = []
    for run in aligned_runs:
        counts = np.bincount(run.crisp_cluster, minlength=run.c)
        minimum_sizes.append(int(np.min(counts)))
    return int(min(minimum_sizes))


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    """Return mean/std, preserving non-finite penalties without NumPy warnings."""
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        return float("inf"), float("inf")
    return float(np.mean(array)), float(np.std(array, ddof=0))


def evaluate_stability(results: dict[tuple[int, float], list[FCMRunResult]]) -> list[ConfigurationSummary]:
    """Summarize validity and stability across seeds for each c/m configuration."""
    summaries: list[ConfigurationSummary] = []
    for (c, m), runs in results.items():
        representative_index, aligned_runs = select_representative_run(runs)
        centroids_stack = np.stack([run.centroids for run in aligned_runs])
        mean_centroids = np.mean(centroids_stack, axis=0)
        centroid_variation = float(
            np.mean([np.linalg.norm(run.centroids - mean_centroids) for run in aligned_runs])
        )
        mean_xb, std_xb = _mean_and_std([run.xie_beni for run in runs])
        min_cluster_size = _minimum_crisp_cluster_size(aligned_runs)

        summaries.append(
            ConfigurationSummary(
                c=c,
                m=m,
                runs=runs,
                representative_seed=aligned_runs[representative_index].seed,
                representative_index=representative_index,
                convergence_rate=float(np.mean([run.converged for run in runs])),
                mean_xie_beni=mean_xb,
                std_xie_beni=std_xb,
                mean_partition_coefficient=float(np.mean([run.partition_coefficient for run in runs])),
                std_partition_coefficient=float(np.std([run.partition_coefficient for run in runs], ddof=0)),
                mean_modified_partition_coefficient=float(
                    np.mean([run.modified_partition_coefficient for run in runs])
                ),
                std_modified_partition_coefficient=float(
                    np.std([run.modified_partition_coefficient for run in runs], ddof=0)
                ),
                mean_partition_entropy=float(np.mean([run.partition_entropy for run in runs])),
                std_partition_entropy=float(np.std([run.partition_entropy for run in runs], ddof=0)),
                mean_final_objective=float(np.mean([run.final_objective for run in runs])),
                std_final_objective=float(np.std([run.final_objective for run in runs], ddof=0)),
                mean_minimum_centroid_distance=float(
                    np.mean([run.minimum_centroid_distance for run in runs])
                ),
                centroid_variation=centroid_variation,
                mean_pairwise_ari=_mean_pairwise_ari(aligned_runs),
                mean_membership_change=_mean_membership_change(aligned_runs),
                minimum_crisp_cluster_size=min_cluster_size,
                empty_crisp_clusters=int(min_cluster_size == 0),
                aligned_runs=aligned_runs,
            )
        )
    return summaries


def configuration_summaries_to_frame(summaries: list[ConfigurationSummary]) -> pd.DataFrame:
    """Build one-row-per-configuration diagnostics for ranking and reporting."""
    return pd.DataFrame(
        [
            {
                "c": summary.c,
                "m": summary.m,
                "mean_xie_beni": summary.mean_xie_beni,
                "mean_partition_coefficient": summary.mean_partition_coefficient,
                "mean_modified_partition_coefficient": summary.mean_modified_partition_coefficient,
                "mean_partition_entropy": summary.mean_partition_entropy,
                "mean_pairwise_ari": summary.mean_pairwise_ari,
                "mean_membership_change": summary.mean_membership_change,
                "centroid_variation": summary.centroid_variation,
                "convergence_rate": summary.convergence_rate,
                "mean_minimum_centroid_distance": summary.mean_minimum_centroid_distance,
                "minimum_centroid_distance": summary.mean_minimum_centroid_distance,
                "minimum_crisp_cluster_size": summary.minimum_crisp_cluster_size,
                "empty_crisp_clusters": summary.empty_crisp_clusters,
            }
            for summary in summaries
        ]
    )


def _rank_to_unit_score(series: pd.Series, ascending: bool) -> pd.Series:
    """Return 0 for the best rank and 1 for the worst rank."""
    ranks = series.rank(method="min", ascending=ascending)
    if len(ranks) <= 1:
        return pd.Series(0.0, index=series.index)
    return (ranks - 1) / (len(ranks) - 1)


def _add_metric_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["score_xie_beni"] = _rank_to_unit_score(scored["mean_xie_beni"], ascending=True)
    scored["score_centroid_distance"] = _rank_to_unit_score(
        scored["mean_minimum_centroid_distance"], ascending=False
    )
    scored["score_mpc"] = _rank_to_unit_score(
        scored["mean_modified_partition_coefficient"], ascending=False
    )
    scored["score_partition_entropy"] = _rank_to_unit_score(
        scored["mean_partition_entropy"], ascending=True
    )
    scored["score_pairwise_ari"] = _rank_to_unit_score(scored["mean_pairwise_ari"], ascending=False)
    scored["score_membership_change"] = _rank_to_unit_score(
        scored["mean_membership_change"], ascending=True
    )
    scored["score_centroid_variation"] = _rank_to_unit_score(
        scored["centroid_variation"], ascending=True
    )
    scored["score_convergence"] = _rank_to_unit_score(scored["convergence_rate"], ascending=False)
    scored["score_empty_clusters"] = scored["empty_crisp_clusters"].astype(float)
    scored["score_centroid_collision"] = (
        scored["mean_minimum_centroid_distance"] <= MIN_CENTROID_DISTANCE_EPS
    ).astype(float)
    scored["hard_diagnostic_penalty"] = (
        (scored["convergence_rate"] < 1.0).astype(float)
        + scored["score_empty_clusters"]
        + scored["score_centroid_collision"]
    )

    scored["structure_score"] = scored[["score_xie_beni", "score_centroid_distance"]].mean(axis=1)
    scored["fuzzy_quality_score"] = scored[["score_mpc", "score_partition_entropy"]].mean(axis=1)
    scored["stability_score"] = scored[
        ["score_pairwise_ari", "score_membership_change", "score_centroid_variation"]
    ].mean(axis=1)
    scored["diagnostic_score"] = scored[
        ["score_convergence", "score_empty_clusters", "score_centroid_collision"]
    ].mean(axis=1)
    return scored


def rank_fcm_configurations(
    summaries: list[ConfigurationSummary],
    scheme_name: str = "balanced",
) -> pd.DataFrame:
    """Rank configurations using grouped metrics to avoid double-counting fuzzy crispness."""
    if scheme_name not in RANKING_SCHEMES:
        raise ValueError(f"Unknown ranking scheme: {scheme_name}")

    df = _add_metric_scores(configuration_summaries_to_frame(summaries))
    weights = RANKING_SCHEMES[scheme_name]
    df["composite_rank_score"] = (
        weights["structure"] * df["structure_score"]
        + weights["stability"] * df["stability_score"]
        + weights["fuzzy_quality"] * df["fuzzy_quality_score"]
        + weights["diagnostic"] * df["diagnostic_score"]
        + df["hard_diagnostic_penalty"]
    )

    ranked = df.sort_values(
        [
            "composite_rank_score",
            "diagnostic_score",
            "structure_score",
            "stability_score",
            "mean_pairwise_ari",
            "mean_xie_beni",
            "c",
            "m",
        ],
        ascending=[True, True, True, True, False, True, True, True],
    ).reset_index(drop=True)
    ranked["selection_order"] = np.arange(1, len(ranked) + 1)
    ranked[f"rank_{scheme_name}"] = ranked["selection_order"]
    return ranked


def build_ranking_sensitivity(summaries: list[ConfigurationSummary]) -> pd.DataFrame:
    """Create ranking results for all weighting schemes."""
    base = _add_metric_scores(configuration_summaries_to_frame(summaries))
    rank_frames = []
    for scheme_name in RANKING_SCHEMES:
        ranked = rank_fcm_configurations(summaries, scheme_name)
        rank_frames.append(ranked[["c", "m", f"rank_{scheme_name}", "composite_rank_score"]].rename(
            columns={"composite_rank_score": f"score_{scheme_name}"}
        ))

    sensitivity = base
    for ranks in rank_frames:
        sensitivity = sensitivity.merge(ranks, on=["c", "m"], how="left")

    rank_columns = [f"rank_{name}" for name in RANKING_SCHEMES]
    sensitivity["selection_consistency_count"] = (sensitivity[rank_columns] <= 3).sum(axis=1)
    sensitivity["selection_consistency"] = sensitivity["selection_consistency_count"].map(
        lambda count: f"Top-3 in {int(count)} of {len(rank_columns)} weighting schemes"
    )
    return sensitivity.sort_values(["rank_balanced", "c", "m"]).reset_index(drop=True)


def save_configuration_rankings(
    summaries: list[ConfigurationSummary],
    summary_path: Path = CONFIGURATION_SUMMARY_PATH,
    sensitivity_path: Path = RANKING_SENSITIVITY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persist transparent configuration diagnostics and sensitivity rankings."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    sensitivity = build_ranking_sensitivity(summaries)
    summary_columns = [
        "c",
        "m",
        "mean_xie_beni",
        "mean_partition_coefficient",
        "mean_modified_partition_coefficient",
        "mean_partition_entropy",
        "mean_pairwise_ari",
        "mean_membership_change",
        "centroid_variation",
        "convergence_rate",
        "minimum_centroid_distance",
        "minimum_crisp_cluster_size",
        "empty_crisp_clusters",
        "structure_score",
        "stability_score",
        "fuzzy_quality_score",
        "diagnostic_score",
        "rank_balanced",
        "rank_validity_focused",
        "rank_stability_focused",
        "rank_fuzzy_quality_focused",
        "selection_consistency",
    ]
    sensitivity.loc[:, summary_columns].to_csv(summary_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    return sensitivity.loc[:, summary_columns], sensitivity


def select_best_configuration(summaries: list[ConfigurationSummary]) -> ConfigurationSummary:
    """Choose the best FCM configuration from balanced grouped ranking."""
    ranking = build_ranking_sensitivity(summaries)
    score_lookup = {
        (int(row.c), float(row.m)): float(row.score_balanced)
        for row in ranking.itertuples(index=False)
    }
    consistency_lookup = {
        (int(row.c), float(row.m)): str(row.selection_consistency)
        for row in ranking.itertuples(index=False)
    }
    for summary in summaries:
        summary.composite_rank_score = score_lookup[(summary.c, summary.m)]
        summary.selection_consistency = consistency_lookup[(summary.c, summary.m)]

    best_row = ranking.sort_values(
        [
            "rank_balanced",
            "empty_crisp_clusters",
            "mean_minimum_centroid_distance",
            "selection_consistency_count",
            "mean_pairwise_ari",
        ],
        ascending=[True, True, False, False, False],
    ).iloc[0]
    best = next(
        summary
        for summary in summaries
        if summary.c == int(best_row["c"]) and summary.m == float(best_row["m"])
    )
    return best


def save_experiment_results(
    results: dict[tuple[int, float], list[FCMRunResult]],
    output_path: Path = EXPERIMENT_RESULTS_PATH,
) -> None:
    """Save one-row-per-run experiment diagnostics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for runs in results.values():
        for run in runs:
            records.append(
                {
                    "c": run.c,
                    "m": run.m,
                    "seed": run.seed,
                    "converged": run.converged,
                    "iterations": run.iterations,
                    "final_objective": run.final_objective,
                    "xie_beni": run.xie_beni,
                    "partition_coefficient": run.partition_coefficient,
                    "modified_partition_coefficient": run.modified_partition_coefficient,
                    "partition_entropy": run.partition_entropy,
                    "minimum_centroid_distance": run.minimum_centroid_distance,
                }
            )
    pd.DataFrame(records).sort_values(["c", "m", "seed"]).to_csv(output_path, index=False)


def _summary_to_dict(summary: ConfigurationSummary) -> dict[str, float | int]:
    return {
        "c": summary.c,
        "m": summary.m,
        "representative_seed": summary.representative_seed,
        "convergence_rate": summary.convergence_rate,
        "mean_xie_beni": summary.mean_xie_beni,
        "std_xie_beni": summary.std_xie_beni,
        "mean_partition_coefficient": summary.mean_partition_coefficient,
        "std_partition_coefficient": summary.std_partition_coefficient,
        "mean_modified_partition_coefficient": summary.mean_modified_partition_coefficient,
        "std_modified_partition_coefficient": summary.std_modified_partition_coefficient,
        "mean_partition_entropy": summary.mean_partition_entropy,
        "std_partition_entropy": summary.std_partition_entropy,
        "mean_final_objective": summary.mean_final_objective,
        "std_final_objective": summary.std_final_objective,
        "mean_minimum_centroid_distance": summary.mean_minimum_centroid_distance,
        "centroid_variation": summary.centroid_variation,
        "mean_pairwise_ari": summary.mean_pairwise_ari,
        "mean_membership_change": summary.mean_membership_change,
        "minimum_crisp_cluster_size": summary.minimum_crisp_cluster_size,
        "empty_crisp_clusters": summary.empty_crisp_clusters,
        "composite_rank_score": summary.composite_rank_score or float("nan"),
    }


def _validate_final_outputs(
    input_df: pd.DataFrame,
    centroids: np.ndarray,
    membership: np.ndarray,
    best_c: int,
) -> None:
    if membership.shape[0] != len(input_df):
        raise ValueError("Membership row count does not match province count.")
    if not np.allclose(membership.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Membership rows do not sum to 1 within tolerance.")
    if not np.isfinite(membership).all() or not np.isfinite(centroids).all():
        raise ValueError("Final membership or centroids contain NaN/infinite values.")
    if ((membership < -1e-8) | (membership > 1 + 1e-8)).any():
        raise ValueError("Membership values must be between 0 and 1.")
    if centroids.shape != (best_c, len(FEATURE_COLUMNS)):
        raise ValueError("Centroid matrix shape is inconsistent with best_c and feature count.")


def save_best_model_outputs(
    input_df: pd.DataFrame,
    summaries: list[ConfigurationSummary],
    best_summary: ConfigurationSummary,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save JSON, final standardized centroids, and final province membership."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    representative = best_summary.aligned_runs[best_summary.representative_index]
    _validate_final_outputs(input_df, representative.centroids, representative.membership, best_summary.c)

    centroid_df = pd.DataFrame(representative.centroids, columns=FEATURE_COLUMNS)
    centroid_df.insert(0, "cluster", [f"cluster_{idx}" for idx in range(1, best_summary.c + 1)])
    centroid_df.to_csv(CENTROIDS_PATH, index=False, float_format="%.6f")

    membership_columns = [
        f"membership_cluster_{idx}" for idx in range(1, best_summary.c + 1)
    ]
    membership_df = pd.DataFrame(representative.membership, columns=membership_columns)
    sorted_membership = np.sort(representative.membership, axis=1)
    membership_df.insert(0, "province_name", input_df["province_name"].to_numpy())
    membership_df["maximum_membership"] = sorted_membership[:, -1]
    membership_df["second_highest_membership"] = sorted_membership[:, -2]
    membership_df["membership_margin"] = (
        membership_df["maximum_membership"] - membership_df["second_highest_membership"]
    )
    membership_df["crisp_cluster"] = np.argmax(representative.membership, axis=1) + 1
    membership_df.to_csv(MEMBERSHIP_PATH, index=False, float_format="%.6f")

    configuration_summary, ranking_sensitivity = save_configuration_rankings(summaries)
    ranking = ranking_sensitivity.sort_values("rank_balanced").head(5).to_dict(orient="records")
    best_row = ranking_sensitivity[
        (ranking_sensitivity["c"] == best_summary.c)
        & (ranking_sensitivity["m"] == best_summary.m)
    ].iloc[0]
    best_parameters = {
        "best_c": best_summary.c,
        "best_m": best_summary.m,
        "representative_seed": best_summary.representative_seed,
        "error_tolerance": ERROR_TOLERANCE,
        "max_iterations": MAX_ITERATIONS,
        "number_of_initializations": len(RANDOM_SEEDS),
        "feature_columns": FEATURE_COLUMNS,
        "selection_method": "Grouped weighted ranking with sensitivity analysis across weighting schemes.",
        "metric_groups": {
            "structure": ["Xie-Beni (lower is better)", "minimum centroid distance (higher is better)"],
            "fuzzy_quality": [
                "Modified Partition Coefficient (higher is better)",
                "Partition Entropy (lower is better)",
                "Partition Coefficient retained as a diagnostic, not a full independent vote.",
            ],
            "stability": [
                "mean pairwise ARI (higher is better)",
                "mean membership change (lower is better)",
                "centroid variation (lower is better)",
            ],
            "diagnostic": [
                "convergence rate",
                "empty crisp clusters",
                "near-colliding centroids",
            ],
        },
        "weighting_scheme": RANKING_SCHEMES["balanced"],
        "ranking_sensitivity": {
            scheme: {
                "rank": int(best_row[f"rank_{scheme}"]),
                "score": float(best_row[f"score_{scheme}"]),
            }
            for scheme in RANKING_SCHEMES
        },
        "selection_consistency": best_summary.selection_consistency,
        "selection_rationale": (
            f"Selected c={best_summary.c}, m={best_summary.m} from the balanced grouped ranking "
            f"(score {best_summary.composite_rank_score:.3f}). The selected configuration "
            f"had convergence rate {best_summary.convergence_rate:.3f}, minimum crisp cluster "
            f"size {best_summary.minimum_crisp_cluster_size}, mean pairwise ARI "
            f"{best_summary.mean_pairwise_ari:.3f}, and no centroid collision. "
            "PC, MPC, and PE were not counted as three independent full-weight votes; "
            "MPC and PE form the fuzzy-quality group while PC remains diagnostic. "
            "Cluster numbers remain arbitrary and are not substantive risk labels."
        ),
        "validity_summary": _summary_to_dict(best_summary),
        "stability_summary": {
            "mean_pairwise_ari": best_summary.mean_pairwise_ari,
            "mean_membership_change": best_summary.mean_membership_change,
            "centroid_variation": best_summary.centroid_variation,
            "convergence_rate": best_summary.convergence_rate,
        },
        "alternative_top_configurations": ranking,
        "top_candidate_configurations": ranking,
    }
    with BEST_PARAMETERS_PATH.open("w", encoding="utf-8") as f:
        json.dump(best_parameters, f, indent=2)

    for path in [
        EXPERIMENT_RESULTS_PATH,
        CONFIGURATION_SUMMARY_PATH,
        RANKING_SENSITIVITY_PATH,
        BEST_PARAMETERS_PATH,
        CENTROIDS_PATH,
        MEMBERSHIP_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Expected output file was not created: {path}")

    return centroid_df, membership_df


def run_model_pipeline(input_path: Path = INPUT_PATH) -> ConfigurationSummary:
    """Run FCM experiments, rank configurations, and persist model artifacts."""
    input_df, x = load_and_validate_data(input_path=input_path)
    results = run_fcm_experiments(x)
    summaries = evaluate_stability(results)
    best_summary = select_best_configuration(summaries)
    save_experiment_results(results)
    save_best_model_outputs(input_df, summaries, best_summary)
    return best_summary


def main() -> int:
    """CLI entry point for reproducible FCM modeling experiments."""
    try:
        best_summary = run_model_pipeline()
        print(f"Best number of clusters: {best_summary.c}")
        print(f"Best fuzziness exponent: {best_summary.m}")
        print(f"Representative seed: {best_summary.representative_seed}")
        print(f"Mean Xie-Beni: {best_summary.mean_xie_beni:.6f}")
        print(f"Mean Partition Coefficient: {best_summary.mean_partition_coefficient:.6f}")
        print(f"Mean Modified Partition Coefficient: {best_summary.mean_modified_partition_coefficient:.6f}")
        print(f"Mean Partition Entropy: {best_summary.mean_partition_entropy:.6f}")
        print(f"Stability score: {best_summary.mean_pairwise_ari:.6f}")
        print(f"Selection consistency: {best_summary.selection_consistency}")
        print("Output files:")
        for path in [
            EXPERIMENT_RESULTS_PATH,
            CONFIGURATION_SUMMARY_PATH,
            RANKING_SENSITIVITY_PATH,
            BEST_PARAMETERS_PATH,
            CENTROIDS_PATH,
            MEMBERSHIP_PATH,
        ]:
            print(f"- {path.relative_to(PROJECT_ROOT)}")
        return 0
    except Exception:
        logger.exception("FCM modeling pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
