from __future__ import annotations

import json
import logging
import os
from pathlib import Path

TEMP_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "pkk-stunting-fcm-cache"
os.environ.setdefault("MPLCONFIGDIR", str(TEMP_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TEMP_CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CENTROIDS_PATH = PROJECT_ROOT / "outputs" / "model" / "cluster_centroids_standardized.csv"
CONFIGURATION_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "model" / "fcm_configuration_summary.csv"
EXPERIMENT_RESULTS_PATH = PROJECT_ROOT / "outputs" / "model" / "fcm_experiment_results.csv"
BEST_PARAMETERS_PATH = PROJECT_ROOT / "outputs" / "model" / "best_fcm_parameters.json"
CLUSTER_PROFILES_PATH = PROJECT_ROOT / "outputs" / "analysis" / "cluster_profiles.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

VALIDITY_PLOT_PATH = FIGURE_DIR / "fcm_validity_plot.png"
CENTROID_HEATMAP_PATH = FIGURE_DIR / "centroid_heatmap.png"
LEGACY_VALIDITY_PLOT_PATH = FIGURE_DIR / "grafik_indeks_validitas.png"
LEGACY_CENTROID_HEATMAP_PATH = FIGURE_DIR / "heatmap_centroid.png"

VARIABLE_LABELS: dict[str, str] = {
    "maternal_age_risk_z": "Risiko Usia\nIbu Hamil",
    "low_knowledge_z": "Rendahnya\nPengetahuan Stunting",
    "water_no_or_unimproved_z": "Air Minum\nTidak Layak / Tanpa Akses",
    "water_limited_z": "Air Minum\nAkses Terbatas",
    "sanitation_babs_z": "Sanitasi\nBABS",
    "sanitation_unimproved_z": "Sanitasi\nTidak Layak",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_best_parameters(path: Path = BEST_PARAMETERS_PATH) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _cluster_label_lookup() -> dict[str, str]:
    if not CLUSTER_PROFILES_PATH.exists():
        return {}
    profiles = pd.read_csv(CLUSTER_PROFILES_PATH)
    if "cluster" not in profiles.columns or "cluster_label" not in profiles.columns:
        return {}
    return {
        f"cluster_{int(row.cluster)}": f"Klaster {int(row.cluster)}\n{row.cluster_label}"
        for row in profiles.itertuples(index=False)
    }


def plot_centroid_heatmap(
    centroids_path: Path = CENTROIDS_PATH,
    output_path: Path = CENTROID_HEATMAP_PATH,
    legacy_output_path: Path = LEGACY_CENTROID_HEATMAP_PATH,
) -> None:
    df = pd.read_csv(centroids_path).set_index("cluster")
    cols = [column for column in VARIABLE_LABELS if column in df.columns]
    if not cols:
        raise ValueError("No known centroid feature columns found for heatmap.")

    plot_df = df[cols].rename(columns=VARIABLE_LABELS)
    label_lookup = _cluster_label_lookup()
    plot_df.index = [label_lookup.get(index, index) for index in plot_df.index]

    height = max(4.5, 1.0 + 0.75 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, height))
    vmax = max(abs(float(plot_df.values.min())), abs(float(plot_df.values.max())), 0.1)
    im = ax.imshow(plot_df.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns, fontsize=9)
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index, fontsize=9)

    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            value = plot_df.values[i, j]
            text_color = "white" if abs(value) > 0.55 * vmax else "black"
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=9, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Z-score centroid", fontsize=9)
    ax.set_title("Karakteristik Centroid Tiap Klaster FCM", fontsize=13, weight="bold", pad=12)
    fig.tight_layout(pad=1.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(legacy_output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _configuration_summary() -> pd.DataFrame:
    if CONFIGURATION_SUMMARY_PATH.exists():
        return pd.read_csv(CONFIGURATION_SUMMARY_PATH)

    experiment = pd.read_csv(EXPERIMENT_RESULTS_PATH)
    summary = (
        experiment.groupby(["c", "m"], as_index=False)
        .agg(
            mean_xie_beni=("xie_beni", "mean"),
            mean_partition_coefficient=("partition_coefficient", "mean"),
            mean_modified_partition_coefficient=("modified_partition_coefficient", "mean"),
            mean_partition_entropy=("partition_entropy", "mean"),
            convergence_rate=("converged", "mean"),
            minimum_centroid_distance=("minimum_centroid_distance", "mean"),
        )
    )
    return summary


def plot_validity_indices(
    output_path: Path = VALIDITY_PLOT_PATH,
    legacy_output_path: Path = LEGACY_VALIDITY_PLOT_PATH,
) -> None:
    df = _configuration_summary()
    best = load_best_parameters()
    best_c = int(best.get("best_c", df.sort_values("mean_xie_beni").iloc[0]["c"]))
    best_m = float(best.get("best_m", df.sort_values("mean_xie_beni").iloc[0]["m"]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Perbandingan Indeks Validitas dan Ranking FCM", fontsize=13, weight="bold", y=0.98)

    sub_m = df[df["m"] == best_m].sort_values("c")
    sub_c = df[df["c"] == best_c].sort_values("m")

    axes[0, 0].plot(sub_m["c"], sub_m["mean_xie_beni"], marker="o", color="#DC2626", lw=2)
    axes[0, 0].axvline(best_c, color="#111111", ls="--", lw=1.2, label=f"c={best_c} terpilih")
    axes[0, 0].set_title("Xie-Beni vs jumlah klaster\nlebih rendah lebih baik", fontsize=10)
    axes[0, 0].set_xlabel("Jumlah klaster (c)")
    axes[0, 0].set_ylabel("Mean Xie-Beni")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        sub_m["c"],
        sub_m["minimum_centroid_distance"],
        marker="o",
        color="#2563EB",
        lw=2,
    )
    axes[0, 1].axvline(best_c, color="#111111", ls="--", lw=1.2)
    axes[0, 1].set_title("Jarak centroid minimum\nlebih tinggi lebih baik", fontsize=10)
    axes[0, 1].set_xlabel("Jumlah klaster (c)")
    axes[0, 1].set_ylabel("Mean min. centroid distance")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(
        sub_c["m"],
        sub_c["mean_modified_partition_coefficient"],
        marker="s",
        color="#16A34A",
        lw=2,
        label="MPC",
    )
    axes[1, 0].plot(
        sub_c["m"],
        sub_c["mean_partition_entropy"],
        marker="s",
        color="#9333EA",
        lw=2,
        label="PE",
    )
    axes[1, 0].axvline(best_m, color="#111111", ls="--", lw=1.2, label=f"m={best_m:g} terpilih")
    axes[1, 0].set_title("Kualitas fuzzy vs fuzzifier pada c terpilih", fontsize=10)
    axes[1, 0].set_xlabel("Fuzziness exponent (m)")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    if "rank_balanced" in df.columns:
        pivot = df.pivot(index="m", columns="c", values="rank_balanced").sort_index()
        im = axes[1, 1].imshow(pivot.values, cmap="viridis_r", aspect="auto")
        axes[1, 1].set_xticks(range(len(pivot.columns)))
        axes[1, 1].set_xticklabels(pivot.columns)
        axes[1, 1].set_yticks(range(len(pivot.index)))
        axes[1, 1].set_yticklabels([f"{value:g}" for value in pivot.index])
        axes[1, 1].set_title("Rank balanced per kombinasi c dan m", fontsize=10)
        axes[1, 1].set_xlabel("Jumlah klaster (c)")
        axes[1, 1].set_ylabel("m")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                axes[1, 1].text(j, i, int(pivot.values[i, j]), ha="center", va="center", color="white")
        fig.colorbar(im, ax=axes[1, 1], shrink=0.85, label="Rank")
    else:
        axes[1, 1].plot(sub_c["m"], sub_c["mean_partition_coefficient"], marker="s", color="#2563EB", lw=2)
        axes[1, 1].set_title("Partition Coefficient vs m", fontsize=10)
        axes[1, 1].grid(alpha=0.25)

    fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(legacy_output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_visualizations() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_centroid_heatmap()
    plot_validity_indices()


def main() -> int:
    try:
        run_visualizations()
        print(f"Visualizations written to: {FIGURE_DIR}")
        return 0
    except Exception:
        logger.exception("Visualization pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
