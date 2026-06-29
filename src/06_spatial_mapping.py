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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEOJSON_PATH = PROJECT_ROOT / "data" / "external" / "indonesia_38_provinces.geojson"
MEMBERSHIP_PATH = PROJECT_ROOT / "outputs" / "model" / "cluster_membership.csv"
ANALYSIS_PATH = PROJECT_ROOT / "outputs" / "analysis" / "province_membership_analysis.csv"
MAPPING_OUT = PROJECT_ROOT / "data" / "interim" / "province_name_mapping.csv"
HARMONIZED_OUT = PROJECT_ROOT / "data" / "interim" / "fcm_membership_geojson.csv"
MAP_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "maps"
LEGACY_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

CLUSTER_MAP_PNG = MAP_OUTPUT_DIR / "fcm_cluster_map.png"
CLUSTER_MAP_PDF = MAP_OUTPUT_DIR / "fcm_cluster_map.pdf"
CERTAINTY_MAP_PNG = MAP_OUTPUT_DIR / "membership_certainty_map.png"
CERTAINTY_MAP_PDF = MAP_OUTPUT_DIR / "membership_certainty_map.pdf"

LEGACY_CLUSTER_MAP = LEGACY_FIGURE_DIR / "peta_klaster_provinsi.png"
LEGACY_CERTAINTY_MAP = LEGACY_FIGURE_DIR / "peta_kepastian_membership.png"

SPECIAL_MAPPING: dict[str, str] = {
    "BANGKA BELITUNG": "Kepulauan Bangka Belitung",
    "DI YOGYAKARTA": "Daerah Istimewa Yogyakarta",
    "DKI JAKARTA": "DKI Jakarta",
}

NO_DATA_COLOR = "#CCCCCC"
EDGE_COLOR_MAP = "#FFFFFF"
EDGE_LW_MAP = 0.5
CLUSTER_PALETTE = [
    "#2563EB",
    "#DC2626",
    "#16A34A",
    "#9333EA",
    "#F59E0B",
    "#0891B2",
    "#DB2777",
]
STATUS_STYLE = {
    "Ambigu tinggi": dict(edgecolor="#111111", linewidth=2.0, linestyle="--"),
    "Transisi moderat": dict(edgecolor="#555555", linewidth=1.2, linestyle="--"),
    "Keanggotaan kuat": dict(edgecolor="#FFFFFF", linewidth=0.4, linestyle="-"),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _import_geopandas():
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for spatial mapping. Install requirements or run with --skip-mapping."
        ) from exc
    return gpd


def load_geojson_names(path: Path = GEOJSON_PATH) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Spatial boundary file not found: {path}")
    with path.open(encoding="utf-8") as f:
        gj = json.load(f)
    return sorted(feature["properties"]["PROVINSI"] for feature in gj["features"])


def caps_to_geojson(caps_name: str, geo_set: set[str]) -> str | None:
    """Convert uppercase FCM province names to the names used by GeoJSON."""
    caps_name = str(caps_name).strip().upper()
    if caps_name in SPECIAL_MAPPING:
        return SPECIAL_MAPPING[caps_name]
    title = caps_name.title()
    if title in geo_set:
        return title
    return None


def make_rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, float(np.clip(alpha, 0.0, 1.0)))


def cluster_colors(cluster_ids: list[int]) -> dict[int, str]:
    return {
        cluster_id: CLUSTER_PALETTE[index % len(CLUSTER_PALETTE)]
        for index, cluster_id in enumerate(sorted(cluster_ids))
    }


def build_harmonized_membership(
    geojson_path: Path = GEOJSON_PATH,
    membership_path: Path = MEMBERSHIP_PATH,
    analysis_path: Path = ANALYSIS_PATH,
    mapping_out: Path = MAPPING_OUT,
    harmonized_out: Path = HARMONIZED_OUT,
) -> pd.DataFrame:
    """Create membership table with GeoJSON province names for spatial joins."""
    geo_names = load_geojson_names(geojson_path)
    geo_set = set(geo_names)
    membership = pd.read_csv(membership_path)
    analysis = pd.read_csv(analysis_path)

    rows = []
    for province in membership["province_name"]:
        mapped = caps_to_geojson(province, geo_set)
        if mapped is None:
            status = f"WARNING: cannot map {province!r}"
        else:
            method = "special" if str(province).strip().upper() in SPECIAL_MAPPING else "titlecase"
            status = f"OK ({method})"
        rows.append(
            {
                "province_name_fcm": province,
                "province_name_geojson": mapped,
                "status": status,
            }
        )

    mapping_df = pd.DataFrame(rows)
    mapping_out.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(mapping_out, index=False)

    slim_columns = [
        column
        for column in [
            "province_name",
            "risk_rank",
            "membership_status",
            "stunting_prevalence_pct",
            "cluster_label",
        ]
        if column in analysis.columns
    ]
    harmonized = membership.copy()
    harmonized["province_name_geojson"] = harmonized["province_name"].map(
        lambda value: caps_to_geojson(value, geo_set)
    )
    harmonized = harmonized.merge(analysis[slim_columns], on="province_name", how="left")
    harmonized_out.parent.mkdir(parents=True, exist_ok=True)
    harmonized.to_csv(harmonized_out, index=False)

    warnings = mapping_df[mapping_df["status"].str.startswith("WARNING")]
    if not warnings.empty:
        logger.warning("Unmatched FCM provinces: %s", warnings["province_name_fcm"].tolist())

    missing_geo = sorted(geo_set - set(mapping_df["province_name_geojson"].dropna()))
    if missing_geo:
        logger.info("GeoJSON provinces without FCM data: %s", missing_geo)
    return harmonized


def load_spatial_joined(
    geojson_path: Path = GEOJSON_PATH,
    harmonized_path: Path = HARMONIZED_OUT,
):
    """Read GeoJSON, validate CRS, and join harmonized FCM membership."""
    gpd = _import_geopandas()
    if not geojson_path.exists():
        raise FileNotFoundError(f"Spatial boundary file not found: {geojson_path}")
    if not harmonized_path.exists():
        build_harmonized_membership(geojson_path=geojson_path, harmonized_out=harmonized_path)

    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        logger.warning("GeoJSON CRS is missing; assuming EPSG:4326 for plotting.")
        gdf = gdf.set_crs(epsg=4326)

    df = pd.read_csv(harmonized_path)
    joined = gdf.merge(df, left_on="PROVINSI", right_on="province_name_geojson", how="left")
    no_data = joined[joined["crisp_cluster"].isna()]
    if not no_data.empty:
        logger.info("Map regions without FCM data: %s", no_data["PROVINSI"].tolist())
    return joined, df


def plot_cluster_map(
    output_png: Path = CLUSTER_MAP_PNG,
    output_pdf: Path = CLUSTER_MAP_PDF,
    legacy_output: Path = LEGACY_CLUSTER_MAP,
) -> None:
    joined, df = load_spatial_joined()
    cluster_ids = sorted(int(value) for value in df["crisp_cluster"].dropna().unique())
    colors = cluster_colors(cluster_ids)
    joined["plot_color"] = joined["crisp_cluster"].map(colors).fillna(NO_DATA_COLOR)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    fig.patch.set_facecolor("white")
    joined.plot(color=joined["plot_color"], edgecolor=EDGE_COLOR_MAP, linewidth=EDGE_LW_MAP, ax=ax)

    n_by_cluster = df.groupby("crisp_cluster").size()
    label_by_cluster = (
        df.dropna(subset=["crisp_cluster"])
        .drop_duplicates("crisp_cluster")
        .set_index("crisp_cluster")["cluster_label"]
        .to_dict()
    )
    handles = []
    for cluster_id in cluster_ids:
        label = label_by_cluster.get(cluster_id, f"Klaster {cluster_id}")
        n = int(n_by_cluster.get(cluster_id, 0))
        handles.append(mpatches.Patch(color=colors[cluster_id], label=f"Klaster {cluster_id}: {label}\n({n} provinsi)"))
    handles.append(mpatches.Patch(color=NO_DATA_COLOR, label="Tidak ada data FCM"))

    ax.legend(handles=handles, loc="lower left", fontsize=8.5, frameon=True, framealpha=0.92)
    ax.set_title("Peta Persebaran Klaster Risiko Stunting per Provinsi", fontsize=14, weight="bold", pad=14)
    ax.axis("off")
    fig.tight_layout(pad=1.5)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    legacy_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(legacy_output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_membership_certainty_map(
    output_png: Path = CERTAINTY_MAP_PNG,
    output_pdf: Path = CERTAINTY_MAP_PDF,
    legacy_output: Path = LEGACY_CERTAINTY_MAP,
    certainty_metric: str = "maximum_membership",
) -> None:
    joined, df = load_spatial_joined()
    if certainty_metric not in {"maximum_membership", "membership_margin"}:
        raise ValueError("certainty_metric must be 'maximum_membership' or 'membership_margin'.")

    cluster_ids = sorted(int(value) for value in df["crisp_cluster"].dropna().unique())
    colors = cluster_colors(cluster_ids)

    gpd = _import_geopandas()
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    fig.patch.set_facecolor("white")

    for _, row in joined.iterrows():
        geom = gpd.GeoSeries([row.geometry], crs=joined.crs)
        if pd.isna(row["crisp_cluster"]):
            geom.plot(ax=ax, color=NO_DATA_COLOR, edgecolor="white", linewidth=0.4)
            continue

        base_hex = colors[int(row["crisp_cluster"])]
        certainty = float(row[certainty_metric])
        alpha = 0.30 + 0.70 * certainty
        status = str(row["membership_status"]) if not pd.isna(row["membership_status"]) else "Keanggotaan kuat"
        style = STATUS_STYLE.get(status, STATUS_STYLE["Keanggotaan kuat"])
        geom.plot(ax=ax, color=make_rgba(base_hex, alpha), **style)

    cluster_handles = [
        mpatches.Patch(color=colors[cluster_id], label=f"Klaster {cluster_id}")
        for cluster_id in cluster_ids
    ]
    status_handles = [
        Line2D([0], [0], color="#111111", linewidth=2.0, linestyle="--", label="Ambigu tinggi"),
        Line2D([0], [0], color="#555555", linewidth=1.2, linestyle="--", label="Transisi moderat"),
        Line2D([0], [0], color="#AAAAAA", linewidth=0.4, linestyle="-", label="Keanggotaan kuat"),
        mpatches.Patch(color=NO_DATA_COLOR, label="Tidak ada data FCM"),
    ]
    ax.legend(
        handles=cluster_handles + status_handles,
        loc="lower left",
        fontsize=8.5,
        frameon=True,
        framealpha=0.92,
        title="Klaster & Status Membership",
    )
    ax.set_title(
        "Peta Kepastian Membership FCM per Provinsi\n"
        f"Opacity berdasarkan {certainty_metric}",
        fontsize=13,
        weight="bold",
        pad=14,
    )
    ax.axis("off")
    fig.tight_layout(pad=1.5)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    legacy_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(legacy_output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_spatial_mapping() -> None:
    """Build harmonized spatial input and write publication-resolution maps."""
    build_harmonized_membership()
    plot_cluster_map()
    plot_membership_certainty_map()


def main() -> int:
    try:
        run_spatial_mapping()
        print(f"Spatial maps written to: {MAP_OUTPUT_DIR}")
        return 0
    except Exception as exc:
        logger.error("Spatial mapping failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
