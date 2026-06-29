from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_input_file(path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"Input file not found: {resolved}")
    return resolved


def validate_expected_outputs(skip_mapping: bool) -> None:
    required = [
        PROJECT_ROOT / "outputs" / "model" / "fcm_experiment_results.csv",
        PROJECT_ROOT / "outputs" / "model" / "fcm_configuration_summary.csv",
        PROJECT_ROOT / "outputs" / "model" / "fcm_ranking_sensitivity.csv",
        PROJECT_ROOT / "outputs" / "model" / "best_fcm_parameters.json",
        PROJECT_ROOT / "outputs" / "model" / "cluster_centroids_standardized.csv",
        PROJECT_ROOT / "outputs" / "model" / "cluster_membership.csv",
        PROJECT_ROOT / "outputs" / "analysis" / "province_membership_analysis.csv",
        PROJECT_ROOT / "outputs" / "analysis" / "ambiguous_provinces.csv",
        PROJECT_ROOT / "outputs" / "analysis" / "cluster_profiles.csv",
        PROJECT_ROOT / "outputs" / "analysis" / "dominant_factors.csv",
        PROJECT_ROOT / "outputs" / "analysis" / "external_validation.csv",
        PROJECT_ROOT / "outputs" / "figures" / "fcm_validity_plot.png",
        PROJECT_ROOT / "outputs" / "figures" / "centroid_heatmap.png",
    ]
    if not skip_mapping:
        required.extend(
            [
                PROJECT_ROOT / "outputs" / "maps" / "fcm_cluster_map.png",
                PROJECT_ROOT / "outputs" / "maps" / "fcm_cluster_map.pdf",
                PROJECT_ROOT / "outputs" / "maps" / "membership_certainty_map.png",
                PROJECT_ROOT / "outputs" / "maps" / "membership_certainty_map.pdf",
            ]
        )

    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Expected pipeline outputs were not created: {missing}")


def run_pipeline(input_path: Path, skip_mapping: bool = False, skip_preprocessing: bool = False) -> None:
    preprocessing = load_module("preprocessing_pipeline", PROJECT_ROOT / "src" / "preprocessing.py")
    fcm_model = load_module("fcm_model_pipeline", PROJECT_ROOT / "src" / "04_fcm_model.py")
    validation = load_module(
        "validation_robustness_pipeline",
        PROJECT_ROOT / "src" / "05_validation_robustness_analysis.py",
    )
    visualization = load_module("visualization_pipeline", PROJECT_ROOT / "src" / "07_visualization.py")

    if not skip_preprocessing:
        logger.info("Step 1/8: preprocessing input data")
        preprocessing.run_preprocessing_pipeline()
    else:
        logger.info("Step 1/8: preprocessing skipped by user")

    model_input = validate_input_file(input_path)
    logger.info("Step 2/8: running FCM experiments and model ranking")
    best_summary = fcm_model.run_model_pipeline(input_path=model_input)
    logger.info("Selected FCM configuration: c=%s, m=%s", best_summary.c, best_summary.m)

    logger.info("Step 3/8: cluster profiles, membership ambiguity, and external validation")
    validation.run_validation_analysis()

    logger.info("Step 4/8: visualizations")
    visualization.run_visualizations()

    if skip_mapping:
        logger.info("Step 5/8: spatial mapping skipped by user")
    else:
        spatial_path = PROJECT_ROOT / "data" / "external" / "indonesia_38_provinces.geojson"
        if not spatial_path.exists():
            logger.warning("Spatial file missing; mapping skipped safely: %s", spatial_path)
        else:
            logger.info("Step 5/8: spatial mapping")
            spatial = load_module("spatial_mapping_pipeline", PROJECT_ROOT / "src" / "06_spatial_mapping.py")
            spatial.run_spatial_mapping()

    logger.info("Step 6/8: validating expected outputs")
    validate_expected_outputs(skip_mapping=skip_mapping)
    logger.info("Pipeline completed successfully")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full stunting-risk FCM pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/fcm_model_matrix_zscore.csv"),
        help="Path to standardized FCM input matrix.",
    )
    parser.add_argument(
        "--skip-mapping",
        action="store_true",
        help="Skip GeoJSON spatial mapping even if spatial data is available.",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Use existing processed input instead of rebuilding it from raw files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_pipeline(
            input_path=args.input,
            skip_mapping=args.skip_mapping,
            skip_preprocessing=args.skip_preprocessing,
        )
        return 0
    except Exception:
        logger.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
