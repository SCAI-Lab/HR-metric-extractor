#!/usr/bin/env python3
"""
Step 1 (ML): Build a subject-level training table by merging
batch-extracted features with ICF targets.

This script is intentionally model-agnostic and focuses only on data assembly.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _safe_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and col not in {"activity_idx", "resting_idx", "t_start", "t_end", "duration_sec"}
    ]


def _aggregate_metrics_file(metrics_path: Path, prefix: str) -> Dict[str, float]:
    if not metrics_path.exists():
        return {}

    df = pd.read_csv(metrics_path)
    if df.empty:
        return {}

    numeric_cols = _safe_numeric_columns(df)
    if not numeric_cols:
        return {}

    features: Dict[str, float] = {}
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        features[f"{prefix}_{col}_mean"] = float(values.mean())
        features[f"{prefix}_{col}_std"] = float(values.std(ddof=0))
        features[f"{prefix}_{col}_median"] = float(values.median())

    features[f"{prefix}_n_windows"] = float(len(df))
    return features


def _collect_subject_features(subject_dir: Path) -> Dict[str, float]:
    features: Dict[str, float] = {}

    # Core activity metrics
    features.update(_aggregate_metrics_file(subject_dir / "propulsion_hr_metrics.csv", "propulsion"))
    features.update(_aggregate_metrics_file(subject_dir / "resting_hr_metrics.csv", "resting"))

    # Extra activities: activity_<name>_hr_metrics.csv
    for file in sorted(subject_dir.glob("activity_*_hr_metrics.csv")):
        stem = file.stem
        # stem format: activity_<name>_hr_metrics
        activity_name = stem.removeprefix("activity_").removesuffix("_hr_metrics")
        prefix = f"custom_{activity_name}"
        features.update(_aggregate_metrics_file(file, prefix))

    return features


def find_latest_batch(batch_root: Path) -> Path:
    if not batch_root.exists():
        raise FileNotFoundError(f"Batch output root not found: {batch_root}")

    candidates = [d for d in batch_root.iterdir() if d.is_dir() and d.name.startswith("batch_")]
    if not candidates:
        raise FileNotFoundError(f"No batch directories found in {batch_root}")

    return sorted(candidates, key=lambda p: p.name)[-1]


def _parse_path_list(value: str | None) -> List[Path]:
    if value is None:
        return []
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def load_icf_targets(icf_csv: Path, id_col: str | int, target_cols: List[str]) -> pd.DataFrame:
    icf_df = pd.read_csv(icf_csv)

    if isinstance(id_col, int):
        if id_col < 0 or id_col >= len(icf_df.columns):
            raise ValueError(f"id_col index {id_col} is out of range for ICF CSV columns")
        subject_col = icf_df.columns[id_col]
    else:
        subject_col = id_col

    if subject_col not in icf_df.columns:
        raise ValueError(f"Subject ID column '{subject_col}' not found in ICF CSV")
    if len(target_cols) != 4:
        raise ValueError(f"Expected exactly 4 target columns, got {len(target_cols)}: {target_cols}")
    missing_targets = [target_col for target_col in target_cols if target_col not in icf_df.columns]
    if missing_targets:
        raise ValueError(f"Target column(s) not found in ICF CSV: {missing_targets}")

    targets = icf_df[[subject_col, *target_cols]].copy()
    targets.rename(columns={subject_col: "subject_id"}, inplace=True)
    targets["subject_id"] = (
        targets["subject_id"]
        .astype(str)
        .str.strip()
        .str.replace("^subj_", "sub_", regex=True)
        .str.replace("^Subject_", "sub_", regex=True)
    )
    for target_col in target_cols:
        targets[target_col] = pd.to_numeric(targets[target_col], errors="coerce")

    targets = targets.dropna(subset=["subject_id", *target_cols]).reset_index(drop=True)
    return targets


def load_combined_icf_targets(icf_csvs: Sequence[Path], id_col: str | int, target_cols: List[str]) -> pd.DataFrame:
    if len(icf_csvs) == 0:
        raise ValueError("At least one ICF CSV must be provided")

    frames = [load_icf_targets(icf_csv=path, id_col=id_col, target_cols=target_cols) for path in icf_csvs]
    combined = pd.concat(frames, axis=0, ignore_index=True)

    duplicate_subjects = combined[combined["subject_id"].duplicated(keep=False)]["subject_id"].unique().tolist()
    if duplicate_subjects:
        raise ValueError(
            "Duplicate subject_id values found across ICF CSVs. "
            f"Ensure subject IDs are unique across cohorts. Duplicates: {sorted(duplicate_subjects)}"
        )

    return combined.reset_index(drop=True)


def _collect_feature_rows(batch_dirs: Sequence[Path], target_subjects: set[str]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    seen_subjects: set[str] = set()

    for batch_dir in batch_dirs:
        if not batch_dir.exists():
            raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

        for subject_dir in sorted(batch_dir.glob("sub_*")):
            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name
            if subject_id not in target_subjects:
                continue

            if subject_id in seen_subjects:
                raise ValueError(
                    f"Duplicate subject directory '{subject_id}' found across batch dirs. "
                    "Ensure IDs are unique or build cohorts separately."
                )

            feature_row = _collect_subject_features(subject_dir)
            if not feature_row:
                logger.warning("No features found for %s (skipping)", subject_id)
                continue

            feature_row["subject_id"] = subject_id
            rows.append(feature_row)
            seen_subjects.add(subject_id)

    return rows


def _drop_sparse_feature_columns(
    merged: pd.DataFrame,
    target_cols: Sequence[str],
    min_feature_non_nan_ratio: float,
) -> tuple[pd.DataFrame, List[str]]:
    if not 0.0 <= min_feature_non_nan_ratio <= 1.0:
        raise ValueError(
            f"min_feature_non_nan_ratio must be in [0, 1], got {min_feature_non_nan_ratio}"
        )

    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [column for column in numeric_cols if column not in target_cols]
    if not feature_cols or min_feature_non_nan_ratio <= 0.0:
        return merged, []

    min_non_nan_count = int(np.ceil(min_feature_non_nan_ratio * len(merged)))
    keep_cols: List[str] = []
    drop_cols: List[str] = []

    for column in feature_cols:
        non_nan_count = int(merged[column].notna().sum())
        if non_nan_count >= min_non_nan_count:
            keep_cols.append(column)
        else:
            drop_cols.append(column)

    if not drop_cols:
        return merged, []

    keep_set = {"subject_id", *target_cols, *keep_cols}
    ordered_cols = [column for column in merged.columns if column in keep_set]
    return merged[ordered_cols].copy(), drop_cols


def build_training_table(
    batch_dir: Path | None = None,
    icf_targets: pd.DataFrame | None = None,
    batch_dirs: Sequence[Path] | None = None,
    min_feature_non_nan_ratio: float = 0.6,
) -> pd.DataFrame:
    if icf_targets is None:
        raise ValueError("icf_targets is required")

    effective_batch_dirs = list(batch_dirs) if batch_dirs is not None else ([batch_dir] if batch_dir is not None else [])
    if len(effective_batch_dirs) == 0:
        raise ValueError("Provide batch_dir or batch_dirs")

    target_subjects = set(icf_targets["subject_id"].unique())
    rows = _collect_feature_rows(batch_dirs=effective_batch_dirs, target_subjects=target_subjects)

    features_df = pd.DataFrame(rows)
    if features_df.empty:
        return pd.DataFrame(columns=icf_targets.columns.tolist())

    merged = features_df.merge(icf_targets, on="subject_id", how="inner")

    target_cols = [column for column in icf_targets.columns if column != "subject_id"]
    merged, dropped_sparse_cols = _drop_sparse_feature_columns(
        merged=merged,
        target_cols=target_cols,
        min_feature_non_nan_ratio=min_feature_non_nan_ratio,
    )
    if dropped_sparse_cols:
        logger.info(
            "Dropped %d sparse feature columns with non-NaN ratio < %.2f",
            len(dropped_sparse_cols),
            min_feature_non_nan_ratio,
        )

    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [column for column in numeric_cols if column not in target_cols]

    if numeric_cols:
        medians = merged[numeric_cols].median()
        merged[numeric_cols] = merged[numeric_cols].fillna(medians)

    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subject-level ML training dataset")
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=Path("./output_batch"),
        help="Root directory containing batch_* folders",
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=None,
        help="Specific batch directory; if omitted, latest batch_* is used",
    )
    parser.add_argument(
        "--batch-dirs",
        default=None,
        help="Comma-separated list of batch directories to combine",
    )
    parser.add_argument(
        "--icf-csv",
        type=Path,
        default=Path(r"C:\Users\Nicla\Documents\ETHZ\Lifelogging\Data\ICF_scores_nursing_home.csv"),
        help="Path to ICF target CSV",
    )
    parser.add_argument(
        "--icf-csvs",
        default=None,
        help="Comma-separated list of ICF target CSVs to combine",
    )
    parser.add_argument(
        "--id-col",
        default="0",
        help="ICF subject ID column name or zero-based index (default: 0)",
    )
    parser.add_argument(
        "--target-cols",
        required=True,
        help="Comma-separated list of 4 ICF score columns (e.g., Basic Movements,Walking,Oral Care,Grooming)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./output_ml/training_table.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--min-feature-non-nan-ratio",
        type=float,
        default=0.6,
        help="Drop feature columns with non-NaN ratio below this threshold before imputation. I.e., at least 60% present values required to keep a feature column.",
    )
    return parser.parse_args()


def _parse_id_col(value: str) -> str | int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    return value


def _parse_target_cols(value: str) -> List[str]:
    columns = [item.strip() for item in value.split(",") if item.strip()]
    if len(columns) != 4:
        raise ValueError(f"Expected exactly 4 target columns in --target-cols, got {len(columns)}: {columns}")
    return columns


def main() -> None:
    args = parse_args()

    explicit_batch_dirs = _parse_path_list(args.batch_dirs)
    if explicit_batch_dirs:
        batch_dirs = explicit_batch_dirs
    else:
        batch_dir = args.batch_dir if args.batch_dir is not None else find_latest_batch(args.batch_root)
        batch_dirs = [batch_dir]

    explicit_icf_csvs = _parse_path_list(args.icf_csvs)
    icf_csvs = explicit_icf_csvs if explicit_icf_csvs else [args.icf_csv]

    id_col = _parse_id_col(str(args.id_col))
    target_cols = _parse_target_cols(args.target_cols)

    logger.info("Using batch directories: %s", [str(path) for path in batch_dirs])
    logger.info("Using ICF CSVs: %s", [str(path) for path in icf_csvs])
    logger.info("Target columns: %s", target_cols)

    icf_targets = load_combined_icf_targets(icf_csvs=icf_csvs, id_col=id_col, target_cols=target_cols)
    train_df = build_training_table(
        batch_dirs=batch_dirs,
        icf_targets=icf_targets,
        min_feature_non_nan_ratio=args.min_feature_non_nan_ratio,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.out, index=False)

    logger.info("Saved training table: %s", args.out)
    logger.info("Subjects in training table: %d", len(train_df))
    logger.info("Feature columns: %d", max(len(train_df.columns) - (1 + len(target_cols)), 0))


if __name__ == "__main__":
    main()
