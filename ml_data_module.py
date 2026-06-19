#!/usr/bin/env python3
"""
Step 2 (ML): Branch-aware PyTorch data module for ICF prediction.

Builds two model inputs from the Step-1 training table:
- HRV/tabular branch (for expert MLP)
- EDA+IMU branch as modality tokens (for transformer input)

PPG features are intentionally excluded by default.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except Exception as import_error:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required for ml_data_module.py. Install with: pip install torch"
    ) from import_error


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


HRV_KEYWORDS = (
    "hrv_",
    "rmssd",
    "sdnn",
    "pnn50",
    "rr_",
    "_rr_",
    "mean_rr",
    "median_rr",
    "mean_hr",
    "stress_index",
    "n_beats",
    "cv_rr",
)


@dataclass
class FeatureSplit:
    hrv_columns: List[str]
    eda_columns: List[str]
    imu_columns: List[str]
    dropped_columns: List[str]

    @property
    def transformer_columns(self) -> List[str]:
        return self.eda_columns + self.imu_columns


@dataclass
class NormalizationStats:
    hrv_mean: np.ndarray
    hrv_std: np.ndarray
    eda_mean: np.ndarray
    eda_std: np.ndarray
    imu_mean: np.ndarray
    imu_std: np.ndarray


class HybridICFDataset(Dataset):
    def __init__(
        self,
        sample_indices: np.ndarray,
        subject_ids: np.ndarray,
        hrv_features: np.ndarray,
        eda_features: np.ndarray,
        imu_features: np.ndarray,
        targets: np.ndarray,
        max_transformer_dim: int,
    ) -> None:
        self.sample_indices = sample_indices.astype(int)
        self.subject_ids = subject_ids
        self.hrv_features = hrv_features.astype(np.float32)
        self.eda_features = eda_features.astype(np.float32)
        self.imu_features = imu_features.astype(np.float32)
        target_values = targets.astype(np.float32)
        if target_values.ndim == 1:
            target_values = target_values.reshape(-1, 1)
        self.targets = target_values
        self.max_transformer_dim = int(max_transformer_dim)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        hrv_vector = torch.from_numpy(self.hrv_features[index])

        transformer_tokens = torch.zeros((2, self.max_transformer_dim), dtype=torch.float32)
        transformer_mask = torch.zeros((2, self.max_transformer_dim), dtype=torch.bool)

        eda_vector = torch.from_numpy(self.eda_features[index])
        imu_vector = torch.from_numpy(self.imu_features[index])

        if eda_vector.numel() > 0:
            transformer_tokens[0, : eda_vector.numel()] = eda_vector
            transformer_mask[0, : eda_vector.numel()] = True

        if imu_vector.numel() > 0:
            transformer_tokens[1, : imu_vector.numel()] = imu_vector
            transformer_mask[1, : imu_vector.numel()] = True

        target = torch.from_numpy(self.targets[index])

        return {
            "row_index": int(self.sample_indices[index]),
            "subject_id": self.subject_ids[index],
            "hrv": hrv_vector,
            "transformer_tokens": transformer_tokens,
            "transformer_mask": transformer_mask,
            "target_score": target,
        }


def split_feature_columns(
    table: pd.DataFrame,
    target_cols: List[str] | None = None,
    subject_col: str = "subject_id",
    drop_ppg: bool = True,
) -> FeatureSplit:
    if target_cols is None:
        target_cols = ["target_score"]

    excluded_columns = set(target_cols) | {subject_col}
    numeric_columns = [
        column
        for column in table.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(table[column])
    ]

    dropped_columns: List[str] = []
    filtered_columns: List[str] = []

    for column in numeric_columns:
        lower_name = column.lower()
        if drop_ppg and "ppg" in lower_name:
            dropped_columns.append(column)
            continue
        filtered_columns.append(column)

    eda_columns = [column for column in filtered_columns if "eda_" in column.lower()]
    imu_columns = [column for column in filtered_columns if "imu_" in column.lower()]

    transformer_set = set(eda_columns + imu_columns)

    hrv_columns = [
        column
        for column in filtered_columns
        if column not in transformer_set and any(keyword in column.lower() for keyword in HRV_KEYWORDS)
    ]

    assigned_set = set(hrv_columns) | transformer_set
    dropped_columns.extend([column for column in filtered_columns if column not in assigned_set])

    return FeatureSplit(
        hrv_columns=sorted(hrv_columns),
        eda_columns=sorted(eda_columns),
        imu_columns=sorted(imu_columns),
        dropped_columns=sorted(dropped_columns),
    )


def split_indices(
    sample_count: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    fraction_sum = train_fraction + val_fraction + test_fraction
    if not np.isclose(fraction_sum, 1.0, atol=1e-6):
        raise ValueError(f"Split fractions must sum to 1.0, got {fraction_sum}")

    rng = np.random.default_rng(seed)
    permuted = rng.permutation(sample_count)

    train_end = int(round(sample_count * train_fraction))
    val_end = train_end + int(round(sample_count * val_fraction))

    train_idx = permuted[:train_end]
    val_idx = permuted[train_end:val_end]
    test_idx = permuted[val_end:]

    if len(train_idx) == 0:
        raise ValueError("Training split is empty. Increase dataset size or train_fraction.")

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def split_indices_by_subject(
    subject_ids: np.ndarray,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    fraction_sum = train_fraction + val_fraction + test_fraction
    if not np.isclose(fraction_sum, 1.0, atol=1e-6):
        raise ValueError(f"Split fractions must sum to 1.0, got {fraction_sum}")

    unique_subjects = np.unique(subject_ids)
    if len(unique_subjects) == 0:
        raise ValueError("No subjects available for splitting")

    rng = np.random.default_rng(seed)
    permuted_subjects = rng.permutation(unique_subjects)

    train_end = int(round(len(permuted_subjects) * train_fraction))
    val_end = train_end + int(round(len(permuted_subjects) * val_fraction))

    train_subjects = set(permuted_subjects[:train_end])
    val_subjects = set(permuted_subjects[train_end:val_end])
    test_subjects = set(permuted_subjects[val_end:])

    train_idx = np.array([idx for idx, sid in enumerate(subject_ids) if sid in train_subjects], dtype=int)
    val_idx = np.array([idx for idx, sid in enumerate(subject_ids) if sid in val_subjects], dtype=int)
    test_idx = np.array([idx for idx, sid in enumerate(subject_ids) if sid in test_subjects], dtype=int)

    if len(train_idx) == 0:
        raise ValueError("Training split is empty after subject-wise split. Increase data or train_fraction.")

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _safe_standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = np.where(std < 1e-8, 1.0, std)
    return (values - mean) / safe_std


def _compute_stats(train_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if train_values.shape[1] == 0:
        return np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    mean_values = train_values.mean(axis=0).astype(np.float32)
    std_values = train_values.std(axis=0).astype(np.float32)
    std_values = np.where(std_values < 1e-8, 1.0, std_values)
    return mean_values, std_values


class ICFHybridDataModule:
    def __init__(
        self,
        table: pd.DataFrame,
        feature_split: FeatureSplit,
        target_cols: List[str] | None = None,
        subject_col: str = "subject_id",
        train_fraction: float = 0.7,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int = 42,
        split_mode: str = "subject",
    ) -> None:
        self.table = table.copy()
        self.feature_split = feature_split
        self.target_cols = target_cols or ["target_score"]
        self.subject_col = subject_col
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.split_mode = split_mode

        self.indices: Dict[str, np.ndarray] = {}
        self.stats: NormalizationStats | None = None
        self.datasets: Dict[str, HybridICFDataset] = {}

        self.max_transformer_dim = max(len(self.feature_split.eda_columns), len(self.feature_split.imu_columns), 1)

    def setup(self) -> None:
        table = self.table

        subject_ids = table[self.subject_col].astype(str).values
        target_values = table[self.target_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

        hrv_values = table[self.feature_split.hrv_columns].to_numpy(dtype=np.float32) if self.feature_split.hrv_columns else np.zeros((len(table), 0), dtype=np.float32)
        eda_values = table[self.feature_split.eda_columns].to_numpy(dtype=np.float32) if self.feature_split.eda_columns else np.zeros((len(table), 0), dtype=np.float32)
        imu_values = table[self.feature_split.imu_columns].to_numpy(dtype=np.float32) if self.feature_split.imu_columns else np.zeros((len(table), 0), dtype=np.float32)

        if self.split_mode == "subject":
            self.indices = split_indices_by_subject(
                subject_ids=subject_ids,
                train_fraction=self.train_fraction,
                val_fraction=self.val_fraction,
                test_fraction=self.test_fraction,
                seed=self.seed,
            )
        elif self.split_mode == "row":
            self.indices = split_indices(
                sample_count=len(table),
                train_fraction=self.train_fraction,
                val_fraction=self.val_fraction,
                test_fraction=self.test_fraction,
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unknown split_mode '{self.split_mode}'. Use 'subject' or 'row'.")

        train_idx = self.indices["train"]
        hrv_mean, hrv_std = _compute_stats(hrv_values[train_idx])
        eda_mean, eda_std = _compute_stats(eda_values[train_idx])
        imu_mean, imu_std = _compute_stats(imu_values[train_idx])

        self.stats = NormalizationStats(
            hrv_mean=hrv_mean,
            hrv_std=hrv_std,
            eda_mean=eda_mean,
            eda_std=eda_std,
            imu_mean=imu_mean,
            imu_std=imu_std,
        )

        hrv_values = _safe_standardize(hrv_values, hrv_mean, hrv_std)
        eda_values = _safe_standardize(eda_values, eda_mean, eda_std)
        imu_values = _safe_standardize(imu_values, imu_mean, imu_std)

        for split_name, split_indices_array in self.indices.items():
            self.datasets[split_name] = HybridICFDataset(
                sample_indices=split_indices_array,
                subject_ids=subject_ids[split_indices_array],
                hrv_features=hrv_values[split_indices_array],
                eda_features=eda_values[split_indices_array],
                imu_features=imu_values[split_indices_array],
                targets=target_values[split_indices_array],
                max_transformer_dim=self.max_transformer_dim,
            )

    def dataloader(self, split_name: str, batch_size: int = 8, shuffle: bool | None = None) -> DataLoader:
        if split_name not in self.datasets:
            raise ValueError(f"Unknown split: {split_name}. Call setup() first.")
        if shuffle is None:
            shuffle = split_name == "train"
        return DataLoader(self.datasets[split_name], batch_size=batch_size, shuffle=shuffle)

    def export_schema(self) -> Dict[str, object]:
        if self.stats is None:
            raise RuntimeError("Call setup() before export_schema().")

        return {
            "target_cols": self.target_cols,
            "subject_col": self.subject_col,
            "split_mode": self.split_mode,
            "n_subjects": int(len(self.table)),
            "splits": {name: int(len(idx)) for name, idx in self.indices.items()},
            "hrv_columns": self.feature_split.hrv_columns,
            "eda_columns": self.feature_split.eda_columns,
            "imu_columns": self.feature_split.imu_columns,
            "dropped_columns": self.feature_split.dropped_columns,
            "max_transformer_dim": int(self.max_transformer_dim),
        }


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Prepare branch-aware PyTorch data for hybrid ICF model")
    parser.add_argument("--table", type=Path, default=Path("./output_ml/training_table.csv"), help="Input training table CSV")
    parser.add_argument("--subject-col", default="subject_id", help="Subject ID column")
    parser.add_argument(
        "--target-cols",
        default="target_1,target_2,target_3,target_4",
        help="Comma-separated target columns (expects 4 for multi-target runs)",
    )
    parser.add_argument("--train-frac", type=float, default=0.7, help="Train split fraction")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation split fraction")
    parser.add_argument("--test-frac", type=float, default=0.15, help="Test split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split-mode", choices=["subject", "row"], default="subject", help="Split strategy")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for smoke-check loader")
    parser.add_argument("--out-schema", type=Path, default=Path("./output_ml/feature_schema.json"), help="Where to save split/schema JSON")

    args = parser.parse_args()
    target_cols = [item.strip() for item in args.target_cols.split(",") if item.strip()]
    if len(target_cols) != 4:
        raise ValueError(f"Expected exactly 4 target columns, got {len(target_cols)}: {target_cols}")

    table = pd.read_csv(args.table)
    split = split_feature_columns(table, target_cols=target_cols, subject_col=args.subject_col, drop_ppg=True)

    data_module = ICFHybridDataModule(
        table=table,
        feature_split=split,
        target_cols=target_cols,
        subject_col=args.subject_col,
        train_fraction=args.train_frac,
        val_fraction=args.val_frac,
        test_fraction=args.test_frac,
        seed=args.seed,
        split_mode=args.split_mode,
    )
    data_module.setup()

    train_loader = data_module.dataloader("train", batch_size=args.batch_size)
    batch = next(iter(train_loader))

    logger.info("Subjects: %d", len(table))
    logger.info("HRV columns: %d", len(split.hrv_columns))
    logger.info("EDA columns: %d", len(split.eda_columns))
    logger.info("IMU columns: %d", len(split.imu_columns))
    logger.info("Dropped columns: %d", len(split.dropped_columns))
    logger.info("Train/Val/Test: %s", {name: len(dataset) for name, dataset in data_module.datasets.items()})
    logger.info("Batch HRV tensor shape: %s", tuple(batch["hrv"].shape))
    logger.info("Batch transformer tensor shape: %s", tuple(batch["transformer_tokens"].shape))
    logger.info("Batch transformer mask shape: %s", tuple(batch["transformer_mask"].shape))

    schema = data_module.export_schema()
    args.out_schema.parent.mkdir(parents=True, exist_ok=True)
    args.out_schema.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    logger.info("Saved schema: %s", args.out_schema)


if __name__ == "__main__":
    run_cli()
