#!/usr/bin/env python3
"""
Step 3 (ML): Training scaffold for hybrid ICF model.

This script provides a minimal end-to-end training loop that wires:
- Step-2 data module
- Step-3 hybrid model
- Theory-informed loss

Current C_a inputs are placeholder-based and can be replaced with your final mapping.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from ml_data_module import HybridICFDataset, ICFHybridDataModule, split_feature_columns
from ml_hybrid_model import (
    CapacityConfig,
    HybridICFModel,
    TheoryInformedICFLoss,
    compute_functional_capacity,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_target_cols(value: str) -> List[str]:
    columns = [item.strip() for item in value.split(",") if item.strip()]
    if len(columns) != 4:
        raise ValueError(f"Expected exactly 4 target columns, got {len(columns)}: {columns}")
    return columns


def _to_icf_class_labels(target_scores: torch.Tensor, n_classes: int = 5) -> torch.Tensor:
    """
    Converts continuous 0-100 target to ordinal bins [0..n_classes-1].
    """
    clipped = target_scores.clamp(0.0, 100.0)
    bin_size = 100.0 / n_classes
    labels = torch.floor(clipped / bin_size).long()
    labels = torch.clamp(labels, 0, n_classes - 1)
    return labels


def _to_icf_class_labels_np(target_scores: np.ndarray, n_classes: int = 5) -> np.ndarray:
    clipped = np.clip(target_scores, 0.0, 100.0)
    bin_size = 100.0 / n_classes
    labels = np.floor(clipped / bin_size).astype(int)
    labels = np.clip(labels, 0, n_classes - 1)
    return labels


def _quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    if len(y_true) == 0:
        return float("nan")

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    conf_mat = np.zeros((n_classes, n_classes), dtype=float)
    for true_class, pred_class in zip(y_true, y_pred):
        conf_mat[true_class, pred_class] += 1.0

    hist_true = conf_mat.sum(axis=1)
    hist_pred = conf_mat.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / max(conf_mat.sum(), 1.0)

    weight = np.zeros((n_classes, n_classes), dtype=float)
    denom = float((n_classes - 1) ** 2)
    for i in range(n_classes):
        for j in range(n_classes):
            weight[i, j] = ((i - j) ** 2) / denom if denom > 0 else 0.0

    observed_weighted = (weight * conf_mat).sum()
    expected_weighted = (weight * expected).sum()

    if expected_weighted <= 1e-12:
        return float("nan")
    return float(1.0 - (observed_weighted / expected_weighted))


def _parse_column_list(value: str | None) -> List[str]:
    if value is None:
        return []
    return [token.strip() for token in value.split(",") if token.strip()]


def _remap_targets(
    table: pd.DataFrame,
    target_col: str,
    target_scale_mode: str,
    target_source_min: float,
    target_source_max: float,
) -> pd.DataFrame:
    if target_col not in table.columns:
        raise ValueError(f"Target column '{target_col}' not found in table")

    if target_source_max <= target_source_min:
        raise ValueError(
            f"Invalid target source range: min={target_source_min} max={target_source_max}; max must be > min"
        )

    work = table.copy()
    raw_target = pd.to_numeric(work[target_col], errors="coerce")
    if raw_target.isna().any():
        missing_count = int(raw_target.isna().sum())
        raise ValueError(f"Target column '{target_col}' contains {missing_count} non-numeric/missing value(s)")

    values = raw_target.to_numpy(dtype=float)
    eps = 1e-6

    should_map = False
    if target_scale_mode == "map_1_5_to_0_100":
        should_map = True
    elif target_scale_mode == "auto":
        should_map = bool(values.min() >= (target_source_min - eps) and values.max() <= (target_source_max + eps))
    elif target_scale_mode == "none":
        should_map = False
    else:
        raise ValueError(
            f"Unsupported target_scale_mode: {target_scale_mode}. Use 'none', 'map_1_5_to_0_100', or 'auto'."
        )

    logger.info(
        "Target '%s' before remap | min=%.4f max=%.4f mean=%.4f",
        target_col,
        float(values.min()),
        float(values.max()),
        float(values.mean()),
    )

    if not should_map:
        logger.info("Target remap: skipped (mode=%s)", target_scale_mode)
        work[target_col] = raw_target.astype(np.float32)
        return work

    denom = float(target_source_max - target_source_min)
    mapped = ((values - target_source_min) / denom) * 100.0
    mapped = np.clip(mapped, 0.0, 100.0)

    work[target_col] = mapped.astype(np.float32)
    logger.info(
        "Target remap applied (%s) using source range [%.3f, %.3f] -> [0, 100]",
        target_scale_mode,
        target_source_min,
        target_source_max,
    )
    logger.info(
        "Target '%s' after remap  | min=%.4f max=%.4f mean=%.4f",
        target_col,
        float(mapped.min()),
        float(mapped.max()),
        float(mapped.mean()),
    )
    return work


def _validate_required_columns(table: pd.DataFrame, columns: List[str], purpose: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"Missing {purpose} columns in table: {missing}")


def _compute_ca_from_columns(
    batch_df: pd.DataFrame,
    prob_columns: List[str],
    duration_columns: List[str],
    capacity_cfg: CapacityConfig,
    device: torch.device,
) -> torch.Tensor:
    if len(prob_columns) == 0 or len(duration_columns) == 0:
        raise ValueError("Both probability and duration column lists must be non-empty for ADL-based C_a")

    if len(prob_columns) != len(duration_columns):
        raise ValueError(
            f"ADL probability and duration columns must have same length, got {len(prob_columns)} vs {len(duration_columns)}"
        )

    probabilities = torch.tensor(
        batch_df[prob_columns].to_numpy(dtype=float),
        device=device,
        dtype=torch.float32,
    )
    durations = torch.tensor(
        batch_df[duration_columns].to_numpy(dtype=float),
        device=device,
        dtype=torch.float32,
    )

    ca_value = compute_functional_capacity(
        probabilities=probabilities,
        durations=durations,
        config=capacity_cfg,
    )
    return torch.clamp(ca_value * 100.0, 0.0, 100.0)


def _compute_ca_for_batch(
    batch: Dict[str, torch.Tensor],
    table: pd.DataFrame,
    args: argparse.Namespace,
    capacity_cfg: CapacityConfig,
    device: torch.device,
    adl_prob_columns: List[str],
    adl_duration_columns: List[str],
) -> tuple[bool, torch.Tensor | None]:
    use_ca_constraint = bool(args.enable_ca)
    if not use_ca_constraint:
        return False, None

    x_hrv = batch["hrv"].to(device)
    row_index_tensor = batch["row_index"]
    if hasattr(row_index_tensor, "tolist"):
        batch_row_indices = [int(item) for item in row_index_tensor.tolist()]
    else:
        batch_row_indices = [int(item) for item in row_index_tensor]

    if args.ca_input_mode == "adl":
        batch_df = table.iloc[batch_row_indices].copy()
        if len(batch_df) != len(batch_row_indices):
            raise ValueError("Could not map all batch rows to source table rows for ADL-based C_a computation")
        ca_value = _compute_ca_from_columns(
            batch_df=batch_df,
            prob_columns=adl_prob_columns,
            duration_columns=adl_duration_columns,
            capacity_cfg=capacity_cfg,
            device=device,
        )
        return True, ca_value

    pseudo_prob = torch.sigmoid(x_hrv)
    pseudo_duration = x_hrv.abs() + 1e-3
    ca_value = compute_functional_capacity(
        probabilities=pseudo_prob,
        durations=pseudo_duration,
        config=capacity_cfg,
    )
    ca_value = torch.clamp(ca_value * 100.0, 0.0, 100.0)
    return True, ca_value


def _run_epoch(
    model: HybridICFModel,
    loader,
    criterion: TheoryInformedICFLoss,
    table: pd.DataFrame,
    args: argparse.Namespace,
    capacity_cfg: CapacityConfig,
    device: torch.device,
    adl_prob_columns: List[str],
    adl_duration_columns: List[str],
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    sum_total = 0.0
    sum_base = 0.0
    sum_ca = 0.0
    sum_ordinal = 0.0
    sample_count = 0

    context_manager = torch.enable_grad if is_training else torch.no_grad
    with context_manager():
        for batch in loader:
            x_hrv = batch["hrv"].to(device)
            x_tokens = batch["transformer_tokens"].to(device)
            y_target = batch["target_score"].to(device)

            outputs = model(x_hrv=x_hrv, transformer_tokens=x_tokens)
            pred_score = outputs["pred_score"]

            use_ca_constraint, ca_value = _compute_ca_for_batch(
                batch=batch,
                table=table,
                args=args,
                capacity_cfg=capacity_cfg,
                device=device,
                adl_prob_columns=adl_prob_columns,
                adl_duration_columns=adl_duration_columns,
            )

            icf_class_label = _to_icf_class_labels(y_target)
            total_loss, l_base, l_ca, l_ordinal = criterion(
                pred_score=pred_score,
                clinical_target=y_target,
                ca_value=ca_value,
                icf_class_label=icf_class_label,
                use_ca_constraint=use_ca_constraint,
            )

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = int(y_target.size(0))
            sample_count += batch_size
            sum_total += float(total_loss.item()) * batch_size
            sum_base += float(l_base.item()) * batch_size
            sum_ca += float(l_ca.item()) * batch_size
            sum_ordinal += float(l_ordinal.item()) * batch_size

    if sample_count == 0:
        return {"total": 0.0, "base": 0.0, "ca": 0.0, "ordinal": 0.0}

    return {
        "total": sum_total / sample_count,
        "base": sum_base / sample_count,
        "ca": sum_ca / sample_count,
        "ordinal": sum_ordinal / sample_count,
    }


def _collect_predictions(
    model: HybridICFModel,
    loader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, float | int | str]] = []

    with torch.no_grad():
        for batch in loader:
            x_hrv = batch["hrv"].to(device)
            x_tokens = batch["transformer_tokens"].to(device)
            y_target = batch["target_score"].to(device)

            outputs = model(x_hrv=x_hrv, transformer_tokens=x_tokens)
            pred_score = outputs["pred_score"]

            target_vals = y_target.detach().cpu().numpy()
            pred_vals = pred_score.detach().cpu().numpy()

            subject_ids = [str(item) for item in batch["subject_id"]]
            row_indices = batch["row_index"]
            if hasattr(row_indices, "tolist"):
                row_indices = row_indices.tolist()

            for index in range(len(subject_ids)):
                for target_idx in range(target_vals.shape[1]):
                    rows.append(
                        {
                            "row_index": int(row_indices[index]),
                            "subject_id": subject_ids[index],
                            "target_idx": int(target_idx),
                            "target_score": float(target_vals[index, target_idx]),
                            "pred_score": float(pred_vals[index, target_idx]),
                        }
                    )

    return pd.DataFrame(rows)


def _evaluate_predictions(predictions_df: pd.DataFrame, n_classes: int) -> Dict[str, float]:
    if predictions_df.empty:
        return {
            "count": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "ordinal_mae": float("nan"),
            "qwk": float("nan"),
        }

    y_true = predictions_df["target_score"].to_numpy(dtype=float)
    y_pred = predictions_df["pred_score"].to_numpy(dtype=float)

    abs_error = np.abs(y_pred - y_true)
    sq_error = (y_pred - y_true) ** 2

    true_cls = _to_icf_class_labels_np(y_true, n_classes=n_classes)
    pred_cls = _to_icf_class_labels_np(y_pred, n_classes=n_classes)
    ordinal_abs_error = np.abs(pred_cls - true_cls)
    qwk = _quadratic_weighted_kappa(true_cls, pred_cls, n_classes=n_classes)

    return {
        "count": float(len(predictions_df)),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(sq_error.mean())),
        "ordinal_mae": float(ordinal_abs_error.mean()),
        "qwk": float(qwk),
    }


def _per_subject_breakdown(predictions_df: pd.DataFrame, n_classes: int) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "n_samples",
                "mae",
                "rmse",
                "ordinal_mae",
                "mean_target",
                "mean_pred",
            ]
        )

    work_df = predictions_df.copy()
    work_df["target_cls"] = _to_icf_class_labels_np(work_df["target_score"].to_numpy(dtype=float), n_classes=n_classes)
    work_df["pred_cls"] = _to_icf_class_labels_np(work_df["pred_score"].to_numpy(dtype=float), n_classes=n_classes)
    work_df["abs_error"] = (work_df["pred_score"] - work_df["target_score"]).abs()
    work_df["sq_error"] = (work_df["pred_score"] - work_df["target_score"]) ** 2
    work_df["ordinal_abs_error"] = (work_df["pred_cls"] - work_df["target_cls"]).abs()

    grouped = work_df.groupby("subject_id", as_index=False).agg(
        n_samples=("row_index", "count"),
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda values: float(np.sqrt(np.mean(values)))),
        ordinal_mae=("ordinal_abs_error", "mean"),
        mean_target=("target_score", "mean"),
        mean_pred=("pred_score", "mean"),
    )

    return grouped.sort_values(by=["mae", "subject_id"], ascending=[False, True]).reset_index(drop=True)


def _safe_standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = np.where(std < 1e-8, 1.0, std)
    return (values - mean) / safe_std


def _compute_stats(train_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train_values.shape[1] == 0:
        return np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    mean_values = train_values.mean(axis=0).astype(np.float32)
    std_values = train_values.std(axis=0).astype(np.float32)
    std_values = np.where(std_values < 1e-8, 1.0, std_values)
    return mean_values, std_values


def _get_subject_sets(indices: Dict[str, np.ndarray], subject_ids: np.ndarray) -> tuple[set[str], set[str], set[str]]:
    train_subjects = {str(subject_ids[i]) for i in indices["train"]}
    val_subjects = {str(subject_ids[i]) for i in indices["val"]}
    test_subjects = {str(subject_ids[i]) for i in indices["test"]}
    return train_subjects, val_subjects, test_subjects


def _print_split_report(indices: Dict[str, np.ndarray], subject_ids: np.ndarray, split_mode_label: str) -> None:
    train_subjects, val_subjects, test_subjects = _get_subject_sets(indices, subject_ids)

    overlap_train_val = train_subjects & val_subjects
    overlap_train_test = train_subjects & test_subjects
    overlap_val_test = val_subjects & test_subjects

    print("\n--- Split Leakage Report ---")
    print(f"split_mode: {split_mode_label}")
    print(f"train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"val subjects ({len(val_subjects)}): {sorted(val_subjects)}")
    print(f"test subjects ({len(test_subjects)}): {sorted(test_subjects)}")
    print(f"overlap train-val ({len(overlap_train_val)}): {sorted(overlap_train_val)}")
    print(f"overlap train-test ({len(overlap_train_test)}): {sorted(overlap_train_test)}")
    print(f"overlap val-test ({len(overlap_val_test)}): {sorted(overlap_val_test)}")
    print("----------------------------\n")


def _build_loaders_for_indices(
    table: pd.DataFrame,
    split,
    target_cols: List[str],
    subject_col: str,
    indices: Dict[str, np.ndarray],
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    subject_ids = table[subject_col].astype(str).values
    target_values = table[target_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    hrv_values = (
        table[split.hrv_columns].to_numpy(dtype=np.float32)
        if split.hrv_columns
        else np.zeros((len(table), 0), dtype=np.float32)
    )
    eda_values = (
        table[split.eda_columns].to_numpy(dtype=np.float32)
        if split.eda_columns
        else np.zeros((len(table), 0), dtype=np.float32)
    )
    imu_values = (
        table[split.imu_columns].to_numpy(dtype=np.float32)
        if split.imu_columns
        else np.zeros((len(table), 0), dtype=np.float32)
    )

    train_idx = indices["train"]
    hrv_mean, hrv_std = _compute_stats(hrv_values[train_idx])
    eda_mean, eda_std = _compute_stats(eda_values[train_idx])
    imu_mean, imu_std = _compute_stats(imu_values[train_idx])

    hrv_values = _safe_standardize(hrv_values, hrv_mean, hrv_std)
    eda_values = _safe_standardize(eda_values, eda_mean, eda_std)
    imu_values = _safe_standardize(imu_values, imu_mean, imu_std)

    max_transformer_dim = max(len(split.eda_columns), len(split.imu_columns), 1)

    datasets = {}
    for split_name in ["train", "val", "test"]:
        split_idx = indices[split_name]
        datasets[split_name] = HybridICFDataset(
            sample_indices=split_idx,
            subject_ids=subject_ids[split_idx],
            hrv_features=hrv_values[split_idx],
            eda_features=eda_values[split_idx],
            imu_features=imu_values[split_idx],
            targets=target_values[split_idx],
            max_transformer_dim=max_transformer_dim,
        )

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, max_transformer_dim


def _split_train_val_subjects(
    remaining_subjects: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    if len(remaining_subjects) == 0:
        raise ValueError("No subjects available for train/val split")

    rng = np.random.default_rng(seed)
    permuted = rng.permutation(remaining_subjects)

    if len(permuted) == 1:
        subject = str(permuted[0])
        logger.warning("Only one non-test subject available; using it for both train and val in this fold")
        return {subject}, {subject}

    val_count = int(round(len(permuted) * val_ratio))
    val_count = max(1, val_count)
    val_count = min(val_count, len(permuted) - 1)

    val_subjects = {str(subject) for subject in permuted[:val_count]}
    train_subjects = {str(subject) for subject in permuted[val_count:]}
    return train_subjects, val_subjects


def _fold_indices_from_subject_sets(
    subject_ids: np.ndarray,
    train_subjects: set[str],
    val_subjects: set[str],
    test_subjects: set[str],
) -> Dict[str, np.ndarray]:
    train_idx = np.array([i for i, sid in enumerate(subject_ids) if str(sid) in train_subjects], dtype=int)
    val_idx = np.array([i for i, sid in enumerate(subject_ids) if str(sid) in val_subjects], dtype=int)
    test_idx = np.array([i for i, sid in enumerate(subject_ids) if str(sid) in test_subjects], dtype=int)

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Invalid fold with empty split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _build_loso_folds(subject_ids: np.ndarray, val_ratio: float, seed: int) -> List[Dict[str, object]]:
    unique_subjects = sorted({str(subject) for subject in subject_ids})
    if len(unique_subjects) < 2:
        raise ValueError("LOSO requires at least 2 unique subjects")

    folds: List[Dict[str, object]] = []
    for fold_idx, test_subject in enumerate(unique_subjects, start=1):
        remaining_subjects = np.array([subject for subject in unique_subjects if subject != test_subject], dtype=object)
        train_subjects, val_subjects = _split_train_val_subjects(
            remaining_subjects=remaining_subjects,
            val_ratio=val_ratio,
            seed=seed + fold_idx,
        )
        test_subjects = {test_subject}
        indices = _fold_indices_from_subject_sets(subject_ids, train_subjects, val_subjects, test_subjects)
        folds.append(
            {
                "fold_index": fold_idx,
                "fold_name": f"fold_{fold_idx:02d}_{test_subject}",
                "test_subjects": sorted(test_subjects),
                "indices": indices,
            }
        )

    return folds


def _build_groupkfold_folds(
    subject_ids: np.ndarray,
    n_folds: int,
    val_ratio: float,
    seed: int,
) -> List[Dict[str, object]]:
    unique_subjects = np.array(sorted({str(subject) for subject in subject_ids}), dtype=object)
    if len(unique_subjects) < 3:
        raise ValueError("Group K-fold requires at least 3 unique subjects")
    if n_folds < 2:
        raise ValueError("cv_folds must be at least 2 for groupkfold")
    if n_folds > len(unique_subjects):
        raise ValueError(f"cv_folds={n_folds} cannot exceed number of subjects={len(unique_subjects)}")

    rng = np.random.default_rng(seed)
    shuffled_subjects = rng.permutation(unique_subjects)
    test_blocks = np.array_split(shuffled_subjects, n_folds)

    folds: List[Dict[str, object]] = []
    for fold_idx, test_block in enumerate(test_blocks, start=1):
        test_subjects = {str(subject) for subject in test_block.tolist()}
        remaining_subjects = np.array([subject for subject in shuffled_subjects if str(subject) not in test_subjects], dtype=object)
        train_subjects, val_subjects = _split_train_val_subjects(
            remaining_subjects=remaining_subjects,
            val_ratio=val_ratio,
            seed=seed + 1000 + fold_idx,
        )

        indices = _fold_indices_from_subject_sets(subject_ids, train_subjects, val_subjects, test_subjects)
        folds.append(
            {
                "fold_index": fold_idx,
                "fold_name": f"fold_{fold_idx:02d}",
                "test_subjects": sorted(test_subjects),
                "indices": indices,
            }
        )

    return folds


def _run_training_once(
    args: argparse.Namespace,
    table: pd.DataFrame,
    split,
    train_loader,
    val_loader,
    test_loader,
    sensor_token_dim: int,
    device: torch.device,
    run_name: str,
    checkpoint_dir: Path,
    history_csv_path: Path,
    report_dir: Path,
    adl_prob_columns: List[str],
    adl_duration_columns: List[str],
) -> Dict[str, object]:
    num_targets = len(args.target_cols)
    model = HybridICFModel(
        hrv_input_dim=max(len(split.hrv_columns), 1),
        sensor_token_dim=sensor_token_dim,
        hrv_hidden_dim=args.hrv_hidden_dim,
        sensor_model_dim=args.sensor_model_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        num_targets=num_targets,
        num_heads=args.num_heads,
        num_sensor_layers=args.num_sensor_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = TheoryInformedICFLoss(alpha=args.alpha, beta=args.beta, margin=args.margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    capacity_cfg = CapacityConfig(
        omega_f=args.omega_f,
        omega_q=args.omega_q,
        tau=args.ca_tau,
        expected_frequency=args.ca_expected_frequency,
        expected_duration=args.ca_expected_duration,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / f"{run_name}_best.pt"
    last_ckpt_path = checkpoint_dir / f"{run_name}_last.pt"
    report_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: List[Dict[str, float | int | bool]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            table=table,
            args=args,
            capacity_cfg=capacity_cfg,
            device=device,
            adl_prob_columns=adl_prob_columns,
            adl_duration_columns=adl_duration_columns,
            optimizer=optimizer,
        )

        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            table=table,
            args=args,
            capacity_cfg=capacity_cfg,
            device=device,
            adl_prob_columns=adl_prob_columns,
            adl_duration_columns=adl_duration_columns,
            optimizer=None,
        )

        logger.info(
            "[%s] Epoch %d/%d | train total=%.4f (base=%.4f, ca=%.4f, ord=%.4f) | val total=%.4f (base=%.4f, ca=%.4f, ord=%.4f)",
            run_name,
            epoch,
            args.epochs,
            train_metrics["total"],
            train_metrics["base"],
            train_metrics["ca"],
            train_metrics["ordinal"],
            val_metrics["total"],
            val_metrics["base"],
            val_metrics["ca"],
            val_metrics["ordinal"],
        )

        improved = val_metrics["total"] < (best_val_loss - args.min_delta)

        if improved:
            best_val_loss = val_metrics["total"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_total_loss": best_val_loss,
                    "args": checkpoint_args,
                    "feature_counts": {
                        "hrv": len(split.hrv_columns),
                        "eda": len(split.eda_columns),
                        "imu": len(split.imu_columns),
                    },
                },
                best_ckpt_path,
            )
            logger.info("[%s] Saved new best checkpoint: %s", run_name, best_ckpt_path)
        else:
            epochs_without_improvement += 1

        history_rows.append(
            {
                "epoch": int(epoch),
                "train_total": float(train_metrics["total"]),
                "train_base": float(train_metrics["base"]),
                "train_ca": float(train_metrics["ca"]),
                "train_ordinal": float(train_metrics["ordinal"]),
                "val_total": float(val_metrics["total"]),
                "val_base": float(val_metrics["base"]),
                "val_ca": float(val_metrics["ca"]),
                "val_ordinal": float(val_metrics["ordinal"]),
                "best_val_so_far": float(best_val_loss),
                "improved": bool(improved),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if epochs_without_improvement >= args.patience:
            logger.info(
                "[%s] Early stopping at epoch %d (no val improvement for %d epoch(s))",
                run_name,
                epoch,
                args.patience,
            )
            break

    if args.save_last:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_total_loss": best_val_loss,
                "args": checkpoint_args,
            },
            last_ckpt_path,
        )
        logger.info("[%s] Saved last checkpoint: %s", run_name, last_ckpt_path)

    if history_rows:
        history_df = pd.DataFrame(history_rows)
        history_csv_path.parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(history_csv_path, index=False)
        logger.info("[%s] Saved training history CSV: %s", run_name, history_csv_path)

    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("[%s] Loaded best checkpoint from epoch %d", run_name, int(ckpt.get("epoch", best_epoch)))

    test_metrics = _run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        table=table,
        args=args,
        capacity_cfg=capacity_cfg,
        device=device,
        adl_prob_columns=adl_prob_columns,
        adl_duration_columns=adl_duration_columns,
        optimizer=None,
    )

    val_predictions = _collect_predictions(model=model, loader=val_loader, device=device)
    test_predictions = _collect_predictions(model=model, loader=test_loader, device=device)

    val_eval = _evaluate_predictions(val_predictions, n_classes=args.n_classes)
    test_eval = _evaluate_predictions(test_predictions, n_classes=args.n_classes)
    per_subject_test = _per_subject_breakdown(test_predictions, n_classes=args.n_classes)

    summary_path = report_dir / f"{run_name}_evaluation_summary.csv"
    test_pred_path = report_dir / f"{run_name}_test_predictions.csv"
    test_subject_path = report_dir / f"{run_name}_test_per_subject.csv"

    summary_df = pd.DataFrame(
        [
            {"split": "val", **val_eval},
            {"split": "test", **test_eval},
        ]
    )
    summary_df.to_csv(summary_path, index=False)
    test_predictions.to_csv(test_pred_path, index=False)
    per_subject_test.to_csv(test_subject_path, index=False)

    logger.info("[%s] Training complete | best val total=%.4f at epoch %d", run_name, best_val_loss, best_epoch)
    logger.info(
        "[%s] Test metrics | total=%.4f | base=%.4f | ca=%.4f | ordinal=%.4f",
        run_name,
        test_metrics["total"],
        test_metrics["base"],
        test_metrics["ca"],
        test_metrics["ordinal"],
    )
    logger.info("[%s] Saved evaluation summary: %s", run_name, summary_path)

    return {
        "run_name": run_name,
        "best_val_total": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "test_total": float(test_metrics["total"]),
        "val_eval": val_eval,
        "test_eval": test_eval,
        "per_subject_test": per_subject_test,
    }


def _run_cross_validation(
    args: argparse.Namespace,
    table: pd.DataFrame,
    split,
    device: torch.device,
    adl_prob_columns: List[str],
    adl_duration_columns: List[str],
) -> None:
    subject_ids = table[args.subject_col].astype(str).values
    non_test_fraction = max(args.train_frac + args.val_frac, 1e-8)
    val_ratio = args.val_frac / non_test_fraction

    if args.cv_mode == "loso":
        folds = _build_loso_folds(subject_ids=subject_ids, val_ratio=val_ratio, seed=args.seed)
    elif args.cv_mode == "groupkfold":
        folds = _build_groupkfold_folds(
            subject_ids=subject_ids,
            n_folds=args.cv_folds,
            val_ratio=val_ratio,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unsupported cv_mode: {args.cv_mode}")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_name = args.run_name or f"hybrid_icf_{args.cv_mode}_{run_stamp}"
    checkpoint_root = Path(args.checkpoint_dir) / base_run_name
    report_root = Path(args.report_dir) / base_run_name

    fold_rows: List[Dict[str, object]] = []
    per_subject_rows: List[pd.DataFrame] = []

    for fold in folds:
        fold_name = str(fold["fold_name"])
        fold_indices = fold["indices"]
        fold_run_name = f"{base_run_name}_{fold_name}"

        _print_split_report(
            indices=fold_indices,
            subject_ids=subject_ids,
            split_mode_label=f"{args.cv_mode}:{fold_name}",
        )

        train_loader, val_loader, test_loader, max_transformer_dim = _build_loaders_for_indices(
            table=table,
            split=split,
            target_cols=args.target_cols,
            subject_col=args.subject_col,
            indices=fold_indices,
            batch_size=args.batch_size,
        )

        if args.history_csv is None:
            fold_history_csv = checkpoint_root / f"{fold_run_name}_history.csv"
        else:
            fold_history_csv = Path(args.history_csv)
            fold_history_csv = fold_history_csv.parent / f"{fold_history_csv.stem}_{fold_name}{fold_history_csv.suffix}"

        result = _run_training_once(
            args=args,
            table=table,
            split=split,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            sensor_token_dim=max_transformer_dim,
            device=device,
            run_name=fold_run_name,
            checkpoint_dir=checkpoint_root,
            history_csv_path=fold_history_csv,
            report_dir=report_root,
            adl_prob_columns=adl_prob_columns,
            adl_duration_columns=adl_duration_columns,
        )

        train_subjects, val_subjects, test_subjects = _get_subject_sets(fold_indices, subject_ids)
        fold_rows.append(
            {
                "fold": fold_name,
                "test_subjects": ",".join(sorted(test_subjects)),
                "n_train_subjects": len(train_subjects),
                "n_val_subjects": len(val_subjects),
                "n_test_subjects": len(test_subjects),
                "best_val_total": result["best_val_total"],
                "best_epoch": result["best_epoch"],
                "test_total": result["test_total"],
                "val_mae": result["val_eval"]["mae"],
                "val_rmse": result["val_eval"]["rmse"],
                "val_ordinal_mae": result["val_eval"]["ordinal_mae"],
                "val_qwk": result["val_eval"]["qwk"],
                "test_mae": result["test_eval"]["mae"],
                "test_rmse": result["test_eval"]["rmse"],
                "test_ordinal_mae": result["test_eval"]["ordinal_mae"],
                "test_qwk": result["test_eval"]["qwk"],
            }
        )

        per_subject_df = result["per_subject_test"].copy()
        per_subject_df.insert(0, "fold", fold_name)
        per_subject_rows.append(per_subject_df)

    fold_df = pd.DataFrame(fold_rows)
    report_root.mkdir(parents=True, exist_ok=True)
    fold_metrics_path = report_root / "cv_fold_metrics.csv"
    fold_df.to_csv(fold_metrics_path, index=False)

    metric_columns = [
        "best_val_total",
        "test_total",
        "val_mae",
        "val_rmse",
        "val_ordinal_mae",
        "val_qwk",
        "test_mae",
        "test_rmse",
        "test_ordinal_mae",
        "test_qwk",
    ]
    agg_rows = []
    for metric in metric_columns:
        values = pd.to_numeric(fold_df[metric], errors="coerce").to_numpy(dtype=float)
        agg_rows.append(
            {
                "metric": metric,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=0)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
        )

    agg_df = pd.DataFrame(agg_rows)
    agg_path = report_root / "cv_aggregate_metrics.csv"
    agg_df.to_csv(agg_path, index=False)

    if per_subject_rows:
        per_subject_all = pd.concat(per_subject_rows, ignore_index=True)
        per_subject_path = report_root / "cv_per_subject_all_folds.csv"
        per_subject_all.to_csv(per_subject_path, index=False)
        logger.info("Saved CV per-subject breakdown: %s", per_subject_path)

    logger.info("Saved CV fold metrics: %s", fold_metrics_path)
    logger.info("Saved CV aggregate metrics: %s", agg_path)


def run_train_step(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logger.info("Using device: %s", device)
    args.target_cols = _parse_target_cols(args.target_cols)

    table = pd.read_csv(args.table)
    for target_col in args.target_cols:
        table = _remap_targets(
            table=table,
            target_col=target_col,
            target_scale_mode=args.target_scale_mode,
            target_source_min=args.target_source_min,
            target_source_max=args.target_source_max,
        )
    split = split_feature_columns(table, target_cols=args.target_cols, subject_col=args.subject_col, drop_ppg=True)

    adl_prob_columns = _parse_column_list(args.ca_prob_columns)
    adl_duration_columns = _parse_column_list(args.ca_duration_columns)

    if args.enable_ca and args.ca_input_mode == "adl":
        _validate_required_columns(table, adl_prob_columns, "ADL probability")
        _validate_required_columns(table, adl_duration_columns, "ADL duration")

    if args.enable_ca:
        if args.ca_input_mode == "adl":
            logger.info("C_a mode: ADL classifier columns (%d activity channels)", len(adl_prob_columns))
        else:
            logger.info("C_a mode: placeholder")
    else:
        logger.info("C_a mode: disabled")

    if args.cv_mode != "none":
        _run_cross_validation(
            args=args,
            table=table,
            split=split,
            device=device,
            adl_prob_columns=adl_prob_columns,
            adl_duration_columns=adl_duration_columns,
        )
        return

    data_module = ICFHybridDataModule(
        table=table,
        feature_split=split,
        target_cols=args.target_cols,
        subject_col=args.subject_col,
        train_fraction=args.train_frac,
        val_fraction=args.val_frac,
        test_fraction=args.test_frac,
        seed=args.seed,
        split_mode=args.split_mode,
    )
    data_module.setup()

    subject_ids = table[args.subject_col].astype(str).values
    _print_split_report(indices=data_module.indices, subject_ids=subject_ids, split_mode_label=args.split_mode)

    train_loader = data_module.dataloader("train", batch_size=args.batch_size, shuffle=True)
    val_loader = data_module.dataloader("val", batch_size=args.batch_size, shuffle=False)
    test_loader = data_module.dataloader("test", batch_size=args.batch_size, shuffle=False)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"hybrid_icf_{run_stamp}"
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_csv_path = Path(args.history_csv) if args.history_csv else (checkpoint_dir / f"{run_name}_history.csv")
    result = _run_training_once(
        args=args,
        table=table,
        split=split,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        sensor_token_dim=data_module.max_transformer_dim,
        device=device,
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        history_csv_path=history_csv_path,
        report_dir=Path(args.report_dir),
        adl_prob_columns=adl_prob_columns,
        adl_duration_columns=adl_duration_columns,
    )

    logger.info("Feature counts -> HRV: %d, EDA: %d, IMU: %d", len(split.hrv_columns), len(split.eda_columns), len(split.imu_columns))
    logger.info(
        "Evaluation (val) | MAE=%.4f | RMSE=%.4f | Ordinal_MAE=%.4f | QWK=%.4f",
        result["val_eval"]["mae"],
        result["val_eval"]["rmse"],
        result["val_eval"]["ordinal_mae"],
        result["val_eval"]["qwk"],
    )
    logger.info(
        "Evaluation (test) | MAE=%.4f | RMSE=%.4f | Ordinal_MAE=%.4f | QWK=%.4f",
        result["test_eval"]["mae"],
        result["test_eval"]["rmse"],
        result["test_eval"]["ordinal_mae"],
        result["test_eval"]["qwk"],
    )


def _build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid ICF model training scaffold")

    parser.add_argument("--table", type=Path, default=Path("./output_ml/training_table.csv"), help="Training table CSV")
    parser.add_argument("--subject-col", default="subject_id", help="Subject ID column")
    parser.add_argument("--target-cols", required=True, help="Comma-separated list of 4 target columns")
    parser.add_argument(
        "--target-scale-mode",
        choices=["none", "map_1_5_to_0_100", "auto"],
        default="auto",
        help="Target scale handling before training",
    )
    parser.add_argument(
        "--target-source-min",
        type=float,
        default=1.0,
        help="Source target minimum used for linear remapping to [0,100]",
    )
    parser.add_argument(
        "--target-source-max",
        type=float,
        default=5.0,
        help="Source target maximum used for linear remapping to [0,100]",
    )

    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["subject", "row"], default="subject")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cv-mode", choices=["none", "loso", "groupkfold"], default="none")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of folds for groupkfold mode")

    parser.add_argument("--hrv-hidden-dim", type=int, default=128)
    parser.add_argument("--sensor-model-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-sensor-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience on val total loss")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum val loss improvement to reset patience")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("./output_ml/checkpoints"))
    parser.add_argument("--run-name", default=None, help="Optional checkpoint run name")
    parser.add_argument("--save-last", action="store_true", help="Save final-epoch checkpoint in addition to best")
    parser.add_argument("--history-csv", type=Path, default=None, help="Optional path for epoch history CSV")
    parser.add_argument("--report-dir", type=Path, default=Path("./output_ml/reports"), help="Directory for evaluation reports")
    parser.add_argument("--n-classes", type=int, default=5, help="Number of ordinal classes for ordinal-aware metrics")

    parser.add_argument("--alpha", type=float, default=0.5, help="Theory-informed C_a loss weight")
    parser.add_argument("--beta", type=float, default=0.3, help="Ordinal contrastive loss weight")
    parser.add_argument("--margin", type=float, default=15.0, help="Ordinal margin")

    parser.add_argument("--omega-f", type=float, default=0.5, help="C_a frequency term weight")
    parser.add_argument("--omega-q", type=float, default=0.5, help="C_a quality term weight")
    parser.add_argument("--ca-tau", type=float, default=0.5, help="C_a threshold")
    parser.add_argument("--ca-expected-frequency", type=float, default=1.0, help="C_a expected frequency")
    parser.add_argument("--ca-expected-duration", type=float, default=1.0, help="C_a expected duration")

    parser.add_argument("--enable-ca", action="store_true", help="Enable C_a regularization term")
    parser.add_argument(
        "--ca-input-mode",
        choices=["placeholder", "adl"],
        default="placeholder",
        help="Source for C_a inputs (placeholder or ADL classifier columns)",
    )
    parser.add_argument(
        "--ca-prob-columns",
        default=None,
        help="Comma-separated ADL classifier probability columns p_i",
    )
    parser.add_argument(
        "--ca-duration-columns",
        default=None,
        help="Comma-separated duration columns d_i aligned with probability columns",
    )

    parser.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA is available")

    if defaults:
        parser.set_defaults(**defaults)

    return parser


def _load_config_defaults(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f) or {}

    if not isinstance(cfg_raw, dict):
        raise ValueError("Config file must contain a YAML mapping/object")

    defaults: dict = {}

    def merge_section(section_name: str, keys: list[str]) -> None:
        section = cfg_raw.get(section_name, {})
        if isinstance(section, dict):
            for key in keys:
                if key in section:
                    defaults[key] = section[key]

    for key in [
        "table",
        "subject_col",
        "target_cols",
        "target_scale_mode",
        "target_source_min",
        "target_source_max",
        "split_mode",
        "batch_size",
        "seed",
        "force_cpu",
        "cv_mode",
        "cv_folds",
    ]:
        if key in cfg_raw:
            defaults[key] = cfg_raw[key]

    merge_section("split", ["train_frac", "val_frac", "test_frac"])
    merge_section(
        "model",
        [
            "hrv_hidden_dim",
            "sensor_model_dim",
            "fusion_hidden_dim",
            "num_heads",
            "num_sensor_layers",
            "dropout",
        ],
    )
    merge_section(
        "optimization",
        [
            "learning_rate",
            "weight_decay",
            "epochs",
            "patience",
            "min_delta",
        ],
    )
    merge_section(
        "output",
        [
            "checkpoint_dir",
            "run_name",
            "save_last",
            "history_csv",
            "report_dir",
        ],
    )
    merge_section(
        "loss",
        ["alpha", "beta", "margin", "n_classes"],
    )
    merge_section(
        "ca",
        [
            "enable_ca",
            "ca_input_mode",
            "ca_prob_columns",
            "ca_duration_columns",
            "omega_f",
            "omega_q",
            "ca_tau",
            "ca_expected_frequency",
            "ca_expected_duration",
        ],
    )

    path_keys = ["table", "checkpoint_dir", "history_csv", "report_dir"]
    for key in path_keys:
        if key in defaults and defaults[key] is not None:
            defaults[key] = Path(defaults[key])

    return defaults


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None, help="Path to YAML experiment config")
    pre_args, _ = pre_parser.parse_known_args()

    config_defaults = {}
    if pre_args.config is not None:
        config_defaults = _load_config_defaults(pre_args.config)

    parser = _build_parser(defaults=config_defaults)
    parser.add_argument("--config", type=Path, default=pre_args.config, help="Path to YAML experiment config")
    args = parser.parse_args()

    if args.config is not None:
        logger.info("Loaded config: %s", args.config)

    return args


if __name__ == "__main__":
    run_train_step(parse_args())
