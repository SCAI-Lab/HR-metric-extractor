#!/usr/bin/env python3
"""
Feature selection workflow for subject-level ICF modeling.

This script supports two complementary selection families:
1) Filter methods: target correlation ranking + inter-feature collinearity pruning.
2) Wrapper methods: greedy forward selection and backward elimination
   using LOOCV multi-output ridge regression MAE.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from upsetplot import from_contents, UpSet

from ml_build_dataset import (
    _parse_path_list,
    build_training_table,
    find_latest_batch,
    load_combined_icf_targets,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_target_cols(value: str | None) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_id_col(value: str) -> str | int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    return value


def _load_or_build_table(args: argparse.Namespace) -> pd.DataFrame:
    if not args.build_from_batch and args.table is not None:
        if not args.table.exists():
            raise FileNotFoundError(f"Training table not found: {args.table}")
        table = pd.read_csv(args.table)
        logger.info("Loaded training table: %s (shape=%s)", args.table, table.shape)
        return table

    if not args.target_cols:
        raise ValueError("--target-cols is required when building table from batch outputs")

    explicit_batch_dirs = _parse_path_list(args.batch_dirs)
    if explicit_batch_dirs:
        batch_dirs = explicit_batch_dirs
    else:
        batch_dir = args.batch_dir if args.batch_dir is not None else find_latest_batch(args.batch_root)
        batch_dirs = [batch_dir]

    explicit_icf_csvs = _parse_path_list(args.icf_csvs)
    if explicit_icf_csvs:
        icf_csvs = explicit_icf_csvs
    else:
        if args.icf_csv is None:
            raise ValueError("--icf-csv or --icf-csvs is required when building table from batch outputs")
        icf_csvs = [args.icf_csv]

    id_col = _parse_id_col(str(args.id_col))
    icf_targets = load_combined_icf_targets(icf_csvs=icf_csvs, id_col=id_col, target_cols=args.target_cols)
    table = build_training_table(
        batch_dirs=batch_dirs,
        icf_targets=icf_targets,
        min_feature_non_nan_ratio=args.min_feature_non_nan_ratio,
    )
    logger.info("Built training table from batch dirs: %s (shape=%s)", [str(path) for path in batch_dirs], table.shape)

    if args.out_table is not None:
        args.out_table.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out_table, index=False)
        logger.info("Saved built training table to: %s", args.out_table)

    return table


def _resolve_target_cols(table: pd.DataFrame, target_cols_arg: Sequence[str]) -> List[str]:
    if target_cols_arg:
        missing = [col for col in target_cols_arg if col not in table.columns]
        if missing:
            raise ValueError(f"Target column(s) missing from table: {missing}")
        return list(target_cols_arg)

    if "target_score" in table.columns:
        logger.info("No --target-cols provided; defaulting to ['target_score']")
        return ["target_score"]

    raise ValueError("Could not infer target columns. Provide --target-cols.")


def _prepare_features(
    table: pd.DataFrame,
    target_cols: Sequence[str],
    subject_col: str,
    min_non_nan_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    if subject_col not in table.columns:
        raise ValueError(f"Subject column '{subject_col}' not found")

    y = table[list(target_cols)].apply(pd.to_numeric, errors="coerce")
    valid_rows = ~y.isna().any(axis=1)
    if valid_rows.sum() < 3:
        raise ValueError("Not enough rows with valid targets for analysis")

    work = table.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    numeric_cols = [
        col
        for col in work.columns
        if col not in set(target_cols) | {subject_col}
        and pd.api.types.is_numeric_dtype(work[col])
    ]

    if not numeric_cols:
        raise ValueError("No numeric feature columns found")

    min_non_nan = int(np.ceil(min_non_nan_ratio * len(work)))
    kept_cols: List[str] = []
    dropped_cols: List[str] = []
    drop_rows: List[Dict[str, float | int | str]] = []

    for col in numeric_cols:
        non_nan = int(work[col].notna().sum())
        if non_nan >= min_non_nan:
            kept_cols.append(col)
        else:
            dropped_cols.append(col)
            drop_rows.append(
                {
                    "feature": col,
                    "stage": "prepare_features",
                    "target": "all",
                    "step": "min_non_nan_ratio",
                    "reason": "insufficient_non_nan_coverage",
                    "non_nan_count": non_nan,
                    "threshold_non_nan": int(min_non_nan),
                }
            )

    x = work[kept_cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True))

    variances = x.var(axis=0, ddof=0)
    variable_cols = variances[variances > 0.0].index.tolist()
    dropped_constant = sorted(set(kept_cols) - set(variable_cols))

    if dropped_constant:
        dropped_cols.extend(dropped_constant)
        for col in dropped_constant:
            drop_rows.append(
                {
                    "feature": col,
                    "stage": "prepare_features",
                    "target": "all",
                    "step": "constant_variance",
                    "reason": "zero_variance",
                    "non_nan_count": int(work[col].notna().sum()) if col in work.columns else np.nan,
                    "threshold_non_nan": int(min_non_nan),
                }
            )

    x = x[variable_cols]

    if x.shape[1] == 0:
        raise ValueError("No usable features remained after NaN/constant filtering")

    logger.info(
        "Prepared matrix with %d samples, %d features (dropped %d)",
        x.shape[0],
        x.shape[1],
        len(dropped_cols),
    )

    dropped_detail_df = pd.DataFrame(drop_rows)
    return x, y, dropped_cols, dropped_detail_df


def _build_feature_drop_audit(
    all_features_after_prepare: Sequence[str],
    dropped_prepare_df: pd.DataFrame,
    corr_selected_by_target: Dict[str, List[str]],
    wrapper_candidates_by_target: Dict[str, List[str]],
    forward_trace: pd.DataFrame,
    backward_trace: pd.DataFrame,
    wrapper_selected_by_target: Dict[str, List[str]],
    collinearity_decisions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float | int | str]] = []

    if not dropped_prepare_df.empty:
        for rec in dropped_prepare_df.to_dict("records"):
            rows.append(
                {
                    "feature": str(rec.get("feature", "")),
                    "stage": str(rec.get("stage", "prepare_features")),
                    "target": str(rec.get("target", "all")),
                    "step": str(rec.get("step", "prepare_features")),
                    "reason": str(rec.get("reason", "dropped")),
                    "ref_feature": "",
                    "score_before": np.nan,
                    "score_after": np.nan,
                    "metric": "",
                }
            )

    all_features_set = set(all_features_after_prepare)

    for target, corr_selected in corr_selected_by_target.items():
        corr_selected_set = set(corr_selected)
        dropped_corr = sorted(all_features_set - corr_selected_set)
        for feature in dropped_corr:
            rows.append(
                {
                    "feature": feature,
                    "stage": "filter",
                    "target": target,
                    "step": "target_corr_top_n_cutoff",
                    "reason": "outside_corr_max_features",
                    "ref_feature": "",
                    "score_before": np.nan,
                    "score_after": np.nan,
                    "metric": "max_abs_corr",
                }
            )

        wrapper_candidates = wrapper_candidates_by_target.get(target, [])
        wrapper_candidates_set = set(wrapper_candidates)
        dropped_wrapper_cutoff = sorted(corr_selected_set - wrapper_candidates_set)
        for feature in dropped_wrapper_cutoff:
            rows.append(
                {
                    "feature": feature,
                    "stage": "wrapper",
                    "target": target,
                    "step": "wrapper_candidate_top_k_cutoff",
                    "reason": "outside_wrapper_candidates",
                    "ref_feature": "",
                    "score_before": np.nan,
                    "score_after": np.nan,
                    "metric": "max_abs_corr",
                }
            )

        selected_final = set(wrapper_selected_by_target.get(target, []))
        dropped_not_selected = sorted(wrapper_candidates_set - selected_final)
        for feature in dropped_not_selected:
            rows.append(
                {
                    "feature": feature,
                    "stage": "wrapper",
                    "target": target,
                    "step": "wrapper_not_in_target_final",
                    "reason": "not_selected_by_forward_backward",
                    "ref_feature": "",
                    "score_before": np.nan,
                    "score_after": np.nan,
                    "metric": "mae",
                }
            )

    if not forward_trace.empty:
        fwd_stop = forward_trace[forward_trace["operation"] == "stop"]
        for rec in fwd_stop.to_dict("records"):
            rows.append(
                {
                    "feature": str(rec.get("feature", "")),
                    "stage": "wrapper",
                    "target": str(rec.get("target", "all")),
                    "step": "forward_stop",
                    "reason": "min_improvement_not_met",
                    "ref_feature": "",
                    "score_before": np.nan,
                    "score_after": float(rec.get("score_mae", np.nan)),
                    "metric": "mae",
                }
            )

    if not backward_trace.empty:
        bwd_remove = backward_trace[backward_trace["operation"] == "remove"]
        for rec in bwd_remove.to_dict("records"):
            rows.append(
                {
                    "feature": str(rec.get("feature", "")),
                    "stage": "wrapper",
                    "target": str(rec.get("target", "all")),
                    "step": "backward_remove",
                    "reason": "removed_by_backward_elimination",
                    "ref_feature": "",
                    "score_before": np.nan,
                    "score_after": float(rec.get("score_mae", np.nan)),
                    "metric": "mae",
                }
            )

        bwd_stop = backward_trace[backward_trace["operation"] == "stop"]
        for rec in bwd_stop.to_dict("records"):
            rows.append(
                {
                    "feature": str(rec.get("feature", "")),
                    "stage": "wrapper",
                    "target": str(rec.get("target", "all")),
                    "step": "backward_stop",
                    "reason": "min_improvement_not_met",
                    "ref_feature": "",
                    "score_before": np.nan,
                    "score_after": float(rec.get("score_mae", np.nan)),
                    "metric": "mae",
                }
            )

    if not collinearity_decisions.empty:
        for rec in collinearity_decisions.to_dict("records"):
            rows.append(
                {
                    "feature": str(rec.get("feature_dropped", "")),
                    "stage": "union_prune",
                    "target": "all",
                    "step": "post_union_collinearity",
                    "reason": "high_collinearity_pair",
                    "ref_feature": str(rec.get("feature_kept", "")),
                    "score_before": float(rec.get("dropped_predictive_power", np.nan)),
                    "score_after": float(rec.get("kept_predictive_power", np.nan)),
                    "metric": "predictive_power",
                }
            )

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        summary_df = pd.DataFrame(
            columns=["stage", "target", "step", "reason", "n_drop_events", "n_unique_features"]
        )
    else:
        summary_df = (
            detail_df.groupby(["stage", "target", "step", "reason"], as_index=False)
            .agg(
                n_drop_events=("feature", "size"),
                n_unique_features=("feature", pd.Series.nunique),
            )
            .sort_values(["stage", "target", "step", "reason"])
            .reset_index(drop=True)
        )

    return detail_df, summary_df


def _feature_target_correlation_table(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []

    for feature in x.columns:
        xv = x[feature].to_numpy(dtype=float)
        for target in y.columns:
            yv = y[target].to_numpy(dtype=float)

            if np.std(xv) == 0.0 or np.std(yv) == 0.0:
                pearson_r = np.nan
                pearson_p = np.nan
                spearman_r = np.nan
                spearman_p = np.nan
            else:
                pearson_r, pearson_p = stats.pearsonr(xv, yv)
                spearman_r, spearman_p = stats.spearmanr(xv, yv)

            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "pearson_r": float(pearson_r) if not pd.isna(pearson_r) else np.nan,
                    "pearson_p": float(pearson_p) if not pd.isna(pearson_p) else np.nan,
                    "spearman_r": float(spearman_r) if not pd.isna(spearman_r) else np.nan,
                    "spearman_p": float(spearman_p) if not pd.isna(spearman_p) else np.nan,
                    "abs_corr_score": float(
                        np.nanmax([abs(pearson_r), abs(spearman_r)])
                        if not (pd.isna(pearson_r) and pd.isna(spearman_r))
                        else np.nan
                    ),
                }
            )

    corr_df = pd.DataFrame(rows)
    return corr_df.sort_values(["abs_corr_score", "feature", "target"], ascending=[False, True, True]).reset_index(drop=True)


def _aggregate_feature_scores(corr_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        corr_df.groupby("feature", as_index=False)
        .agg(
            max_abs_corr=("abs_corr_score", "max"),
            mean_abs_corr=("abs_corr_score", "mean"),
            best_target=("target", lambda s: s.iloc[int(corr_df.loc[s.index, "abs_corr_score"].argmax())]),
        )
        .sort_values(["max_abs_corr", "mean_abs_corr"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return agg


def _safe_target_name(target: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(target).strip())
    return safe.strip("_") or "target"


def _build_feature_priority_rows(
    target_cols: Sequence[str],
    score_by_target: Dict[str, pd.DataFrame],
    wrapper_selected_by_target: Dict[str, List[str]],
    wrapper_best_score_by_target: Dict[str, float],
) -> pd.DataFrame:
    """
    Build feature-level priority metadata used for post-union collinearity pruning.

    Predictive power is a combined score using:
    - target-specific max absolute correlation
    - wrapper model quality when the feature is wrapper-selected for that target
    """
    rows: List[Dict[str, float | str | bool]] = []
    score_maps = {
        target: dict(zip(df["feature"].tolist(), df["max_abs_corr"].tolist()))
        for target, df in score_by_target.items()
    }
    wrapper_sets = {
        target: set(features)
        for target, features in wrapper_selected_by_target.items()
    }

    all_features = sorted(
        {
            feature
            for target in target_cols
            for feature in score_maps.get(target, {}).keys()
        }
    )

    for feature in all_features:
        best_row: Dict[str, float | str | bool] | None = None
        best_key: Tuple[int, float, float] | None = None

        for target in target_cols:
            corr_value = score_maps.get(target, {}).get(feature, np.nan)
            corr_value = float(corr_value) if pd.notna(corr_value) else float("-inf")
            wrapper_selected = feature in wrapper_sets.get(target, set())

            wrapper_score = float(wrapper_best_score_by_target.get(target, float("inf")))
            wrapper_strength = 1.0 / (1.0 + wrapper_score) if wrapper_selected and np.isfinite(wrapper_score) else float("-inf")
            predictive_power = max(corr_value, wrapper_strength)

            # Prefer wrapper-selected features first, then higher combined power, then higher correlation.
            rank_key = (1 if wrapper_selected else 0, predictive_power, corr_value)
            if best_key is None or rank_key > best_key:
                best_key = rank_key
                best_row = {
                    "feature": feature,
                    "best_target": target,
                    "best_abs_corr": corr_value,
                    "wrapper_selected": bool(wrapper_selected),
                    "wrapper_best_score_mae": wrapper_score if np.isfinite(wrapper_score) else np.nan,
                    "predictive_power": predictive_power,
                }

        if best_row is not None:
            rows.append(best_row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "best_target",
                "best_abs_corr",
                "wrapper_selected",
                "wrapper_best_score_mae",
                "predictive_power",
            ]
        )

    return pd.DataFrame(rows).sort_values("predictive_power", ascending=False).reset_index(drop=True)


def _high_collinearity_pairs(
    x: pd.DataFrame,
    feature_order: Sequence[str],
    collinearity_scan_top_n: int,
    threshold: float,
) -> pd.DataFrame:
    candidate_cols = list(feature_order[:collinearity_scan_top_n])
    if len(candidate_cols) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"])

    corr = x[candidate_cols].corr().abs()
    pairs: List[Dict[str, float | str]] = []

    for i, a in enumerate(candidate_cols):
        for b in candidate_cols[i + 1 :]:
            value = corr.at[a, b]
            if pd.notna(value) and value >= threshold:
                pairs.append({"feature_a": a, "feature_b": b, "abs_corr": float(value)})

    if not pairs:
        return pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"])

    return pd.DataFrame(pairs).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def _post_union_collinearity_prune(
    x: pd.DataFrame,
    union_features: Sequence[str],
    feature_priority: pd.DataFrame,
    threshold: float,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """
    Apply collinearity pruning only on the union feature set.

    For each highly-collinear pair, keep the feature with higher predictive power.
    Returns:
    - final selected feature list
    - high-correlation pair table restricted to union set
    - prune decision table
    """
    candidate_cols = [feature for feature in union_features if feature in x.columns]
    if len(candidate_cols) < 2:
        return sorted(candidate_cols), pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"]), pd.DataFrame(
            columns=[
                "feature_kept",
                "feature_dropped",
                "abs_corr",
                "kept_predictive_power",
                "dropped_predictive_power",
                "kept_target",
                "dropped_target",
                "decision_rule",
            ]
        )

    corr = x[candidate_cols].corr().abs()
    pair_rows: List[Dict[str, float | str]] = []
    for i, feature_a in enumerate(candidate_cols):
        for feature_b in candidate_cols[i + 1 :]:
            value = corr.at[feature_a, feature_b]
            if pd.notna(value) and value >= threshold:
                pair_rows.append({"feature_a": feature_a, "feature_b": feature_b, "abs_corr": float(value)})

    if not pair_rows:
        return sorted(candidate_cols), pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"]), pd.DataFrame(
            columns=[
                "feature_kept",
                "feature_dropped",
                "abs_corr",
                "kept_predictive_power",
                "dropped_predictive_power",
                "kept_target",
                "dropped_target",
                "decision_rule",
            ]
        )

    high_pairs_df = pd.DataFrame(pair_rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)

    priority_map = feature_priority.set_index("feature").to_dict("index") if not feature_priority.empty else {}

    active = set(candidate_cols)
    decisions: List[Dict[str, float | str]] = []

    for row in high_pairs_df.itertuples(index=False):
        feature_a = str(row.feature_a)
        feature_b = str(row.feature_b)
        abs_corr = float(row.abs_corr)

        if feature_a not in active or feature_b not in active:
            continue

        meta_a = priority_map.get(feature_a, {})
        meta_b = priority_map.get(feature_b, {})

        power_a = float(meta_a.get("predictive_power", float("-inf")))
        power_b = float(meta_b.get("predictive_power", float("-inf")))
        corr_a = float(meta_a.get("best_abs_corr", float("-inf")))
        corr_b = float(meta_b.get("best_abs_corr", float("-inf")))

        # Keep stronger feature by predictive power; break ties by correlation then name.
        rank_a = (power_a, corr_a, feature_a)
        rank_b = (power_b, corr_b, feature_b)
        if rank_a >= rank_b:
            kept, dropped = feature_a, feature_b
            kept_meta, dropped_meta = meta_a, meta_b
            kept_power, dropped_power = power_a, power_b
        else:
            kept, dropped = feature_b, feature_a
            kept_meta, dropped_meta = meta_b, meta_a
            kept_power, dropped_power = power_b, power_a

        active.remove(dropped)
        decisions.append(
            {
                "feature_kept": kept,
                "feature_dropped": dropped,
                "abs_corr": abs_corr,
                "kept_predictive_power": kept_power,
                "dropped_predictive_power": dropped_power,
                "kept_target": str(kept_meta.get("best_target", "")),
                "dropped_target": str(dropped_meta.get("best_target", "")),
                "decision_rule": "keep_higher_predictive_power",
            }
        )

    final_selected = sorted(active)
    decisions_df = pd.DataFrame(decisions)
    return final_selected, high_pairs_df, decisions_df


def _correlation_filter_select(
    x: pd.DataFrame,
    ranked_features: Sequence[str],
    threshold: float,
    max_features: int,
) -> List[str]:
    selected: List[str] = []

    for feat in ranked_features:
        if len(selected) >= max_features:
            break

        keep = True
        for selected_feat in selected:
            corr = x[[feat, selected_feat]].corr().iloc[0, 1]
            if pd.notna(corr) and abs(corr) >= threshold:
                keep = False
                break

        if keep:
            selected.append(feat)

    return selected


def _ridge_predict_loocv_mae(x: np.ndarray, y: np.ndarray, alpha: float) -> float:
    n_samples = x.shape[0]

    if n_samples < 3:
        return float("inf")

    errors: List[float] = []

    for i in range(n_samples):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[i] = False

        x_train = x[train_mask]
        y_train = y[train_mask]
        x_test = x[~train_mask]
        y_test = y[~train_mask]

        if x_train.shape[1] == 0:
            y_pred = np.repeat(y_train.mean(axis=0, keepdims=True), repeats=1, axis=0)
        else:
            x_mean = x_train.mean(axis=0, keepdims=True)
            x_std = x_train.std(axis=0, keepdims=True)
            x_std[x_std == 0.0] = 1.0

            x_train_z = (x_train - x_mean) / x_std
            x_test_z = (x_test - x_mean) / x_std

            xtx = x_train_z.T @ x_train_z
            reg = alpha * np.eye(xtx.shape[0], dtype=float)
            xty = x_train_z.T @ y_train

            try:
                beta = np.linalg.solve(xtx + reg, xty)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(xtx + reg) @ xty

            intercept = y_train.mean(axis=0, keepdims=True)
            y_pred = x_test_z @ beta + intercept

        mae = np.mean(np.abs(y_pred - y_test))
        errors.append(float(mae))

    return float(np.mean(errors))


def _forward_selection(
    x: pd.DataFrame,
    y: pd.DataFrame,
    candidate_features: Sequence[str],
    max_features: int,
    alpha: float,
    min_improvement: float,
) -> Tuple[pd.DataFrame, List[str], float]:
    selected: List[str] = []
    remaining = list(candidate_features)

    y_np = y.to_numpy(dtype=float)
    baseline_score = _ridge_predict_loocv_mae(np.zeros((len(y), 0), dtype=float), y_np, alpha=alpha)

    trace_rows: List[Dict[str, float | int | str]] = [
        {
            "step": 0,
            "n_features": 0,
            "operation": "baseline",
            "feature": "",
            "score_mae": baseline_score,
            "improvement": 0.0,
        }
    ]

    best_score = baseline_score
    best_set: List[str] = []

    step = 0
    while remaining and len(selected) < max_features:
        step += 1

        best_feature = None
        best_candidate_score = float("inf")

        for feature in remaining:
            cols = selected + [feature]
            score = _ridge_predict_loocv_mae(x[cols].to_numpy(dtype=float), y_np, alpha=alpha)
            if score < best_candidate_score:
                best_candidate_score = score
                best_feature = feature

        if best_feature is None:
            break

        improvement = best_score - best_candidate_score
        if improvement < min_improvement:
            trace_rows.append(
                {
                    "step": step,
                    "n_features": len(selected),
                    "operation": "stop",
                    "feature": str(best_feature),
                    "score_mae": float(best_candidate_score),
                    "improvement": float(improvement),
                }
            )
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        best_score = best_candidate_score

        if best_score <= min(row["score_mae"] for row in trace_rows if isinstance(row["score_mae"], float)):
            best_set = selected.copy()

        trace_rows.append(
            {
                "step": step,
                "n_features": len(selected),
                "operation": "add",
                "feature": best_feature,
                "score_mae": float(best_score),
                "improvement": float(improvement),
            }
        )

    if not best_set:
        best_set = selected.copy()

    return pd.DataFrame(trace_rows), best_set, best_score


def _backward_elimination(
    x: pd.DataFrame,
    y: pd.DataFrame,
    initial_features: Sequence[str],
    min_features: int,
    alpha: float,
    min_improvement: float,
) -> Tuple[pd.DataFrame, List[str], float]:
    selected = list(initial_features)
    y_np = y.to_numpy(dtype=float)

    if not selected:
        empty_trace = pd.DataFrame(
            [{"step": 0, "n_features": 0, "operation": "empty", "feature": "", "score_mae": np.nan, "improvement": 0.0}]
        )
        return empty_trace, [], float("inf")

    current_score = _ridge_predict_loocv_mae(x[selected].to_numpy(dtype=float), y_np, alpha=alpha)

    trace_rows: List[Dict[str, float | int | str]] = [
        {
            "step": 0,
            "n_features": len(selected),
            "operation": "start",
            "feature": "",
            "score_mae": float(current_score),
            "improvement": 0.0,
        }
    ]

    best_score = current_score
    best_set = selected.copy()

    step = 0
    while len(selected) > min_features:
        step += 1

        best_feature_to_remove = None
        best_candidate_score = float("inf")

        for feature in selected:
            candidate = [f for f in selected if f != feature]
            score = _ridge_predict_loocv_mae(x[candidate].to_numpy(dtype=float), y_np, alpha=alpha)
            if score < best_candidate_score:
                best_candidate_score = score
                best_feature_to_remove = feature

        if best_feature_to_remove is None:
            break

        improvement = current_score - best_candidate_score
        if improvement < min_improvement:
            trace_rows.append(
                {
                    "step": step,
                    "n_features": len(selected),
                    "operation": "stop",
                    "feature": str(best_feature_to_remove),
                    "score_mae": float(best_candidate_score),
                    "improvement": float(improvement),
                }
            )
            break

        selected.remove(best_feature_to_remove)
        current_score = best_candidate_score

        if current_score < best_score:
            best_score = current_score
            best_set = selected.copy()

        trace_rows.append(
            {
                "step": step,
                "n_features": len(selected),
                "operation": "remove",
                "feature": best_feature_to_remove,
                "score_mae": float(current_score),
                "improvement": float(improvement),
            }
        )

    return pd.DataFrame(trace_rows), best_set, float(best_score)


# ---------------------------------------------------------------------------
# Plot / report generator
# ---------------------------------------------------------------------------

def _generate_plots(
    out_dir: Path,
    corr_df: pd.DataFrame,
    score_df: pd.DataFrame,
    high_corr_pairs: pd.DataFrame,
    forward_trace: pd.DataFrame,
    backward_trace: pd.DataFrame,
    corr_selected: List[str],
    final_selected: List[str],
    wrapper_selected_by_target: Dict[str, List[str]] | None = None,
    x: pd.DataFrame | None = None,
    y: pd.DataFrame | None = None,
    top_n: int = 25,
    plot_scale: float = 1.0,
) -> None:
    """Write diagnostic plots into *out_dir*/plots/."""
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _short_name(value: str, max_len: int = 60) -> str:
        return value[:max_len] + "..." if len(value) > max_len else value

    def _figsize(width: float, height: float) -> Tuple[float, float]:
        scale = max(0.2, float(plot_scale))
        return (max(1.0, width * scale), max(1.0, height * scale))

    # ------------------------------------------------------------------
    # 1. Top-feature correlation bar chart (max_abs_corr across targets)
    # ------------------------------------------------------------------
    top_scores = score_df.head(top_n).copy()
    # Shorten feature names for readability
    short_names = [_short_name(str(f), max_len=60) for f in top_scores["feature"]]

    fig, ax = plt.subplots(figsize=_figsize(10, max(4, top_n * 0.35)))
    colors = [
        "#2196F3" if f in final_selected else
        "#4CAF50" if f in corr_selected else "#BDBDBD"
        for f in top_scores["feature"]
    ]
    bars = ax.barh(range(len(top_scores)), top_scores["max_abs_corr"], color=colors)
    ax.set_yticks(range(len(top_scores)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Max |correlation| across targets")
    ax.set_title(f"Top {top_n} Features by Correlation Strength")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Wrapper-selected"),
        Patch(facecolor="#4CAF50", label="Filter-selected"),
        Patch(facecolor="#BDBDBD", label="Not selected"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "top_feature_correlations.png", dpi=130)
    plt.close(fig)
    logger.info("Saved top_feature_correlations.png")

    # ------------------------------------------------------------------
    # 2. Per-target top-correlation heatmap
    # ------------------------------------------------------------------
    top_features = score_df["feature"].head(top_n).tolist()
    heatmap_df = (
        corr_df[corr_df["feature"].isin(top_features)]
        .pivot_table(index="feature", columns="target", values="pearson_r")
        .reindex(top_features)  # keep ranking order
    )

    n_targets = heatmap_df.shape[1]
    fig, ax = plt.subplots(figsize=_figsize(max(4, n_targets * 1.4), max(4, top_n * 0.35)))
    vmax = float(heatmap_df.abs().max().max())
    im = ax.imshow(heatmap_df.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_targets))
    ax.set_xticklabels(heatmap_df.columns.tolist(), rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(
        [f[:60] + "..." if len(f) > 60 else f for f in top_features], fontsize=7
    )
    ax.set_title(f"Pearson r — Top {top_n} Features × Targets")
    plt.colorbar(im, ax=ax, label="Pearson r")
    # Mark wrapper-selected features
    for row_idx, feat in enumerate(top_features):
        if feat in final_selected:
            ax.get_yticklabels()[row_idx].set_fontweight("bold")
            ax.get_yticklabels()[row_idx].set_color("#1565C0")
    fig.tight_layout()
    fig.savefig(plots_dir / "correlation_heatmap.png", dpi=130)
    plt.close(fig)
    logger.info("Saved correlation_heatmap.png")

    # ------------------------------------------------------------------
    # 3. Target-specific top-correlation bars
    # ------------------------------------------------------------------
    if "target" in corr_df.columns:
        targets = sorted(corr_df["target"].dropna().unique().tolist())
    else:
        targets = []

    if targets:
        n_targets = len(targets)
        ncols = 2 if n_targets > 1 else 1
        nrows = int(np.ceil(n_targets / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=_figsize(12, max(3.8, 3.3 * nrows)))
        axes_flat = np.atleast_1d(axes).ravel()

        for idx, target in enumerate(targets):
            ax_t = axes_flat[idx]
            score_target = _aggregate_feature_scores(corr_df[corr_df["target"] == target])
            top_target = score_target.head(top_n)

            target_features = top_target["feature"].tolist()
            labels = [_short_name(str(feat), max_len=42) for feat in target_features]
            values = top_target["max_abs_corr"].tolist()
            colors = ["#1976D2" if feat in final_selected else "#90CAF9" for feat in target_features]

            ax_t.barh(range(len(top_target)), values, color=colors)
            ax_t.set_yticks(range(len(top_target)))
            ax_t.set_yticklabels(labels, fontsize=7)
            ax_t.invert_yaxis()
            ax_t.set_xlim(0.0, 1.0)
            ax_t.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax_t.set_title(f"{target}", fontsize=10)
            ax_t.grid(axis="x", alpha=0.25)

        for idx in range(n_targets, len(axes_flat)):
            fig.delaxes(axes_flat[idx])

        fig.suptitle(f"Top {top_n} Correlations per Target", y=0.995)
        fig.tight_layout()
        fig.savefig(plots_dir / "top_correlations_by_target.png", dpi=130)
        plt.close(fig)
        logger.info("Saved top_correlations_by_target.png")

    # ------------------------------------------------------------------
    # 4. Wrapper score-vs-feature-count curves (combined and faceted)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=_figsize(9, 5))

    if "target" in forward_trace.columns:
        targets_fwd = sorted(forward_trace["target"].dropna().unique().tolist())
    else:
        targets_fwd = ["all_targets"]
        forward_trace = forward_trace.copy()
        forward_trace["target"] = "all_targets"

    if "target" in backward_trace.columns:
        targets_bwd = sorted(backward_trace["target"].dropna().unique().tolist())
    else:
        targets_bwd = ["all_targets"]
        backward_trace = backward_trace.copy()
        backward_trace["target"] = "all_targets"

    targets = sorted(set(targets_fwd) | set(targets_bwd))
    cmap = plt.get_cmap("tab10")

    for idx, target in enumerate(targets):
        color = cmap(idx % 10)

        fwd_plot = forward_trace[
            (forward_trace["target"] == target)
            & (forward_trace["operation"].isin(["baseline", "add"]))
        ].copy()
        if not fwd_plot.empty:
            ax.plot(
                fwd_plot["n_features"].tolist(),
                fwd_plot["score_mae"].tolist(),
                marker="o",
                label=f"Forward: {target}",
                color=color,
                linewidth=1.8,
            )

        bwd_plot = backward_trace[
            (backward_trace["target"] == target)
            & (backward_trace["operation"].isin(["start", "remove"]))
        ].copy()
        if not bwd_plot.empty:
            ax.plot(
                bwd_plot["n_features"].tolist(),
                bwd_plot["score_mae"].tolist(),
                marker="s",
                label=f"Backward: {target}",
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
            )

    ax.set_xlabel("Number of features")
    ax.set_ylabel("LOOCV MAE (single-target ridge)")
    ax.set_title("Wrapper Method: Score vs Feature Count by Target")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "wrapper_score_curve.png", dpi=130)
    plt.close(fig)
    logger.info("Saved wrapper_score_curve.png")

    if targets:
        n_targets = len(targets)
        ncols = 2 if n_targets > 1 else 1
        nrows = int(np.ceil(n_targets / ncols))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=_figsize(12, max(3.6, 3.2 * nrows)),
            sharey=True,
        )
        axes_flat = np.atleast_1d(axes).ravel()

        for idx, target in enumerate(targets):
            ax_t = axes_flat[idx]
            color = cmap(idx % 10)

            fwd_plot = forward_trace[
                (forward_trace["target"] == target)
                & (forward_trace["operation"].isin(["baseline", "add"]))
            ].copy()
            if not fwd_plot.empty:
                ax_t.plot(
                    fwd_plot["n_features"].tolist(),
                    fwd_plot["score_mae"].tolist(),
                    marker="o",
                    color=color,
                    linewidth=1.8,
                    label="forward",
                )

            bwd_plot = backward_trace[
                (backward_trace["target"] == target)
                & (backward_trace["operation"].isin(["start", "remove"]))
            ].copy()
            if not bwd_plot.empty:
                ax_t.plot(
                    bwd_plot["n_features"].tolist(),
                    bwd_plot["score_mae"].tolist(),
                    marker="s",
                    color=color,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.9,
                    label="backward",
                )

            ax_t.set_title(_short_name(target, max_len=35), fontsize=10)
            ax_t.set_xlabel("n features")
            ax_t.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax_t.grid(True, alpha=0.3)

        for idx in range(n_targets, len(axes_flat)):
            fig.delaxes(axes_flat[idx])

        if len(axes_flat) > 0:
            axes_flat[0].set_ylabel("LOOCV MAE")
            handles, labels = axes_flat[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9)

        fig.suptitle("Wrapper Score Curves by Target", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(plots_dir / "wrapper_score_curve_by_target.png", dpi=130)
        plt.close(fig)
        logger.info("Saved wrapper_score_curve_by_target.png")

    # ------------------------------------------------------------------
    # 5. Top collinearity pairs (if any)
    # ------------------------------------------------------------------
    if not high_corr_pairs.empty:
        top_pairs = high_corr_pairs.head(min(30, len(high_corr_pairs)))
        labels = [
            f"{str(r.feature_a)[:30]}… × {str(r.feature_b)[:30]}…"
            if (len(str(r.feature_a)) > 30 or len(str(r.feature_b)) > 30)
            else f"{r.feature_a} × {r.feature_b}"
            for _, r in top_pairs.iterrows()
        ]
        fig, ax = plt.subplots(figsize=_figsize(9, max(3, len(top_pairs) * 0.32)))
        ax.barh(range(len(top_pairs)), top_pairs["abs_corr"].tolist(), color="#FF7043")
        ax.set_yticks(range(len(top_pairs)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Absolute inter-feature correlation")
        ax.set_title("High Collinearity Feature Pairs")
        ax.axvline(x=0.9, color="k", linestyle=":", linewidth=0.8, label="threshold=0.9")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / "collinearity_pairs.png", dpi=130)
        plt.close(fig)
        logger.info("Saved collinearity_pairs.png")

    # ------------------------------------------------------------------
    # 6. UpSet plot — feature overlap across targets
    # ------------------------------------------------------------------
    if wrapper_selected_by_target and any(wrapper_selected_by_target.values()):
        # upsetplot needs at least 2 non-empty categories
        non_empty = {k: v for k, v in wrapper_selected_by_target.items() if v}
        if len(non_empty) >= 2:
            try:
                upset_data = from_contents(non_empty)
                fig_up = plt.figure(figsize=_figsize(max(10, len(non_empty) * 2), 6))
                upset = UpSet(
                    upset_data,
                    subset_size="count",
                    show_counts=True,
                    sort_by="cardinality",
                    element_size=40,
                )
                upset.plot(fig=fig_up)
                fig_up.suptitle("Feature Overlap Across Targets", fontsize=13, y=1.01)
                fig_up.savefig(plots_dir / "upset_feature_intersections.png", dpi=130, bbox_inches="tight")
                plt.close(fig_up)
                logger.info("Saved upset_feature_intersections.png")
            except Exception as exc:
                logger.warning("UpSet plot skipped: %s", exc)

    # ------------------------------------------------------------------
    # 7. Clustered heatmap — union features × targets
    # ------------------------------------------------------------------
    if x is not None and y is not None and final_selected:
        available = [f for f in final_selected if f in x.columns]
        if available and not y.empty:
            try:
                x_sel = x[available]
                combined = pd.concat([x_sel, y], axis=1)
                full_corr = combined.corr(method="pearson")
                target_names = [col for col in y.columns if col in full_corr.columns]
                feature_target_corr = full_corr.loc[available, target_names]

                fig_height = max(16, len(available) * 0.38)
                cg = sns.clustermap(
                    feature_target_corr,
                    cmap="coolwarm",
                    center=0,
                    annot=len(available) <= 40,
                    fmt=".2f",
                    figsize=_figsize(max(15, len(target_names) * 1.6), fig_height),
                    col_cluster=False,
                    row_cluster=True,
                    linewidths=0.4,
                    cbar_pos=(0.02, 0.82, 0.04, 0.15),
                    yticklabels=[
                        _short_name(str(f), max_len=100)
                        for f in feature_target_corr.index
                    ],
                )
                cg.ax_heatmap.set_title(
                    "Feature–Target Pearson r (Union, Clustered)",
                    fontsize=12,
                    pad=16,
                )
                cg.ax_heatmap.set_xticklabels(
                    cg.ax_heatmap.get_xticklabels(), rotation=30, ha="right", fontsize=9
                )
                cg.ax_heatmap.set_yticklabels(
                    cg.ax_heatmap.get_yticklabels(), fontsize=7
                )
                cg.figure.savefig(
                    plots_dir / "clustered_heatmap_union_features.png",
                    dpi=130,
                    bbox_inches="tight",
                )
                plt.close(cg.figure)
                logger.info("Saved clustered_heatmap_union_features.png")
            except Exception as exc:
                logger.warning("Clustered heatmap skipped: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature selection with filter + wrapper methods")

    parser.add_argument("--table", type=Path, default=Path("./output_ml/training_table.csv"), help="Existing training table CSV")
    parser.add_argument("--build-from-batch", action="store_true", help="Ignore --table and build training table from batch outputs")
    parser.add_argument("--batch-root", type=Path, default=Path("./output_batch"), help="Root with batch_* dirs")
    parser.add_argument("--batch-dir", type=Path, default=None, help="Specific batch dir; used if --table is omitted")
    parser.add_argument("--batch-dirs", default=None, help="Comma-separated list of batch dirs to combine")
    parser.add_argument("--icf-csv", type=Path, default=None, help="ICF targets CSV (required if --table omitted)")
    parser.add_argument("--icf-csvs", default=None, help="Comma-separated list of ICF target CSVs to combine")
    parser.add_argument("--id-col", default="0", help="ICF subject id column name/index (for batch build mode)")
    parser.add_argument("--target-cols", default=None, help="Comma-separated target columns; defaults to target_score when present")
    parser.add_argument("--subject-col", default="subject_id", help="Subject id column in table")
    parser.add_argument("--out-dir", type=Path, default=Path("./output_ml/feature_selection"), help="Output directory")
    parser.add_argument("--out-table", type=Path, default=None, help="Save built table here (batch build mode only)")
    parser.add_argument(
        "--min-feature-non-nan-ratio",
        type=float,
        default=0.6,
        help="Drop feature columns below this non-NaN ratio when building a table from batch outputs",
    )

    parser.add_argument("--min-non-nan-ratio", type=float, default=0.6, help="Minimum non-NaN ratio per feature")
    parser.add_argument("--collinearity-threshold", type=float, default=0.9, help="Absolute correlation threshold for collinearity")
    parser.add_argument("--collinearity-scan-top-n", type=int, default=200, help="How many top-ranked features to scan for pairwise collinearity report")
    parser.add_argument("--corr-max-features", type=int, default=80, help="Maximum number of filter-selected features")

    parser.add_argument("--wrapper-candidates-top-k", type=int, default=30, help="Top filter-ranked features considered by wrappers")
    parser.add_argument("--wrapper-max-features", type=int, default=12, help="Max features added by forward selection")
    parser.add_argument("--wrapper-min-features", type=int, default=4, help="Min features retained by backward elimination")
    parser.add_argument("--wrapper-alpha", type=float, default=1.0, help="Ridge regularization strength")
    parser.add_argument("--wrapper-min-improvement", type=float, default=1e-4, help="Minimum LOOCV MAE improvement required to keep step")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating diagnostic plots")
    parser.add_argument("--plots-only", action="store_true", help="Reuse existing artifacts in --out-dir and regenerate plots only")
    parser.add_argument("--plot-top-n", type=int, default=25, help="Number of top features to show in correlation plots")
    parser.add_argument("--plot-scale", type=float, default=1.0, help="Scale factor applied to all figure sizes")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.target_cols = _parse_target_cols(args.target_cols)

    if args.plots_only:
        args.out_dir.mkdir(parents=True, exist_ok=True)

        corr_path = args.out_dir / "feature_target_correlations.csv"
        score_path = args.out_dir / "feature_scores_aggregated.csv"
        fwd_path = args.out_dir / "wrapper_forward_selection_trace.csv"
        bwd_path = args.out_dir / "wrapper_backward_elimination_trace.csv"
        final_path = args.out_dir / "final_selected_union_features.csv"
        corr_sel_path = args.out_dir / "correlation_filter_selected_features.csv"
        high_pairs_path = args.out_dir / "high_collinearity_pairs.csv"
        summary_path = args.out_dir / "feature_selection_summary.json"

        required = [corr_path, score_path, fwd_path, bwd_path]
        missing_required = [str(path) for path in required if not path.exists()]
        if missing_required:
            raise FileNotFoundError(f"Missing required artifact files for --plots-only: {missing_required}")

        corr_df = pd.read_csv(corr_path)
        score_df = pd.read_csv(score_path)
        forward_trace = pd.read_csv(fwd_path)
        backward_trace = pd.read_csv(bwd_path)
        high_corr_pairs = pd.read_csv(high_pairs_path) if high_pairs_path.exists() else pd.DataFrame(
            columns=["feature_a", "feature_b", "abs_corr"]
        )

        if final_path.exists():
            final_selected = pd.read_csv(final_path)["feature"].dropna().astype(str).tolist()
        elif corr_sel_path.exists():
            final_selected = pd.read_csv(corr_sel_path)["feature"].dropna().astype(str).tolist()
        else:
            final_selected = []

        corr_selected = (
            pd.read_csv(args.out_dir / "union_master_features_pre_prune.csv")["feature"].dropna().astype(str).tolist()
            if (args.out_dir / "union_master_features_pre_prune.csv").exists()
            else final_selected
        )

        wrapper_selected_by_target: Dict[str, List[str]] = {}
        target_cols: List[str] = list(args.target_cols)
        if summary_path.exists():
            with summary_path.open("r", encoding="ascii") as fp:
                summary_data = json.load(fp)
            target_cols = list(summary_data.get("target_cols", target_cols))
            wrapper_selected_by_target = {
                str(k): [str(v) for v in vals]
                for k, vals in summary_data.get("wrapper", {}).get("selected_by_target", {}).items()
            }

        table_path_for_plots = args.table if args.table is not None and args.table.exists() else (args.out_dir / "training_table_selected_features.csv")
        x = None
        y = None
        if table_path_for_plots.exists():
            table_for_plots = pd.read_csv(table_path_for_plots)
            available_targets = [col for col in target_cols if col in table_for_plots.columns]
            if available_targets:
                y = table_for_plots[available_targets].apply(pd.to_numeric, errors="coerce")
                feature_cols = [
                    col
                    for col in table_for_plots.columns
                    if col not in set(available_targets) | {args.subject_col}
                ]
                if feature_cols:
                    x = table_for_plots[feature_cols].apply(pd.to_numeric, errors="coerce")
                    x = x.fillna(x.median(numeric_only=True))

        _generate_plots(
            out_dir=args.out_dir,
            corr_df=corr_df,
            score_df=score_df,
            high_corr_pairs=high_corr_pairs,
            forward_trace=forward_trace,
            backward_trace=backward_trace,
            corr_selected=corr_selected,
            final_selected=final_selected,
            wrapper_selected_by_target=wrapper_selected_by_target,
            x=x,
            y=y,
            top_n=args.plot_top_n,
            plot_scale=args.plot_scale,
        )
        logger.info("Plots regenerated from artifacts in: %s", args.out_dir)
        return

    table = _load_or_build_table(args)
    target_cols = _resolve_target_cols(table, args.target_cols)

    x, y, dropped_cols, dropped_prepare_df = _prepare_features(
        table=table,
        target_cols=target_cols,
        subject_col=args.subject_col,
        min_non_nan_ratio=args.min_non_nan_ratio,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    corr_frames: List[pd.DataFrame] = []
    score_by_target: Dict[str, pd.DataFrame] = {}
    corr_selected_by_target: Dict[str, List[str]] = {}
    wrapper_selected_by_target: Dict[str, List[str]] = {}
    wrapper_best_score_by_target: Dict[str, float] = {}
    wrapper_candidates_by_target: Dict[str, List[str]] = {}
    forward_traces: List[pd.DataFrame] = []
    backward_traces: List[pd.DataFrame] = []
    union_features: set[str] = set()
    per_target_summary: List[Dict[str, float | int | str | List[str]]] = []

    targets_dir = args.out_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    for target in target_cols:
        y_target = y[[target]].copy()
        target_dir = targets_dir / _safe_target_name(target)
        target_dir.mkdir(parents=True, exist_ok=True)

        corr_target = _feature_target_correlation_table(x=x, y=y_target)
        corr_target.to_csv(target_dir / "feature_target_correlations.csv", index=False)
        corr_frames.append(corr_target)

        score_target = _aggregate_feature_scores(corr_target)
        score_target.to_csv(target_dir / "feature_scores_aggregated.csv", index=False)
        score_by_target[target] = score_target

        ranked_target = score_target["feature"].tolist()
        corr_selected_target = ranked_target[: args.corr_max_features]
        corr_selected_by_target[target] = corr_selected_target
        pd.DataFrame(
            {
                "rank": list(range(1, len(corr_selected_target) + 1)),
                "feature": corr_selected_target,
            }
        ).to_csv(target_dir / "correlation_filter_selected_features.csv", index=False)

        wrapper_candidates = corr_selected_target[: args.wrapper_candidates_top_k]
        wrapper_candidates_by_target[target] = list(wrapper_candidates)
        forward_trace, forward_best_set, forward_best_score = _forward_selection(
            x=x,
            y=y_target,
            candidate_features=wrapper_candidates,
            max_features=min(args.wrapper_max_features, len(wrapper_candidates)),
            alpha=args.wrapper_alpha,
            min_improvement=args.wrapper_min_improvement,
        )
        forward_trace = forward_trace.copy()
        forward_trace["target"] = target
        forward_trace.to_csv(target_dir / "wrapper_forward_selection_trace.csv", index=False)
        forward_traces.append(forward_trace)

        backward_start = forward_best_set if forward_best_set else wrapper_candidates
        backward_trace, backward_best_set, backward_best_score = _backward_elimination(
            x=x,
            y=y_target,
            initial_features=backward_start,
            min_features=max(1, min(args.wrapper_min_features, len(backward_start))),
            alpha=args.wrapper_alpha,
            min_improvement=args.wrapper_min_improvement,
        )
        backward_trace = backward_trace.copy()
        backward_trace["target"] = target
        backward_trace.to_csv(target_dir / "wrapper_backward_elimination_trace.csv", index=False)
        backward_traces.append(backward_trace)

        final_target_features = backward_best_set if backward_best_set else forward_best_set
        wrapper_selected_by_target[target] = final_target_features
        wrapper_best_score_by_target[target] = float(backward_best_score)
        union_features.update(final_target_features)

        per_target_summary.append(
            {
                "target": target,
                "corr_selected_count": int(len(corr_selected_target)),
                "wrapper_candidate_count": int(len(wrapper_candidates)),
                "forward_best_score_mae": float(forward_best_score),
                "forward_best_features": forward_best_set,
                "backward_best_score_mae": float(backward_best_score),
                "backward_best_features": backward_best_set,
                "final_selected_features": final_target_features,
            }
        )

    corr_df = pd.concat(corr_frames, ignore_index=True)
    corr_df.to_csv(args.out_dir / "feature_target_correlations.csv", index=False)

    score_df = _aggregate_feature_scores(corr_df)
    score_df.to_csv(args.out_dir / "feature_scores_aggregated.csv", index=False)

    pd.DataFrame(per_target_summary).to_csv(args.out_dir / "target_selection_summary.csv", index=False)

    feature_priority_df = _build_feature_priority_rows(
        target_cols=target_cols,
        score_by_target=score_by_target,
        wrapper_selected_by_target=wrapper_selected_by_target,
        wrapper_best_score_by_target=wrapper_best_score_by_target,
    )
    feature_priority_df.to_csv(args.out_dir / "feature_priority_by_target.csv", index=False)

    union_master = sorted(union_features)
    pd.DataFrame(
        {
            "rank": list(range(1, len(union_master) + 1)),
            "feature": union_master,
        }
    ).to_csv(args.out_dir / "union_master_features_pre_prune.csv", index=False)

    final_selected, high_corr_pairs, collinearity_decisions = _post_union_collinearity_prune(
        x=x,
        union_features=union_master,
        feature_priority=feature_priority_df,
        threshold=args.collinearity_threshold,
    )

    high_corr_pairs.to_csv(args.out_dir / "high_collinearity_pairs.csv", index=False)
    collinearity_decisions.to_csv(args.out_dir / "union_collinearity_prune_decisions.csv", index=False)
    pd.DataFrame(
        {
            "rank": list(range(1, len(final_selected) + 1)),
            "feature": final_selected,
        }
    ).to_csv(args.out_dir / "final_selected_union_features.csv", index=False)

    # Keep this path stable for downstream dataloader consumers.
    pd.DataFrame(
        {
            "rank": list(range(1, len(final_selected) + 1)),
            "feature": final_selected,
        }
    ).to_csv(args.out_dir / "correlation_filter_selected_features.csv", index=False)

    forward_trace = pd.concat(forward_traces, ignore_index=True) if forward_traces else pd.DataFrame()
    backward_trace = pd.concat(backward_traces, ignore_index=True) if backward_traces else pd.DataFrame()
    forward_trace.to_csv(args.out_dir / "wrapper_forward_selection_trace.csv", index=False)
    backward_trace.to_csv(args.out_dir / "wrapper_backward_elimination_trace.csv", index=False)

    drop_audit_df, drop_summary_df = _build_feature_drop_audit(
        all_features_after_prepare=x.columns.tolist(),
        dropped_prepare_df=dropped_prepare_df,
        corr_selected_by_target=corr_selected_by_target,
        wrapper_candidates_by_target=wrapper_candidates_by_target,
        forward_trace=forward_trace,
        backward_trace=backward_trace,
        wrapper_selected_by_target=wrapper_selected_by_target,
        collinearity_decisions=collinearity_decisions,
    )
    drop_audit_df.to_csv(args.out_dir / "feature_drop_audit.csv", index=False)
    drop_summary_df.to_csv(args.out_dir / "feature_drop_summary.csv", index=False)

    reduced_cols = [args.subject_col, *target_cols, *final_selected]
    reduced_cols = [col for col in reduced_cols if col in table.columns]
    reduced_table = table[reduced_cols].copy()
    reduced_table.to_csv(args.out_dir / "training_table_selected_features.csv", index=False)

    summary = {
        "n_samples": int(x.shape[0]),
        "n_features_initial": int(x.shape[1]),
        "n_dropped_features": int(len(dropped_cols)),
        "target_cols": target_cols,
        "correlation": {
            "collinearity_threshold": float(args.collinearity_threshold),
            "selection_mode": "target_specific_then_union",
            "per_target": per_target_summary,
        },
        "wrapper": {
            "per_target_best_score_mae": wrapper_best_score_by_target,
            "selected_by_target": wrapper_selected_by_target,
            "union_master_count_pre_prune": int(len(union_master)),
            "union_master_features_pre_prune": union_master,
            "final_count_post_prune": int(len(final_selected)),
            "final_selected_features": final_selected,
        },
        "collinearity_pruning": {
            "pairs_above_threshold": int(len(high_corr_pairs)),
            "drop_decisions": int(len(collinearity_decisions)),
            "decisions_file": "union_collinearity_prune_decisions.csv",
        },
        "feature_drop_audit": {
            "audit_file": "feature_drop_audit.csv",
            "summary_file": "feature_drop_summary.csv",
            "drop_events": int(len(drop_audit_df)),
            "dropped_unique_features": int(drop_audit_df["feature"].nunique()) if not drop_audit_df.empty else 0,
        },
    }

    with (args.out_dir / "feature_selection_summary.json").open("w", encoding="ascii") as fp:
        json.dump(summary, fp, indent=2)

    logger.info("Feature selection outputs saved to: %s", args.out_dir)
    logger.info("Final selected features (%d): %s", len(final_selected), final_selected)

    if not args.no_plots:
        _generate_plots(
            out_dir=args.out_dir,
            corr_df=corr_df,
            score_df=score_df,
            high_corr_pairs=high_corr_pairs,
            forward_trace=forward_trace,
            backward_trace=backward_trace,
            corr_selected=union_master,
            final_selected=final_selected,
            wrapper_selected_by_target=wrapper_selected_by_target,
            x=x,
            y=y,
            top_n=args.plot_top_n,
            plot_scale=args.plot_scale,
        )
        logger.info("Plots saved to: %s", args.out_dir / "plots")


if __name__ == "__main__":
    main()
