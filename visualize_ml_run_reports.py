"""
Create visualizations for ML outputs produced by training, CV, and tuning.

Supported modes:
- single: one training run from ml_train_hybrid.py outputs
- cv: one cross-validation report directory
- tuning: one hyperparameter tuning study directory

Writes PNG figures to output_ml/figures/<scope_name>/ by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _infer_latest_run_name(reports_dir: Path) -> str:
    candidates = list(reports_dir.glob("*_evaluation_summary.csv"))
    if not candidates:
        raise FileNotFoundError(f"No *_evaluation_summary.csv files found in {reports_dir}")

    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    suffix = "_evaluation_summary.csv"
    if not latest.name.endswith(suffix):
        raise ValueError(f"Could not infer run name from file: {latest.name}")
    return latest.name[: -len(suffix)]


def _infer_latest_cv_dir(search_root: Path) -> Path:
    candidates = list(search_root.glob("**/cv_fold_metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No cv_fold_metrics.csv found under {search_root}")
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest.parent


def _infer_latest_tuning_dir(search_root: Path) -> Path:
    candidates = list(search_root.glob("**/leaderboard.csv"))
    if not candidates:
        raise FileNotFoundError(f"No leaderboard.csv found under {search_root}")
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest.parent


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _plot_history(history_df: pd.DataFrame, out_path: Path, dpi: int) -> bool:
    if history_df.empty or "epoch" not in history_df.columns:
        return False

    metric_names = ["total", "base", "ca", "ordinal"]
    pairs = []
    for metric in metric_names:
        train_col = f"train_{metric}"
        val_col = f"val_{metric}"
        if train_col in history_df.columns and val_col in history_df.columns:
            pairs.append((metric, train_col, val_col))

    if not pairs:
        return False

    n_plots = len(pairs)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.5 * n_rows), squeeze=False)
    epoch = history_df["epoch"].to_numpy(dtype=float)

    for index, (metric, train_col, val_col) in enumerate(pairs):
        row_index = index // n_cols
        col_index = index % n_cols
        axis = axes[row_index][col_index]

        axis.plot(epoch, history_df[train_col], label=f"train_{metric}", linewidth=1.8)
        axis.plot(epoch, history_df[val_col], label=f"val_{metric}", linewidth=1.8)
        axis.set_title(f"{metric} loss")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.3)
        axis.legend()

    for index in range(n_plots, n_rows * n_cols):
        row_index = index // n_cols
        col_index = index % n_cols
        axes[row_index][col_index].axis("off")

    fig.suptitle("Training/Validation Loss Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_predictions_scatter(pred_df: pd.DataFrame, out_path: Path, dpi: int) -> bool:
    required = {"target_score", "pred_score"}
    if pred_df.empty or not required.issubset(pred_df.columns):
        return False

    target = pred_df["target_score"].to_numpy(dtype=float)
    pred = pred_df["pred_score"].to_numpy(dtype=float)

    lo = float(min(np.min(target), np.min(pred)))
    hi = float(max(np.max(target), np.max(pred)))
    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))

    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(target, pred, alpha=0.8)
    axis.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="black", label="ideal")
    axis.set_title("Test Predictions: Predicted vs Target")
    axis.set_xlabel("Target score")
    axis.set_ylabel("Predicted score")
    axis.grid(alpha=0.3)
    axis.legend()
    axis.text(
        0.02,
        0.98,
        f"n={len(pred_df)}\nMAE={mae:.3f}\nRMSE={rmse:.3f}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_error_histogram(pred_df: pd.DataFrame, out_path: Path, dpi: int) -> bool:
    required = {"target_score", "pred_score"}
    if pred_df.empty or not required.issubset(pred_df.columns):
        return False

    error = pred_df["pred_score"].to_numpy(dtype=float) - pred_df["target_score"].to_numpy(dtype=float)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(error, bins=20, alpha=0.85)
    axis.axvline(0.0, linestyle="--", linewidth=1.2, color="black")
    axis.set_title("Test Prediction Error Distribution")
    axis.set_xlabel("Prediction error (pred - target)")
    axis.set_ylabel("Count")
    axis.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_per_subject(per_subject_df: pd.DataFrame, out_path: Path, dpi: int, top_n: int) -> bool:
    required = {"subject_id", "mae"}
    if per_subject_df.empty or not required.issubset(per_subject_df.columns):
        return False

    work_df = per_subject_df.sort_values("mae", ascending=False).head(top_n).copy()
    work_df = work_df.iloc[::-1]  # reverse for readable horizontal bar plot

    fig_height = max(4.0, 0.45 * len(work_df) + 1.5)
    fig, axis = plt.subplots(figsize=(9, fig_height))
    axis.barh(work_df["subject_id"], work_df["mae"], alpha=0.9)
    axis.set_title(f"Per-Subject MAE (Top {len(work_df)})")
    axis.set_xlabel("MAE")
    axis.set_ylabel("Subject")
    axis.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_summary(summary_df: pd.DataFrame, out_path: Path, dpi: int) -> bool:
    required = {"split", "mae", "rmse", "ordinal_mae", "qwk"}
    if summary_df.empty or not required.issubset(summary_df.columns):
        return False

    plot_df = summary_df.copy()
    plot_df["qwk"] = pd.to_numeric(plot_df["qwk"], errors="coerce")

    metrics = ["mae", "rmse", "ordinal_mae", "qwk"]
    x_labels = plot_df["split"].astype(str).tolist()
    x = np.arange(len(x_labels), dtype=float)
    width = 0.18

    fig, axis = plt.subplots(figsize=(9, 5))
    for index, metric in enumerate(metrics):
        offset = (index - (len(metrics) - 1) / 2.0) * width
        values = pd.to_numeric(plot_df[metric], errors="coerce").to_numpy(dtype=float)
        axis.bar(x + offset, values, width=width, label=metric)

    axis.set_xticks(x)
    axis.set_xticklabels(x_labels)
    axis.set_title("Evaluation Summary Metrics by Split")
    axis.set_ylabel("Metric value")
    axis.grid(alpha=0.25, axis="y")
    axis.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_cv_fold_metrics(cv_fold_df: pd.DataFrame, out_path: Path, dpi: int) -> bool:
    if cv_fold_df.empty:
        return False

    preferred = ["test_mae", "test_rmse", "val_mae", "val_rmse", "best_val_total", "test_total"]
    metrics = [name for name in preferred if name in cv_fold_df.columns]
    if not metrics:
        return False

    plot_df = cv_fold_df.copy()
    for metric in metrics:
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")

    fold_labels = plot_df["fold"].astype(str).tolist() if "fold" in plot_df.columns else [str(i + 1) for i in range(len(plot_df))]
    x = np.arange(len(plot_df), dtype=float)
    width = min(0.13, 0.8 / max(len(metrics), 1))

    fig, axis = plt.subplots(figsize=(max(10.0, 0.45 * len(plot_df) + 4.0), 5.8))
    for idx, metric in enumerate(metrics):
        offset = (idx - (len(metrics) - 1) / 2.0) * width
        vals = plot_df[metric].to_numpy(dtype=float)
        axis.bar(x + offset, vals, width=width, label=metric)

    axis.set_xticks(x)
    axis.set_xticklabels(fold_labels, rotation=45, ha="right")
    axis.set_title("Cross-Validation Metrics by Fold")
    axis.set_ylabel("Metric value")
    axis.grid(alpha=0.25, axis="y")
    axis.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_cv_per_subject(cv_subject_df: pd.DataFrame, out_path: Path, dpi: int, top_n: int) -> bool:
    required = {"subject_id", "mae"}
    if cv_subject_df.empty or not required.issubset(cv_subject_df.columns):
        return False

    work_df = cv_subject_df.copy()
    work_df["mae"] = pd.to_numeric(work_df["mae"], errors="coerce")
    work_df = work_df.dropna(subset=["mae"])
    if work_df.empty:
        return False

    grouped = work_df.groupby("subject_id", as_index=False).agg(mean_mae=("mae", "mean"), n_folds=("mae", "count"))
    grouped = grouped.sort_values("mean_mae", ascending=False).head(top_n)
    grouped = grouped.iloc[::-1]

    fig_height = max(4.0, 0.45 * len(grouped) + 1.5)
    fig, axis = plt.subplots(figsize=(9.2, fig_height))
    axis.barh(grouped["subject_id"], grouped["mean_mae"], alpha=0.9)
    axis.set_title(f"CV Mean MAE per Subject (Top {len(grouped)})")
    axis.set_xlabel("Mean MAE across folds")
    axis.set_ylabel("Subject")
    axis.grid(alpha=0.25, axis="x")

    for i, (_, row) in enumerate(grouped.iterrows()):
        axis.text(float(row["mean_mae"]), i, f"  n={int(row['n_folds'])}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_cv_aggregate(cv_agg_df: pd.DataFrame, out_path: Path, dpi: int) -> bool:
    required = {"metric", "mean"}
    if cv_agg_df.empty or not required.issubset(cv_agg_df.columns):
        return False

    work_df = cv_agg_df.copy()
    work_df["mean"] = pd.to_numeric(work_df["mean"], errors="coerce")
    work_df["std"] = pd.to_numeric(work_df.get("std", np.nan), errors="coerce")
    work_df = work_df.dropna(subset=["mean"])
    if work_df.empty:
        return False

    fig_height = max(4.0, 0.45 * len(work_df) + 1.5)
    fig, axis = plt.subplots(figsize=(8.8, fig_height))
    y = np.arange(len(work_df), dtype=float)
    axis.barh(y, work_df["mean"].to_numpy(dtype=float), xerr=work_df["std"].to_numpy(dtype=float), alpha=0.9)
    axis.set_yticks(y)
    axis.set_yticklabels(work_df["metric"].astype(str).tolist())
    axis.set_title("CV Aggregate Metrics (mean +/- std)")
    axis.set_xlabel("Value")
    axis.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_tuning_objective(leaderboard_df: pd.DataFrame, out_path: Path, dpi: int, maximize: bool) -> bool:
    required = {"trial", "objective", "status"}
    if leaderboard_df.empty or not required.issubset(leaderboard_df.columns):
        return False

    work_df = leaderboard_df.copy()
    work_df["trial"] = pd.to_numeric(work_df["trial"], errors="coerce")
    work_df["objective"] = pd.to_numeric(work_df["objective"], errors="coerce")
    ok_df = work_df[work_df["status"].astype(str) == "ok"].dropna(subset=["trial", "objective"]).sort_values("trial")
    if ok_df.empty:
        return False

    objective_vals = ok_df["objective"].to_numpy(dtype=float)
    best_so_far = np.maximum.accumulate(objective_vals) if maximize else np.minimum.accumulate(objective_vals)

    fig, axis = plt.subplots(figsize=(9.2, 5.2))
    axis.plot(ok_df["trial"], objective_vals, marker="o", linewidth=1.4, label="trial objective")
    axis.plot(ok_df["trial"], best_so_far, linewidth=2.2, label="best-so-far")
    axis.set_title("Tuning Objective by Trial")
    axis.set_xlabel("Trial")
    axis.set_ylabel("Objective")
    axis.grid(alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_tuning_status(leaderboard_df: pd.DataFrame, out_path: Path, dpi: int) -> bool:
    if leaderboard_df.empty or "status" not in leaderboard_df.columns:
        return False

    counts = leaderboard_df["status"].astype(str).value_counts(dropna=False)
    if counts.empty:
        return False

    fig, axis = plt.subplots(figsize=(6.8, 4.8))
    axis.bar(counts.index.tolist(), counts.values.tolist(), alpha=0.9)
    axis.set_title("Tuning Trial Status Counts")
    axis.set_xlabel("Status")
    axis.set_ylabel("Number of trials")
    axis.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_tuning_top_trials(
    leaderboard_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
    top_n: int,
    maximize: bool,
) -> bool:
    required = {"run_name", "objective", "status"}
    if leaderboard_df.empty or not required.issubset(leaderboard_df.columns):
        return False

    work_df = leaderboard_df.copy()
    work_df["objective"] = pd.to_numeric(work_df["objective"], errors="coerce")
    work_df = work_df[work_df["status"].astype(str) == "ok"].dropna(subset=["objective"])
    if work_df.empty:
        return False

    sorted_df = work_df.sort_values("objective", ascending=not maximize).head(top_n)
    sorted_df = sorted_df.iloc[::-1]

    fig_height = max(4.0, 0.45 * len(sorted_df) + 1.5)
    fig, axis = plt.subplots(figsize=(10.0, fig_height))
    axis.barh(sorted_df["run_name"], sorted_df["objective"], alpha=0.9)
    axis.set_title(f"Top {len(sorted_df)} Tuning Trials by Objective")
    axis.set_xlabel("Objective")
    axis.set_ylabel("Trial run")
    axis.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_tuning_repeat_variability(
    leaderboard_df: pd.DataFrame,
    repeat_df: pd.DataFrame | None,
    out_path: Path,
    dpi: int,
    top_n: int,
    maximize: bool,
) -> bool:
    """
    Plot per-trial objective mean with variability across repeated seeds.

    Preference order:
    1) Compute mean/std from repeat_metrics.csv when available.
    2) Fallback to leaderboard objective + objective_std if present.
    """
    descending = bool(maximize)

    if repeat_df is not None and not repeat_df.empty:
        required = {"trial", "objective", "status"}
        if required.issubset(repeat_df.columns):
            work_df = repeat_df.copy()
            work_df["trial"] = pd.to_numeric(work_df["trial"], errors="coerce")
            work_df["objective"] = pd.to_numeric(work_df["objective"], errors="coerce")
            work_df = work_df[work_df["status"].astype(str) == "ok"].dropna(subset=["trial", "objective"])

            if not work_df.empty:
                agg = work_df.groupby("trial", as_index=False).agg(
                    objective_mean=("objective", "mean"),
                    objective_std=("objective", "std"),
                    n_repeats=("objective", "count"),
                )
                agg["objective_std"] = agg["objective_std"].fillna(0.0)
                agg = agg.sort_values("objective_mean", ascending=not descending).head(top_n)
                agg = agg.iloc[::-1]

                fig_height = max(4.0, 0.45 * len(agg) + 1.5)
                fig, axis = plt.subplots(figsize=(9.8, fig_height))
                y = np.arange(len(agg), dtype=float)

                axis.barh(
                    y,
                    agg["objective_mean"].to_numpy(dtype=float),
                    xerr=agg["objective_std"].to_numpy(dtype=float),
                    alpha=0.9,
                )
                labels = [f"trial {int(t)}" for t in agg["trial"].to_numpy(dtype=float)]
                axis.set_yticks(y)
                axis.set_yticklabels(labels)
                axis.set_title("Top Trials: Objective Mean +/- Std Across Seeds")
                axis.set_xlabel("Objective")
                axis.set_ylabel("Trial")
                axis.grid(alpha=0.25, axis="x")

                for i, (_, row) in enumerate(agg.iterrows()):
                    axis.text(float(row["objective_mean"]), i, f"  n={int(row['n_repeats'])}", va="center", fontsize=9)

                fig.tight_layout()
                fig.savefig(out_path, dpi=dpi)
                plt.close(fig)
                return True

    required_lb = {"trial", "objective", "status", "objective_std"}
    if leaderboard_df.empty or not required_lb.issubset(leaderboard_df.columns):
        return False

    work_lb = leaderboard_df.copy()
    work_lb["trial"] = pd.to_numeric(work_lb["trial"], errors="coerce")
    work_lb["objective"] = pd.to_numeric(work_lb["objective"], errors="coerce")
    work_lb["objective_std"] = pd.to_numeric(work_lb["objective_std"], errors="coerce")
    work_lb = work_lb[work_lb["status"].astype(str) == "ok"].dropna(subset=["trial", "objective"])
    if work_lb.empty:
        return False

    work_lb["objective_std"] = work_lb["objective_std"].fillna(0.0)
    sorted_lb = work_lb.sort_values("objective", ascending=not descending).head(top_n)
    sorted_lb = sorted_lb.iloc[::-1]

    fig_height = max(4.0, 0.45 * len(sorted_lb) + 1.5)
    fig, axis = plt.subplots(figsize=(9.8, fig_height))
    y = np.arange(len(sorted_lb), dtype=float)
    axis.barh(
        y,
        sorted_lb["objective"].to_numpy(dtype=float),
        xerr=sorted_lb["objective_std"].to_numpy(dtype=float),
        alpha=0.9,
    )
    labels = [f"trial {int(t)}" for t in sorted_lb["trial"].to_numpy(dtype=float)]
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.set_title("Top Trials: Objective +/- Std Across Seeds")
    axis.set_xlabel("Objective")
    axis.set_ylabel("Trial")
    axis.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _visualize_single_mode(args: argparse.Namespace) -> tuple[str, Path, list[Path]]:
    run_name = args.run_name or _infer_latest_run_name(args.reports_dir)

    history_path = args.checkpoints_dir / f"{run_name}_history.csv"
    summary_path = args.reports_dir / f"{run_name}_evaluation_summary.csv"
    test_pred_path = args.reports_dir / f"{run_name}_test_predictions.csv"
    per_subject_path = args.reports_dir / f"{run_name}_test_per_subject.csv"

    summary_df = _load_optional_csv(summary_path)
    test_pred_df = _load_optional_csv(test_pred_path)
    per_subject_df = _load_optional_csv(per_subject_path)
    history_df = _load_optional_csv(history_path)

    if summary_df is None and test_pred_df is None and per_subject_df is None and history_df is None:
        raise FileNotFoundError(
            "Could not find any run report files. "
            f"Checked under reports={args.reports_dir} and checkpoints={args.checkpoints_dir} for run={run_name}"
        )

    output_dir = args.figures_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []

    if history_df is not None and _plot_history(history_df, output_dir / "loss_curves.png", dpi=args.dpi):
        generated_paths.append(output_dir / "loss_curves.png")
    if test_pred_df is not None and _plot_predictions_scatter(test_pred_df, output_dir / "test_pred_vs_target.png", dpi=args.dpi):
        generated_paths.append(output_dir / "test_pred_vs_target.png")
    if test_pred_df is not None and _plot_error_histogram(test_pred_df, output_dir / "test_error_histogram.png", dpi=args.dpi):
        generated_paths.append(output_dir / "test_error_histogram.png")
    if per_subject_df is not None and _plot_per_subject(
        per_subject_df,
        output_dir / "test_per_subject_mae.png",
        dpi=args.dpi,
        top_n=max(1, int(args.top_subjects)),
    ):
        generated_paths.append(output_dir / "test_per_subject_mae.png")
    if summary_df is not None and _plot_summary(summary_df, output_dir / "evaluation_summary_metrics.png", dpi=args.dpi):
        generated_paths.append(output_dir / "evaluation_summary_metrics.png")

    return f"single:{run_name}", output_dir, generated_paths


def _visualize_cv_mode(args: argparse.Namespace) -> tuple[str, Path, list[Path]]:
    cv_dir = args.cv_report_dir if args.cv_report_dir is not None else _infer_latest_cv_dir(args.search_root)

    fold_df = _load_optional_csv(cv_dir / "cv_fold_metrics.csv")
    agg_df = _load_optional_csv(cv_dir / "cv_aggregate_metrics.csv")
    per_subject_df = _load_optional_csv(cv_dir / "cv_per_subject_all_folds.csv")
    if fold_df is None and agg_df is None and per_subject_df is None:
        raise FileNotFoundError(f"No CV report CSV files found in {cv_dir}")

    output_dir = args.figures_dir / f"cv_{cv_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []

    if fold_df is not None and _plot_cv_fold_metrics(fold_df, output_dir / "cv_fold_metrics.png", dpi=args.dpi):
        generated_paths.append(output_dir / "cv_fold_metrics.png")
    if agg_df is not None and _plot_cv_aggregate(agg_df, output_dir / "cv_aggregate_metrics.png", dpi=args.dpi):
        generated_paths.append(output_dir / "cv_aggregate_metrics.png")
    if per_subject_df is not None and _plot_cv_per_subject(
        per_subject_df,
        output_dir / "cv_per_subject_mean_mae.png",
        dpi=args.dpi,
        top_n=max(1, int(args.top_subjects)),
    ):
        generated_paths.append(output_dir / "cv_per_subject_mean_mae.png")

    return f"cv:{cv_dir}", output_dir, generated_paths


def _visualize_tuning_mode(args: argparse.Namespace) -> tuple[str, Path, list[Path]]:
    tuning_dir = args.tuning_dir if args.tuning_dir is not None else _infer_latest_tuning_dir(args.search_root)
    leaderboard_df = _load_optional_csv(tuning_dir / "leaderboard.csv")
    repeat_df = _load_optional_csv(tuning_dir / "repeat_metrics.csv")
    if leaderboard_df is None:
        raise FileNotFoundError(f"No leaderboard.csv found in tuning directory {tuning_dir}")

    output_dir = args.figures_dir / f"tuning_{tuning_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []

    if _plot_tuning_objective(leaderboard_df, output_dir / "tuning_objective_by_trial.png", dpi=args.dpi, maximize=args.maximize_objective):
        generated_paths.append(output_dir / "tuning_objective_by_trial.png")
    if _plot_tuning_top_trials(
        leaderboard_df,
        output_dir / "tuning_top_trials.png",
        dpi=args.dpi,
        top_n=max(1, int(args.top_trials)),
        maximize=args.maximize_objective,
    ):
        generated_paths.append(output_dir / "tuning_top_trials.png")
    if _plot_tuning_status(leaderboard_df, output_dir / "tuning_trial_status.png", dpi=args.dpi):
        generated_paths.append(output_dir / "tuning_trial_status.png")
    if _plot_tuning_repeat_variability(
        leaderboard_df,
        repeat_df,
        output_dir / "tuning_repeat_variability.png",
        dpi=args.dpi,
        top_n=max(1, int(args.top_trials)),
        maximize=args.maximize_objective,
    ):
        generated_paths.append(output_dir / "tuning_repeat_variability.png")

    return f"tuning:{tuning_dir}", output_dir, generated_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ML reports from single-run, CV, or tuning outputs")
    parser.add_argument("--mode", choices=["auto", "single", "cv", "tuning"], default="auto")
    parser.add_argument("--run-name", default=None, help="Run name prefix (single mode; e.g., hybrid_icf_20260306_234802)")
    parser.add_argument("--reports-dir", type=Path, default=Path("./output_ml/reports"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("./output_ml/checkpoints"))
    parser.add_argument("--figures-dir", type=Path, default=Path("./output_ml/figures"))
    parser.add_argument("--search-root", type=Path, default=Path("./output_ml"), help="Root used for auto-discovery in cv/tuning mode")
    parser.add_argument("--cv-report-dir", type=Path, default=None, help="Directory containing CV files (cv_fold_metrics.csv, etc.)")
    parser.add_argument("--tuning-dir", type=Path, default=None, help="Directory containing leaderboard.csv")
    parser.add_argument("--top-subjects", type=int, default=20, help="Max subjects in per-subject MAE plot")
    parser.add_argument("--top-trials", type=int, default=15, help="Max trials in top-trial plot (tuning mode)")
    parser.add_argument("--maximize-objective", action="store_true", help="Use max objective as better (e.g., QWK). Default assumes lower is better")
    parser.add_argument("--dpi", type=int, default=150)

    args = parser.parse_args()

    selected_mode = args.mode
    if selected_mode == "auto":
        if args.run_name is not None:
            selected_mode = "single"
        elif args.cv_report_dir is not None:
            selected_mode = "cv"
        elif args.tuning_dir is not None:
            selected_mode = "tuning"
        else:
            try:
                _ = _infer_latest_run_name(args.reports_dir)
                selected_mode = "single"
            except Exception:
                try:
                    _ = _infer_latest_cv_dir(args.search_root)
                    selected_mode = "cv"
                except Exception:
                    selected_mode = "tuning"

    if selected_mode == "single":
        scope, output_dir, generated_paths = _visualize_single_mode(args)
    elif selected_mode == "cv":
        scope, output_dir, generated_paths = _visualize_cv_mode(args)
    elif selected_mode == "tuning":
        scope, output_dir, generated_paths = _visualize_tuning_mode(args)
    else:
        raise ValueError(f"Unsupported mode: {selected_mode}")

    if not generated_paths:
        raise ValueError(f"Mode {selected_mode} found files, but no figures could be generated due to missing/invalid columns")

    print(f"Scope: {scope}")
    print(f"Mode: {selected_mode}")
    print(f"Output directory: {output_dir}")
    print("Generated figures:")
    for path in generated_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
