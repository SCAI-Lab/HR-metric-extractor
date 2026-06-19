#!/usr/bin/env python3
"""
Hyperparameter tuning helper for ml_train_hybrid.py.

Runs random search trials by generating temporary config files,
executing training, and collecting validation metrics.
"""

from __future__ import annotations

import argparse
import copy
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


@dataclass
class ParamSpec:
    path: tuple[str, ...]
    kind: str  # 'choice', 'float', 'log_float'
    values: list[Any] | None = None
    low: float | None = None
    high: float | None = None


def _default_search_space() -> List[ParamSpec]:
    return [
        ParamSpec(path=("optimization", "learning_rate"), kind="log_float", low=1e-4, high=5e-3),
        ParamSpec(path=("optimization", "weight_decay"), kind="log_float", low=1e-6, high=1e-3),
        ParamSpec(path=("model", "dropout"), kind="float", low=0.1, high=0.4),
        ParamSpec(path=("model", "hrv_hidden_dim"), kind="choice", values=[64, 128, 256]),
        ParamSpec(path=("model", "sensor_model_dim"), kind="choice", values=[64, 128, 256]),
        ParamSpec(path=("model", "fusion_hidden_dim"), kind="choice", values=[64, 128, 256]),
        ParamSpec(path=("model", "num_heads"), kind="choice", values=[2, 4, 8]),
        ParamSpec(path=("model", "num_sensor_layers"), kind="choice", values=[1, 2, 3]),
        ParamSpec(path=("batch_size",), kind="choice", values=[4, 8, 16]),
        ParamSpec(path=("loss", "alpha"), kind="float", low=0.1, high=0.8),
        ParamSpec(path=("loss", "beta"), kind="float", low=0.1, high=0.8),
        ParamSpec(path=("loss", "margin"), kind="choice", values=[5.0, 10.0, 15.0, 20.0]),
    ]


def _set_nested(config: Dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = config
    for key in path[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[path[-1]] = value


def _sample_param(spec: ParamSpec, rng: random.Random) -> Any:
    if spec.kind == "choice":
        if not spec.values:
            raise ValueError(f"Param {spec.path} has empty choice set")
        return rng.choice(spec.values)
    if spec.kind == "float":
        if spec.low is None or spec.high is None:
            raise ValueError(f"Param {spec.path} missing low/high")
        return float(rng.uniform(spec.low, spec.high))
    if spec.kind == "log_float":
        if spec.low is None or spec.high is None:
            raise ValueError(f"Param {spec.path} missing low/high")
        lo = float(spec.low)
        hi = float(spec.high)
        if lo <= 0 or hi <= 0:
            raise ValueError(f"log_float bounds must be > 0 for {spec.path}")
        import math

        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    raise ValueError(f"Unknown param kind: {spec.kind}")


def _objective_is_maximized(objective: str) -> bool:
    return objective in {"val_qwk"}


def _extract_objective(summary_csv: Path, objective: str) -> float:
    df = pd.read_csv(summary_csv)
    val_row = df[df["split"] == "val"]
    if len(val_row) == 0:
        raise ValueError(f"No val row found in {summary_csv}")

    metric_col = objective.replace("val_", "")
    if metric_col not in val_row.columns:
        raise ValueError(f"Objective column '{metric_col}' not found in {summary_csv}")
    return float(val_row.iloc[0][metric_col])


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def run_tuning(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    base_config = _load_yaml(args.base_config)
    specs = _default_search_space()

    study_root = Path(args.output_dir) / args.study_name
    trial_cfg_dir = study_root / "trial_configs"
    trial_report_dir = study_root / "trial_reports"
    trial_ckpt_dir = study_root / "trial_checkpoints"
    study_root.mkdir(parents=True, exist_ok=True)
    trial_cfg_dir.mkdir(parents=True, exist_ok=True)
    trial_report_dir.mkdir(parents=True, exist_ok=True)
    trial_ckpt_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_rows: List[Dict[str, Any]] = []
    repeat_rows: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None
    best_trial_config: Dict[str, Any] | None = None

    maximize = _objective_is_maximized(args.objective)

    print(f"Starting tuning study '{args.study_name}' with {args.trials} trial(s)")
    print(f"Objective: {args.objective} ({'maximize' if maximize else 'minimize'})")
    print(f"Repeats per trial: {args.repeats_per_trial} (seed stride: {args.repeat_seed_stride})")

    for trial_idx in range(1, args.trials + 1):
        run_name = f"{args.study_name}_trial_{trial_idx:03d}"
        trial_config = copy.deepcopy(base_config)

        sampled: Dict[str, Any] = {}
        for spec in specs:
            value = _sample_param(spec, rng)
            sampled[".".join(spec.path)] = value
            _set_nested(trial_config, spec.path, value)

        _set_nested(trial_config, ("optimization", "epochs"), int(args.epochs))
        _set_nested(trial_config, ("output", "checkpoint_dir"), str(trial_ckpt_dir))
        _set_nested(trial_config, ("output", "report_dir"), str(trial_report_dir))

        trial_template_config = copy.deepcopy(trial_config)

        successful_objectives: List[float] = []
        successful_repeats = 0

        for repeat_idx in range(1, args.repeats_per_trial + 1):
            repeat_run_name = run_name if args.repeats_per_trial == 1 else f"{run_name}_r{repeat_idx:02d}"
            repeat_seed = int(args.seed + (trial_idx * args.repeat_seed_stride) + (repeat_idx - 1))

            repeat_config = copy.deepcopy(trial_template_config)
            _set_nested(repeat_config, ("seed",), repeat_seed)
            _set_nested(repeat_config, ("output", "run_name"), repeat_run_name)

            cfg_path = trial_cfg_dir / f"{repeat_run_name}.yaml"
            _save_yaml(cfg_path, repeat_config)

            cmd = [sys.executable, "ml_train_hybrid.py", "--config", str(cfg_path)]
            result = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True)

            summary_csv = trial_report_dir / f"{repeat_run_name}_evaluation_summary.csv"
            repeat_status = "ok" if result.returncode == 0 and summary_csv.exists() else "failed"

            repeat_row: Dict[str, Any] = {
                "trial": trial_idx,
                "repeat": repeat_idx,
                "run_name": repeat_run_name,
                "repeat_seed": repeat_seed,
                "status": repeat_status,
                "return_code": int(result.returncode),
            }

            if repeat_status == "ok":
                objective_value = _extract_objective(summary_csv, args.objective)
                repeat_row["objective"] = objective_value
                successful_objectives.append(objective_value)
                successful_repeats += 1
                print(
                    f"Trial {trial_idx:03d} repeat {repeat_idx:02d}/{args.repeats_per_trial}: "
                    f"objective={objective_value:.6f} (seed={repeat_seed})"
                )
            else:
                repeat_row["objective"] = float("nan")
                stderr_tail = (result.stderr or "").strip().splitlines()[-5:]
                repeat_row["error_tail"] = " | ".join(stderr_tail)
                print(
                    f"Trial {trial_idx:03d} repeat {repeat_idx:02d}/{args.repeats_per_trial}: "
                    f"failed (seed={repeat_seed})"
                )

            repeat_row.update(sampled)
            repeat_rows.append(repeat_row)

        row: Dict[str, Any] = {
            "trial": trial_idx,
            "run_name": run_name,
            "status": "ok" if successful_repeats > 0 else "failed",
            "repeats_requested": int(args.repeats_per_trial),
            "repeats_ok": int(successful_repeats),
            "repeats_failed": int(args.repeats_per_trial - successful_repeats),
        }
        row.update(sampled)

        if successful_objectives:
            row["objective"] = float(statistics.mean(successful_objectives))
            row["objective_std"] = float(statistics.pstdev(successful_objectives)) if len(successful_objectives) > 1 else 0.0
            row["objective_min"] = float(min(successful_objectives))
            row["objective_max"] = float(max(successful_objectives))
            print(
                f"Trial {trial_idx:03d} aggregate: objective_mean={row['objective']:.6f}, "
                f"std={row['objective_std']:.6f}, ok={successful_repeats}/{args.repeats_per_trial}"
            )
        else:
            row["objective"] = float("nan")
            row["objective_std"] = float("nan")
            row["objective_min"] = float("nan")
            row["objective_max"] = float("nan")
            print(f"Trial {trial_idx:03d} aggregate: failed (all repeats failed)")

        leaderboard_rows.append(row)

        if row["status"] == "ok":
            if best_row is None:
                best_row = row
                best_trial_config = copy.deepcopy(trial_template_config)
            else:
                current = float(row["objective"])
                incumbent = float(best_row["objective"])
                better = (current > incumbent) if maximize else (current < incumbent)
                if better:
                    best_row = row
                    best_trial_config = copy.deepcopy(trial_template_config)

    leaderboard_csv = study_root / "leaderboard.csv"
    pd.DataFrame(leaderboard_rows).to_csv(leaderboard_csv, index=False)
    print(f"Saved leaderboard: {leaderboard_csv}")

    repeats_csv = study_root / "repeat_metrics.csv"
    pd.DataFrame(repeat_rows).to_csv(repeats_csv, index=False)
    print(f"Saved repeat metrics: {repeats_csv}")

    if best_row is None:
        print("No successful trials. Inspect leaderboard and trial logs.")
        return

    best_export_path = study_root / "best_config.yaml"
    if best_trial_config is None:
        raise RuntimeError("Internal error: best trial config is missing")

    best_config = copy.deepcopy(best_trial_config)
    _set_nested(best_config, ("seed",), int(args.seed))
    _set_nested(best_config, ("output", "run_name"), str(best_row["run_name"]))
    _save_yaml(best_export_path, best_config)

    summary_txt = study_root / "best_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"study_name: {args.study_name}\n")
        f.write(f"objective: {args.objective}\n")
        f.write(f"best_run: {best_row['run_name']}\n")
        f.write(f"best_objective_mean: {best_row['objective']}\n")
        f.write(f"best_objective_std: {best_row.get('objective_std', float('nan'))}\n")
        f.write(f"repeats_per_trial: {args.repeats_per_trial}\n")

    print(
        f"Best run: {best_row['run_name']} "
        f"({args.objective}_mean={best_row['objective']:.6f}, std={best_row.get('objective_std', float('nan')):.6f})"
    )
    print(f"Saved best config: {best_export_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-search tuner for ml_train_hybrid.py")
    parser.add_argument("--base-config", type=Path, default=Path("experiment_config.example.yaml"))
    parser.add_argument("--study-name", default="hybrid_tuning")
    parser.add_argument("--output-dir", type=Path, default=Path("./output_ml/tuning"))
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--objective",
        choices=["val_mae", "val_rmse", "val_ordinal_mae", "val_qwk"],
        default="val_mae",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats-per-trial", type=int, default=3)
    parser.add_argument("--repeat-seed-stride", type=int, default=1000)
    args = parser.parse_args()
    if args.repeats_per_trial < 1:
        parser.error("--repeats-per-trial must be >= 1")
    if args.repeat_seed_stride < 1:
        parser.error("--repeat-seed-stride must be >= 1")
    return args


if __name__ == "__main__":
    run_tuning(parse_args())
