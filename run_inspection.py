#!/usr/bin/env python3
"""
Data Inspection Pipeline - Main Script

Comprehensive pipeline for:
1. Activity extraction (propulsion, resting, etc.)
2. HR metrics computation during activities
3. Baseline-activity comparisons
4. Window overlap and delay analysis
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Optional

from activity_extraction import (
    parse_adl_file, identify_activity_intervals,
    extract_propulsion_activities, extract_resting_activities,
    add_baseline_reference, extract_custom_activities
)
from hr_metrics import (
    extract_rr_intervals_from_ecg, compute_hr_metrics_for_window, 
    compute_differential_metrics, extract_hr_metrics_from_timeseries,
    check_signal_quality
)
from window_overlap_analysis import (
    segment_activity_into_phases, extract_phases_from_data,
    compute_optimal_windows_for_metrics, create_window_overlap_report
)
from data_loading import (
    load_timeseries_data, load_hr_metrics, extract_window_data,
    estimate_sampling_frequency, create_data_summary,
    load_ppg_data, load_imu_sensors, load_eda_bioz_data
)
from feature_extraction import (
    extract_activity_eda_features,
    extract_activity_imu_features,
    extract_activity_sensor_features,
    extract_eda_sensor_features,
)
from data_loading import load_sensor_hr_data, load_sensor_rr_intervals


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def configure_logging(log_level: str) -> None:
    """Set the root and module logger levels from a user-provided string."""
    level_name = str(log_level).strip().upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported log level: {log_level}")

    logging.getLogger().setLevel(level)
    logger.setLevel(level)


def _compute_ecg_segments(t_sec: np.ndarray, gap_factor: float = 10.0):
    """Identify continuous ECG segments based on large time gaps."""
    if t_sec is None or len(t_sec) < 2:
        return []
    t = np.asarray(t_sec)
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return [(float(t[0]), float(t[-1]))]

    median_dt = np.median(dt)
    gap_threshold = median_dt * gap_factor
    breaks = np.where(np.diff(t) > gap_threshold)[0]

    segments = []
    start_idx = 0
    for b in breaks:
        end_idx = b
        segments.append((float(t[start_idx]), float(t[end_idx])))
        start_idx = b + 1
    segments.append((float(t[start_idx]), float(t[-1])))
    return segments


def _total_overlap(activities: pd.DataFrame, segments, offset: float) -> float:
    if activities is None or len(activities) == 0 or not segments:
        return 0.0
    total = 0.0
    for _, row in activities.iterrows():
        t_start = row['t_start'] + offset
        t_end = row['t_end'] + offset
        for seg_start, seg_end in segments:
            overlap = max(0.0, min(t_end, seg_end) - max(t_start, seg_start))
            total += overlap
    return total


def _count_windows_with_ecg_samples(
    activities_df: pd.DataFrame,
    ecg_t_sec: np.ndarray,
    min_samples: int = 100,
) -> int:
    """Count activity windows that contain at least ``min_samples`` ECG samples.

    This avoids misleading diagnostics from coarse global min/max range checks
    when ECG recordings contain large gaps.
    """
    if activities_df is None or len(activities_df) == 0 or ecg_t_sec is None or len(ecg_t_sec) == 0:
        return 0

    t = np.asarray(ecg_t_sec, dtype=float)
    if len(t) == 0:
        return 0
    t = np.sort(t)

    count = 0
    for _, row in activities_df.iterrows():
        left = np.searchsorted(t, float(row['t_start']), side='left')
        right = np.searchsorted(t, float(row['t_end']), side='right')
        if (right - left) >= int(min_samples):
            count += 1
    return count


def _offset_candidates_sec_from_hours(hours_cfg) -> List[float]:
    """Build unique offset candidates in seconds from config hours.

    Defaults are timezone-aligned hour offsets to avoid odd auto offsets.
    """
    # Keep timezone-aligned defaults so auto-estimation can recover known
    # clock offsets (for example +/-7h or +/-8h) in partially unsynced data.
    default_hours = [-8.0, -7.0, 0.0, 7.0, 8.0]

    if hours_cfg is None:
        hours = default_hours
    elif isinstance(hours_cfg, (int, float, str)):
        hours = [hours_cfg]
    elif isinstance(hours_cfg, (list, tuple)):
        hours = list(hours_cfg)
    else:
        logger.warning("Unsupported time_offset_candidates_hours type (%s); using defaults", type(hours_cfg).__name__)
        hours = default_hours

    parsed_hours = []
    for value in hours:
        try:
            parsed_hours.append(float(value))
        except Exception:
            logger.warning("Ignoring invalid offset hour candidate: %r", value)

    if not parsed_hours:
        parsed_hours = default_hours

    # Keep deterministic ordering and remove duplicates after rounding.
    unique_hours = sorted({round(hour, 6) for hour in parsed_hours})
    return [hour * 3600.0 for hour in unique_hours]


def _expand_intervals_to_sliding_windows(
    activities_df: pd.DataFrame,
    window_duration_sec: float,
    overlap_percent: float,
) -> pd.DataFrame:
    """Expand interval rows into fixed-duration sliding windows.

    Windows are generated within each interval's [t_start, t_end] bounds.
    Intervals shorter than ``window_duration_sec`` are skipped.
    """
    if activities_df is None or len(activities_df) == 0:
        return activities_df.copy() if isinstance(activities_df, pd.DataFrame) else pd.DataFrame()

    win_sec = float(window_duration_sec)
    ov = float(overlap_percent)
    if win_sec <= 0:
        raise ValueError(f"window_duration_sec must be > 0, got {window_duration_sec}")
    if ov < 0 or ov >= 100:
        raise ValueError(f"overlap_percent must be in [0, 100), got {overlap_percent}")

    step_sec = win_sec * (1.0 - ov / 100.0)
    if step_sec <= 0:
        raise ValueError(
            f"Invalid window step computed from window_duration_sec={win_sec} and overlap_percent={ov}"
        )

    window_rows = []
    for original_idx, row in activities_df.iterrows():
        start = float(row['t_start'])
        end = float(row['t_end'])
        duration = end - start
        if not np.isfinite(start) or not np.isfinite(end) or duration < win_sec:
            continue

        win_i = 0
        t0 = start
        # Small epsilon avoids dropping the mathematically last aligned window
        # due to floating point accumulation for high overlaps.
        while (t0 + win_sec) <= (end + 1e-9):
            t1 = t0 + win_sec
            out = row.to_dict()
            out['t_start'] = t0
            out['t_end'] = t1
            out['duration_sec'] = win_sec
            out['source_activity_idx'] = int(original_idx)
            out['window_idx_within_activity'] = int(win_i)
            window_rows.append(out)
            win_i += 1
            t0 += step_sec

    if not window_rows:
        cols = list(activities_df.columns)
        for extra_col in ('source_activity_idx', 'window_idx_within_activity'):
            if extra_col not in cols:
                cols.append(extra_col)
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(window_rows)


def _resolve_subject_path_from_ecg_path(ecg_path_value: str) -> Path:
    """Resolve subject directory from an ECG config path (file or directory)."""
    ecg_cfg_path = Path(ecg_path_value)

    if ecg_cfg_path.exists() and ecg_cfg_path.is_dir():
        return ecg_cfg_path.parent

    if ecg_cfg_path.suffix.lower() in {'.gz', '.csv'}:
        return ecg_cfg_path.parent.parent

    if ecg_cfg_path.name.lower().startswith('vivalnk_vv330_ecg'):
        return ecg_cfg_path.parent

    return ecg_cfg_path.parent


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_default_config() -> dict:
    """Create default configuration template."""
    return {
        'project': {
            'name': 'data-inspection',
            'output_dir': './output',
        },
        'data': {
            'adl_path': 'D:/ETHZ/Lifelogging/interim/scai-ncgg/sim_elderly_2/scai_app/ADLs_1.csv.gz',  # Path to ADL CSV
            'ecg_path': 'D:/ETHZ/Lifelogging/interim/scai-ncgg/sim_elderly_2/vivalnk_vv330_ecg/data_1.csv.gz',  # Path to PPG CSV
            'hr_metrics_path': None,  # Path to pre-computed HR metrics (optional)
        },
        'activities': {
            'propulsion_keywords': ['level walking', 'walker', 'propulsion'],
            'resting_keywords': ['sitting', 'rest', 'lying'],
            'min_duration_sec': 30.0,
            'baseline_min_duration_sec': 35.0,
            'extra': {
                # Example custom short activity
                'washing_hands': {
                    'keywords': ['wash hands', 'washing hands', 'hand wash'],
                    'min_duration_sec': 15.0,
                }
            },
        },
        'signal': {
            'signal_type': 'ecg',  # One of: ppg, ecg, hr
            'sampling_frequency_hz': 128.0,
        },
        'analysis': {
            'compute_baseline_comparison': True,
            'compute_window_overlap': True,
            'analyze_delays': True,
            'max_delay_sec': 300.0,
            'recovery_window_sec': 300.0,
            'baseline_window_sec': 120.0,
        }
    }


def run_inspection_pipeline(config_path: str) -> None:
    """
    Main pipeline execution.
    
    Args:
        config_path: Path to YAML configuration file
    """
    logger.info("=" * 80)
    logger.info("Data Inspection Pipeline")
    logger.info("=" * 80)
    
    # Load configuration
    cfg = load_config(config_path)
    
    # Create output directory
    output_dir = Path(cfg['project']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # ========================================================================
    # STEP 1: Load and parse ADL data
    # ========================================================================
    logger.info("\n[STEP 1] Loading ADL data...")
    adl_path = Path(cfg['data']['adl_path'])
    adl_df = parse_adl_file(adl_path)
    logger.info(f"  Loaded {len(adl_df)} ADL events")
    
    # Identify activity intervals
    adl_intervals = identify_activity_intervals(adl_df)
    logger.info(f"  Identified {len(adl_intervals)} activity intervals")
    
    # ========================================================================
    # STEP 2: Extract activity types
    # ========================================================================
    logger.info("\n[STEP 2] Extracting activity types...")
    
    propulsion = extract_propulsion_activities(
        adl_intervals,
        min_duration_sec=cfg['activities'].get('min_duration_sec', 30.0),
        keywords=cfg['activities'].get('propulsion_keywords', ['level walking','walking','walker','self propulsion','propulsion','assisted propulsion'])
    )
    logger.info(f"  Propulsion activities: {len(propulsion)}")
    
    resting = extract_resting_activities(
        adl_intervals,
        min_duration_sec=cfg['activities'].get('baseline_min_duration_sec', 40.0),
        keywords=cfg['activities'].get('resting_keywords', ['sitting','rest','lying','seated'])
    )
    logger.info(f"  Resting activities: {len(resting)}")
    
    # Save activity extracts
    propulsion.to_csv(output_dir / 'propulsion_activities.csv', index=False)
    resting.to_csv(output_dir / 'resting_activities.csv', index=False)
    logger.info(f"  Saved activity extracts to output_dir/")

    # Optional: extract and save custom activities
    extra_cfg = cfg['activities'].get('extra', {})
    custom_activities = extract_custom_activities(adl_intervals, extra_cfg)
    for name, df in custom_activities.items():
        safe_name = str(name).strip().lower().replace(' ', '_')
        df.to_csv(output_dir / f'activity_{safe_name}.csv', index=False)
        logger.info(f"  Saved custom activity '{name}' with {len(df)} intervals")
    
    # ========================================================================
    # STEP 3: Load PPG/HR data
    # ========================================================================
    logger.info("\n[STEP 3] Loading physiological data...")
    
    # Get subject path for fallback sensor loading (PPG/IMU/EDA).
    # Support ecg_path being either a file (.../subject/vivalnk_vv330_ecg/data_1.csv.gz)
    # or a directory (.../subject/vivalnk_vv330_ecg).
    subject_path = _resolve_subject_path_from_ecg_path(cfg['data']['ecg_path'])
    
    # Try to load ECG, assess quality, and fallback to PPG if needed
    ecg_data = None
    signal_source = None
    sensor_quality = {}
    ppg_fallback_used = False  # Track if we fall back to PPG due to bad ECG
    
    # Load ECG and check quality
    try:
        ecg_path = Path(cfg['data']['ecg_path'])
        ecg_data = load_timeseries_data(ecg_path)
        
        if ecg_data is not None and len(ecg_data) > 0:
            # Check ECG quality
            ecg_quality = check_signal_quality(ecg_data['value'].values)
            sensor_quality['ecg'] = ecg_quality
            logger.info(f"  ECG loaded: {len(ecg_data)} samples, quality_score={ecg_quality['quality_score']:.3f}")
            
            # If ECG quality is poor (flat signal), try PPG fallback
            if ecg_quality['is_flat']:
                logger.warning(f"  ECG signal is flat (std={ecg_quality['std']:.2e}) - attempting PPG fallback")
                ecg_data = None
                ppg_fallback_used = True
            else:
                signal_source = 'ecg'
    except Exception as e:
        logger.warning(f"  Failed to load ECG: {str(e)} - attempting PPG fallback")
        ecg_data = None
        ppg_fallback_used = True
    
    # If ECG failed or is poor quality, try PPG sensors
    if ecg_data is None or signal_source is None:
        ppg_channels = ['green', 'infrared', 'red']
        best_ppg = None
        best_ppg_channel = None
        best_ppg_quality = -1
        
        logger.info(f"  Attempting to load PPG data as fallback...")
        for channel in ppg_channels:
            ppg_data = load_ppg_data(subject_path, channel)
            if ppg_data is not None and len(ppg_data) > 0:
                ppg_quality = check_signal_quality(ppg_data['ppg'].values)
                sensor_quality[f'ppg_{channel}'] = ppg_quality
                logger.info(f"    PPG ({channel}): {len(ppg_data)} samples, quality_score={ppg_quality['quality_score']:.3f}")
                
                # Keep track of best PPG channel
                if ppg_quality['quality_score'] > best_ppg_quality:
                    best_ppg = ppg_data
                    best_ppg_channel = channel
                    best_ppg_quality = ppg_quality['quality_score']
        
        if best_ppg is not None:
            logger.info(f"  Using PPG ({best_ppg_channel}) as signal source (quality_score={best_ppg_quality:.3f})")
            # Convert PPG data to match ECG format
            ecg_data = best_ppg.copy()
            ecg_data.columns = ['t_sec', 'value']
            signal_source = f'ppg_{best_ppg_channel}'
        else:
            logger.error(f"  No usable PPG data found - proceeding with empty ECG data")
            ecg_data = pd.DataFrame(columns=['t_sec', 'value'])
            signal_source = 'none'
    
    logger.info(f"  Signal source: {signal_source}")
    logger.info(f"  Sensor quality scores: {sensor_quality}")

    # Load IMU data (multiple sensors, each kept separate)
    imu_sensors = {}
    enable_imu_features = bool(cfg.get('analysis', {}).get('enable_imu_features', True))
    if not enable_imu_features:
        logger.info("  IMU feature extraction disabled by config (analysis.enable_imu_features=false)")
    else:
        try:
            imu_cfg_paths = cfg.get('data', {}).get('imu_paths', None)
            if imu_cfg_paths is None:
                imu_cfg_paths = {}
            elif isinstance(imu_cfg_paths, str):
                imu_cfg_paths = {'imu': imu_cfg_paths}

            imu_sensors = load_imu_sensors(subject_path=subject_path, imu_config=imu_cfg_paths)
            if len(imu_sensors) > 0:
                logger.info(f"  IMU sensors loaded: {len(imu_sensors)} sensor(s)")
                for sensor_name, sensor_df in imu_sensors.items():
                    duration_sec = sensor_df['t_sec'].max() - sensor_df['t_sec'].min()
                    logger.info(f"    - {sensor_name}: {len(sensor_df)} samples, duration={duration_sec:.1f}s")
            else:
                logger.info("  IMU data not found (skipping IMU for this run)")
        except Exception as e:
            logger.warning(f"  Failed to load IMU data: {str(e)}")
            imu_sensors = {}

    # Load EDA/BioZ data
    eda_bioz_data = None
    try:
        eda_bioz_cfg_path = cfg.get('data', {}).get('eda_bioz_path')
        eda_bioz_data = load_eda_bioz_data(subject_path=subject_path, eda_bioz_path=eda_bioz_cfg_path)
        if eda_bioz_data is not None and len(eda_bioz_data) > 0:
            logger.info(f"  EDA/BioZ loaded: {len(eda_bioz_data)} samples")
        else:
            logger.info("  EDA/BioZ data not found (skipping EDA for this run)")
    except Exception as e:
        logger.warning(f"  Failed to load EDA/BioZ data: {str(e)}")
        eda_bioz_data = None

    # Load sensor HR data (vivalnk_vv330_heart_rate)
    sensor_hr_data = None
    try:
        hr_sensor_cfg_path = cfg.get('data', {}).get('hr_sensor_path')
        if hr_sensor_cfg_path:
            hr_sensor_path = Path(hr_sensor_cfg_path)
        else:
            hr_sensor_path = subject_path / 'vivalnk_vv330_heart_rate'
        sensor_hr_data = load_sensor_hr_data(hr_sensor_path)
        if sensor_hr_data is not None and len(sensor_hr_data) > 0:
            logger.info(
                f"  Sensor HR loaded: {len(sensor_hr_data)} samples, "
                f"range {sensor_hr_data['value'].min():.0f}–{sensor_hr_data['value'].max():.0f} bpm"
            )
            logger.info(
                f"    Time range: {sensor_hr_data['t_sec'].min():.1f} - {sensor_hr_data['t_sec'].max():.1f}"
            )
        else:
            logger.info("  Sensor HR data not found (HR sensor features will be NaN)")
    except Exception as e:
        logger.warning(f"  Failed to load sensor HR data: {str(e)}")
        sensor_hr_data = None

    # Load Corsano wrist/BioZ RR intervals (preferred source for HRV)
    sensor_rr_data = None
    try:
        rr_cfg_path = cfg.get('data', {}).get('rr_intervals_path')
        sensor_rr_data = load_sensor_rr_intervals(
            subject_path=subject_path,
            rr_path=Path(rr_cfg_path) if rr_cfg_path else None,
        )
        if sensor_rr_data is not None and len(sensor_rr_data) > 0:
            logger.info(
                f"  Sensor RR intervals loaded: {len(sensor_rr_data)} beats, "
                f"range {sensor_rr_data['rr_ms'].min():.0f}–{sensor_rr_data['rr_ms'].max():.0f} ms"
            )
            logger.info(
                f"    Time range: {sensor_rr_data['t_sec'].min():.1f} - {sensor_rr_data['t_sec'].max():.1f}"
            )
        else:
            logger.info("  Sensor RR intervals not found; HRV will be derived from sensor HR")
    except Exception as e:
        logger.warning(f"  Failed to load sensor RR intervals: {str(e)}")
        sensor_rr_data = None
    
    # Determine sampling frequency
    cfg_fs = cfg.get('signal', {}).get('sampling_frequency_hz', None)
    if cfg_fs is not None and cfg_fs > 0:
        fs = float(cfg_fs)
        logger.info(f"  Using configured sampling frequency: {fs:.2f} Hz")
    elif len(ecg_data) > 0:
        fs = estimate_sampling_frequency(ecg_data['t_sec'].values)
        logger.info(f"  Estimated sampling frequency: {fs:.2f} Hz")
    else:
        fs = 128.0  # Default fallback
        logger.info(f"  Estimated sampling frequency: {fs:.2f} Hz")

    # Precompute auxiliary sensor sampling rates and provide a shared extractor
    eda_fs = None
    if eda_bioz_data is not None and len(eda_bioz_data) > 0:
        eda_fs = estimate_sampling_frequency(eda_bioz_data['t_sec'].to_numpy())

    hr_sensor_fs = 1.0  # vivalnk HR is typically 1 Hz
    if sensor_hr_data is not None and len(sensor_hr_data) > 1:
        hr_sensor_fs = estimate_sampling_frequency(sensor_hr_data['t_sec'].to_numpy())

    imu_fs_map = {}
    if imu_sensors:
        for sensor_name, sensor_df in imu_sensors.items():
            if sensor_df is None or len(sensor_df) == 0 or 't_sec' not in sensor_df.columns:
                continue
            imu_fs_map[sensor_name] = estimate_sampling_frequency(sensor_df['t_sec'].to_numpy())

    # Cache IMU arrays once so per-window extraction can use fast searchsorted
    # instead of O(N) boolean masks over the full sensor DataFrame.
    imu_cache: Dict[str, Dict[str, np.ndarray]] = {}
    if imu_sensors:
        for sensor_name, sensor_df in imu_sensors.items():
            if sensor_df is None or len(sensor_df) == 0:
                continue
            if 't_sec' not in sensor_df.columns or 'imu_magnitude' not in sensor_df.columns:
                continue

            imu_sorted = sensor_df.sort_values('t_sec', kind='mergesort').reset_index(drop=True)
            sensor_cache: Dict[str, np.ndarray] = {
                't': imu_sorted['t_sec'].to_numpy(dtype=float),
                'mag': imu_sorted['imu_magnitude'].to_numpy(dtype=float),
            }
            if {'imu_x', 'imu_y', 'imu_z'}.issubset(imu_sorted.columns):
                sensor_cache['axes'] = imu_sorted[['imu_x', 'imu_y', 'imu_z']].to_numpy(dtype=float)
            imu_cache[sensor_name] = sensor_cache

    def extract_aux_features(
        t_start: float,
        t_end: float,
        duration_sec: float,
        precomputed_hr_win: Optional[np.ndarray] = None,
    ) -> Dict:
        aux_features = {}
        activity_context = {
            't_start': t_start,
            't_end': t_end,
            'duration_sec': duration_sec,
        }

        # Keep legacy metric columns present even when no HR/RR can be computed.
        for key in ('mean_hr', 'rmssd', 'sdnn', 'mean_rr_ms', 'stress_index', 'n_beats'):
            aux_features[key] = np.nan

        # Sensor HR features (HR time-series + HRV from sensor estimates)
        if sensor_hr_data is not None and len(sensor_hr_data) > 0:
            # Re-use the already-extracted window from the quality gate when available.
            hr_win = precomputed_hr_win if precomputed_hr_win is not None else extract_window_data(sensor_hr_data, t_start, t_end)[0]
            # Pull EDA window for the same time span
            eda_win = None
            if eda_bioz_data is not None and len(eda_bioz_data) > 0:
                eda_win, _ = extract_window_data(eda_bioz_data, t_start, t_end, signal_col='eda_bioz')
                if len(eda_win) < 10:
                    eda_win = None
            # Pull RR interval window (Corsano PPG fallback HRV source)
            rr_win = None
            if sensor_rr_data is not None and len(sensor_rr_data) > 0:
                rr_win, _ = extract_window_data(sensor_rr_data, t_start, t_end, signal_col='rr_ms')
                if len(rr_win) < 4:
                    rr_win = None
            aux_features.update(
                extract_activity_sensor_features(
                    hr_values=hr_win,
                    eda_signal=eda_win,
                    rr_intervals_ms=rr_win,
                    hr_fs=hr_sensor_fs,
                    eda_fs=eda_fs if eda_fs is not None else 25.0,
                )
            )

            # Populate legacy HR metric keys so downstream summaries/comparisons
            # remain meaningful when ECG extraction is skipped.
            hr_valid = np.asarray(hr_win, dtype=float)
            hr_valid = hr_valid[hr_valid > 0]
            rr_for_legacy = np.array([], dtype=float)
            if len(hr_valid) >= 2:
                rr_for_legacy = 60000.0 / hr_valid
            elif rr_win is not None and len(rr_win) >= 2:
                rr_for_legacy = np.asarray(rr_win, dtype=float)

            if len(rr_for_legacy) >= 2:
                legacy = compute_hr_metrics_for_window(rr_for_legacy)
                for key in ('mean_rr_ms', 'rmssd', 'sdnn', 'mean_hr', 'stress_index', 'n_beats'):
                    aux_features[key] = legacy.get(key, np.nan)
        elif sensor_rr_data is not None and len(sensor_rr_data) > 0:
            # No VivaLNK HR at all — use Corsano RR intervals as sole HRV source.
            # Pass an empty HR array so _assess_hr_quality returns 0 and the Corsano
            # RR path is taken inside extract_activity_sensor_features.
            rr_win, _ = extract_window_data(sensor_rr_data, t_start, t_end, signal_col='rr_ms')
            eda_win = None
            if eda_bioz_data is not None and len(eda_bioz_data) > 0:
                eda_win, _ = extract_window_data(eda_bioz_data, t_start, t_end, signal_col='eda_bioz')
                if len(eda_win) < 10:
                    eda_win = None
            if len(rr_win) >= 4:
                aux_features.update(
                    extract_activity_sensor_features(
                        hr_values=np.array([], dtype=float),
                        eda_signal=eda_win,
                        rr_intervals_ms=rr_win,
                        hr_fs=hr_sensor_fs,
                        eda_fs=eda_fs if eda_fs is not None else 25.0,
                    )
                )

                legacy = compute_hr_metrics_for_window(np.asarray(rr_win, dtype=float))
                for key in ('mean_rr_ms', 'rmssd', 'sdnn', 'mean_hr', 'stress_index', 'n_beats'):
                    aux_features[key] = legacy.get(key, np.nan)
            elif eda_win is not None:
                aux_features.update(
                    extract_eda_sensor_features(eda_win, fs=eda_fs if eda_fs is not None else 25.0)
                )
        elif eda_bioz_data is not None and len(eda_bioz_data) > 0:
            # No sensor HR and no RR data — compute EDA features standalone.
            eda_signal, eda_time = extract_window_data(eda_bioz_data, t_start, t_end, signal_col='eda_bioz')
            if len(eda_signal) >= 10:
                aux_features.update(
                    extract_eda_sensor_features(
                        eda_signal,
                        fs=eda_fs if eda_fs is not None else 25.0,
                    )
                )

        if imu_sensors:
            for sensor_name, sensor_cache in imu_cache.items():
                imu_time_all = sensor_cache['t']
                lo = np.searchsorted(imu_time_all, t_start, side='left')
                hi = np.searchsorted(imu_time_all, t_end, side='right')
                if lo >= hi:
                    continue

                imu_signal = sensor_cache['mag'][lo:hi]
                imu_time = imu_time_all[lo:hi]
                if len(imu_signal) < 64:
                    continue

                imu_axes_window = None
                if 'axes' in sensor_cache:
                    imu_axes_window = sensor_cache['axes'][lo:hi]

                aux_features.update(
                    extract_activity_imu_features(
                        imu_signal,
                        imu_time,
                        activity_context,
                        sensor_name=sensor_name,
                        imu_axes_window=imu_axes_window,
                        fs=imu_fs_map.get(sensor_name, 1.0),
                    )
                )

        return aux_features

    def extract_activity_metrics_for_windows(
        activities_df: pd.DataFrame,
        index_col_name: str,
        label_prefix: str,
        default_activity_name: str = 'unknown',
        reset_index: bool = False,
        log_progress_every: Optional[int] = None,
        progress_suffix: str = 'activities',
        log_warnings: bool = True,
    ):
        metrics_rows = []
        skipped = {'insufficient_data': 0, 'outside_range': 0}
        hr_quality_threshold = float(cfg.get('analysis', {}).get('sensor_hr_quality_threshold', 0.5))
        n_total = int(len(activities_df))
        loop_t0 = time.perf_counter()
        t_gate = 0.0
        t_ecg_metrics = 0.0
        t_aux = 0.0

        # Determine the broadest available time range from each sensor source
        hr_data_min = sensor_hr_data['t_sec'].min() if sensor_hr_data is not None and len(sensor_hr_data) > 0 else None
        hr_data_max = sensor_hr_data['t_sec'].max() if sensor_hr_data is not None and len(sensor_hr_data) > 0 else None
        rr_data_min = sensor_rr_data['t_sec'].min() if sensor_rr_data is not None and len(sensor_rr_data) > 0 else None
        rr_data_max = sensor_rr_data['t_sec'].max() if sensor_rr_data is not None and len(sensor_rr_data) > 0 else None
       
        # Log sensor data availability for this batch (for custom activities, help diagnose NaN issues)
        if label_prefix.startswith('Custom'):
            if hr_data_min is not None:
                logger.debug(f"  [{label_prefix}] Sensor HR time range: {hr_data_min:.1f} - {hr_data_max:.1f}")
            if rr_data_min is not None:
                logger.debug(f"  [{label_prefix}] Sensor RR time range: {rr_data_min:.1f} - {rr_data_max:.1f}")

        iterable = activities_df.reset_index(drop=True).iterrows() if reset_index else activities_df.iterrows()

        for row_num, (idx, activity) in enumerate(iterable, start=1):
            t_start = activity['t_start']
            t_end = activity['t_end']

            within_ecg = (t_start >= ecg_min and t_end <= ecg_max)
            within_hr = (
                hr_data_min is not None
                and t_start >= hr_data_min
                and t_end <= hr_data_max
            )
            within_rr = (
                rr_data_min is not None
                and t_start >= rr_data_min
                and t_end <= rr_data_max
            )
            if not (within_ecg or within_hr or within_rr):
                if log_warnings:
                    logger.warning(f"  {label_prefix} {idx}: Outside available signal time range")
                skipped['outside_range'] += 1
                continue

            # Sensor HR quality gate: prefer sensor-HR-based metrics when quality is acceptable.
            # Also cache the extracted window to avoid re-extracting it later in extract_aux_features.
            hr_quality = 0.0
            cached_hr_win = None
            gate_t0 = time.perf_counter()
            if within_hr and sensor_hr_data is not None and len(sensor_hr_data) > 0:
                cached_hr_win, _ = extract_window_data(sensor_hr_data, t_start, t_end, signal_col='value')
                if len(cached_hr_win) > 0:
                    hr_arr = np.asarray(cached_hr_win, dtype=float)
                    hr_quality = float(np.sum(hr_arr > 0)) / float(len(hr_arr))

            prefer_sensor_hr = within_hr and hr_quality >= hr_quality_threshold

            # ECG quality assessment: check actual signal quality in the activity window
            ecg_signal = None
            ecg_time = None
            ecg_quality = 0.0
            has_good_ecg = False
            ecg_quality_threshold = float(cfg.get('analysis', {}).get('ecg_quality_threshold', 0.1))

            if within_ecg and not prefer_sensor_hr:
                ecg_signal, ecg_time = extract_window_data(ecg_data, t_start, t_end)
                if len(ecg_signal) >= 100:
                    signal_quality_info = check_signal_quality(ecg_signal)
                    ecg_quality = signal_quality_info.get('quality_score', 0.0)
                    has_good_ecg = ecg_quality >= ecg_quality_threshold
                    if log_warnings and not has_good_ecg:
                        logger.debug(
                            f"  {label_prefix} {idx}: ECG quality {ecg_quality:.4f} < threshold {ecg_quality_threshold:.4f}; "
                            f"will use sensor HR instead"
                        )
                else:
                    if log_warnings:
                        logger.debug(
                            f"  {label_prefix} {idx}: Insufficient ECG samples in window "
                            f"({len(ecg_signal)} < 100); will use sensor HR instead"
                        )

            use_ecg_metrics = has_good_ecg and (not within_hr or hr_quality < hr_quality_threshold)
            t_gate += (time.perf_counter() - gate_t0)

            # ECG-based HR metrics (only if quality is acceptable and HR quality is poor)
            activity_metrics: Dict = {}
            if use_ecg_metrics:
                ecg_t0 = time.perf_counter()
                signal_std = np.std(ecg_signal)
                if signal_std >= 1e-6:
                    signal_type = (
                        'ppg' if (signal_source and signal_source.startswith('ppg'))
                        else cfg['signal'].get('signal_type', 'ecg')
                    )
                    activity_metrics = extract_hr_metrics_from_timeseries(
                        ecg_signal, ecg_time, signal_type=signal_type, fs=fs,
                    )
                    if activity_metrics.get('n_beats', 0) == 0 and log_warnings:
                        logger.warning(
                            f"  {label_prefix} {idx}: No peaks detected in "
                            f"{signal_type.upper()} signal (ecg_quality={ecg_quality:.4f})"
                        )
                else:
                    if log_warnings:
                        logger.debug(
                            f"  {label_prefix} {idx}: ECG signal flat (std={signal_std:.6f}); using sensor HR only"
                        )
                t_ecg_metrics += (time.perf_counter() - ecg_t0)
            elif within_hr and hr_quality >= hr_quality_threshold and log_warnings:
                logger.debug(
                    f"  {label_prefix} {idx}: Using sensor HR; quality {hr_quality:.2f} >= threshold {hr_quality_threshold:.2f}"
                )

            # Skip windows with no usable data from any sensor source
            if not activity_metrics and not within_hr and not within_rr:
                skipped['insufficient_data'] += 1
                continue

            result = {
                index_col_name: idx,
                'activity_name': activity.get('activity', default_activity_name),
                't_start': t_start,
                't_end': t_end,
                'duration_sec': activity['duration_sec'],
            }
            # Sensor-based features (HR, HRV, EDA) are added via extract_aux_features.
            # Pass the cached HR window to avoid extracting it a second time.
            aux_t0 = time.perf_counter()
            result.update(extract_aux_features(t_start, t_end, activity['duration_sec'], precomputed_hr_win=cached_hr_win))
            t_aux += (time.perf_counter() - aux_t0)
            # ECG-based metrics (may be empty dict); come after aux so they don't
            # shadow the newly named sensor features
            result.update(activity_metrics)
            metrics_rows.append(result)

            # For custom activities, log when no sensor HR data is available (helps diagnose NaN issues)
            if label_prefix.startswith('Custom') and log_warnings:
                mean_hr_val = result.get('mean_hr', np.nan)
                if pd.isna(mean_hr_val):
                    within_ranges = []
                    if within_hr:
                        within_ranges.append('HR')
                    if within_rr:
                        within_ranges.append('RR')
                    if within_ecg:
                        within_ranges.append('ECG')
                    ranges_str = ', '.join(within_ranges) if within_ranges else 'NONE'
                    logger.debug(
                        f"  {label_prefix} {idx}: No sensor HR data (NaN); "
                        f"time range {t_start:.1f}-{t_end:.1f}; "
                        f"within sensor ranges: {ranges_str}"
                    )

            if log_progress_every and (row_num % log_progress_every == 0):
                elapsed = time.perf_counter() - loop_t0
                rate = (row_num / elapsed) if elapsed > 0 else 0.0
                remaining = max(0, n_total - row_num)
                eta_sec = (remaining / rate) if rate > 0 else np.nan
                logger.info(
                    "  Progress %s: %d/%d | %.1f windows/s | ETA %.1fs",
                    progress_suffix,
                    row_num,
                    n_total,
                    rate,
                    eta_sec,
                )

        total_elapsed = time.perf_counter() - loop_t0
        processed = len(metrics_rows)
        logger.info(
            "  %s timing summary: total=%.1fs for %d/%d windows (%.1f windows/s)",
            label_prefix,
            total_elapsed,
            processed,
            n_total,
            (n_total / total_elapsed) if total_elapsed > 0 else 0.0,
        )
        logger.info(
            "    Time split: gate=%.1fs, ecg_metrics=%.1fs, aux_features=%.1fs",
            t_gate,
            t_ecg_metrics,
            t_aux,
        )

        return metrics_rows, skipped
    
    # Apply time offset to align ADL times with ECG times
    # If offset is None or 'auto', estimate it from data; otherwise use configured value
    # SPECIAL CASE: When PPG fallback is used due to flatline ECG, always auto-estimate
    # the offset to ensure activities align with the (potentially much shorter) PPG time range
    time_offset_sec = cfg['activities'].get('time_offset_sec', None)
    configured_offset_sec = time_offset_sec  # Remember the original configured value

    # Preserve raw activity times for offset optimization
    propulsion_raw = propulsion.copy()
    resting_raw = resting.copy()
    custom_raw = {name: df.copy() for name, df in custom_activities.items()}
    
    # Force auto-estimation when PPG fallback was used, since the hardcoded offset may not align with PPG time range
    if ppg_fallback_used and isinstance(time_offset_sec, (int, float)):
        logger.warning(
            f"  PPG fallback was used due to flatline ECG. Forcing offset auto-estimation "
            f"(configured offset {time_offset_sec/3600:.2f}h may not align with PPG time range)"
        )
        time_offset_sec = 'auto'
    
    if time_offset_sec is None or time_offset_sec == 'auto':
        # Auto-estimate offset: align ADL activities to ECG segments based on ADL start time and ECG recording times
        if len(adl_intervals) > 0 and len(ecg_data) > 0:
            adl_min = adl_intervals['t_start'].min()
            ecg_min = ecg_data['t_sec'].min()
            ecg_max = ecg_data['t_sec'].max()

            # Compute ECG segments to avoid aligning activities into gaps
            segments = _compute_ecg_segments(ecg_data['t_sec'].values)

            # Use timezone-aligned offset candidates (hours) to avoid odd offsets.
            candidate_offsets = _offset_candidates_sec_from_hours(
                cfg.get('activities', {}).get('time_offset_candidates_hours', None)
            )

            # If an explicit numeric offset was configured, still try it but don't force priority
            # when PPG fallback is used (the configured offset may be designed for ECG, not PPG)
            if isinstance(configured_offset_sec, (int, float)) and not ppg_fallback_used:
                candidate_offsets = [float(configured_offset_sec)] + candidate_offsets

            # Deduplicate candidates while preserving order.
            seen = set()
            deduped_candidates = []
            for off in candidate_offsets:
                key = round(float(off), 6)
                if key in seen:
                    continue
                seen.add(key)
                deduped_candidates.append(float(off))
            candidate_offsets = deduped_candidates

            ecg_aligned_offset = ecg_min - adl_min
            best_offset = candidate_offsets[0] if candidate_offsets else 0.0
            best_overlap = -1.0
            candidate_scores = []
            for off in candidate_offsets:
                overlap = _total_overlap(propulsion_raw, segments, off) + _total_overlap(resting_raw, segments, off)
                for _, custom_df in custom_raw.items():
                    overlap += _total_overlap(custom_df, segments, off)
                candidate_scores.append((off, overlap))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_offset = off

            # If overlap ties or all overlaps are zero, choose candidate closest to raw alignment.
            if len(candidate_scores) > 1:
                max_overlap = max(score for _, score in candidate_scores)
                tied = [off for off, score in candidate_scores if abs(score - max_overlap) < 1e-9]
                if len(tied) > 1:
                    best_offset = min(tied, key=lambda off: abs(off - ecg_aligned_offset))

            time_offset_sec = best_offset
            logger.info("  Offset candidates (hours): %s", [round(off / 3600.0, 3) for off in candidate_offsets])
            logger.info(
                "  Candidate overlap scores: %s",
                [f"{off/3600.0:.3f}h={score:.1f}s" for off, score in candidate_scores],
            )
            logger.info(f"  Auto-estimated time offset: {time_offset_sec:.3f} sec ({time_offset_sec/3600:.3f} hours)")
            logger.info(f"    (ADL start: {adl_min:.1f}, ECG range: {ecg_min:.1f} to {ecg_max:.1f})")
            if ppg_fallback_used:
                logger.info(f"    Note: PPG fallback used; offset adjusted from configured {configured_offset_sec/3600:.2f}h")
        else:
            time_offset_sec = 0.0
            logger.warning("  No activities to offset; using 0.0")
            logger.warning("  No activities to offset; using 0.0")
    
    if time_offset_sec != 0.0:
        propulsion['t_start'] = propulsion['t_start'] + time_offset_sec
        propulsion['t_end'] = propulsion['t_end'] + time_offset_sec
        resting['t_start'] = resting['t_start'] + time_offset_sec
        resting['t_end'] = resting['t_end'] + time_offset_sec
        for name, df in custom_activities.items():
            if df is not None and len(df) > 0:
                df['t_start'] = df['t_start'] + time_offset_sec
                df['t_end'] = df['t_end'] + time_offset_sec
        logger.info(f"  Applied time offset: {time_offset_sec:.3f} sec")

    # Optional: convert event intervals into fixed-size sliding windows.
    window_cfg = cfg.get('analysis', {}).get('windowing', {})
    if window_cfg is None:
        window_cfg = {}
    windowing_enabled = bool(window_cfg.get('enabled', False))
    if windowing_enabled:
        window_duration_sec = float(window_cfg.get('window_duration_sec', 10.0))
        overlap_percent = float(window_cfg.get('overlap_percent', 90.0))
        logger.info(
            "  Applying sliding windows: duration=%.2fs, overlap=%.1f%%",
            window_duration_sec,
            overlap_percent,
        )

        propulsion = _expand_intervals_to_sliding_windows(
            propulsion,
            window_duration_sec=window_duration_sec,
            overlap_percent=overlap_percent,
        )
        resting = _expand_intervals_to_sliding_windows(
            resting,
            window_duration_sec=window_duration_sec,
            overlap_percent=overlap_percent,
        )
        for name, df in list(custom_activities.items()):
            custom_activities[name] = _expand_intervals_to_sliding_windows(
                df,
                window_duration_sec=window_duration_sec,
                overlap_percent=overlap_percent,
            )

        propulsion = propulsion.reset_index(drop=True)
        resting = resting.reset_index(drop=True)
        for name, df in list(custom_activities.items()):
            custom_activities[name] = df.reset_index(drop=True)

        logger.info(
            "  Windowed interval counts -> propulsion: %d, resting: %d",
            len(propulsion),
            len(resting),
        )
    
    # ========================================================================
    # STEP 4: Compute HR metrics from signal
    # ========================================================================
    logger.info("\n[STEP 4] Computing HR metrics from signal...")
    
    # Check if pre-computed metrics available
    hr_metrics_path = cfg['data'].get('hr_metrics_path')
    if hr_metrics_path and Path(hr_metrics_path).exists():
        logger.info("  Using pre-computed HR metrics")
        hr_metrics = load_hr_metrics(Path(hr_metrics_path))
    else:
        logger.info("  Computing HR metrics from ECG signal (this may take a while)...")
        # For now, just compute metrics for activities
        hr_metrics = None
    
    # ========================================================================
    # STEP 5: Extract HR metrics for propulsion activities
    # ========================================================================
    logger.info("\n[STEP 5] Extracting HR metrics for propulsion activities...")
    
    propulsion_metrics = []
    skipped_prop = {'insufficient_data': 0, 'outside_range': 0}

    # Diagnostic: log data time range and sample activity intervals
    logger.info(f"  ECG time range: {ecg_data['t_sec'].min():.1f} - {ecg_data['t_sec'].max():.1f}")
    if len(propulsion) > 0:
        logger.info("  Sample propulsion intervals:")
        for i, row in propulsion.head(5).iterrows():
            logger.info(f"    idx={i} t_start={row['t_start']} t_end={row['t_end']} duration={row['duration_sec']}")
    # Report overlap statistics
    prop_min = propulsion['t_start'].min() if len(propulsion)>0 else np.nan
    prop_max = propulsion['t_end'].max() if len(propulsion)>0 else np.nan
    resting_min = resting['t_start'].min() if len(resting)>0 else np.nan
    resting_max = resting['t_end'].max() if len(resting)>0 else np.nan
    logger.info(f"  Propulsion time range: {prop_min} - {prop_max}")
    logger.info(f"  Resting time range: {resting_min} - {resting_max}")

    # Count how many activities fall within ECG time range
    ecg_min = ecg_data['t_sec'].min()
    ecg_max = ecg_data['t_sec'].max()
    prop_in_range = propulsion[(propulsion['t_start'] >= ecg_min) & (propulsion['t_end'] <= ecg_max)]
    rest_in_range = resting[(resting['t_start'] >= ecg_min) & (resting['t_end'] <= ecg_max)]
    logger.info(f"  Propulsion intervals within ECG range: {len(prop_in_range)}/{len(propulsion)}")
    logger.info(f"  Resting intervals within ECG range: {len(rest_in_range)}/{len(resting)}")
    prop_with_ecg_samples = _count_windows_with_ecg_samples(propulsion, ecg_data['t_sec'].values, min_samples=100)
    rest_with_ecg_samples = _count_windows_with_ecg_samples(resting, ecg_data['t_sec'].values, min_samples=100)
    logger.info(f"  Propulsion intervals with >=100 ECG samples: {prop_with_ecg_samples}/{len(propulsion)}")
    logger.info(f"  Resting intervals with >=100 ECG samples: {rest_with_ecg_samples}/{len(resting)}")

    default_progress_every = 250 if windowing_enabled else 5
    progress_log_every = int(cfg.get('analysis', {}).get('progress_log_every_windows', default_progress_every))
    if progress_log_every < 1:
        progress_log_every = default_progress_every
    
    step5_t0 = time.perf_counter()
    propulsion_metrics, skipped_prop = extract_activity_metrics_for_windows(
        activities_df=propulsion,
        index_col_name='activity_idx',
        label_prefix='Activity',
        default_activity_name='unknown',
        reset_index=False,
        log_progress_every=progress_log_every,
        progress_suffix='activities',
        log_warnings=True,
    )
    step5_dt = time.perf_counter() - step5_t0
    
    logger.info(f"  Propulsion HR metrics extraction complete:")
    logger.info(f"    Successfully processed: {len(propulsion_metrics)}")
    logger.info(f"    Skipped - outside available sensor range: {skipped_prop['outside_range']}")
    logger.info(f"    Skipped - insufficient sensor data: {skipped_prop['insufficient_data']}")
    logger.info(f"    Elapsed time: {step5_dt:.1f} sec")
    
    propulsion_metrics_df = pd.DataFrame(propulsion_metrics)
    # Ensure expected index column exists even if DataFrame is empty
    if 'activity_idx' not in propulsion_metrics_df.columns:
        propulsion_metrics_df['activity_idx'] = pd.Series(dtype='int')
    # Ensure expected time and metric columns exist so later joins/selections don't KeyError
    _required_cols = ['t_start', 't_end', 'mean_hr', 'rmssd', 'stress_index', 'mean_rr_ms', 'n_beats']
    for _c in _required_cols:
        if _c not in propulsion_metrics_df.columns:
            propulsion_metrics_df[_c] = pd.Series(dtype='float')

    propulsion_metrics_df.to_csv(output_dir / 'propulsion_hr_metrics.csv', index=False)
    prop_valid_hr = int(propulsion_metrics_df['mean_hr'].notna().sum()) if 'mean_hr' in propulsion_metrics_df.columns else 0
    logger.info(f"  Propulsion rows with valid mean_hr: {prop_valid_hr}/{len(propulsion_metrics_df)}")
    logger.info(f"  Computed HR metrics for {len(propulsion_metrics_df)} propulsion activities")
    
    # ========================================================================
    # STEP 6: Extract HR metrics for resting activities
    # ========================================================================
    logger.info("\n[STEP 6] Extracting HR metrics for resting activities...")
    
    step6_t0 = time.perf_counter()
    resting_metrics, skipped_rest = extract_activity_metrics_for_windows(
        activities_df=resting,
        index_col_name='resting_idx',
        label_prefix='Resting',
        default_activity_name='unknown',
        reset_index=False,
        log_progress_every=progress_log_every,
        progress_suffix='resting activities',
        log_warnings=True,
    )
    step6_dt = time.perf_counter() - step6_t0
    
    logger.info(f"  Resting HR metrics extraction complete:")
    logger.info(f"    Successfully processed: {len(resting_metrics)}")
    logger.info(f"    Skipped - outside available sensor range: {skipped_rest['outside_range']}")
    logger.info(f"    Skipped - insufficient sensor data: {skipped_rest['insufficient_data']}")
    logger.info(f"    Elapsed time: {step6_dt:.1f} sec")
    
    resting_metrics_df = pd.DataFrame(resting_metrics)
    # Ensure expected index column exists even if DataFrame is empty
    if 'resting_idx' not in resting_metrics_df.columns:
        resting_metrics_df['resting_idx'] = pd.Series(dtype='int')
    # Ensure expected time and metric columns exist so later selections don't KeyError
    _required_cols = ['t_start', 't_end', 'mean_hr', 'rmssd', 'stress_index', 'mean_rr_ms', 'n_beats']
    for _c in _required_cols:
        if _c not in resting_metrics_df.columns:
            resting_metrics_df[_c] = pd.Series(dtype='float')
    resting_metrics_df.to_csv(output_dir / 'resting_hr_metrics.csv', index=False)
    rest_valid_hr = int(resting_metrics_df['mean_hr'].notna().sum()) if 'mean_hr' in resting_metrics_df.columns else 0
    logger.info(f"  Resting rows with valid mean_hr: {rest_valid_hr}/{len(resting_metrics_df)}")
    logger.info(f"  Computed HR metrics for {len(resting_metrics_df)} resting activities")

    # ========================================================================
    # STEP 6B: Extract HR metrics for custom activities
    # ========================================================================
    custom_metrics_dfs = {}
    if custom_activities:
        logger.info("\n[STEP 6B] Extracting HR metrics for custom activities...")

    for name, activities_df in custom_activities.items():
        safe_name = str(name).strip().lower().replace(' ', '_')
        custom_t0 = time.perf_counter()
        activity_metrics_list, skipped_custom = extract_activity_metrics_for_windows(
            activities_df=activities_df,
            index_col_name='activity_idx',
            label_prefix=f"Custom {name}",
            default_activity_name=str(name),
            reset_index=True,
            log_progress_every=None,
            log_warnings=False,
        )
        custom_dt = time.perf_counter() - custom_t0

        metrics_df = pd.DataFrame(activity_metrics_list)
        if 'activity_idx' not in metrics_df.columns:
            metrics_df['activity_idx'] = pd.Series(dtype='int')
        _required_cols = ['t_start', 't_end', 'mean_hr', 'rmssd', 'stress_index', 'mean_rr_ms', 'n_beats']
        for _c in _required_cols:
            if _c not in metrics_df.columns:
                metrics_df[_c] = pd.Series(dtype='float')

        metrics_df.to_csv(output_dir / f'activity_{safe_name}_hr_metrics.csv', index=False)
        custom_metrics_dfs[name] = metrics_df
        
        valid_hr_count = int(metrics_df['mean_hr'].notna().sum()) if 'mean_hr' in metrics_df.columns else 0
        logger.info(
            f"  Custom activity '{name}': {len(metrics_df)} metrics ({valid_hr_count} with valid HR); "
            f"elapsed {custom_dt:.1f}s; "
            f"skipped outside range={skipped_custom['outside_range']}, "
            f"insufficient data={skipped_custom['insufficient_data']}"
        )
    
    # ========================================================================
    # STEP 7: Baseline-Activity Comparison
    # ========================================================================
    logger.info("\n[STEP 7] Computing baseline-activity comparisons...")
    
    # Pair propulsion with preceding resting baseline
    propulsion_with_baseline = add_baseline_reference(propulsion, resting)
    
    # Compute differential metrics
    comparisons = []
    for idx, activity in propulsion_with_baseline.iterrows():
        if pd.isna(activity.get('baseline_t_start')):
            continue
        
        # Find corresponding metrics
        activity_metrics_row = propulsion_metrics_df[
            propulsion_metrics_df['activity_idx'] == idx
        ]
        
        baseline_row = resting_metrics_df[
            (resting_metrics_df['t_start'] >= activity['baseline_t_start']) &
            (resting_metrics_df['t_end'] <= activity['baseline_t_end'])
        ]
        
        if len(activity_metrics_row) == 0 or len(baseline_row) == 0:
            continue
        
        activity_metrics = activity_metrics_row.iloc[0].to_dict()
        baseline_metrics = baseline_row.iloc[0].to_dict()
        
        # Compute differentials
        diff_metrics = compute_differential_metrics(activity_metrics, baseline_metrics)
        
        comparison = {
            'activity_idx': idx,
            'activity_name': activity.get('activity', 'unknown'),
            'propulsion_t_start': activity['t_start'],
            'propulsion_t_end': activity['t_end'],
            'propulsion_duration_sec': activity['duration_sec'],
            'baseline_t_start': activity['baseline_t_start'],
            'baseline_t_end': activity['baseline_t_end'],
            'baseline_duration_sec': activity['baseline_t_end'] - activity['baseline_t_start'],
            'time_gap_sec': activity['baseline_time_before_sec'],
        }
        comparison.update(diff_metrics)
        comparisons.append(comparison)
    
    comparisons_df = pd.DataFrame(comparisons)
    comparisons_df.to_csv(output_dir / 'baseline_activity_comparisons.csv', index=False)
    logger.info(f"  Created {len(comparisons_df)} baseline-activity comparisons")

    # Save HR differentials based on baseline comparison pairing
    if len(comparisons_df) > 0 and 'delta_mean_hr' in comparisons_df.columns:
        propulsion_vs_resting_df = comparisons_df[[
            'activity_idx',
            'activity_name',
            'delta_mean_hr',
            'propulsion_t_start',
            'propulsion_t_end',
            'baseline_t_start',
            'baseline_t_end'
        ]].copy()
        propulsion_vs_resting_df.rename(columns={'delta_mean_hr': 'hr_differential'}, inplace=True)
        propulsion_vs_resting_df.to_csv(output_dir / 'propulsion_vs_resting_differential.csv', index=False)
        logger.info(f"  Saved {len(propulsion_vs_resting_df)} propulsion vs resting differentials")
    else:
        pd.DataFrame().to_csv(output_dir / 'propulsion_vs_resting_differential.csv', index=False)
        logger.info("  No propulsion vs resting differentials available")

    # Baseline comparisons for custom activities
    for name, activities_df in custom_activities.items():
        safe_name = str(name).strip().lower().replace(' ', '_')
        custom_metrics_df = custom_metrics_dfs.get(name, pd.DataFrame())

        if len(activities_df) == 0 or len(custom_metrics_df) == 0:
            pd.DataFrame().to_csv(output_dir / f'activity_{safe_name}_baseline_comparisons.csv', index=False)
            continue

        custom_with_baseline = add_baseline_reference(activities_df.reset_index(drop=True), resting)
        custom_comparisons = []
        for idx, activity in custom_with_baseline.iterrows():
            if pd.isna(activity.get('baseline_t_start')):
                continue

            activity_metrics_row = custom_metrics_df[
                custom_metrics_df['activity_idx'] == idx
            ]
            baseline_row = resting_metrics_df[
                (resting_metrics_df['t_start'] >= activity['baseline_t_start']) &
                (resting_metrics_df['t_end'] <= activity['baseline_t_end'])
            ]

            if len(activity_metrics_row) == 0 or len(baseline_row) == 0:
                continue

            activity_metrics = activity_metrics_row.iloc[0].to_dict()
            baseline_metrics = baseline_row.iloc[0].to_dict()
            diff_metrics = compute_differential_metrics(activity_metrics, baseline_metrics)

            comparison = {
                'activity_idx': idx,
                'activity_type': name,
                'activity_name': activity.get('activity', str(name)),
                'activity_t_start': activity['t_start'],
                'activity_t_end': activity['t_end'],
                'activity_duration_sec': activity['duration_sec'],
                'baseline_t_start': activity['baseline_t_start'],
                'baseline_t_end': activity['baseline_t_end'],
                'baseline_duration_sec': activity['baseline_t_end'] - activity['baseline_t_start'],
                'time_gap_sec': activity['baseline_time_before_sec'],
            }
            comparison.update(diff_metrics)
            custom_comparisons.append(comparison)

        custom_comparisons_df = pd.DataFrame(custom_comparisons)
        custom_comparisons_df.to_csv(output_dir / f'activity_{safe_name}_baseline_comparisons.csv', index=False)
        logger.info(f"  Custom activity '{name}': {len(custom_comparisons_df)} baseline comparisons")
        
        # Save custom activity vs resting differentials based on baseline pairing
        if len(custom_comparisons_df) > 0 and 'delta_mean_hr' in custom_comparisons_df.columns:
            custom_vs_resting_df = custom_comparisons_df[[
                'activity_idx',
                'activity_name',
                'delta_mean_hr',
                'activity_t_start',
                'activity_t_end',
                'baseline_t_start',
                'baseline_t_end'
            ]].copy()
            custom_vs_resting_df.rename(columns={'delta_mean_hr': 'hr_differential'}, inplace=True)
            custom_vs_resting_df.to_csv(output_dir / f'activity_{safe_name}_vs_resting_differential.csv', index=False)
            logger.info(f"  Saved {len(custom_vs_resting_df)} {name} vs resting differentials")
        else:
            pd.DataFrame().to_csv(output_dir / f'activity_{safe_name}_vs_resting_differential.csv', index=False)
            logger.info(f"  No {name} vs resting differentials available")
    
    # ========================================================================
    # STEP 8: Window Overlap and Delay Analysis
    # ========================================================================
    if cfg['analysis'].get('compute_window_overlap', True) and hr_metrics is not None:
        logger.info("\n[STEP 8] Analyzing window overlaps and delays...")
        
        overlap_reports = []
        
        for idx, activity in propulsion.iterrows():
            # Segment into phases
            phases = segment_activity_into_phases(
                (activity['t_start'], activity['t_end']),
                baseline_before_sec=cfg['analysis'].get('baseline_window_sec', 120.0),
                recovery_after_sec=cfg['analysis'].get('recovery_window_sec', 300.0)
            )
            
            # Create overlap report
            activity_dict = activity.to_dict()
            report = create_window_overlap_report(
                activity_dict, phases, hr_metrics,
                hr_metric_col='rmssd'
            )
            overlap_reports.append(report)
        
        if overlap_reports:
            full_overlap_report = pd.concat(overlap_reports, ignore_index=True)
            full_overlap_report.to_csv(output_dir / 'window_overlap_report.csv', index=False)
            logger.info(f"  Created window overlap report with {len(full_overlap_report)} rows")
    
    # ========================================================================
    # STEP 9: Summary Statistics and Reporting
    # ========================================================================
    logger.info("\n[STEP 9] Generating summary report...")
    
    summary = {
        'total_adl_events': len(adl_df),
        'total_activity_intervals': len(adl_intervals),
        'propulsion_count': len(propulsion),
        'resting_count': len(resting),
        'propulsion_with_metrics': len(propulsion_metrics_df),
        'resting_with_metrics': len(resting_metrics_df),
        'baseline_comparisons': len(comparisons_df),
        'ecg_data_samples': len(ecg_data),
        'ecg_data_duration_sec': ecg_data['t_sec'].max() - ecg_data['t_sec'].min(),
        'ecg_estimated_fs_hz': fs,
        'imu_sensors_loaded': len(imu_sensors),
        'eda_bioz_loaded': bool(eda_bioz_data is not None and len(eda_bioz_data) > 0),
    }

    for sensor_name, sensor_df in imu_sensors.items():
        safe_sensor = str(sensor_name).strip().lower().replace(' ', '_')
        summary[f'imu_{safe_sensor}_samples'] = len(sensor_df)
        summary[f'imu_{safe_sensor}_duration_sec'] = sensor_df['t_sec'].max() - sensor_df['t_sec'].min()
        summary[f'imu_{safe_sensor}_fs_hz'] = estimate_sampling_frequency(sensor_df['t_sec'].values)
    
    # Add propulsion metrics summary
    if len(propulsion_metrics_df) > 0:
        summary['propulsion_mean_hr'] = propulsion_metrics_df['mean_hr'].mean()
        summary['propulsion_mean_rmssd'] = propulsion_metrics_df['rmssd'].mean()
        summary['propulsion_mean_stress_index'] = propulsion_metrics_df['stress_index'].mean()
    
    # Add resting metrics summary
    if len(resting_metrics_df) > 0:
        summary['resting_mean_hr'] = resting_metrics_df['mean_hr'].mean()
        summary['resting_mean_rmssd'] = resting_metrics_df['rmssd'].mean()
        summary['resting_mean_stress_index'] = resting_metrics_df['stress_index'].mean()

    # Add custom activity summaries
    if custom_activities:
        custom_total_count = 0
        custom_total_with_metrics = 0
        for name, activities_df in custom_activities.items():
            safe_name = str(name).strip().lower().replace(' ', '_')
            metrics_df = custom_metrics_dfs.get(name, pd.DataFrame())

            summary[f'custom_{safe_name}_count'] = len(activities_df)
            summary[f'custom_{safe_name}_with_metrics'] = len(metrics_df)

            if len(metrics_df) > 0:
                summary[f'custom_{safe_name}_mean_hr'] = metrics_df['mean_hr'].mean()
                summary[f'custom_{safe_name}_mean_rmssd'] = metrics_df['rmssd'].mean()
                summary[f'custom_{safe_name}_mean_stress_index'] = metrics_df['stress_index'].mean()

            custom_total_count += len(activities_df)
            custom_total_with_metrics += len(metrics_df)

        summary['custom_total_count'] = custom_total_count
        summary['custom_total_with_metrics'] = custom_total_with_metrics
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'pipeline_summary.csv', index=False)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)
    for key, value in summary.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Output saved to: {output_dir}")
    logger.info("=" * 80)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Data Inspection Pipeline for Activity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run with existing config
            python run_inspection.py --config config.yaml
            
            # Create default config template
            python run_inspection.py --create-config config.yaml
            """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create default config template'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help='Logging level: DEBUG, INFO, WARNING, ERROR, or CRITICAL (default: INFO)'
    )
    
    args = parser.parse_args()

    configure_logging(args.log_level)
    
    config_path = args.config
    
    if args.create_config:
        # Create default config
        config = create_default_config()
        import os
        os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Created default config template: {config_path}")
        print("Please edit the config file with your data paths and settings.")
        return
    
    # Run pipeline
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    run_inspection_pipeline(config_path)


if __name__ == '__main__':
    main()
