#!/usr/bin/env python3
"""
Data Loading Module

Utilities for loading and preparing physiological data from various formats.
Supports PPG, ECG, HR, and ADL (Activities of Daily Living) data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


def _load_single_timeseries_file(data_path: Path,
                                 time_col: str = 't_sec',
                                 signal_col: str = 'value',
                                 compression: Optional[str] = 'infer',
                                 verbose: bool = True) -> pd.DataFrame:
    """Load a single timeseries CSV file with flexible column detection."""
    data_path = Path(data_path)

    # Load data
    try:
        df = pd.read_csv(data_path, compression='gzip')
    except Exception as e:
        raise IOError(f"Failed to read {data_path}: {str(e)}")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Find time column
    time_options = ['t_sec', 'time', 'timestamp', 't']
    time_col_actual = None
    for opt in time_options:
        if opt in df.columns:
            time_col_actual = opt
            break

    if time_col_actual is None:
        raise ValueError(f"No time column found. Columns: {df.columns.tolist()}")

    # Find signal column
    signal_options = ['value', 'signal', 'ppg', 'ecg', 'hr', 'eda', 'imu']
    signal_col_actual = None
    for opt in signal_options:
        if opt in df.columns:
            signal_col_actual = opt
            break

    if signal_col_actual is None:
        # Use first numeric column after time
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != time_col_actual]
        if numeric_cols:
            signal_col_actual = numeric_cols[0]
        else:
            raise ValueError(f"No numeric signal column found. Columns: {df.columns.tolist()}")

    # Extract relevant columns
    result = df[[time_col_actual, signal_col_actual]].copy()
    result.columns = ['t_sec', 'value']

    # Convert to numeric
    result['t_sec'] = pd.to_numeric(result['t_sec'], errors='coerce')
    result['value'] = pd.to_numeric(result['value'], errors='coerce')

    # Remove NaN rows
    result = result.dropna()

    # Sort by time
    result = result.sort_values('t_sec').reset_index(drop=True)

    if len(result) == 0:
        raise ValueError(f"No valid data after cleaning: {data_path}")

    if verbose:
        print(f"Loaded {len(result)} samples from {data_path}")
        print(f"  Time range: {result['t_sec'].min():.1f} - {result['t_sec'].max():.1f} seconds")
        print(f"  Value range: {result['value'].min():.2f} - {result['value'].max():.2f}")

    return result


def load_timeseries_data(data_path: Path,
                         time_col: str = 't_sec',
                         signal_col: str = 'value',
                         compression: Optional[str] = 'infer') -> pd.DataFrame:
    """
    Load timeseries data from CSV with flexible column detection.
    
    Args:
        data_path: Path to CSV file
        time_col: Name of time column (will try variations if not found)
        signal_col: Name of signal column (will try variations if not found)
        compression: Compression type ('infer', 'gzip', None, etc.)
        
    Returns:
        DataFrame with [time_col, signal_col] columns, sorted by time
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if data_path.is_dir():
        # Load and concatenate all CSV.GZ files in directory (recursively)
        files = sorted(data_path.glob('**/*.csv.gz'))
        if len(files) == 0:
            raise FileNotFoundError(f"No CSV.GZ files found in directory: {data_path}")

        frames = []
        for f in files:
            try:
                df_part = _load_single_timeseries_file(f, time_col=time_col, signal_col=signal_col, compression=compression, verbose=False)
                frames.append(df_part)
            except Exception:
                continue

        if len(frames) == 0:
            raise ValueError(f"No valid data after cleaning: {data_path}")

        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values('t_sec').reset_index(drop=True)

        print(f"Loaded {len(result)} samples from {data_path} ({len(frames)} files)")
        print(f"  Time range: {result['t_sec'].min():.1f} - {result['t_sec'].max():.1f} seconds")
        print(f"  Value range: {result['value'].min():.2f} - {result['value'].max():.2f}")
        return result

    # Single file path
    return _load_single_timeseries_file(data_path, time_col=time_col, signal_col=signal_col, compression=compression, verbose=True)


def load_hr_metrics(metrics_path: Path) -> pd.DataFrame:
    """
    Load pre-computed HR metrics (RMSSD, SDNN, HR, etc.) from CSV.
    
    Expected columns: t_sec, rmssd, sdnn, mean_hr, stress_index, etc.
    
    Args:
        metrics_path: Path to CSV with HR metrics
        
    Returns:
        DataFrame with HR metrics timeseries
    """
    metrics_path = Path(metrics_path)
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    df = pd.read_csv(metrics_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Ensure t_sec column exists
    if 't_sec' not in df.columns:
        if 'time' in df.columns:
            df['t_sec'] = pd.to_numeric(df['time'], errors='coerce')
        else:
            raise ValueError("No time column found in metrics")
    
    # Convert numeric columns
    for col in df.columns:
        if col != 't_sec':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by time
    df = df.sort_values('t_sec').reset_index(drop=True)
    
    print(f"Loaded HR metrics: {len(df)} samples")
    print(f"  Columns: {[c for c in df.columns if c != 't_sec']}")
    
    return df


def extract_window_data(data_df: pd.DataFrame,
                        t_start: float,
                        t_end: float,
                        time_col: str = 't_sec',
                        signal_col: str = 'value',
                        margin_sec: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract signal data for a specific time window.
    
    Args:
        data_df: DataFrame with time and signal columns
        t_start: Start time (seconds)
        t_end: End time (seconds)
        time_col: Name of time column
        signal_col: Name of signal column
        margin_sec: Margin to add on both sides of window
        
    Returns:
        Tuple of (signal_values, time_values)
    """
    t_start_adj = t_start - margin_sec
    t_end_adj = t_end + margin_sec

    time_arr = data_df[time_col].to_numpy()
    lo = np.searchsorted(time_arr, t_start_adj, side='left')
    hi = np.searchsorted(time_arr, t_end_adj, side='right')

    if lo >= hi:
        return np.array([]), np.array([])

    return data_df[signal_col].to_numpy()[lo:hi], time_arr[lo:hi]


def estimate_sampling_frequency(time_array: np.ndarray) -> float:
    """
    Estimate sampling frequency from time array.
    
    Args:
        time_array: Array of time samples
        
    Returns:
        Estimated sampling frequency (Hz)
    """
    if len(time_array) < 2:
        return 1.0
    
    time_diffs = np.diff(time_array)
    # Remove outliers (might be gaps)
    time_diffs = time_diffs[time_diffs > 0]
    
    if len(time_diffs) == 0:
        return 1.0
    
    # Use median to be robust to outliers
    median_dt = np.median(time_diffs)
    fs = 1.0 / median_dt if median_dt > 0 else 1.0
    
    return fs


def create_data_summary(data_df: pd.DataFrame,
                       time_col: str = 't_sec',
                       signal_col: str = 'value') -> Dict:
    """
    Create summary statistics for a dataset.
    
    Args:
        data_df: Data DataFrame
        time_col: Name of time column
        signal_col: Name of signal column
        
    Returns:
        Dict with summary statistics
    """
    summary = {
        'n_samples': len(data_df),
        'duration_sec': data_df[time_col].max() - data_df[time_col].min(),
        'time_start': data_df[time_col].min(),
        'time_end': data_df[time_col].max(),
        'signal_mean': data_df[signal_col].mean(),
        'signal_std': data_df[signal_col].std(),
        'signal_min': data_df[signal_col].min(),
        'signal_max': data_df[signal_col].max(),
        'signal_range': data_df[signal_col].max() - data_df[signal_col].min(),
        'nan_count': data_df[signal_col].isna().sum(),
    }
    
    # Estimate sampling rate
    summary['estimated_fs_hz'] = estimate_sampling_frequency(data_df[time_col].values)
    
    return summary


def _normalize_imu_columns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Normalize common IMU column names to a standard schema.

    Returns DataFrame with columns [t_sec, imu_x, imu_y, imu_z, imu_magnitude]
    or None if no valid IMU columns are found.
    """
    if df is None or len(df) == 0:
        return None

    work_df = df.copy()
    work_df.columns = [c.strip().lower() for c in work_df.columns]

    # Time column detection
    time_col = None
    for col in ['t_sec', 'time', 'timestamp', 't', 'time_sec']:
        if col in work_df.columns:
            time_col = col
            break

    if time_col is None:
        return None

    axis_candidates = {
        'imu_x': ['accX', 'imu_x', 'x', 'acc_x', 'ax', 'accel_x', 'accelerometer_x', 'gyro_x', 'gyr_x'],
        'imu_y': ['accY', 'imu_y', 'y', 'acc_y', 'ay', 'accel_y', 'accelerometer_y', 'gyro_y', 'gyr_y'],
        'imu_z': ['accZ', 'imu_z', 'z', 'acc_z', 'az', 'accel_z', 'accelerometer_z', 'gyro_z', 'gyr_z'],
    }

    selected_cols = {'t_sec': time_col}
    for target_col, options in axis_candidates.items():
        for option in options:
            if option in work_df.columns and option != time_col:
                selected_cols[target_col] = option
                break

    # Fallback: if axes not found, use up to 3 numeric columns after time
    if len(selected_cols) == 1:
        numeric_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != time_col]
        for idx, col_name in enumerate(numeric_cols[:3]):
            selected_cols[f'imu_{"xyz"[idx]}'] = col_name

    if len(selected_cols) == 1:
        return None

    out_cols = [selected_cols['t_sec']]
    for axis in ['imu_x', 'imu_y', 'imu_z']:
        if axis in selected_cols:
            out_cols.append(selected_cols[axis])

    result = work_df[out_cols].copy()
    rename_map = {selected_cols['t_sec']: 't_sec'}
    for axis in ['imu_x', 'imu_y', 'imu_z']:
        if axis in selected_cols:
            rename_map[selected_cols[axis]] = axis

    result = result.rename(columns=rename_map)

    result['t_sec'] = pd.to_numeric(result['t_sec'], errors='coerce')
    for axis in ['imu_x', 'imu_y', 'imu_z']:
        if axis in result.columns:
            result[axis] = pd.to_numeric(result[axis], errors='coerce')

    # Ensure all axis columns exist (NaN when unavailable)
    for axis in ['imu_x', 'imu_y', 'imu_z']:
        if axis not in result.columns:
            result[axis] = np.nan

    result = result.dropna(subset=['t_sec'])
    if len(result) == 0:
        return None

    result['imu_magnitude'] = np.sqrt(
        result['imu_x'].fillna(0.0) ** 2 +
        result['imu_y'].fillna(0.0) ** 2 +
        result['imu_z'].fillna(0.0) ** 2
    )

    result = result.sort_values('t_sec').reset_index(drop=True)
    return result


def _discover_imu_sensors(subject_path: Path) -> Dict[str, List[Path]]:
    """Discover IMU sensor directories and files organized by sensor type.
    
    Looks for patterns like:
    - corsano_bioz_acc/ (contains date folders with CSV files)
    - corsano_wrist_acc/ (contains date folders with CSV files)
    - vivalnk_vv330_acceleration/ (contains CSV files directly)
    
    Returns a dict mapping sensor_name -> list of candidate file/folder paths.
    """
    if subject_path is None or not Path(subject_path).exists():
        return {}
    
    subject_path = Path(subject_path)
    sensor_map: Dict[str, List[Path]] = {}
    
    # Known sensor folder patterns
    imu_sensor_patterns = [
        'corsano_bioz_acc',
        'corsano_wrist_acc',
        'vivalnk_vv330_acceleration',
        'vivalnk_vv330_accel',
        # 'imu',
    ]
    
    for folder in subject_path.iterdir():
        if not folder.is_dir():
            continue
        
        folder_name = folder.name.lower()
        
        # Check if folder matches any known sensor pattern
        matched_sensor = None
        for pattern in imu_sensor_patterns:
            if pattern in folder_name:
                matched_sensor = folder.name  # Use actual folder name (preserves case)
                break
        
        if matched_sensor is None:
            continue
        
        # Collect all CSV files within this sensor directory (recursively for date folders)
        csv_files = list(folder.glob('**/*.csv.gz')) + list(folder.glob('**/*.csv'))
        if csv_files:
            sensor_map[matched_sensor] = sorted(csv_files)
    
    return sensor_map


def load_imu_sensors(subject_path: Path,
                     imu_config: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
    """Load multiple IMU sensor streams as separate DataFrames.
    
    Discovers IMU sensors in subject directory or uses explicit config paths.
    Each sensor is kept separate to preserve independent time synchronization
    and signal characteristics.
    
    Args:
        subject_path: Path to subject directory for auto-discovery
        imu_config: Optional dict mapping sensor_name -> sensor_path
                   (e.g., {'corsano_bioz_acc': '/path/to/corsano_bioz_acc'})
    
    Returns:
        Dict mapping sensor_name -> DataFrame with columns [t_sec, imu_x, imu_y, imu_z, imu_magnitude]
        Sensors with no/invalid data are omitted from the dict.
    """
    result_sensors: Dict[str, pd.DataFrame] = {}
    
    if imu_config is None:
        imu_config = {}
    
    # First, try to load sensors from explicit config paths
    for sensor_name, sensor_path in imu_config.items():
        if sensor_path is None:
            continue
        
        sensor_path = Path(sensor_path)
        candidate_files: List[Path] = []
        
        if sensor_path.is_file():
            candidate_files = [sensor_path]
        elif sensor_path.is_dir():
            candidate_files = sorted(list(sensor_path.glob('**/*.csv.gz')) + list(sensor_path.glob('**/*.csv')))
        
        if not candidate_files:
            continue
        
        # Merge all files for this sensor
        frames = []
        for file_path in candidate_files:
            try:
                compression = 'gzip' if str(file_path).lower().endswith('.gz') else None
                raw_df = pd.read_csv(file_path, compression=compression)
                imu_df = _normalize_imu_columns(raw_df)
                if imu_df is not None and len(imu_df) > 0:
                    frames.append(imu_df)
            except Exception:
                continue
        
        if frames:
            sensor_df = pd.concat(frames, ignore_index=True)
            sensor_df = sensor_df.sort_values('t_sec').reset_index(drop=True)
            result_sensors[sensor_name] = sensor_df
    
    # Second, auto-discover sensors in subject_path (if not already in config)
    auto_sensors = _discover_imu_sensors(subject_path)
    for sensor_name, candidate_files in auto_sensors.items():
        if sensor_name in result_sensors:
            continue  # Already loaded from config
        
        frames = []
        for file_path in candidate_files:
            try:
                compression = 'gzip' if str(file_path).lower().endswith('.gz') else None
                raw_df = pd.read_csv(file_path, compression=compression)
                imu_df = _normalize_imu_columns(raw_df)
                if imu_df is not None and len(imu_df) > 0:
                    frames.append(imu_df)
            except Exception:
                continue
        
        if frames:
            sensor_df = pd.concat(frames, ignore_index=True)
            sensor_df = sensor_df.sort_values('t_sec').reset_index(drop=True)
            result_sensors[sensor_name] = sensor_df
    
    return result_sensors


def load_eda_bioz_data(subject_path: Path,
                       eda_bioz_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load EDA/BioZ data from corsano_bioz_bioz sensor.
    
    Searches for corsano_bioz_bioz directory with date subfolders containing CSV files.
    All files are merged into a single DataFrame.
    
    Args:
        subject_path: Path to subject directory for auto-discovery
        eda_bioz_path: Optional explicit path to corsano_bioz_bioz folder or specific file
        
    Returns:
        DataFrame with columns [t_sec, eda_bioz] or None if not found
    """
    candidate_files: List[Path] = []
    
    # If explicit path provided, use it
    if eda_bioz_path is not None:
        eda_bioz_path = Path(eda_bioz_path)
        if eda_bioz_path.is_file():
            candidate_files = [eda_bioz_path]
        elif eda_bioz_path.is_dir():
            # Recursively find CSV files in date folders
            candidate_files = sorted(list(eda_bioz_path.glob('**/*.csv.gz')) + list(eda_bioz_path.glob('**/*.csv')))
    elif subject_path is not None:
        # Auto-discover corsano_bioz_bioz folder
        subject_path = Path(subject_path)
        bioz_dir = subject_path / 'corsano_bioz_bioz'
        if bioz_dir.exists():
            candidate_files = sorted(list(bioz_dir.glob('**/*.csv.gz')) + list(bioz_dir.glob('**/*.csv')))
    
    if not candidate_files:
        return None
    
    frames = []
    for file_path in candidate_files:
        try:
            compression = 'gzip' if str(file_path).lower().endswith('.gz') else None
            df = pd.read_csv(file_path, compression=compression)
            
            # Normalize column names
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Find time column
            time_col = None
            for col in ['t_sec', 'time', 'timestamp', 't', 'time_sec']:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col is None:
                continue
            
            # Find signal column (BioZ/EDA typically single-valued)
            signal_col = None
            for col in ['value', 'bioz', 'eda', 'bioimpedance', 'impedance', 'conductance', 'gsr']:
                if col in df.columns:
                    signal_col = col
                    break
            
            # Fallback: use first numeric column after time
            if signal_col is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c != time_col]
                if numeric_cols:
                    signal_col = numeric_cols[0]
                else:
                    continue
            
            # Extract, rename, and clean
            result = df[[time_col, signal_col]].copy()
            result.columns = ['t_sec', 'eda_bioz']
            
            result['t_sec'] = pd.to_numeric(result['t_sec'], errors='coerce')
            result['eda_bioz'] = pd.to_numeric(result['eda_bioz'], errors='coerce')
            result = result.dropna()
            result = result.sort_values('t_sec').reset_index(drop=True)
            
            if len(result) > 0:
                frames.append(result)
        except Exception:
            continue
    
    if not frames:
        return None
    
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values('t_sec').reset_index(drop=True)
    return result if len(result) > 0 else None


def load_ppg_data(subject_path: Path, channel: str = 'green') -> Optional[pd.DataFrame]:
    """
    Load PPG data for a subject.
    
    Args:
        subject_path: Path to subject directory
        channel: PPG channel - 'green', 'infrared' (or 'infra_red'), or 'red'
        
    Returns:
        DataFrame with columns [t_sec, ppg] or None if not found
    """
    subject_path = Path(subject_path)
    
    # Map channel names to directory patterns
    channel_map = {
        'green': 'corsano_wrist_ppg2_green_6',
        'infrared': 'corsano_wrist_ppg2_infra_red_22',
        'red': 'corsano_wrist_ppg2_red_182',
    }
    
    if channel.lower() not in channel_map:
        return None
    
    ppg_dir = subject_path / channel_map[channel.lower()]
    
    if not ppg_dir.exists():
        return None
    
    # Find PPG files in the directory (support flat and nested layouts)
    csv_files = sorted(list(ppg_dir.glob('*.csv.gz')) + list(ppg_dir.glob('**/*.csv.gz')))
    # Deduplicate paths when recursive glob includes top-level files
    csv_files = list(dict.fromkeys(csv_files))
    if not csv_files:
        return None

    frames = []
    for ppg_path in csv_files:
        try:
            df = pd.read_csv(ppg_path, compression='gzip')

            # Normalize column names
            df.columns = [c.strip().lower() for c in df.columns]

            # Find time and signal columns
            time_col = None
            signal_col = None

            for col in df.columns:
                if col in ['t_sec', 'time', 'timestamp', 't']:
                    time_col = col
                elif col in ['value', 'ppg', 'signal', 'data']:
                    signal_col = col

            # If we couldn't find standard column names, use first two columns
            if time_col is None or signal_col is None:
                if len(df.columns) >= 2:
                    time_col = df.columns[0]
                    signal_col = df.columns[1]
                else:
                    continue

            result = df[[time_col, signal_col]].copy()
            result.columns = ['t_sec', 'ppg']
            result['t_sec'] = pd.to_numeric(result['t_sec'], errors='coerce')
            result['ppg'] = pd.to_numeric(result['ppg'], errors='coerce')
            result = result.dropna()

            if len(result) > 0:
                frames.append(result)

        except Exception:
            continue

    if not frames:
        return None

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values('t_sec').reset_index(drop=True)
    return result if len(result) > 0 else None


def load_best_available_signal(subject_path: Path, sensor_priority: list = None) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load the best available physiological signal (ECG, PPG) for a subject.
    Tries sensors in order of priority.
    
    Args:
        subject_path: Path to subject directory
        sensor_priority: List of sensors to try in order. Default: ['ecg', 'ppg_green', 'ppg_infrared', 'ppg_red']
        
    Returns:
        Tuple of (DataFrame, sensor_name) or (None, None) if no data found
    """
    if sensor_priority is None:
        sensor_priority = ['ecg', 'ppg_green', 'ppg_infrared', 'ppg_red']
    
    subject_path = Path(subject_path)
    
    for sensor in sensor_priority:
        if sensor == 'ecg':
            ecg_dir = subject_path / 'vivalnk_vv330_ecg'
            if ecg_dir.exists():
                # Try direct path first
                direct_ecg = ecg_dir / 'data_1.csv.gz'
                if direct_ecg.exists():
                    try:
                        df = pd.read_csv(direct_ecg, compression='gzip')
                        df.columns = [c.strip().lower() for c in df.columns]
                        if 'time' in df.columns and 'ecg' in df.columns:
                            result = df[['time', 'ecg']].copy()
                            result.columns = ['t_sec', 'signal']
                            result['t_sec'] = pd.to_numeric(result['t_sec'], errors='coerce')
                            result['signal'] = pd.to_numeric(result['signal'], errors='coerce')
                            result = result.dropna().sort_values('t_sec').reset_index(drop=True)
                            if len(result) > 0:
                                return result, 'ecg'
                    except:
                        pass
                # Try date subfolders
                try:
                    for item in ecg_dir.glob('*/*.csv.gz'):
                        df = pd.read_csv(item, compression='gzip')
                        df.columns = [c.strip().lower() for c in df.columns]
                        if 'time' in df.columns and 'ecg' in df.columns:
                            result = df[['time', 'ecg']].copy()
                            result.columns = ['t_sec', 'signal']
                            result['t_sec'] = pd.to_numeric(result['t_sec'], errors='coerce')
                            result['signal'] = pd.to_numeric(result['signal'], errors='coerce')
                            result = result.dropna().sort_values('t_sec').reset_index(drop=True)
                            if len(result) > 0:
                                return result, 'ecg'
                        break
                except:
                    pass
        
        elif sensor.startswith('ppg_'):
            channel = sensor.split('_')[1]
            ppg_df = load_ppg_data(subject_path, channel)
            if ppg_df is not None:
                ppg_df.columns = ['t_sec', 'signal']
                return ppg_df, f'ppg_{channel}'
    
    return None, None


def load_sensor_hr_data(data_path: Path) -> Optional[pd.DataFrame]:
    """Load sensor HR data from a vivalnk_vv330_heart_rate directory.

    Handles either a directory of ``data_*.csv.gz`` files or a single file.
    Invalid readings (``hr <= 0``) are removed.

    Args:
        data_path: Path to ``vivalnk_vv330_heart_rate`` directory or a single CSV(.gz) file.

    Returns:
        DataFrame with columns ``[t_sec, value]`` where ``value`` is HR in bpm,
        sorted by time, or ``None`` if no valid data was found.
    """
    data_path = Path(data_path)

    if data_path.is_dir():
        candidate_files = sorted(
            list(data_path.glob('**/*.csv.gz')) + list(data_path.glob('**/*.csv'))
        )
        if not candidate_files:
            return None
    else:
        candidate_files = [data_path]

    frames = []
    for file_path in candidate_files:
        try:
            compression = 'gzip' if str(file_path).lower().endswith('.gz') else None
            df = pd.read_csv(file_path, compression=compression)
            df.columns = [c.strip().lower() for c in df.columns]

            # Detect time column
            time_col = None
            for col in ['t_sec', 'time', 'timestamp', 't']:
                if col in df.columns:
                    time_col = col
                    break
            if time_col is None:
                continue

            # Detect HR column
            hr_col = None
            for col in ['hr', 'heart_rate', 'heartrate', 'value']:
                if col in df.columns:
                    hr_col = col
                    break
            if hr_col is None:
                continue

            part = df[[time_col, hr_col]].copy()
            part.columns = ['t_sec', 'value']
            part['t_sec'] = pd.to_numeric(part['t_sec'], errors='coerce')
            part['value'] = pd.to_numeric(part['value'], errors='coerce')
            part = part.dropna()
            # Filter out invalid/artifact HR readings
            part = part[part['value'] > 0]
            if len(part) > 0:
                frames.append(part)
        except Exception:
            continue

    if not frames:
        return None

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values('t_sec').reset_index(drop=True)
    return result if len(result) > 0 else None


def load_sensor_rr_intervals(subject_path: Path,
                             rr_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load pre-measured RR intervals from a Corsano sensor directory.

    Searches (in priority order) the following directory names:
    - ``corsano_wrist_rr_intervals``  (plural, older cohort)
    - ``corsano_wrist_rr_interval``   (singular, newer cohort)
    - ``corsano_bioz_rr_interval``    (BioZ-based wristband)

    All ``*.csv.gz`` / ``*.csv`` files in the matched directory (including
    date-named subdirectories) are concatenated.  RR values outside the
    physiologically plausible range 250–2500 ms are removed.

    Args:
        subject_path: Top-level subject directory for auto-discovery.
        rr_path: Optional explicit path to the RR interval directory or single file.
                 When provided, ``subject_path`` is not searched.

    Returns:
        DataFrame with columns ``[t_sec, rr_ms]`` sorted by time, or ``None``
        if no valid data was found.
    """
    _RR_DIR_CANDIDATES = [
        'corsano_wrist_rr_intervals',
        'corsano_wrist_rr_interval',
        'corsano_bioz_rr_interval',
    ]

    candidate_files: List[Path] = []

    if rr_path is not None:
        rr_path = Path(rr_path)
        if rr_path.is_file():
            candidate_files = [rr_path]
        elif rr_path.is_dir():
            candidate_files = sorted(
                list(rr_path.glob('**/*.csv.gz')) + list(rr_path.glob('**/*.csv'))
            )
    else:
        subject_path = Path(subject_path)
        for dir_name in _RR_DIR_CANDIDATES:
            rr_dir = subject_path / dir_name
            if rr_dir.exists():
                candidate_files = sorted(
                    list(rr_dir.glob('**/*.csv.gz')) + list(rr_dir.glob('**/*.csv'))
                )
                if candidate_files:
                    break  # Use first matching directory

    if not candidate_files:
        return None

    frames = []
    for file_path in candidate_files:
        try:
            compression = 'gzip' if str(file_path).lower().endswith('.gz') else None
            df = pd.read_csv(file_path, compression=compression)
            df.columns = [c.strip().lower() for c in df.columns]

            # Time column
            time_col = None
            for col in ['t_sec', 'time', 'timestamp', 't']:
                if col in df.columns:
                    time_col = col
                    break
            if time_col is None:
                continue

            # RR column (milliseconds)
            rr_col = None
            for col in ['rr', 'rr_ms', 'rr_interval', 'rr_interval_ms', 'nn', 'nn_ms']:
                if col in df.columns:
                    rr_col = col
                    break
            if rr_col is None:
                continue

            part = df[[time_col, rr_col]].copy()
            part.columns = ['t_sec', 'rr_ms']
            part['t_sec'] = pd.to_numeric(part['t_sec'], errors='coerce')
            part['rr_ms'] = pd.to_numeric(part['rr_ms'], errors='coerce')
            part = part.dropna()
            # Filter physiologically implausible values
            part = part[(part['rr_ms'] >= 250) & (part['rr_ms'] <= 2500)]
            if len(part) > 0:
                frames.append(part)
        except Exception:
            continue

    if not frames:
        return None

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values('t_sec').reset_index(drop=True)
    return result if len(result) > 0 else None
