"""
Plot raw heart rate series over activity windows for a subject.

Loads ECG data only around activity windows and overlays the activity interval
on the HR series so window selection can be inspected quickly.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loading import estimate_sampling_frequency
from hr_metrics import extract_rr_intervals_from_ecg


TIME_COL_OPTIONS = ['t_sec', 'time', 'timestamp', 't']
SIGNAL_COL_OPTIONS = ['ecg', 'ppg', 'value', 'signal', 'hr']


def get_latest_batch_dir(base_dir: Path) -> Path:
    batch_dirs = sorted(base_dir.glob('batch_*'), reverse=True)
    if not batch_dirs:
        raise FileNotFoundError(f"No batch_* directories found in {base_dir}")
    return batch_dirs[0]


def resolve_ecg_path(data_base: Path, subject_id: str) -> Path:
    subject_path = data_base / subject_id
    ecg_dir = subject_path / 'vivalnk_vv330_ecg'
    if not ecg_dir.exists():
        raise FileNotFoundError(f"ECG directory not found for subject: {subject_id}")

    direct_ecg = ecg_dir / 'data_1.csv.gz'
    if direct_ecg.exists():
        return direct_ecg

    return ecg_dir


def _pick_time_signal_columns(sample_df: pd.DataFrame) -> tuple[str, str]:
    cols = [c.strip().lower() for c in sample_df.columns]
    lower_to_orig = {c.strip().lower(): c for c in sample_df.columns}

    time_col = None
    for opt in TIME_COL_OPTIONS:
        if opt in cols:
            time_col = lower_to_orig[opt]
            break

    if time_col is None:
        raise ValueError(f"No time column found. Columns: {sample_df.columns.tolist()}")

    signal_col = None
    for opt in SIGNAL_COL_OPTIONS:
        if opt in cols:
            signal_col = lower_to_orig[opt]
            break

    if signal_col is None:
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != time_col]
        if numeric_cols:
            signal_col = numeric_cols[0]
        else:
            raise ValueError(f"No numeric signal column found. Columns: {sample_df.columns.tolist()}")

    return time_col, signal_col


def _read_signal_window_from_file(file_path: Path,
                                  t_start: float,
                                  t_end: float,
                                  margin_sec: float,
                                  chunk_size: int) -> pd.DataFrame:
    t_min = t_start - margin_sec
    t_max = t_end + margin_sec

    try:
        sample = pd.read_csv(file_path, compression='gzip', nrows=5)
    except Exception:
        return pd.DataFrame()

    try:
        time_col, signal_col = _pick_time_signal_columns(sample)
    except ValueError:
        return pd.DataFrame()

    rows = []
    try:
        for chunk in pd.read_csv(
            file_path,
            compression='gzip',
            usecols=[time_col, signal_col],
            chunksize=chunk_size
        ):
            chunk = chunk.rename(columns={time_col: 't_sec', signal_col: 'value'})
            chunk['t_sec'] = pd.to_numeric(chunk['t_sec'], errors='coerce')
            chunk['value'] = pd.to_numeric(chunk['value'], errors='coerce')
            chunk = chunk.dropna()

            window_mask = (chunk['t_sec'] >= t_min) & (chunk['t_sec'] <= t_max)
            if window_mask.any():
                rows.append(chunk.loc[window_mask])
    except Exception:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    data = pd.concat(rows, ignore_index=True)
    data = data.sort_values('t_sec').reset_index(drop=True)
    return data


def load_signal_for_window(ecg_path: Path,
                           t_start: float,
                           t_end: float,
                           margin_sec: float = 30.0,
                           chunk_size: int = 200000) -> pd.DataFrame:
    if ecg_path.is_dir():
        files = sorted(ecg_path.glob('**/*.csv.gz'))
        frames = []
        for f in files:
            df_part = _read_signal_window_from_file(f, t_start, t_end, margin_sec, chunk_size)
            if not df_part.empty:
                frames.append(df_part)
        if not frames:
            return pd.DataFrame()
        data = pd.concat(frames, ignore_index=True)
        return data.sort_values('t_sec').reset_index(drop=True)

    return _read_signal_window_from_file(ecg_path, t_start, t_end, margin_sec, chunk_size)


def load_activity_windows(batch_dir: Path,
                          subject_id: str,
                          activities: list[str],
                          max_windows: int | None) -> list[dict]:
    activity_files = {
        'resting': 'resting_activities.csv',
        'propulsion': 'propulsion_activities.csv',
        'washing_hands': 'activity_washing_hands.csv',
    }

    subject_dir = batch_dir / subject_id
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject output not found in batch: {subject_dir}")

    windows = []
    for activity in activities:
        file_name = activity_files.get(activity)
        if not file_name:
            continue

        activity_path = subject_dir / file_name
        if not activity_path.exists():
            continue

        try:
            df = pd.read_csv(activity_path)
        except pd.errors.EmptyDataError:
            continue

        if df.empty:
            continue

        if max_windows is not None:
            df = df.head(max_windows)

        for idx, row in df.iterrows():
            windows.append({
                'activity': activity,
                'activity_idx': idx,
                't_start': float(row['t_start']),
                't_end': float(row['t_end']),
                'duration_sec': float(row['duration_sec'])
            })

    windows.sort(key=lambda x: x['t_start'])
    return windows


def compute_hr_series(time_sec: np.ndarray,
                      signal: np.ndarray,
                      fs: float | None) -> tuple[np.ndarray, np.ndarray]:
    if len(time_sec) < 2 or len(signal) < 2:
        return np.array([]), np.array([])

    if fs is None:
        fs = estimate_sampling_frequency(time_sec)

    rr_intervals, peaks = extract_rr_intervals_from_ecg(signal, fs=fs)
    if len(peaks) < 2:
        return np.array([]), np.array([])

    # Align RR intervals with their corresponding peak times
    peak_intervals = np.diff(peaks)
    rr_ms_all = (peak_intervals / fs) * 1000.0
    valid_mask = (rr_ms_all >= 300) & (rr_ms_all <= 2000)
    if not np.any(valid_mask):
        return np.array([]), np.array([])

    rr_ms = rr_ms_all[valid_mask]
    hr_values = 60000.0 / rr_ms
    hr_times = time_sec[peaks[1:]][valid_mask]
    return hr_times, hr_values


def plot_window_hr(window: dict,
                   data: pd.DataFrame,
                   output_dir: Path,
                   subject_id: str,
                   margin_sec: float,
                   fs: float | None,
                   relative_time: bool = True) -> None:
    if data.empty:
        return

    signal = data['value'].values
    time_sec = data['t_sec'].values
    hr_times, hr_values = compute_hr_series(time_sec, signal, fs)

    if len(hr_times) == 0:
        return

    t_start = window['t_start']
    t_end = window['t_end']

    if relative_time:
        hr_times_plot = hr_times - t_start
        activity_start = 0.0
        activity_end = t_end - t_start
        x_label = 'Time from activity start (sec)'
    else:
        hr_times_plot = hr_times
        activity_start = t_start
        activity_end = t_end
        x_label = 'Time (sec)'

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hr_times_plot, hr_values, color='black', linewidth=1.2)
    ax.axvspan(activity_start, activity_end, color='orange', alpha=0.25, label='Activity window')

    ax.set_title(
        f"{subject_id} | {window['activity']} | idx {window['activity_idx']} | "
        f"dur {window['duration_sec']:.1f}s"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel('Heart Rate (bpm)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right')

    if relative_time:
        ax.set_xlim(-margin_sec, (t_end - t_start) + margin_sec)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{window['activity']}_idx_{window['activity_idx']}.png"
    out_path = output_dir / out_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize HR around activity windows')
    parser.add_argument('--subject-id', required=True, help='Subject ID (e.g., sub_0303)')
    parser.add_argument('--batch-dir', default=None, help='Batch output directory')
    parser.add_argument('--data-base', default='./data', help='Dataset base directory')
    parser.add_argument('--activities', default='resting,propulsion,washing_hands', help='Comma-separated list of activities')
    parser.add_argument('--margin-sec', type=float, default=30.0, help='Padding around activity window in seconds')
    parser.add_argument('--max-windows', type=int, default=None, help='Max windows per activity')
    parser.add_argument('--fs', type=float, default=None, help='Sampling frequency in Hz (optional)')
    parser.add_argument('--relative-time', action='store_true', help='Plot time relative to activity start')
    parser.add_argument('--window-only', action='store_true', help='Only plot activity windows (default behavior)')
    parser.add_argument('--output-dir', default='output/overlays', help='Output directory for plots')

    args = parser.parse_args()

    batch_base = Path('./output_batch')
    batch_dir = Path(args.batch_dir) if args.batch_dir else get_latest_batch_dir(batch_base)

    activities = [a.strip() for a in args.activities.split(',') if a.strip()]

    windows = load_activity_windows(batch_dir, args.subject_id, activities, args.max_windows)
    if not windows:
        print('No activity windows found to plot.')
        return

    ecg_path = resolve_ecg_path(Path(args.data_base), args.subject_id)

    output_dir = Path(args.output_dir) / args.subject_id

    for window in windows:
        data = load_signal_for_window(ecg_path, window['t_start'], window['t_end'], args.margin_sec)
        plot_window_hr(
            window,
            data,
            output_dir,
            args.subject_id,
            args.margin_sec,
            args.fs,
            relative_time=args.relative_time
        )

    print(f"Saved {len(windows)} overlay plots to {output_dir}")


if __name__ == '__main__':
    main()
