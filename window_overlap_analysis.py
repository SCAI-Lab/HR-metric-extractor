#!/usr/bin/env python3
"""
Window Overlap and Delay Analysis Module

Analyzes the relationship between activity windows and HR metric response.
Useful for understanding physiological delays and finding optimal observation windows.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from scipy.signal import correlate
from scipy.stats import pearsonr, spearmanr


def analyze_hr_response_delay(activity_signal: np.ndarray,
                               hr_signal: np.ndarray,
                               time_sec: np.ndarray,
                               fs: float = 1.0,
                               max_delay_sec: float = 300.0) -> Dict:
    """
    Analyze the delay between activity onset and HR response.
    
    Uses cross-correlation to find optimal delay where HR response peaks.
    
    Args:
        activity_signal: Binary or continuous activity signal (activity level)
        hr_signal: HR or HRV metric timeseries
        time_sec: Time in seconds (must be regularly sampled)
        fs: Sampling frequency (Hz)
        max_delay_sec: Maximum delay to check (seconds)
        
    Returns:
        Dict with:
            - peak_delay_sec: Time of maximum correlation
            - peak_correlation: Correlation coefficient at peak
            - correlation_curve: Correlation values at different lags
            - lags_sec: Time lags tested
    """
    if len(activity_signal) < 100 or len(hr_signal) != len(activity_signal):
        return {
            'peak_delay_sec': np.nan,
            'peak_correlation': np.nan,
            'correlation_curve': np.array([]),
            'lags_sec': np.array([]),
        }
    
    # Normalize signals
    activity_norm = (activity_signal - np.mean(activity_signal)) / (np.std(activity_signal) + 0.0001)
    hr_norm = (hr_signal - np.mean(hr_signal)) / (np.std(hr_signal) + 0.0001)
    
    # Compute cross-correlation
    correlation = correlate(activity_norm, hr_norm, mode='full')
    
    # Compute lags in seconds
    lags = np.arange(-len(activity_signal) + 1, len(activity_signal)) / fs
    
    # Restrict to max delay
    max_lag_samples = int(max_delay_sec * fs)
    center = len(lags) // 2
    valid_range = (lags >= 0) & (lags <= max_delay_sec)
    
    lags_sec = lags[valid_range]
    correlation_curve = correlation[valid_range] / np.max(np.abs(correlation))
    
    if len(correlation_curve) > 0:
        peak_idx = np.argmax(np.abs(correlation_curve))
        peak_delay_sec = lags_sec[peak_idx]
        peak_correlation = correlation_curve[peak_idx]
    else:
        peak_delay_sec = np.nan
        peak_correlation = np.nan
    
    return {
        'peak_delay_sec': peak_delay_sec,
        'peak_correlation': peak_correlation,
        'correlation_curve': correlation_curve,
        'lags_sec': lags_sec,
    }


def segment_activity_into_phases(activity_interval: Tuple[float, float],
                                  baseline_before_sec: float = 120.0,
                                  activity_duration_sec: Optional[float] = None,
                                  recovery_after_sec: float = 300.0) -> Dict[str, Tuple[float, float]]:
    """
    Segment an activity into analysis phases for detailed HR response analysis.
    
    Phases:
    1. Pre-activity baseline: baseline_before_sec before activity start
    2. Activity: from activity start to end
    3. Immediate recovery: recovery_after_sec after activity end
    4. Late recovery: longer window for complete recovery assessment
    
    Args:
        activity_interval: Tuple of (t_start, t_end) for the activity
        baseline_before_sec: Seconds of baseline before activity
        activity_duration_sec: If provided, override activity duration
        recovery_after_sec: Seconds of recovery after activity
        
    Returns:
        Dict with phase segments: {phase_name: (t_start, t_end), ...}
    """
    t_start, t_end = activity_interval
    
    if activity_duration_sec is not None:
        t_end = t_start + activity_duration_sec
    
    phases = {
        'baseline': (t_start - baseline_before_sec, t_start),
        'activity': (t_start, t_end),
        'recovery_immediate': (t_end, t_end + recovery_after_sec),
        'recovery_extended': (t_end, t_end + recovery_after_sec * 2),
        'recovery_complete': (t_end, t_end + recovery_after_sec * 3),
    }
    
    return phases


def extract_phases_from_data(data_df: pd.DataFrame,
                             phases: Dict[str, Tuple[float, float]],
                             time_col: str = 't_sec',
                             signal_col: str = 'signal') -> Dict[str, pd.DataFrame]:
    """
    Extract phase data from a timeseries DataFrame.
    
    Args:
        data_df: DataFrame with time and signal columns
        phases: Dict mapping phase names to (t_start, t_end) tuples
        time_col: Name of time column
        signal_col: Name of signal column
        
    Returns:
        Dict mapping phase names to extracted DataFrames
    """
    phase_data = {}
    
    for phase_name, (t_start, t_end) in phases.items():
        mask = (data_df[time_col] >= t_start) & (data_df[time_col] <= t_end)
        phase_df = data_df[mask].copy()
        
        if len(phase_df) > 0:
            # Normalize time relative to phase start
            phase_df['phase_time'] = phase_df[time_col] - t_start
            phase_data[phase_name] = phase_df
        else:
            phase_data[phase_name] = None
    
    return phase_data


def compute_optimal_windows_for_metrics(activity_phases: Dict[str, pd.DataFrame],
                                        metric_col: str = 'rmssd') -> Dict[str, Dict]:
    """
    Find optimal observation windows for HR metrics during activity phases.
    
    Useful for determining:
    - When does HR peak during activity?
    - How long to wait for HR to stabilize?
    - How long for recovery?
    
    Args:
        activity_phases: Dict with phase data {phase_name: DataFrame}
        metric_col: Column name for the metric to analyze
        
    Returns:
        Dict with window recommendations:
            {phase_name: {
                'min_time_sec': time of minimum,
                'max_time_sec': time of maximum,
                'stabilization_time_sec': when metric stabilizes,
                'magnitude_change': absolute change during phase
            }}
    """
    recommendations = {}
    
    for phase_name, phase_df in activity_phases.items():
        if phase_df is None or len(phase_df) < 10:
            recommendations[phase_name] = {
                'min_time_sec': np.nan,
                'max_time_sec': np.nan,
                'stabilization_time_sec': np.nan,
                'magnitude_change': np.nan,
            }
            continue
        
        if metric_col not in phase_df.columns:
            recommendations[phase_name] = None
            continue
        
        signal = phase_df[metric_col].values
        time = phase_df['phase_time'].values
        
        # Find extrema
        min_idx = np.nanargmin(signal)
        max_idx = np.nanargmax(signal)
        
        # Find stabilization (when metric stops changing significantly)
        if len(signal) > 20:
            rolling_std = pd.Series(signal).rolling(window=10, center=True).std().values
            # Stabilization when std becomes low
            threshold = np.nanmean(rolling_std) * 0.1
            stable_mask = rolling_std < threshold
            if np.any(stable_mask):
                stabilization_idx = np.where(stable_mask)[0][0]
                stabilization_time = time[stabilization_idx]
            else:
                stabilization_time = np.nan
        else:
            stabilization_time = np.nan
        
        recommendations[phase_name] = {
            'min_time_sec': time[min_idx] if not np.isnan(signal[min_idx]) else np.nan,
            'max_time_sec': time[max_idx] if not np.isnan(signal[max_idx]) else np.nan,
            'stabilization_time_sec': stabilization_time,
            'magnitude_change': np.nanmax(signal) - np.nanmin(signal),
        }
    
    return recommendations


def create_window_overlap_report(activity: Dict,
                                 phases: Dict[str, Tuple[float, float]],
                                 hr_metrics_df: pd.DataFrame,
                                 hr_time_col: str = 't_sec',
                                 hr_metric_col: str = 'rmssd') -> pd.DataFrame:
    """
    Create a comprehensive report of activity phases overlapped with HR metrics.
    
    Args:
        activity: Dict with activity info (activity, t_start, t_end, etc.)
        phases: Phase definitions from segment_activity_into_phases()
        hr_metrics_df: DataFrame with HR metrics timeseries
        hr_time_col: Name of time column in hr_metrics_df
        hr_metric_col: Name of HR metric column
        
    Returns:
        DataFrame with one row per phase containing statistics
    """
    report = []
    
    for phase_name, (phase_start, phase_end) in phases.items():
        mask = (hr_metrics_df[hr_time_col] >= phase_start) & (hr_metrics_df[hr_time_col] <= phase_end)
        phase_data = hr_metrics_df[mask]
        
        if len(phase_data) > 0 and hr_metric_col in phase_data.columns:
            metric_values = phase_data[hr_metric_col].dropna()
            
            if len(metric_values) > 0:
                report.append({
                    'activity': activity.get('activity', 'unknown'),
                    'activity_t_start': activity.get('t_start', np.nan),
                    'activity_t_end': activity.get('t_end', np.nan),
                    'phase': phase_name,
                    'phase_t_start': phase_start,
                    'phase_t_end': phase_end,
                    'phase_duration_sec': phase_end - phase_start,
                    'n_samples': len(phase_data),
                    'n_valid_metrics': len(metric_values),
                    'metric_mean': metric_values.mean(),
                    'metric_std': metric_values.std(),
                    'metric_min': metric_values.min(),
                    'metric_max': metric_values.max(),
                })
            else:
                report.append({
                    'activity': activity.get('activity', 'unknown'),
                    'activity_t_start': activity.get('t_start', np.nan),
                    'activity_t_end': activity.get('t_end', np.nan),
                    'phase': phase_name,
                    'phase_t_start': phase_start,
                    'phase_t_end': phase_end,
                    'phase_duration_sec': phase_end - phase_start,
                    'n_samples': len(phase_data),
                    'n_valid_metrics': 0,
                    'metric_mean': np.nan,
                    'metric_std': np.nan,
                    'metric_min': np.nan,
                    'metric_max': np.nan,
                })
        else:
            report.append({
                'activity': activity.get('activity', 'unknown'),
                'activity_t_start': activity.get('t_start', np.nan),
                'activity_t_end': activity.get('t_end', np.nan),
                'phase': phase_name,
                'phase_t_start': phase_start,
                'phase_t_end': phase_end,
                'phase_duration_sec': phase_end - phase_start,
                'n_samples': 0,
                'n_valid_metrics': 0,
                'metric_mean': np.nan,
                'metric_std': np.nan,
                'metric_min': np.nan,
                'metric_max': np.nan,
            })
    
    return pd.DataFrame(report)
