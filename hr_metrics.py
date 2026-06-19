import numpy as np
import logging
import pandas as pd
from scipy.signal import find_peaks
from preprocessing import preprocess_ecg

# Optional NeuroKit2 integration (cleaning + peak detection)
try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except Exception:
    HAS_NEUROKIT = False

logger = logging.getLogger(__name__)


def check_signal_quality(signal: np.ndarray) -> dict:
    """
    Assess the quality of a physiological signal.
    
    Args:
        signal: 1-D array of signal values
        
    Returns:
        dict with keys: 'is_flat', 'std', 'range', 'unique_values', 'quality_score'
        quality_score: 0-1 where 1 is best
    """
    sig = np.asarray(signal).astype(float)
    if len(sig) == 0:
        return {'is_flat': True, 'std': 0, 'range': 0, 'unique_values': 0, 'quality_score': 0}
    
    signal_std = np.std(sig)
    signal_range = np.max(sig) - np.min(sig)
    unique_values = len(np.unique(sig))
    
    # Signal is flat if std is very small
    is_flat = signal_std < 1e-6
    
    # Quality score based on signal variance (simplified), maybe arbitrary, use NK score instead?
    # Typical ECG/PPG signals have std in reasonable range
    quality_score = min(1.0, signal_std / 5.0) if signal_std > 0 else 0
    
    return {
        'is_flat': is_flat,
        'std': signal_std,
        'range': signal_range,
        'unique_values': unique_values,
        'quality_score': quality_score
    }


def extract_rr_intervals_from_ecg(ecg_signal: np.ndarray, fs: float = 32.0, min_distance_samples: int = 15, preprocess: bool = True):
    """
    Attempt to extract RR intervals manually from an ECG signal using simple peak detection.
    
    :param ecg_signal: 1-D array of ECG signal values
    :type ecg_signal: np.ndarray
    :param fs: sampling frequency in Hz (used for peak detection)
    :type fs: float
    :param min_distance_samples: minimum number of samples between consecutive peaks
    :type min_distance_samples: int
    :param preprocess: whether to apply preprocessing to the ECG signal
    :type preprocess: bool
    """
    if len(ecg_signal) < 100:
        return np.array([]), np.array([])
    
    # Apply preprocessing if enabled
    if preprocess:
        signal = preprocess_ecg(ecg_signal, fs=fs)
    else:
        signal = ecg_signal.copy()
    
    # Check if signal has any variation
    signal_std = np.std(signal)
    if signal_std < 1e-6:
        # Signal is essentially flat - cannot detect peaks
        return np.array([]), np.array([])
    
    signal = -signal if np.mean(signal) > 0 else signal
    threshold = np.mean(signal) + 0.5 * np.std(signal)
    peaks, _ = find_peaks(signal, height=threshold, distance=min_distance_samples)
    if len(peaks) < 2:
        return np.array([]), np.array([])
    peak_intervals = np.diff(peaks)
    rr_intervals = (peak_intervals / fs) * 1000.0
    rr_intervals = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]
    return rr_intervals, peaks


def extract_rr_intervals_with_neurokit(signal: np.ndarray, fs: float = 32.0, signal_type: str = 'ecg'):
    """
    Attempt to extract RR intervals using NeuroKit2 cleaning and peak detection.
    If ECG processing fails or returns no peaks, fallback to PPG processing.
    Returns (rr_intervals_ms, peaks_indices) or (None, None) if NeuroKit2 unavailable
    or extraction failed.
    """
    if not HAS_NEUROKIT:
        return None, None

    try:
        sig = np.asarray(signal).astype(float)
        if sig.size == 0:
            return np.array([]), np.array([])

        if signal_type == 'ecg':
            # Use NeuroKit2's ecg_process for cleaning and R-peak detection
            proc = nk.ecg_process(sig, sampling_rate=fs)
            # nk.ecg_process returns (signals_df, info_dict)
            if isinstance(proc, tuple) and len(proc) == 2:
                signals_df, info = proc
            else:
                logger.warning("ECG process returned unexpected format; attempting PPG fallback")
                return _extract_rr_with_ppg_fallback(sig, fs)

            # Get R-peak indices from DataFrame
            if isinstance(signals_df, pd.DataFrame) and 'ECG_R_Peaks' in signals_df.columns:
                # ECG_R_Peaks is a boolean Series; convert to indices
                rpeaks_bool = signals_df['ECG_R_Peaks'].values
                rpeaks_idx = np.where(rpeaks_bool)[0]
            elif isinstance(info, dict) and 'ECG_R_Peaks' in info:
                # Fallback: check info dict
                rpeaks_idx = np.asarray(info.get('ECG_R_Peaks', []))
            else:
                logger.warning("ECG: No R-peaks found; attempting PPG fallback")
                return _extract_rr_with_ppg_fallback(sig, fs)

            if len(rpeaks_idx) < 2:
                logger.warning(f"ECG: Found only {len(rpeaks_idx)} peaks; attempting PPG fallback")
                return _extract_rr_with_ppg_fallback(sig, fs)

            peak_intervals = np.diff(rpeaks_idx)
            rr_intervals = (peak_intervals / fs) * 1000.0
            rr_intervals = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]
            
            if len(rr_intervals) == 0:
                logger.warning("ECG: No valid RR intervals after filtering; attempting PPG fallback")
                return _extract_rr_with_ppg_fallback(sig, fs)
            
            logger.info(f"ECG extraction successful: {len(rpeaks_idx)} peaks, {len(rr_intervals)} valid RR intervals")
            return rr_intervals, rpeaks_idx

        elif signal_type == 'ppg':
            return _extract_rr_with_ppg_fallback(sig, fs)

    except Exception as e:
        logger.warning(f"ECG extraction failed with exception: {e}; attempting PPG fallback")
        return _extract_rr_with_ppg_fallback(signal, fs)


def _extract_rr_with_ppg_fallback(sig: np.ndarray, fs: float = 32.0):
    """
    Extract RR intervals using PPG processing as a fallback.
    """
    try:
        logger.info("Falling back to PPG processing")
        # Use NeuroKit2's ppg_process
        proc = nk.ppg_process(sig, sampling_rate=fs)
        if isinstance(proc, tuple) and len(proc) == 2:
            signals_df, info = proc
        else:
            return np.array([]), np.array([])

        # Get PPG peaks from DataFrame
        peaks = None
        if isinstance(signals_df, pd.DataFrame):
            if 'PPG_Peaks' in signals_df.columns:
                peaks_bool = signals_df['PPG_Peaks'].values
                peaks = np.where(peaks_bool)[0]
            elif 'PPG_Onsets' in signals_df.columns:
                onsets_bool = signals_df['PPG_Onsets'].values
                peaks = np.where(onsets_bool)[0]

        # Fallback to simple peak detection on cleaned PPG signal
        if peaks is None or len(peaks) < 2:
            cleaned = None
            if isinstance(signals_df, pd.DataFrame) and 'PPG_Clean' in signals_df.columns:
                cleaned = signals_df['PPG_Clean'].values
            else:
                cleaned = nk.ppg_clean(sig, sampling_rate=fs)
            min_dist = max(1, int(0.4 * fs))
            peaks, _ = find_peaks(cleaned, distance=min_dist)

        if len(peaks) < 2:
            logger.warning("PPG: Not enough peaks detected after fallback attempts")
            return np.array([]), np.array([])

        peak_intervals = np.diff(peaks)
        rr_intervals = (peak_intervals / fs) * 1000.0
        rr_intervals = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]
        
        if len(rr_intervals) > 0:
            logger.info(f"PPG fallback successful: {len(peaks)} peaks, {len(rr_intervals)} valid RR intervals")
        
        return rr_intervals, peaks
    
    except Exception as e:
        logger.warning(f"PPG fallback also failed: {e}")
        return np.array([]), np.array([])


def compute_rmssd(rr_intervals: np.ndarray):
    if len(rr_intervals) < 2:
        return np.nan
    diff = np.diff(rr_intervals)
    return np.sqrt(np.mean(diff ** 2))


def compute_hrv_summary_with_neurokit(rr_intervals: np.ndarray, peaks, fs: float = 32.0) -> dict:
    """Compute HRV summary using NeuroKit2 when available, with safe fallbacks.

    Returns a dict with keys: 'pnn50','median_rr_ms','cv_rr_percent','mean_rr_ms','rmssd','sdnn','mean_hr'
    """
    out = {
        'pnn50': np.nan,
        'median_rr_ms': np.nan,
        'cv_rr_percent': np.nan,
        'mean_rr_ms': np.nan,
        'rmssd': np.nan,
        'sdnn': np.nan,
        'mean_hr': np.nan,
    }

    if rr_intervals is None or len(rr_intervals) == 0:
        return out

    # Computed defaults
    out['mean_rr_ms'] = np.mean(rr_intervals)
    out['rmssd'] = compute_rmssd(rr_intervals)
    out['sdnn'] = compute_sdnn(rr_intervals)
    out['mean_hr'] = compute_mean_hr(rr_intervals)
    out['median_rr_ms'] = np.median(rr_intervals)
    diffs = np.abs(np.diff(rr_intervals))
    out['pnn50'] = (np.sum(diffs > 50) / len(diffs) * 100.0) if len(diffs) > 0 else np.nan
    out['cv_rr_percent'] = (out['sdnn'] / out['median_rr_ms'] * 100.0) if out['median_rr_ms'] != 0 else np.nan

    if not HAS_NEUROKIT:
        return out

    # Try NeuroKit2's hrv if peaks are available
    try:
        # NeuroKit expects a rpeaks dict as returned by nk.ecg_peaks
        rpeaks = {'ECG_R_Peaks': np.asarray(peaks)} if peaks is not None else None
        if rpeaks is None or len(rpeaks['ECG_R_Peaks']) < 2:
            return out

        df = nk.hrv(rpeaks, sampling_rate=fs)
        if df is None or len(df) == 0:
            return out
        s = df.iloc[0]

        def pick(*names):
            for n in names:
                if n in s.index:
                    val = s[n]
                    try:
                        if np.isfinite(val):
                            return val
                    except Exception:
                        return val
            return None

        # Map common NeuroKit outputs to our keys
        meannn = pick('HRV_MeanNN', 'HRV_Mean_NN', 'MeanNN', 'mean_nni')
        if meannn is not None:
            out['mean_rr_ms'] = float(meannn)

        rmssd_nk = pick('HRV_RMSSD', 'RMSSD')
        if rmssd_nk is not None:
            out['rmssd'] = float(rmssd_nk)

        sdnn_nk = pick('HRV_SDNN', 'SDNN')
        if sdnn_nk is not None:
            out['sdnn'] = float(sdnn_nk)

        median_nn = pick('HRV_MedianNN', 'HRV_Median_NN', 'MedianNN')
        if median_nn is not None:
            out['median_rr_ms'] = float(median_nn)

        pnn50_nk = pick('HRV_pNN50', 'pNN50')
        if pnn50_nk is not None:
            out['pnn50'] = float(pnn50_nk)

        meanhr = pick('HRV_MeanHR', 'MeanHR')
        if meanhr is not None:
            out['mean_hr'] = float(meanhr)

        # Recompute CV if we have sdnn and median
        if not np.isnan(out['sdnn']) and not np.isnan(out['median_rr_ms']) and out['median_rr_ms'] != 0:
            out['cv_rr_percent'] = (out['sdnn'] / out['median_rr_ms']) * 100.0

    except Exception:
        # On any failure, return the computed defaults
        return out

    return out


def compute_sdnn(rr_intervals: np.ndarray):
    if len(rr_intervals) < 2:
        return np.nan
    return np.std(rr_intervals)


def compute_mean_hr(rr_intervals: np.ndarray):
    if len(rr_intervals) < 1:
        return np.nan
    return 60000.0 / np.mean(rr_intervals)


def compute_baevsky_stress_index(rr_intervals: np.ndarray):
    if len(rr_intervals) < 10:
        return np.nan
    hist, bins = np.histogram(rr_intervals, bins=50)
    mode_bin = np.argmax(hist)
    amo = (bins[mode_bin] + bins[mode_bin+1]) / 2.0
    mx = np.max(rr_intervals)
    mn = np.min(rr_intervals)
    mx_dmn = mx - mn
    if mx_dmn == 0:
        return np.nan
    si = (amo / (2.0 * mx_dmn)) * 100.0
    return si


def compute_additional_hrv_metrics(rr_intervals: np.ndarray) -> dict:
    """Compute additional HRV summary metrics from RR intervals (ms).

    Returns pnn50, median_rr_ms, cv_rr_percent.
    """
    out = {
        'pnn50': np.nan,
        'median_rr_ms': np.nan,
        'cv_rr_percent': np.nan,
    }
    if rr_intervals is None or len(rr_intervals) < 2:
        return out

    diffs = np.abs(np.diff(rr_intervals))
    pnn50 = np.sum(diffs > 50) / len(diffs) * 100.0
    median_rr = np.median(rr_intervals)
    sdnn = compute_sdnn(rr_intervals)
    cv = (sdnn / median_rr) * 100.0 if median_rr != 0 else np.nan

    out['pnn50'] = pnn50
    out['median_rr_ms'] = median_rr
    out['cv_rr_percent'] = cv
    return out


def preprocess_eda(signal: np.ndarray, fs: float = 32.0):
    """
    Preprocess EDA signal using NeuroKit2 if available.

    Returns (signals, info) from NeuroKit2 or (None, None) if unavailable.
    """
    if not HAS_NEUROKIT:
        return None, None
    try:
        sig = np.asarray(signal).astype(float)
        if sig.size == 0:
            return None, None
        proc = nk.eda_process(sig, sampling_rate=fs)
        if isinstance(proc, tuple) and len(proc) == 2:
            signals, info = proc
        elif isinstance(proc, dict):
            signals = proc.get('signals', {})
            info = proc.get('summary', {}) or proc.get('info', {})
        else:
            return None, None
        return signals, info
    except Exception:
        return None, None


def compute_hr_metrics_for_window(rr_intervals: np.ndarray):
    return {
        'n_beats': len(rr_intervals),
        'mean_rr_ms': np.mean(rr_intervals) if len(rr_intervals)>0 else np.nan,
        'rmssd': compute_rmssd(rr_intervals),
        'sdnn': compute_sdnn(rr_intervals),
        'mean_hr': compute_mean_hr(rr_intervals),
        'stress_index': compute_baevsky_stress_index(rr_intervals),
    }


def compute_differential_metrics(activity_metrics: dict, baseline_metrics: dict) -> dict:
    """Compute absolute and percent differences between activity and baseline metrics.

    Returns a flat dict containing keys like 'delta_mean_hr' and 'pct_mean_hr'.
    Missing or invalid baseline values produce NaN percent changes.
    """
    keys = ['mean_hr', 'rmssd', 'sdnn', 'stress_index', 'mean_rr_ms', 'n_beats']
    out = {}
    for k in keys:
        a = activity_metrics.get(k, np.nan)
        b = baseline_metrics.get(k, np.nan)
        try:
            a_val = np.float64(a)
        except Exception:
            a_val = np.nan
        try:
            b_val = np.float64(b)
        except Exception:
            b_val = np.nan

        delta = np.nan
        pct = np.nan
        if not np.isnan(a_val) and not np.isnan(b_val):
            delta = a_val - b_val
            if b_val != 0:
                pct = (delta / b_val) * 100.0
            else:
                pct = np.nan

        out[f'delta_{k}'] = delta
        out[f'pct_{k}'] = pct

    return out


def extract_hr_metrics_from_timeseries(signal, time=None, signal_type: str = 'ecg', fs: float = 32.0) -> dict:
    """Extract HR metrics from a timeseries window.

    Args:
        signal: 1-D array-like signal (ECG/PPG) or HR (bpm) series when signal_type=='hr'.
        time: optional time vector in seconds (not required by current implementation).
        signal_type: one of 'ecg', 'ppg', or 'hr'.
        fs: sampling frequency in Hz (used for peak detection on ECG/PPG).

    Returns:
        dict: metrics produced by `compute_hr_metrics_for_window` and a few helpers.
    """
    sig = np.asarray(signal)
    if sig.size == 0:
        return {
            'n_beats': 0,
            'mean_rr_ms': np.nan,
            'rmssd': np.nan,
            'sdnn': np.nan,
            'mean_hr': np.nan,
            'stress_index': np.nan,
        }

    if signal_type in ('ecg', 'ppg'):
        # Try NeuroKit2-based extraction first when available
        rr_intervals = None
        peaks = np.array([])
        if HAS_NEUROKIT:
            try:
                rr_intervals_nk, peaks_nk = extract_rr_intervals_with_neurokit(sig, fs=fs, signal_type=signal_type)
                if rr_intervals_nk is not None:
                    rr_intervals = rr_intervals_nk
                    peaks = peaks_nk if peaks_nk is not None else np.array([])
            except Exception:
                rr_intervals = None

        # Fallback to existing ECG/PPG extraction
        if rr_intervals is None:
            rr_intervals, peaks = extract_rr_intervals_from_ecg(sig, fs=fs)

        metrics = compute_hr_metrics_for_window(rr_intervals)
        metrics['n_peaks'] = len(peaks)
        metrics['rr_intervals_ms'] = rr_intervals.tolist() if len(rr_intervals) > 0 else []
        # Add NeuroKit HRV summary where possible (falls back to computed defaults)
        extra = compute_hrv_summary_with_neurokit(rr_intervals, peaks, fs=fs)
        metrics.update(extra)
        return metrics

    if signal_type == 'hr':
        # signal is instantaneous HR (bpm). Convert to RR (ms) where possible.
        hr_vals = sig.astype(float)
        hr_vals = hr_vals[~np.isnan(hr_vals)]
        if hr_vals.size == 0:
            return {
                'n_beats': 0,
                'mean_rr_ms': np.nan,
                'rmssd': np.nan,
                'sdnn': np.nan,
                'mean_hr': np.nan,
                'stress_index': np.nan,
            }

        # Convert HR (bpm) to RR (ms)
        rr_from_hr = 60000.0 / hr_vals
        metrics = compute_hr_metrics_for_window(rr_from_hr)
        # Ensure mean_hr is consistent with provided HR values
        metrics['mean_hr'] = np.nanmean(hr_vals) if hr_vals.size > 0 else np.nan
        extra = compute_hrv_summary_with_neurokit(rr_from_hr, peaks=None, fs=fs)
        metrics.update(extra)
        return metrics

    # Fallback: attempt to treat as ECG
    rr_intervals, peaks = extract_rr_intervals_from_ecg(sig, fs=fs)
    metrics = compute_hr_metrics_for_window(rr_intervals)
    metrics['n_peaks'] = len(peaks)
    extra = compute_hrv_summary_with_neurokit(rr_intervals, peaks, fs=fs)
    metrics.update(extra)
    return metrics
