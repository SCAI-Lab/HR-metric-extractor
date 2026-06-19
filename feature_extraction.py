#!/usr/bin/env python3
"""
Feature Extraction Module

Computes multimodal features for machine learning using Neurokit2 and other signal processing libraries:
- HRV features from ECG/RR intervals (time-domain, frequency-domain, non-linear)
- EDA features from electrodermal activity signal (SCL, SCR, tonic/phasic decomposition)
- PPG features from photoplethysmography (later)
- IMU features from accelerometer data (via Tifex, later)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


_TIFEX_API = None
_TIFEX_IMPORT_ERROR_LOGGED = False


def _get_tifex_api():
    """Lazily import and cache Tifex API symbols."""
    global _TIFEX_API
    global _TIFEX_IMPORT_ERROR_LOGGED

    if _TIFEX_API is not None:
        return _TIFEX_API

    try:
        from tifex_py.feature_extraction.extraction import (
            calculate_statistical_features,
            calculate_spectral_features,
            calculate_time_frequency_features,
            StatisticalFeatureParams,
            SpectralFeatureParams,
            TimeFrequencyFeatureParams,
        )
    except ImportError:
        if not _TIFEX_IMPORT_ERROR_LOGGED:
            logger.error("Tifex-Py not installed. Install via: pip install tifex-py")
            _TIFEX_IMPORT_ERROR_LOGGED = True
        return None

    _TIFEX_API = (
        calculate_statistical_features,
        calculate_spectral_features,
        calculate_time_frequency_features,
        StatisticalFeatureParams,
        SpectralFeatureParams,
        TimeFrequencyFeatureParams,
    )
    return _TIFEX_API


TIFEX_TOP_FEATURES_50: List[str] = [
    'mean_of_auto_corr_lag_1_to_23',
    'higuchi_fractal_dimensions_k=10',
    'mean',
    'no._of_slope_sign_changes',
    'shape_factor',
    'moment_order_3',
    'spectrum_linear_slope',
    'higuchi_fractal_dimensions_k=20',
    'hjorth_complexity',
    'higuchi_fractal_dimensions_k=5',
    'min',
    'max_of_tkeo',
    'iqr_of_wav_coeffs_lvl_4',
    'no._of_mean_crossings_of_tkeo',
    'no._of_zero_crossings',
    'harmonic_mean_of_abs',
    'min_of_abs',
    'geometric_mean',
    'max',
    'max_of_wav_coeffs_lvl_0',
    'geometric_mean_of_abs',
    'coefficient_of_variation_of_tkeo',
    'min_of_wav_coeffs_lvl_0',
    'svd_entropy',
    'no._of_mean_crossings',
    'rms',
    'no._of_zero_crossings_of_tkeo',
    'hjorth_mobility',
    'permutation_entropy',
    'entropy',
    'no._of_slope_sign_changes_of_tkeo',
    'coefficient_of_variation',
    'spectral_rel_power_band_[0.6, 4]',
    'median',
    'max_of_abs',
    'trimmed_mean_thresh_0.1',
    'higuchi_fractal_dimensions_k=40',
    'max_of_wav_coeffs_lvl_4',
    'skewness',
    'median_of_wav_coeffs_lvl_0',
    'skewness_of_abs',
    'iqr',
    'skewness_of_tkeo',
    'median_of_abs',
    'median_abs_deviation',
    'kurtosis',
    'std_of_abs',
    'kurtosis_of_wav_coeffs_4',
    'skewness_of_wav_coeffs_4',
    'min_of_wav_coeffs_lvl_4',
]


TIFEX_TO_REQUESTED_KEY_MAP: Dict[str, str] = {
    'mean_of_auto_corr_lags': 'mean_of_auto_corr_lag_1_to_23',
    'spectral_slope_linear': 'spectrum_linear_slope',
    'tkeo_max': 'max_of_tkeo',
    'wave_coeffs_lvl_4_iqr': 'iqr_of_wav_coeffs_lvl_4',
    'tkeo_no._of_mean_crossings': 'no._of_mean_crossings_of_tkeo',
    'no._of_zero_crossings': 'no._of_zero_crossings',
    'harmonic_mean_of_abs': 'harmonic_mean_of_abs',
    'min_of_abs': 'min_of_abs',
    'geometric_mean': 'geometric_mean',
    'max': 'max',
    'wave_coeffs_lvl_0_max': 'max_of_wav_coeffs_lvl_0',
    'geometric_mean_of_abs': 'geometric_mean_of_abs',
    'tkeo_spectral_coefficient_of_variation': 'coefficient_of_variation_of_tkeo',
    'wave_coeffs_lvl_0_min': 'min_of_wav_coeffs_lvl_0',
    'svd_entropy': 'svd_entropy',
    'no._of_mean_crossings': 'no._of_mean_crossings',
    'rms': 'rms',
    'tkeo_no._of_zero_crossings': 'no._of_zero_crossings_of_tkeo',
    'hjorth_mobility': 'hjorth_mobility',
    'permutation_entropy': 'permutation_entropy',
    'entropy': 'entropy',
    'tkeo_no._of_slope_sign_changes': 'no._of_slope_sign_changes_of_tkeo',
    'spectral_coefficient_of_variation': 'coefficient_of_variation',
    'relative_band_power_[0.6, 4]': 'spectral_rel_power_band_[0.6, 4]',
    'median': 'median',
    'max_of_abs': 'max_of_abs',
    'trimmed_mean_0.1': 'trimmed_mean_thresh_0.1',
    'higuchi_fractal_dimensions_k=40': 'higuchi_fractal_dimensions_k=40',
    'wave_coeffs_lvl_4_max': 'max_of_wav_coeffs_lvl_4',
    'skewness': 'skewness',
    'wave_coeffs_lvl_0_median': 'median_of_wav_coeffs_lvl_0',
    'skewness_of_abs': 'skewness_of_abs',
    'iqr': 'iqr',
    'tkeo_skewness': 'skewness_of_tkeo',
    'median_of_abs': 'median_of_abs',
    'median_absolute_deviation': 'median_abs_deviation',
    'mean_abs_deviation': 'median_abs_deviation',
    'kurtosis': 'kurtosis',
    'std_of_abs': 'std_of_abs',
    'wave_coeffs_lvl_4_kurtosis': 'kurtosis_of_wav_coeffs_4',
    'wave_coeffs_lvl_4_skewness': 'skewness_of_wav_coeffs_4',
    'wave_coeffs_lvl_4_min': 'min_of_wav_coeffs_lvl_4',
    'higuchi_fractal_dimensions_k=10': 'higuchi_fractal_dimensions_k=10',
    'mean': 'mean',
    'no._of_slope_sign_changes': 'no._of_slope_sign_changes',
    'shape_factor': 'shape_factor',
    'moment_order_3': 'moment_order_3',
    'higuchi_fractal_dimensions_k=20': 'higuchi_fractal_dimensions_k=20',
    'hjorth_complexity': 'hjorth_complexity',
    'higuchi_fractal_dimensions_k=5': 'higuchi_fractal_dimensions_k=5',
    'min': 'min',
}


def extract_hrv_features(rr_intervals_ms: np.ndarray,
                         fs: Optional[float] = None,
                         method: str = 'neurokit2') -> Dict[str, float]:
    """
    Compute HRV (Heart Rate Variability) features from RR intervals.
    
    Uses Neurokit2's hrv() and hrv_analyze() to calculate comprehensive HRV features:
    time-domain (SDNN, RMSSD, CVNN, pNN50, pNN20, etc.),
    frequency-domain (LF, HF, VLF, LF/HF ratio, etc.),
    and non-linear (SampEn, ApEn, DFA, etc.).
    
    Args:
        rr_intervals_ms: RR intervals in milliseconds (1D array)
        fs: Sampling frequency (Hz) - used for frequency domain analysis. If None, estimated from RR intervals.
        method: Feature computation method ('neurokit2' default)
        
    Returns:
        Dict mapping feature names to values. Includes:
        - Time-domain: SDNN, RMSSD, CVNN, pNN50, pNN20, etc.
        - Frequency-domain: LF, HF, VLF, LF/HF ratio, etc.
        - Non-linear: SampEn, ApEn, DFA, etc.
    """
    if rr_intervals_ms is None or len(rr_intervals_ms) < 10:
        logger.warning(f"Insufficient RR intervals ({len(rr_intervals_ms) if rr_intervals_ms is not None else 0}) for HRV extraction")
        return {}
    
    try:
        import neurokit2 as nk
    except ImportError:
        logger.error("Neurokit2 not installed. Install via: pip install neurokit2")
        return {}
    
    try:
        # Neurokit2 expects RR intervals in seconds
        rr_intervals_sec = rr_intervals_ms / 1000.0
        
        # Estimate sampling frequency if not provided
        if fs is None:
            # Approximate fs for frequency domain analysis
            # Use median RR interval to estimate
            median_rr_sec = np.median(rr_intervals_sec)
            fs_estimate = 1.0 / median_rr_sec if median_rr_sec > 0 else 1.0
        else:
            fs_estimate = fs
        
        # Convert RR intervals to peak indices (samples)
        # nk.hrv() requires peak indices, not RR intervals
        peak_indices = nk.intervals_to_peaks(rr_intervals_sec)
        
        # Compute HRV metrics using Neurokit2's comprehensive hrv() function
        # Pass peak indices and sampling rate for frequency domain analysis
        hrv_metrics = nk.hrv(peak_indices, sampling_rate=int(fs_estimate), show=False)
        
        # Extract features from result
        features = {}
        if isinstance(hrv_metrics, pd.DataFrame):
            if len(hrv_metrics) > 0:
                features = hrv_metrics.iloc[0].to_dict()
        elif isinstance(hrv_metrics, dict):
            features = hrv_metrics
        
        # Convert numpy types to Python types for serialization
        features_clean = {}
        for key, val in features.items():
            if isinstance(val, (np.integer, np.floating)):
                features_clean[key] = float(val)
            elif pd.isna(val):
                features_clean[key] = None
            else:
                features_clean[key] = val
        
        logger.debug(f"Extracted {len(features_clean)} HRV features")
        return features_clean
        
    except Exception as e:
        logger.warning(f"HRV feature extraction failed: {str(e)}")
        return {}


def extract_eda_features(eda_signal: np.ndarray,
                        time_array: np.ndarray,
                        fs: float = 1.0,
                        method: str = 'neurokit2') -> Dict[str, float]:
    """
    Compute EDA (Electrodermal Activity) features from raw EDA/BioZ signal.
    
    Uses Neurokit2's eda_analyze() to decompose signal into tonic and phasic components,
    and extract features like SCL, SCR amplitude/latency, etc.
    
    Args:
        eda_signal: Raw EDA/BioZ signal (1D array)
        time_array: Time vector corresponding to signal (1D array)
        fs: Sampling frequency (Hz, default 1.0 for pre-sampled data)
        method: Feature computation method ('neurokit2' default)
        
    Returns:
        Dict mapping feature names to values. Includes:
        - SCL (Skin Conductance Level) - mean tonic component
        - SCR (Skin Conductance Response) features - phasic component analysis
        - Signal variability metrics
    """
    if eda_signal is None or len(eda_signal) < 10:
        logger.warning(f"Insufficient EDA samples ({len(eda_signal) if eda_signal is not None else 0}) for feature extraction")
        return {}
    
    try:
        import neurokit2 as nk
    except ImportError:
        logger.error("Neurokit2 not installed. Install via: pip install neurokit2")
        return {}
    
    try:
        # Remove NaN values
        valid_mask = ~np.isnan(eda_signal)
        if valid_mask.sum() < 10:
            logger.warning(f"Insufficient valid EDA samples after NaN removal")
            return {}
        
        clean_signal = eda_signal[valid_mask]
        
        # Process and analyze EDA signal using Neurokit2
        eda_processed, info = nk.eda_process(clean_signal, sampling_rate=int(fs))
        
        # Use Neurokit2's built-in eda_analyze() for comprehensive feature extraction
        eda_features = nk.eda_analyze(eda_processed, sampling_rate=int(fs))
        
        # Extract features from result (should be a DataFrame)
        features = {}
        if isinstance(eda_features, pd.DataFrame):
            if len(eda_features) > 0:
                features = eda_features.iloc[0].to_dict()
        elif isinstance(eda_features, dict):
            features = eda_features
        
        # Add basic signal statistics if not already present
        if 'EDA_Mean' not in features:
            features['EDA_Mean'] = float(np.mean(clean_signal))
        if 'EDA_Std' not in features:
            features['EDA_Std'] = float(np.std(clean_signal))
        if 'EDA_Min' not in features:
            features['EDA_Min'] = float(np.min(clean_signal))
        if 'EDA_Max' not in features:
            features['EDA_Max'] = float(np.max(clean_signal))
        if 'EDA_Range' not in features:
            features['EDA_Range'] = float(np.max(clean_signal) - np.min(clean_signal))
        
        # Convert numpy types to Python types for serialization
        features_clean = {}
        for key, val in features.items():
            if isinstance(val, (np.integer, np.floating)):
                features_clean[key] = float(val)
            elif pd.isna(val):
                features_clean[key] = None
            else:
                features_clean[key] = val
        
        logger.debug(f"Extracted {len(features_clean)} EDA features")
        return features_clean
        
    except Exception as e:
        logger.warning(f"EDA feature extraction failed: {str(e)}")
        return {}


def extract_ppg_features(ppg_signal: np.ndarray,
                         time_array: np.ndarray,
                         fs: float = 1.0,
                         method: str = 'neurokit2') -> Dict[str, float]:
    """
    Compute PPG (Photoplethysmography) features from raw PPG signal.
    
    Uses Neurokit2's ppg_process() and ppg_analyze() to extract cardiac and pulse-related features:
    - Heart rate from PPG
    - Pulse features
    - Signal quality metrics
    
    Args:
        ppg_signal: Raw PPG signal (1D array)
        time_array: Time vector corresponding to signal (1D array)
        fs: Sampling frequency (Hz, default 1.0 for pre-sampled data)
        method: Feature computation method ('neurokit2' default)
        
    Returns:
        Dict mapping feature names to values. Includes:
        - PPG_Heart_Rate: Heart rate extracted from PPG
        - PPG pulse-related metrics
        - Signal quality indicators
    """
    if ppg_signal is None or len(ppg_signal) < 10:
        logger.warning(f"Insufficient PPG samples ({len(ppg_signal) if ppg_signal is not None else 0}) for feature extraction")
        return {}
    
    try:
        import neurokit2 as nk
    except ImportError:
        logger.error("Neurokit2 not installed. Install via: pip install neurokit2")
        return {}
    
    try:
        # Remove NaN values
        valid_mask = ~np.isnan(ppg_signal)
        if valid_mask.sum() < 10:
            logger.warning(f"Insufficient valid PPG samples after NaN removal")
            return {}
        
        clean_signal = ppg_signal[valid_mask]
        
        # Process PPG signal using Neurokit2
        ppg_processed, info = nk.ppg_process(clean_signal, sampling_rate=int(fs))
        
        # Use Neurokit2's built-in ppg_analyze() for comprehensive feature extraction
        ppg_features = nk.ppg_analyze(ppg_processed, sampling_rate=int(fs))
        
        # Extract features from result (should be a DataFrame)
        features = {}
        if isinstance(ppg_features, pd.DataFrame):
            if len(ppg_features) > 0:
                features = ppg_features.iloc[0].to_dict()
        elif isinstance(ppg_features, dict):
            features = ppg_features
        
        # Add basic signal statistics if not already present
        if 'PPG_Mean' not in features:
            features['PPG_Mean'] = float(np.mean(clean_signal))
        if 'PPG_Std' not in features:
            features['PPG_Std'] = float(np.std(clean_signal))
        if 'PPG_Min' not in features:
            features['PPG_Min'] = float(np.min(clean_signal))
        if 'PPG_Max' not in features:
            features['PPG_Max'] = float(np.max(clean_signal))
        if 'PPG_Range' not in features:
            features['PPG_Range'] = float(np.max(clean_signal) - np.min(clean_signal))
        
        # Convert numpy types to Python types for serialization
        features_clean = {}
        for key, val in features.items():
            if isinstance(val, (np.integer, np.floating)):
                features_clean[key] = float(val)
            elif pd.isna(val):
                features_clean[key] = None
            else:
                features_clean[key] = val
        
        logger.debug(f"Extracted {len(features_clean)} PPG features")
        return features_clean
        
    except Exception as e:
        logger.warning(f"PPG feature extraction failed: {str(e)}")
        return {}


def extract_activity_hrv_features(rr_intervals_ms: np.ndarray,
                                  activity_dict: Dict,
                                  fs: Optional[float] = None) -> Dict[str, float]:
    """
    Extract HRV features for a single activity window.
    
    Wraps extract_hrv_features and adds activity context.
    
    Args:
        rr_intervals_ms: RR intervals in milliseconds
        activity_dict: Activity metadata (t_start, t_end, etc.)
        fs: Sampling frequency (Hz)
        
    Returns:
        Dict with HRV features, prefixed 'hrv_'
    """
    features = extract_hrv_features(rr_intervals_ms, fs=fs)
    
    # Prefix all feature names with 'hrv_' for clarity
    hrv_features = {f'hrv_{k}': v for k, v in features.items()}
    
    return hrv_features


def extract_activity_eda_features(eda_signal: np.ndarray,
                                  time_array: np.ndarray,
                                  activity_dict: Dict,
                                  fs: float = 1.0) -> Dict[str, float]:
    """
    Extract EDA features for a single activity window.
    
    Wraps extract_eda_features and adds activity context.
    
    Args:
        eda_signal: Raw EDA/BioZ signal
        time_array: Time vector
        activity_dict: Activity metadata (t_start, t_end, duration_sec, etc.)
        fs: Sampling frequency (Hz)
        
    Returns:
        Dict with EDA features, prefixed 'eda_'
    """
    features = extract_eda_features(eda_signal, time_array, fs=fs)
    
    # Prefix all feature names with 'eda_' for clarity
    eda_features = {f'eda_{k.lower()}': v for k, v in features.items()}
    
    return eda_features


def extract_activity_ppg_features(ppg_signal: np.ndarray,
                                  time_array: np.ndarray,
                                  activity_dict: Dict,
                                  fs: float = 1.0) -> Dict[str, float]:
    """
    Extract PPG features for a single activity window.
    
    Wraps extract_ppg_features and adds activity context.
    
    Args:
        ppg_signal: Raw PPG signal
        time_array: Time vector
        activity_dict: Activity metadata (t_start, t_end, duration_sec, etc.)
        fs: Sampling frequency (Hz)
        
    Returns:
        Dict with PPG features, prefixed 'ppg_'
    """
    features = extract_ppg_features(ppg_signal, time_array, fs=fs)
    
    # Prefix all feature names with 'ppg_' for clarity
    ppg_features = {f'ppg_{k.lower()}': v for k, v in features.items()}
    
    return ppg_features


def extract_difficulty_features(imu_window: np.ndarray) -> Dict[str, float]:
    """Compute compact IMU difficulty proxies from one window.

    Returns:
        - ``mean_enmo`` / ``std_enmo`` from ENMO (euclidean norm minus 1g, clipped at 0)
        - ``rms_jerk`` from first differences (uses xyz axes when provided)
        - ``spectral_entropy`` on the window magnitude as a frequency-complexity proxy
    """
    if imu_window is None:
        return {}

    arr = np.asarray(imu_window, dtype=float)
    if arr.size == 0:
        return {}

    axes = None
    if arr.ndim == 1:
        magnitude = arr
    elif arr.ndim == 2:
        if arr.shape[1] >= 3:
            axes = arr[:, :3]
            magnitude = np.linalg.norm(axes, axis=1)
        elif arr.shape[1] == 1:
            magnitude = arr[:, 0]
        else:
            return {}
    else:
        return {}

    valid_mag = magnitude[np.isfinite(magnitude)]
    if len(valid_mag) < 8:
        return {}

    enmo = np.maximum(valid_mag - 1.0, 0.0)
    out: Dict[str, float] = {
        'mean_enmo': float(np.mean(enmo)),
        'std_enmo': float(np.std(enmo)),
    }

    if axes is not None:
        valid_axes = axes[np.all(np.isfinite(axes), axis=1)]
        if len(valid_axes) >= 2:
            axis_jerk = np.diff(valid_axes, axis=0)
            jerk_series = np.linalg.norm(axis_jerk, axis=1)
        else:
            jerk_series = np.array([], dtype=float)
    else:
        jerk_series = np.diff(valid_mag) if len(valid_mag) >= 2 else np.array([], dtype=float)

    out['rms_jerk'] = float(np.sqrt(np.mean(jerk_series ** 2))) if len(jerk_series) > 0 else np.nan

    centered = valid_mag - np.mean(valid_mag)
    if len(centered) >= 4:
        power = np.abs(np.fft.rfft(centered)) ** 2
        if len(power) > 1:
            power = power[1:]  # remove DC bin from entropy calculation
        power_sum = float(np.sum(power))
        if power_sum > 0 and len(power) > 1:
            p = power / power_sum
            p = p[p > 0]
            out['spectral_entropy'] = float(-np.sum(p * np.log2(p)) / np.log2(len(power)))
        else:
            out['spectral_entropy'] = np.nan
    else:
        out['spectral_entropy'] = np.nan

    return out


def extract_imu_tifex_top_features(imu_signal: np.ndarray,
                                   fs: float,
                                   feature_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Extract a constrained set of top-performing IMU features using Tifex-Py.

    This function computes only a targeted subset of features (default: top-50
    list) to keep runtime practical.
    """
    if imu_signal is None or len(imu_signal) < 64:
        logger.debug("Insufficient IMU samples for Tifex feature extraction")
        return {}

    selected_features = set(feature_names or TIFEX_TOP_FEATURES_50)

    tifex_api = _get_tifex_api()
    if tifex_api is None:
        return {}

    (
        calculate_statistical_features,
        calculate_spectral_features,
        calculate_time_frequency_features,
        StatisticalFeatureParams,
        SpectralFeatureParams,
        TimeFrequencyFeatureParams,
    ) = tifex_api

    try:
        valid_mask = ~np.isnan(imu_signal)
        clean_signal = np.asarray(imu_signal[valid_mask], dtype=float)
        if len(clean_signal) < 64:
            return {}

        window_size = len(clean_signal)
        fs_safe = max(1, int(round(float(fs))))

        stat_params = StatisticalFeatureParams(
            window_size=window_size,
            n_lags_auto_correlation=23,
            moment_orders=[3],
            trimmed_mean_thresholds=[0.1],
            higuchi_k_values=[5, 10, 20, 40],
            calculators=[
                'mean', 'higuchi_fractal_dimensions', 'slope_sign_change',
                'shape_factor', 'higher_order_moments', 'hjorth_mobility_and_complexity',
                'min', 'zero_crossings', 'harmonic_mean_abs', 'min_abs',
                'geometric_mean', 'max', 'geometric_mean_abs', 'coefficient_of_variation',
                'svd_entropy', 'mean_crossing', 'root_mean_square',
                'permutation_entropy', 'entropy', 'median', 'max_abs',
                'trimmed_mean', 'skewness', 'skewness_abs', 'interquartile_range',
                'median_abs', 'median_absolute_deviation', 'kurtosis', 'std_abs',
                'mean_auto_correlation',
            ],
        )

        tkeo_sf_params = StatisticalFeatureParams(
            window_size=window_size,
            calculators=['max', 'mean_crossing', 'zero_crossings', 'slope_sign_change', 'coefficient_of_variation', 'skewness'],
        )
        wavelet_sf_params = StatisticalFeatureParams(
            window_size=window_size,
            calculators=['interquartile_range', 'max', 'min', 'median', 'kurtosis', 'skewness'],
        )
        tf_params = TimeFrequencyFeatureParams(
            window_size=window_size,
            decomposition_level=5,
            tkeo_sf_params=tkeo_sf_params,
            wavelet_sf_params=wavelet_sf_params,
            calculators=['tkeo_features', 'wavelet_features'],
        )

        spec_params = SpectralFeatureParams(
            fs=fs_safe,
            f_bands=[[0.6, 4]],
            calculators=['spectral_slope_linear', 'band_power'],
        )

        stat_df = calculate_statistical_features(clean_signal, params=stat_params)
        tf_df = calculate_time_frequency_features(clean_signal, params=tf_params)
        spec_df = calculate_spectral_features(clean_signal, params=spec_params)

        raw_features: Dict[str, float] = {}
        for feature_df in (stat_df, tf_df, spec_df):
            if isinstance(feature_df, pd.DataFrame) and len(feature_df) > 0:
                raw_features.update(feature_df.iloc[0].to_dict())

        mapped_features: Dict[str, float] = {}
        for tifex_key, requested_key in TIFEX_TO_REQUESTED_KEY_MAP.items():
            if requested_key not in selected_features:
                continue
            if tifex_key not in raw_features:
                continue
            value = raw_features[tifex_key]
            if pd.isna(value):
                continue
            if isinstance(value, (np.integer, np.floating)):
                mapped_features[requested_key] = float(value)
            else:
                mapped_features[requested_key] = value

        return mapped_features

    except Exception as e:
        logger.warning(f"IMU Tifex feature extraction failed: {str(e)}")
        return {}


def extract_activity_imu_features(imu_signal: np.ndarray,
                                  time_array: np.ndarray,
                                  activity_dict: Dict,
                                  sensor_name: str,
                                  imu_axes_window: Optional[np.ndarray] = None,
                                  fs: float = 1.0,
                                  feature_names: Optional[List[str]] = None) -> Dict[str, float]:
    """Extract selected IMU features for a single activity window and sensor."""
    features = extract_imu_tifex_top_features(imu_signal, fs=fs, feature_names=feature_names)
    diff_input = imu_axes_window if imu_axes_window is not None else imu_signal
    features.update(extract_difficulty_features(diff_input))
    sensor_key = str(sensor_name).strip().lower().replace(' ', '_')
    imu_features = {f'imu_{sensor_key}_{k}': v for k, v in features.items()}
    return imu_features


def merge_feature_dicts(*feature_dicts) -> Dict[str, float]:
    """Merge multiple feature dictionaries, handling conflicts by taking first non-None value."""
    result = {}
    for feat_dict in feature_dicts:
        if feat_dict is None:
            continue
        for key, val in feat_dict.items():
            if key not in result and val is not None:
                result[key] = val
    return result


# ---------------------------------------------------------------------------
# Sensor-based HR and HRV feature extraction
# ---------------------------------------------------------------------------

def _number_peaks(x: np.ndarray, n: int) -> int:
    """Count the number of peaks with support *n* (tsfresh-compatible strict local maxima).

    A sample at position ``i`` is a peak iff ``x[i] > x[i-j]`` and
    ``x[i] > x[i+j]`` for all ``1 <= j <= n``.
    """
    x = np.asarray(x, dtype=float)
    if n < 1 or len(x) < 2 * n + 1:
        return 0
    x_reduced = x[n:-n]
    mask = np.ones(len(x_reduced), dtype=bool)
    for i in range(1, n + 1):
        left = x[n - i : len(x) - n - i]         # length == len(x_reduced)
        right = x[n + i : len(x) - (n - i) if (n - i) > 0 else len(x)]
        mask &= (x_reduced > left) & (x_reduced > right)
    return int(mask.sum())


def extract_hr_sensor_features(hr_values: np.ndarray) -> Dict[str, float]:
    """Compute time-series features directly on the sensor's HR estimates (bpm).

    Produces ``HR__``, ``HR_1st_deriv__``, and ``HR_2nd_deriv__`` features.
    Invalid readings (``hr <= 0``) are removed before computation.

    Args:
        hr_values: 1-D array of HR values in bpm; may contain numbers ≤ 0
                   (sensor artifact markers) which will be filtered out.

    Returns:
        Dict with keys:
        ``HR__abs_energy``, ``HR__maximum``, ``HR__root_mean_square``,
        ``HR_1st_deriv__abs_energy``, ``HR_1st_deriv__maximum``,
        ``HR_1st_deriv__minimum``, ``HR_1st_deriv__number_peaks__n_1``,
        ``HR_1st_deriv__root_mean_square``,
        ``HR_2nd_deriv__abs_energy``, ``HR_2nd_deriv__maximum``,
        ``HR_2nd_deriv__minimum``, ``HR_2nd_deriv__root_mean_square``.
    """
    _nan_result: Dict[str, float] = {
        'HR__abs_energy': np.nan,
        'HR__maximum': np.nan,
        'HR__root_mean_square': np.nan,
        'HR_1st_deriv__abs_energy': np.nan,
        'HR_1st_deriv__maximum': np.nan,
        'HR_1st_deriv__minimum': np.nan,
        'HR_1st_deriv__number_peaks__n_1': np.nan,
        'HR_1st_deriv__root_mean_square': np.nan,
        'HR_2nd_deriv__abs_energy': np.nan,
        'HR_2nd_deriv__maximum': np.nan,
        'HR_2nd_deriv__minimum': np.nan,
        'HR_2nd_deriv__root_mean_square': np.nan,
    }

    hr = np.asarray(hr_values, dtype=float)
    hr = hr[hr > 0]  # remove invalid sensor artifact codes
    if len(hr) < 3:
        logger.debug("Insufficient valid HR samples for feature extraction")
        return _nan_result

    d1 = np.diff(hr)
    d2 = np.diff(d1)

    return {
        'HR__abs_energy': float(np.sum(hr ** 2)),
        'HR__maximum': float(np.max(hr)),
        'HR__root_mean_square': float(np.sqrt(np.mean(hr ** 2))),
        'HR_1st_deriv__abs_energy': float(np.sum(d1 ** 2)),
        'HR_1st_deriv__maximum': float(np.max(d1)),
        'HR_1st_deriv__minimum': float(np.min(d1)),
        'HR_1st_deriv__number_peaks__n_1': float(_number_peaks(d1, 1)),
        'HR_1st_deriv__root_mean_square': float(np.sqrt(np.mean(d1 ** 2))),
        'HR_2nd_deriv__abs_energy': float(np.sum(d2 ** 2)) if len(d2) > 0 else np.nan,
        'HR_2nd_deriv__maximum': float(np.max(d2)) if len(d2) > 0 else np.nan,
        'HR_2nd_deriv__minimum': float(np.min(d2)) if len(d2) > 0 else np.nan,
        'HR_2nd_deriv__root_mean_square': float(np.sqrt(np.mean(d2 ** 2))) if len(d2) > 0 else np.nan,
    }


def _compute_hrv_features_from_rr_ms(
    rr_ms: np.ndarray, fs: float = 1.0
) -> Dict[str, float]:
    """Shared HRV computation on an array of RR intervals already in ms.

    Computes ``HRV_MinNN``, ``HRV_SDRMSSD`` from plain statistics and
    attempts ``HRV_LFn`` / ``HRV_TINN`` via NeuroKit2.

    ``rr_ms`` must already be filtered to physiologically plausible values
    before calling; this function does no additional filtering.
    """
    out: Dict[str, float] = {
        'HRV_MinNN': np.nan,
        'HRV_SDRMSSD': np.nan,
        'HRV_LFn': np.nan,
        'HRV_TINN': np.nan,
    }

    if len(rr_ms) < 4:
        return out

    out['HRV_MinNN'] = float(np.min(rr_ms))

    sdnn = float(np.std(rr_ms, ddof=1))
    diffs = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else np.nan
    if not np.isnan(rmssd) and rmssd > 0:
        out['HRV_SDRMSSD'] = float(sdnn / rmssd)

    try:
        import neurokit2 as nk

        rr_sec = rr_ms / 1000.0
        peaks = nk.intervals_to_peaks(rr_sec)
        nk_fs = int(max(1, round(fs)))

        # Use targeted functions instead of nk.hrv() to avoid computing
        # ~50 metrics (including expensive nonlinear ones) when we only need 2.
        try:
            freq_df = nk.hrv_frequency(peaks, sampling_rate=nk_fs, show=False)
            if isinstance(freq_df, pd.DataFrame) and len(freq_df) > 0:
                val = freq_df.iloc[0].get('HRV_LFn', None)
                if val is not None and pd.notna(val):
                    out['HRV_LFn'] = float(val)
        except Exception as freq_e:
            logger.debug(f"HRV_LFn computation failed: {freq_e}")

        try:
            time_df = nk.hrv_time(peaks, sampling_rate=nk_fs, show=False)
            if isinstance(time_df, pd.DataFrame) and len(time_df) > 0:
                val = time_df.iloc[0].get('HRV_TINN', None)
                if val is not None and pd.notna(val):
                    out['HRV_TINN'] = float(val)
        except Exception as time_e:
            logger.debug(f"HRV_TINN computation failed: {time_e}")

    except Exception as e:
        logger.debug(f"NeuroKit2 HRV computation failed: {e}")

    return out


def _assess_hr_quality(hr_values: np.ndarray) -> float:
    """Assess heart rate data quality as a fraction of valid readings.

    Valid HR readings are > 0 bpm (invalid sensor readings are ≤ 0).

    Args:
        hr_values: 1-D array of HR values in bpm.

    Returns:
        Quality score in [0.0, 1.0] where 1.0 means all readings are valid.
        Returns 0.0 if input is empty.
    """
    hr = np.asarray(hr_values, dtype=float)
    if len(hr) == 0:
        return 0.0
    valid_count = float(np.sum(hr > 0))
    return valid_count / len(hr)


def extract_hrv_from_rr_intervals(
    rr_ms_window: np.ndarray, fs: float = 1.0
) -> Dict[str, float]:
    """Compute HRV features from pre-measured RR intervals (Corsano PPG sensor).

    Used as a FALLBACK when vivalnk HR data quality is poor.
    The intervals are directly measured rather than inferred from
    peak detection or HR→RR conversion.

    Computes:
    - ``HRV_MinNN``: minimum NN interval (ms)
    - ``HRV_SDRMSSD``: SDNN / RMSSD ratio
    - ``HRV_LFn``: normalised LF power (via NeuroKit2)
    - ``HRV_TINN``: triangular interpolation of NN histogram (via NeuroKit2)

    Args:
        rr_ms_window: 1-D array of RR intervals in **milliseconds** as
                      returned by ``extract_window_data`` on the
                      ``corsano_wrist_rr_interval*`` / ``corsano_bioz_rr_interval``
                      DataFrame (column ``rr_ms``).
        fs: Effective sampling rate to pass to NeuroKit2 for frequency-domain
            analysis.  The Corsano sensors deliver roughly 1 interval per
            heartbeat (~1 Hz); the default is fine for most use-cases.

    Returns:
        Dict with keys ``HRV_MinNN``, ``HRV_SDRMSSD``, ``HRV_LFn``, ``HRV_TINN``.
        Features that cannot be computed are ``NaN``.
    """
    rr = np.asarray(rr_ms_window, dtype=float)
    rr = rr[~np.isnan(rr)]
    rr = rr[(rr >= 250) & (rr <= 2500)]
    return _compute_hrv_features_from_rr_ms(rr, fs=fs)


def extract_hrv_from_sensor_hr(hr_values: np.ndarray, fs: float = 1.0) -> Dict[str, float]:
    """Compute HRV features from the sensor's HR estimates (VivaLNK, primary source).

    This is the PREFERRED HRV computation path. Converts HR → RR intervals and
    delegates to :func:`_compute_hrv_features_from_rr_ms`.

    Computes the same four features as :func:`extract_hrv_from_rr_intervals`:
    ``HRV_MinNN``, ``HRV_SDRMSSD``, ``HRV_LFn``, ``HRV_TINN``.

    Invalid HR readings (≤ 0) are removed before conversion.
    """
    hr = np.asarray(hr_values, dtype=float)
    hr = hr[hr > 0]
    if len(hr) < 4:
        return {'HRV_MinNN': np.nan, 'HRV_SDRMSSD': np.nan,
                'HRV_LFn': np.nan, 'HRV_TINN': np.nan}

    rr_ms = 60000.0 / hr
    rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
    return _compute_hrv_features_from_rr_ms(rr_ms, fs=fs)


def extract_eda_sensor_features(eda_signal: np.ndarray, fs: float = 25.0) -> Dict[str, float]:
    """Compute the specific EDA feature set from a raw EDA/BioZ signal.

    Features:
    - ``EDA_MAVFD``: Mean Absolute Value of First Difference.
    - ``EDA_Range``: signal range (max − min).
    - ``EDA_Tonic_SD``: standard deviation of the tonic (SCL) component.
    - ``EDA_Sympathetic``: sympathetic tone index (via NeuroKit2).
    - ``EDA_Phasic__number_peaks__n_1``: peaks with support 1 in phasic component.
    - ``EDA_Phasic__number_peaks__n_5``: peaks with support 5 in phasic component.

    Args:
        eda_signal: Raw 1-D EDA/BioZ signal.
        fs: Sampling frequency in Hz (default 25 Hz for Corsano BioZ).

    Returns:
        Dict with the six feature keys above.  Any feature that cannot be
        computed is ``NaN``.
    """
    out: Dict[str, float] = {
        'EDA_MAVFD': np.nan,
        'EDA_Phasic__number_peaks__n_1': np.nan,
        'EDA_Phasic__number_peaks__n_5': np.nan,
        'EDA_Range': np.nan,
        'EDA_Sympathetic': np.nan,
        'EDA_Tonic_SD': np.nan,
    }

    sig = np.asarray(eda_signal, dtype=float)
    valid = sig[~np.isnan(sig)]
    if len(valid) < 10:
        logger.debug("Insufficient valid EDA samples for sensor feature extraction")
        return out

    out['EDA_MAVFD'] = float(np.mean(np.abs(np.diff(valid))))
    out['EDA_Range'] = float(np.max(valid) - np.min(valid))

    try:
        import neurokit2 as nk

        proc = nk.eda_process(valid, sampling_rate=int(max(1, round(fs))))
        if not (isinstance(proc, tuple) and len(proc) == 2):
            return out
        signals_df, _info = proc

        if not isinstance(signals_df, pd.DataFrame):
            return out

        if 'EDA_Tonic' in signals_df.columns:
            out['EDA_Tonic_SD'] = float(np.std(signals_df['EDA_Tonic'].values))

        if 'EDA_Phasic' in signals_df.columns:
            phasic = signals_df['EDA_Phasic'].values
            out['EDA_Phasic__number_peaks__n_1'] = float(_number_peaks(phasic, 1))
            out['EDA_Phasic__number_peaks__n_5'] = float(_number_peaks(phasic, 5))

        try:
            sym_result = nk.eda_sympathetic(valid, sampling_rate=int(max(1, round(fs))))
            if isinstance(sym_result, dict) and 'EDA_Sympathetic' in sym_result:
                out['EDA_Sympathetic'] = float(sym_result['EDA_Sympathetic'])
            elif isinstance(sym_result, pd.DataFrame) and 'EDA_Sympathetic' in sym_result.columns:
                out['EDA_Sympathetic'] = float(sym_result['EDA_Sympathetic'].iloc[0])
        except Exception as sym_err:
            logger.debug(f"EDA sympathetic index computation failed: {sym_err}")

    except Exception as e:
        logger.warning(f"EDA sensor feature extraction failed: {e}")

    return out


def extract_activity_sensor_features(
    hr_values: np.ndarray,
    eda_signal: Optional[np.ndarray] = None,
    rr_intervals_ms: Optional[np.ndarray] = None,
    hr_fs: float = 1.0,
    eda_fs: float = 25.0,
) -> Dict[str, float]:
    """Convenience wrapper: compute all sensor-based features for one activity window.

    Combines :func:`extract_hr_sensor_features`, HRV features, and
    (optionally) :func:`extract_eda_sensor_features`.

    For HRV, uses hierarchical quality-based selection:
    1. **Primary**: :func:`extract_hrv_from_sensor_hr` from VivaLNK HR estimates
       (if quality ≥ 50% valid readings).
    2. **Fallback**: :func:`extract_hrv_from_rr_intervals` from Corsano PPG RR
       intervals (only if HR quality is poor and RR data is available).
    3. **None**: Returns ``NaN`` for all HRV features if both paths fail.

    Args:
        hr_values: Sensor HR signal (bpm) from VivaLNK for the window.
        eda_signal: Optional EDA/BioZ signal for the window.
        rr_intervals_ms: Optional pre-measured RR intervals in ms from Corsano PPG.
        hr_fs: Sampling frequency of the HR signal (Hz).
        eda_fs: Sampling frequency of the EDA signal (Hz).

    Returns:
        Merged feature dict containing HR, HRV, and EDA features.
    """
    features: Dict[str, float] = {}
    features.update(extract_hr_sensor_features(hr_values))

    # Quality-based HRV hierarchy: assess vivalnk HR quality first
    hr_quality = _assess_hr_quality(hr_values)
    hr_quality_threshold = 0.5  # require ≥ 50% valid HR readings

    if hr_quality >= hr_quality_threshold:
        # Primary path: use VivaLNK HR (ECG-based or fallback to PPG)
        features.update(extract_hrv_from_sensor_hr(hr_values, fs=hr_fs))
    else:
        # HR quality is poor; try Corsano PPG RR intervals as fallback
        rr = np.asarray(rr_intervals_ms, dtype=float) if rr_intervals_ms is not None else None
        if rr is not None and len(rr[~np.isnan(rr)]) >= 4:
            features.update(extract_hrv_from_rr_intervals(rr, fs=hr_fs))
        else:
            # Both paths failed; return NaN HRV features
            features.update({
                'HRV_MinNN': np.nan,
                'HRV_SDRMSSD': np.nan,
                'HRV_LFn': np.nan,
                'HRV_TINN': np.nan,
            })

    if eda_signal is not None and len(eda_signal) >= 10:
        features.update(extract_eda_sensor_features(eda_signal, fs=eda_fs))
    return features

