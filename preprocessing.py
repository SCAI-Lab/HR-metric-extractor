#!/usr/bin/env python3
"""
Signal Preprocessing Module

Provides basic ECG/PPG signal preprocessing including:
- Baseline drift removal (high-pass filtering)
- High-frequency noise removal (low-pass filtering)
- Powerline interference removal (notch filtering)
- Signal normalization
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch, filtfilt
from typing import Tuple, Optional


def apply_butterworth_filter(signal: np.ndarray, 
                            lowcut: Optional[float] = None,
                            highcut: Optional[float] = None,
                            fs: float = 128.0,
                            order: int = 4) -> np.ndarray:
    """
    Apply Butterworth bandpass or high/low-pass filter to signal.
    
    Args:
        signal: Input signal array
        lowcut: Low cutoff frequency (Hz). If None, no high-pass filtering.
        highcut: High cutoff frequency (Hz). If None, no low-pass filtering.
        fs: Sampling frequency (Hz)
        order: Filter order (default 4)
        
    Returns:
        Filtered signal
    """
    if lowcut is None and highcut is None:
        return signal.copy()
    
    nyquist = fs / 2.0
    
    # Validate cutoff frequencies
    if lowcut is not None and lowcut >= nyquist:
        raise ValueError(f"lowcut ({lowcut} Hz) must be < Nyquist ({nyquist} Hz)")
    if highcut is not None and highcut >= nyquist:
        raise ValueError(f"highcut ({highcut} Hz) must be < Nyquist ({nyquist} Hz)")
    
    if lowcut is not None and highcut is not None:
        # Bandpass filter
        if lowcut >= highcut:
            raise ValueError("lowcut must be < highcut")
        sos = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band', output='sos')
    elif lowcut is not None:
        # High-pass filter
        sos = butter(order, lowcut / nyquist, btype='high', output='sos')
    else:
        # Low-pass filter
        sos = butter(order, highcut / nyquist, btype='low', output='sos')
    
    # Apply filter with initial conditions to reduce transients
    zi = sosfilt_zi(sos) * signal[0]
    filtered, _ = sosfilt(sos, signal, zi=zi)
    
    return filtered


def apply_notch_filter(signal: np.ndarray,
                      notch_freq: float = 50.0,
                      fs: float = 128.0,
                      quality: int = 30) -> np.ndarray:
    """
    Apply notch filter to remove powerline interference (50/60 Hz).
    
    Args:
        signal: Input signal array
        notch_freq: Notch frequency (Hz) - typically 50 or 60 Hz
        fs: Sampling frequency (Hz)
        quality: Quality factor (higher = narrower notch)
        
    Returns:
        Filtered signal
    """
    nyquist = fs / 2.0
    
    if notch_freq >= nyquist:
        raise ValueError(f"notch_freq ({notch_freq} Hz) must be < Nyquist ({nyquist} Hz)")
    
    # iirnotch returns (b, a) coefficients - use filtfilt for zero-phase filtering
    b, a = iirnotch(notch_freq, quality, fs=fs)
    filtered = filtfilt(b, a, signal)
    
    return filtered


def preprocess_ecg(signal: np.ndarray,
                  fs: float = 128.0,
                  remove_baseline: bool = True,
                  remove_noise: bool = True,
                  remove_powerline: bool = True,
                  normalize: bool = False) -> np.ndarray:
    """
    Apply standard ECG preprocessing pipeline.
    
    Default settings optimized for ECG analysis:
    - Remove baseline drift: High-pass filter at 0.5 Hz
    - Remove high-frequency noise: Low-pass filter at 40 Hz
    - Remove powerline interference: Notch filter at 50 Hz (or 60 Hz)
    - Optional: Z-score normalization
    
    Args:
        signal: Raw ECG signal
        fs: Sampling frequency (Hz)
        remove_baseline: Apply high-pass filter to remove baseline drift
        remove_noise: Apply low-pass filter to remove high-frequency noise
        remove_powerline: Apply notch filter to remove 50/60 Hz interference
        normalize: Apply z-score normalization after filtering
        
    Returns:
        Preprocessed signal
    """
    filtered = signal.copy()
    
    # Remove baseline drift (high-pass at 0.5 Hz typical for ECG)
    if remove_baseline:
        filtered = apply_butterworth_filter(filtered, lowcut=0.5, fs=fs, order=4)
    
    # Remove high-frequency noise (low-pass at 40 Hz typical for ECG)
    if remove_noise:
        filtered = apply_butterworth_filter(filtered, highcut=40.0, fs=fs, order=4)
    
    # Remove powerline interference (50 Hz or 60 Hz)
    if remove_powerline:
        # Try 50 Hz first (common in Europe)
        filtered = apply_notch_filter(filtered, notch_freq=50.0, fs=fs, quality=30)
    
    # Normalize (z-score)
    if normalize:
        mean = np.mean(filtered)
        std = np.std(filtered)
        if std > 0:
            filtered = (filtered - mean) / std
    
    return filtered


def preprocess_ppg(signal: np.ndarray,
                  fs: float = 64.0,
                  remove_baseline: bool = True,
                  remove_noise: bool = True,
                  remove_powerline: bool = True,
                  normalize: bool = False) -> np.ndarray:
    """
    Apply standard PPG preprocessing pipeline.
    
    Default settings optimized for PPG analysis:
    - Remove baseline drift: High-pass filter at 0.5 Hz
    - Remove high-frequency noise: Low-pass filter at 5 Hz (PPG has lower frequency content)
    - Remove powerline interference: Notch filter at 50 Hz
    - Optional: Z-score normalization
    
    Args:
        signal: Raw PPG signal
        fs: Sampling frequency (Hz)
        remove_baseline: Apply high-pass filter
        remove_noise: Apply low-pass filter (narrower than ECG)
        remove_powerline: Apply notch filter
        normalize: Apply z-score normalization
        
    Returns:
        Preprocessed signal
    """
    filtered = signal.copy()
    
    # Remove baseline drift
    if remove_baseline:
        filtered = apply_butterworth_filter(filtered, lowcut=0.5, fs=fs, order=4)
    
    # Remove high-frequency noise (PPG has lower frequency content, 0.5-5 Hz)
    if remove_noise:
        filtered = apply_butterworth_filter(filtered, highcut=5.0, fs=fs, order=4)
    
    # Remove powerline interference -- might need to change that to 60Hz (Japan)
    if remove_powerline:
        filtered = apply_notch_filter(filtered, notch_freq=50.0, fs=fs, quality=30)
    
    # Normalize
    if normalize:
        mean = np.mean(filtered)
        std = np.std(filtered)
        if std > 0:
            filtered = (filtered - mean) / std
    
    return filtered
