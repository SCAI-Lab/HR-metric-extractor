import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict


def _normalize_keywords(keywords: List[str]) -> List[str]:
    """Normalize keyword list for case-insensitive substring matching."""
    if not keywords:
        return []
    return [str(k).strip().lower() for k in keywords if str(k).strip()]


def parse_adl_file(adl_path: Path) -> pd.DataFrame:
    """
    Load and parse ADL (Activity of Daily Living) data from CSV.
    
    Supports three formats:
    1. Event-based: time/timestamp column with 'start activity'/'end activity' events
    2. Interval-based (legacy): t_start and t_end columns
    3. Interval-based (new synchronized): start_time and end_time columns with label
    
    Args:
        adl_path: Path to ADL CSV file (supports gzip compression)
        
    Returns:
        DataFrame with columns ['t_sec', 'activity'] for event-based data,
        or ['t_start', 't_end', 'activity'] for interval-based data
    """
    from pathlib import Path
    import gzip
    
    adl_path = Path(adl_path)

    # macOS can create AppleDouble sidecar files prefixed with '._'; prefer the real file.
    if adl_path.name.startswith('._'):
        sibling = adl_path.with_name(adl_path.name[2:])
        if sibling.exists():
            adl_path = sibling
    
    # Load data (handle gzip compression)
    if adl_path.suffix == '.gz':
        try:
            with gzip.open(adl_path, 'rt', encoding='utf-8', errors='ignore') as f:
                df = pd.read_csv(f)
        except gzip.BadGzipFile as e:
            raise ValueError(
                f"Invalid gzip ADL file: {adl_path}. "
                "This is often a sidecar file (for example names starting with '._')."
            ) from e
    else:
        df = pd.read_csv(adl_path)
    
    df.columns = [c.strip().lower() for c in df.columns]
    
    # ===== Detect data format =====
    
    # Check for new synchronized format (start_time, end_time, label)
    if 'start_time' in df.columns and 'end_time' in df.columns and 'label' in df.columns:
        # New synchronized interval format from healthy controls dataset
        result = pd.DataFrame()
        result['t_start'] = pd.to_numeric(df['start_time'], errors='coerce')
        result['t_end'] = pd.to_numeric(df['end_time'], errors='coerce')
        result['activity'] = df['label'].astype(str).str.strip().str.lower()
        result['duration_sec'] = result['t_end'] - result['t_start']
        result = result.dropna(subset=['t_start', 't_end'])
        return result[['t_start', 't_end', 'activity', 'duration_sec']]
    
    # Check for legacy interval format (t_start, t_end)
    elif 't_start' in df.columns and 't_end' in df.columns:
        result = pd.DataFrame()
        result['t_start'] = pd.to_numeric(df['t_start'], errors='coerce')
        result['t_end'] = pd.to_numeric(df['t_end'], errors='coerce')
        
        # Find activity column
        activity_col = None
        for col in ['activity', 'adl', 'adls', 'label', 'event']:
            if col in df.columns:
                activity_col = col
                break
        
        if activity_col is None:
            raise ValueError('No activity column found (expected: activity, adl, adls, label, or event)')
        
        result['activity'] = df[activity_col].astype(str).str.strip().str.lower()
        result['duration_sec'] = result['t_end'] - result['t_start']
        result = result.dropna(subset=['t_start', 't_end'])
        return result[['t_start', 't_end', 'activity', 'duration_sec']]
    
    # Event-based format (original format with start/end events)
    else:
        # Find time column
        time_col = None
        for col in ['time', 'timestamp', 't_sec']:
            if col in df.columns:
                time_col = col
                break
        
        if time_col is None:
            raise ValueError('No time column found (expected: time, timestamp, or t_sec)')
        
        # Find activity column
        activity_col = None
        for col in ['adls', 'adl', 'activity', 'event']:
            if col in df.columns:
                activity_col = col
                break
        
        if activity_col is None:
            raise ValueError('No activity column found (expected: adls, adl, activity, or event)')
        
        result = pd.DataFrame()
        result['t_sec'] = pd.to_numeric(df[time_col], errors='coerce')
        result['activity'] = df[activity_col].astype(str).str.strip().str.lower()
        result = result.dropna(subset=['t_sec'])
        return result[['t_sec', 'activity']]


def identify_activity_intervals(adl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert event-based ADL data to intervals, or return as-is if already interval-based.
    
    Args:
        adl_df: DataFrame from parse_adl_file. Can be either:
                - Event-based: columns ['t_sec', 'activity'] with 'start'/'end' in activity
                - Interval-based: columns ['t_start', 't_end', 'activity', 'duration_sec']
    
    Returns:
        DataFrame with columns ['activity', 't_start', 't_end', 'duration_sec']
    """
    # If already in interval format, return as-is
    if 't_start' in adl_df.columns and 't_end' in adl_df.columns:
        result = adl_df[['activity', 't_start', 't_end', 'duration_sec']].copy()
        result = result[result['duration_sec'] > 0].reset_index(drop=True)
        return result
    
    # Convert event-based format to intervals
    events = []
    active = {}
    for _, row in adl_df.iterrows():
        a = row['activity']
        t = row['t_sec']
        if 'start' in a:
            name = a.replace('start', '').strip()
            active[name] = t
        elif 'end' in a:
            name = a.replace('end', '').strip()
            if name in active:
                duration = t - active[name]
                events.append({
                    'activity': name, 
                    't_start': active[name], 
                    't_end': t, 
                    'duration_sec': duration
                })
                del active[name]
    
    result = pd.DataFrame(events)
    if len(result) > 0:
        result = result[['activity', 't_start', 't_end', 'duration_sec']]
    return result


def extract_propulsion_activities(adl_intervals: pd.DataFrame, min_duration_sec: float = 30.0, keywords: list = None) -> pd.DataFrame:
    if keywords is None:
        keywords = ['level walking','walking','walker','self propulsion','propulsion','assisted propulsion']
    keywords = _normalize_keywords(keywords)
    mask = adl_intervals['activity'].str.lower().apply(lambda x: any(kw in x for kw in keywords))
    out = adl_intervals[mask].copy()
    out = out[out['duration_sec'] >= min_duration_sec].reset_index(drop=True)
    return out


def extract_resting_activities(adl_intervals: pd.DataFrame, min_duration_sec: float = 60.0, keywords: list = None) -> pd.DataFrame:
    if keywords is None:
        keywords = ['sitting','rest','lying']
    keywords = _normalize_keywords(keywords)
    mask = adl_intervals['activity'].str.lower().apply(lambda x: any(kw in x for kw in keywords))
    out = adl_intervals[mask].copy()
    out = out[out['duration_sec'] >= min_duration_sec].reset_index(drop=True)
    return out


def extract_custom_activities(adl_intervals: pd.DataFrame, activities_config: Dict) -> Dict[str, pd.DataFrame]:
    """Extract custom activities with per-activity keyword and duration settings.

    activities_config example:
    {
        'washing_hands': {
            'keywords': ['washing hands', 'hand wash'],
            'min_duration_sec': 15.0
        },
        'stairs': {
            'keywords': ['stairs'],
            'min_duration_sec': 20.0
        }
    }
    """
    results: Dict[str, pd.DataFrame] = {}
    if not activities_config:
        return results

    for name, cfg in activities_config.items():
        if not isinstance(cfg, dict):
            continue
        keywords = _normalize_keywords(cfg.get('keywords', []))
        min_duration = float(cfg.get('min_duration_sec', 0.0))
        if not keywords:
            results[name] = pd.DataFrame(columns=adl_intervals.columns)
            continue

        mask = adl_intervals['activity'].str.lower().apply(lambda x: any(kw in x for kw in keywords))
        out = adl_intervals[mask].copy()
        out = out[out['duration_sec'] >= min_duration].reset_index(drop=True)
        results[name] = out

    return results


def add_baseline_reference(activities: pd.DataFrame, baseline_activities: pd.DataFrame) -> pd.DataFrame:
    res = activities.copy()
    res['baseline_t_start'] = np.nan
    res['baseline_t_end'] = np.nan
    res['baseline_time_before_sec'] = np.nan
    for i, row in res.iterrows():
        t_start = row['t_start']
        preceding = baseline_activities[baseline_activities['t_end'] <= t_start]
        if len(preceding) > 0:
            nb = preceding.iloc[-1]
            res.at[i,'baseline_t_start'] = nb['t_start']
            res.at[i,'baseline_t_end'] = nb['t_end']
            res.at[i,'baseline_time_before_sec'] = t_start - nb['t_end']
    return res
