#!/usr/bin/env python3
"""
Batch Processing Pipeline for Multiple Subjects

Processes all available subjects from the SCAI-NCGG dataset,
extracting HR metrics and generating comparative analysis.
"""

import logging
import yaml
import pandas as pd
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import gzip
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _is_valid_scai_app_file(file_path: Path) -> bool:
    """Return True for real ADL data files and False for sidecar/hidden files."""
    return file_path.is_file() and not file_path.name.startswith('.') and not file_path.name.startswith('._')

# Data paths
# DATA_BASE_PATH = Path(r'C:\Users\Nicla\Documents\ETHZ\Lifelogging\Data\interim\sim')
# DATA_BASE_PATH = Path(r'C:\Users\Nicla\Documents\ETHZ\Lifelogging\Data\interim\scai-ncgg-synced')
DATA_BASE_PATH = Path(r'C:\Users\Nicla\Documents\ETHZ\Lifelogging\Data\interim\scai-ncgg-hc-synced')
SUBJECTS = [
#        'sim_elderly_1', 'sim_elderly_2', 'sim_elderly_3', 'sim_elderly_4', 'sim_elderly_5',
#        'sim_healthy_1', 'sim_healthy_2', 'sim_healthy_3', 'sim_healthy_4', 'sim_healthy_5',
#        'sim_severe_1', 'sim_severe_2', 'sim_severe_3', 'sim_severe_4', 'sim_severe_5',
#     'sub_0103', 'sub_0301', 'sub_0301_2', 'sub_0302', 'sub_0303', 'sub_0304', 'sub_0305',
#     'sub_SS', 'sub_S1', 'sub_OE', 'sub_N05', 'sub_N04', 'sub_MI2', 'sub_M1', 'sub_IM', 
#     'sub_F', 'sub_EI', 'sub_B4', 'sub_B3', 'sub_B2', 'sub_B1', 'sub_A3', 'sub_A2', 'sub_A1',
        'sub_001', 'sub_002', 'sub_003', 'sub_004', 'sub_005', 'sub_011', 'sub_012', 'sub_017',
        'sub_018', 'sub_019', 'sub_020', 'sub_022', 'sub_023', 'sub_024', 'sub_025', 'sub_026', 
        'sub_027', 'sub_028', 'sub_029', 'sub_030'
    ]

# Per-subject time offset overrides (seconds).
# Use these when the ECG clock is known to be a fixed amount ahead of the ADL clock.
# Set to None to fall back to auto-estimation from the candidate list.
# Negative value: ECG timestamps are earlier than ADL timestamps (shift ADL backward).
SUBJECT_TIME_OFFSET_OVERRIDES: dict = {
#     # sim_1 ECG clock is ~7 h behind ADL clock; offset ADL by -7 h.
#     'sim_elderly_1': -8 * 3600,
#     # All other simulated subjects have an ~8 h gap.
#     'sim_elderly_2':  -8 * 3600,
#     'sim_elderly_3':  -8 * 3600,
#     'sim_elderly_4':  -8 * 3600,
#     'sim_elderly_5':  -8 * 3600,
#     'sim_healthy_1':  -8 * 3600,
#     'sim_healthy_2':  -8 * 3600,
#     'sim_healthy_3':  -8 * 3600,
#     'sim_healthy_4':  -8 * 3600,
#     'sim_healthy_5':  -8 * 3600,
#     'sim_severe_1':   -8 * 3600,
#     'sim_severe_2':   -8 * 3600,
#     'sim_severe_3':   -8 * 3600,
#     'sim_severe_4':   -8 * 3600,
#     'sim_severe_5':   -8 * 3600,
}

def check_subject_data(subject_id: str) -> dict:
    """
    Check if subject has required ECG and ADL data.
    
    Returns:
        dict with keys: 'has_ecg', 'has_adl', 'ecg_path', 'adl_path', 'subject_type',
                        'hr_sensor_path'
    """
    subject_path = DATA_BASE_PATH / subject_id
    ecg_dir = subject_path / 'vivalnk_vv330_ecg'
    adl_path = None
    has_adl = False
    ecg_path = None
    has_ecg = False
    adl_time_min = None
    adl_time_max = None
    
    # Sensor HR path (vivalnk_vv330_heart_rate)
    hr_sensor_dir = subject_path / 'vivalnk_vv330_heart_rate'
    hr_sensor_path = str(hr_sensor_dir) if hr_sensor_dir.exists() else None
    
    # Check for ADL data in scai_app (primary location)
    scai_app_path = subject_path / 'scai_app'
    adl_file = None
    if scai_app_path.exists():
        # Keep filename matching lenient but ensure a file path is selected.
        for pattern in ('*ADL*', '*adl*'):
            for file in scai_app_path.glob(pattern):
                if _is_valid_scai_app_file(file):
                    adl_file = file
                    break
            if adl_file is not None:
                break

        # Fallback: use any CSV/CSV.GZ file if no ADL-named file was found.
        if adl_file is None:
            for pattern in ('*.csv.gz', '*.csv'):
                candidate = next((f for f in scai_app_path.glob(pattern) if _is_valid_scai_app_file(f)), None)
                if candidate is not None:
                    adl_file = candidate
                    break

    if adl_file is not None:
        adl_path = str(adl_file)
        has_adl = True

    # If ADL exists, load time range to select the best matching ECG file
    if has_adl and adl_path is not None:
        try:
            adl_file_path = Path(adl_path)
            if adl_file_path.suffix == '.gz':
                with gzip.open(adl_file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    adl_df = pd.read_csv(f)
            else:
                adl_df = pd.read_csv(adl_file_path)
            adl_df.columns = [c.strip().lower() for c in adl_df.columns]

            if 'time' in adl_df.columns:
                adl_times = pd.to_numeric(adl_df['time'], errors='coerce')
            elif 't_start' in adl_df.columns and 't_end' in adl_df.columns:
                adl_times = pd.to_numeric(adl_df[['t_start', 't_end']].stack(), errors='coerce')
            elif 'start_time' in adl_df.columns and 'end_time' in adl_df.columns:
                # New synchronized format (healthy controls)
                adl_times = pd.to_numeric(adl_df[['start_time', 'end_time']].stack(), errors='coerce')
            else:
                adl_times = None

            if adl_times is not None:
                adl_times = adl_times.dropna()
                if len(adl_times) > 0:
                    adl_time_min = float(adl_times.min())
                    adl_time_max = float(adl_times.max())
        except Exception:
            adl_time_min = None
            adl_time_max = None

    # Check for ECG data - could be in root or in date subfolders
    if ecg_dir.exists():
        is_simulated = subject_id.startswith('sim_')

        # Collect candidates from both root and nested folders.
        root_files = sorted(ecg_dir.glob('*.csv.gz'))
        nested_files = sorted(ecg_dir.glob('*/*.csv.gz'))
        all_candidates = root_files + nested_files

        has_any = False
        for item in all_candidates:
            try:
                with gzip.open(item, 'rt', encoding='utf-8', errors='ignore') as f:
                    header = f.readline()
                    first_data_line = f.readline()
                    if header and first_data_line:
                        has_any = True
                        break
            except Exception:
                continue

        if has_any:
            if is_simulated:
                # Simulated subjects are typically single-file recordings.
                direct_ecg = next(ecg_dir.glob('data_*.csv.gz'), None)
                if direct_ecg is None:
                    direct_ecg = next(iter(root_files), None)
                if direct_ecg is not None:
                    ecg_path = str(direct_ecg)
                    has_ecg = True
            else:
                # Real subjects may contain many chunks/files; let loader handle full directory.
                ecg_path = str(ecg_dir)
                has_ecg = True
    
    result = {
        'subject_id': subject_id,
        'has_ecg': has_ecg,
        'has_adl': has_adl,
        'ecg_path': ecg_path,
        'adl_path': adl_path,
        'hr_sensor_path': hr_sensor_path,
        'subject_type': 'simulated' if subject_id.startswith('sim_') else 'real'
    }
    return result


def create_subject_config(subject_id: str, output_dir: Path, data_check: dict) -> Path:
    """
    Create a temporary config file for a specific subject.
    
    Args:
        subject_id: Subject identifier
        output_dir: Directory for outputs
        data_check: Result from check_subject_data() with ADL path
        
    Returns:
        Path to created config file (absolute path)
    """
    config = {
        'project': {
            'name': f'multi-subject-analysis-{subject_id}',
            'output_dir': str(output_dir / subject_id)
        },
        'data': {
            'adl_path': data_check['adl_path'],
            'ecg_path': data_check['ecg_path'],
            'hr_metrics_path': None,
            'hr_sensor_path': data_check.get('hr_sensor_path'),
        },
        'activities': {
            # Use a hard-coded offset when known, otherwise auto-estimate.
            'time_offset_sec': SUBJECT_TIME_OFFSET_OVERRIDES.get(subject_id, None),
            # Restrict auto-offset to timezone-aligned candidates (hours)
            'time_offset_candidates_hours': [-8, -7, 0, 7, 8],
            # d4500 Walking short distances
            'propulsion_keywords': ['level walking', 'walking', 'walker', 'self propulsion', 'propulsion', 'assisted propulsion'],
            # d4150 Maintaining a lying position
            'resting_keywords': ['sitting', 'rest', 'lying'],
            'min_duration_sec': 30.0,
            'baseline_min_duration_sec': 35.0,
            # additional activities - iterate through entire SENSEI protocol?
            'extra': {
                # Basic Mobility:
                # d4154 Maintaining a standing position
                'standing': {
                    'keywords': ['stand'],
                    'min_duration_sec': 10.0
                },
                # d4200 Transferring oneself while sitting
                'transfer': {
                    'keywords': ['transfer to bed', 'transfer from bed', 'sit to stand', 'stand to sit', 'Sit to Lying', 'bed transfer'],
                    'min_duration_sec': 3.0
                },
                # d4201 Transferring oneself while lying
                'bed_transfer': {
                    'keywords': ['Turn Over (right)', 'Turn Over (left)', 'Lying to Sit'],
                    'min_duration_sec': 10.0
                },
                # Self-care: Grooming, washing hands, washing face, dental care, hair care
                # d5100 Washing body parts
                'washing_hands': {
                    'keywords': ['wash hands', 'washing hands', 'hand wash'],
                    'min_duration_sec': 10.0
                },
                # d5100 Washing body parts
                'washing_face': {
                    'keywords': ['wash face', 'washing face', 'face wash'],
                    'min_duration_sec': 10.0
                },
                # d5201 Caring for teeth
                'dental_care': {
                    'keywords': ['put toothpaste', 'brush teeth', 'rinse mouth'],
                    'min_duration_sec': 8.0
                },
                # d5202 Caring for hair
                'hair_care': {
                    'keywords': ['Style Beard/Hair'],
                    'min_duration_sec': 15.0
                }
            }
        },
        'signal': {
            'signal_type': 'ECG',
            'sampling_frequency_hz': 128.0
        },
        'analysis': {
            'compute_baseline_comparison': True,
            'compute_window_overlap': True,
            'analyze_delays': True,
            'max_delay_sec': 300.0,
            'recovery_window_sec': 300.0,
            'baseline_window_sec': 120.0,
            'windowing': {
                'enabled': True,
                'window_duration_sec': 10.0,
                'overlap_percent': 90.0,
            },
        },
        'visualization': {
            'enable_overlays': True,
            'activities': ['propulsion', 'resting', 'washing_hands'],
            'margin_sec': 30.0,
            'max_windows_per_activity': 5,
            'relative_time': True,
            'output_dir': 'overlays'
        }
    }
    # TMUX, Byobu, terminal multiplexer
    
    # Save to batch directory (not subject-specific directory which may not exist yet)
    config_path = output_dir / f'config_{subject_id}.yaml'
    config_path = config_path.resolve()  # Convert to absolute path
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


def process_subject(subject_id: str, batch_output_dir: Path, timeout_sec: int = 4 * 3600) -> dict:
    """
    Process a single subject through the pipeline.
    
    Returns:
        dict with processing status and metrics
    """
    logger.info(f"Processing subject: {subject_id}")
    
    # Check data availability
    data_check = check_subject_data(subject_id)
    if not data_check['has_ecg'] or not data_check['has_adl']:
        logger.warning(f"  ✗ Missing data - ECG: {data_check['has_ecg']}, ADL: {data_check['has_adl']}")
        return {
            'subject_id': subject_id,
            'status': 'SKIPPED',
            'reason': 'Missing ECG or ADL data',
            'subject_type': data_check['subject_type']
        }
    
    # Create subject-specific config
    subject_config = create_subject_config(subject_id, batch_output_dir, data_check)
    
    # Create subject output directory
    subject_output_dir = batch_output_dir / subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    subject_t0 = time.perf_counter()
    
    try:
        # Run the pipeline for this subject
        logger.info(f"  Running pipeline...")
        result = subprocess.run(
            [sys.executable, '-u', 'run_inspection.py', '--config', str(subject_config)],
            cwd=Path.cwd(),
            timeout=int(timeout_sec)
        )
        
        if result.returncode == 0:
            logger.info(f"  ✓ Pipeline completed successfully")
            
            # Try to extract summary metrics
            metrics = extract_subject_metrics(subject_output_dir)
            
            return {
                'subject_id': subject_id,
                'status': 'SUCCESS',
                'subject_type': data_check['subject_type'],
                'runtime_sec': time.perf_counter() - subject_t0,
                **metrics
            }
        else:
            logger.error(f"  ✗ Pipeline failed with return code {result.returncode}")
            logger.error("  See streamed pipeline output above for error details")
            return {
                'subject_id': subject_id,
                'status': 'FAILED',
                'reason': 'Pipeline execution error',
                'subject_type': data_check['subject_type'],
                'runtime_sec': time.perf_counter() - subject_t0,
            }
    
    except subprocess.TimeoutExpired:
        if timeout_sec >= 3600:
            timeout_label = f"{(float(timeout_sec) / 3600.0):.1f} hours"
        else:
            timeout_label = f"{(float(timeout_sec) / 60.0):.1f} minutes"
        logger.error(f"  ✗ Pipeline timeout ({timeout_label})")
        return {
            'subject_id': subject_id,
            'status': 'TIMEOUT',
            'subject_type': data_check['subject_type'],
            'runtime_sec': time.perf_counter() - subject_t0,
        }
    
    except Exception as e:
        logger.error(f"  ✗ Unexpected error: {str(e)}")
        return {
            'subject_id': subject_id,
            'status': 'ERROR',
            'reason': str(e),
            'subject_type': data_check['subject_type'],
            'runtime_sec': time.perf_counter() - subject_t0,
        }
    
    finally:
        # Clean up temporary config
        if subject_config.exists():
            deleted = False
            for attempt in range(3):
                try:
                    subject_config.unlink()
                    deleted = True
                    break
                except PermissionError:
                    # On timeout, Windows can briefly keep file handles open.
                    time.sleep(1.0)
                except Exception as e:
                    logger.warning(f"  Could not remove temporary config {subject_config}: {e}")
                    break
            if not deleted and subject_config.exists():
                logger.warning(f"  Temporary config kept (file locked): {subject_config}")


def extract_subject_metrics(subject_output_dir: Path) -> dict:
    """
    Extract summary metrics from subject's pipeline output.
    
    Returns:
        dict with summary metrics
    """
    metrics = {
        'propulsion_count': 0,
        'resting_count': 0,
        'propulsion_mean_hr': None,
        'resting_mean_hr': None,
        'stress_index_delta': None
    }
    
    try:
        # Read propulsion activities (detected activities, not necessarily with HR metrics)
        prop_act_file = subject_output_dir / 'propulsion_activities.csv'
        if prop_act_file.exists():
            prop_act_df = pd.read_csv(prop_act_file)
            metrics['propulsion_count'] = len(prop_act_df)
        
        # Read propulsion HR metrics (subset with successful HR extraction)
        prop_file = subject_output_dir / 'propulsion_hr_metrics.csv'
        if prop_file.exists():
            prop_df = pd.read_csv(prop_file)
            if len(prop_df) > 0:
                metrics['propulsion_mean_hr'] = prop_df['mean_hr'].mean()
        
        # Read resting activities
        rest_act_file = subject_output_dir / 'resting_activities.csv'
        if rest_act_file.exists():
            rest_act_df = pd.read_csv(rest_act_file)
            metrics['resting_count'] = len(rest_act_df)
        
        # Read resting HR metrics
        rest_file = subject_output_dir / 'resting_hr_metrics.csv'
        if rest_file.exists():
            rest_df = pd.read_csv(rest_file)
            if len(rest_df) > 0:
                metrics['resting_mean_hr'] = rest_df['mean_hr'].mean()
        
        # Calculate delta
        if metrics['propulsion_mean_hr'] is not None and metrics['resting_mean_hr'] is not None:
            metrics['stress_index_delta'] = (
                metrics['propulsion_mean_hr'] - metrics['resting_mean_hr']
            )
    
    except Exception as e:
        logger.warning(f"Could not extract metrics: {e}")
    
    return metrics


def main(subject_ids: list = None, max_workers: int = 1, timeout_hours: float = 4.0):
    """
    Process multiple subjects and generate summary report.
    
    Args:
        subject_ids: List of subject IDs to process. If None, processes all.
        max_workers: Number of parallel workers (currently sequential only)
    """
    logger.info("=" * 80)
    logger.info("BATCH PROCESSING PIPELINE - SCAI-NCGG Dataset")
    logger.info("=" * 80)
    
    # Determine which subjects to process
    if subject_ids is None:
        subject_ids = SUBJECTS
    
    logger.info(f"Subjects to process: {len(subject_ids)}")
    for sid in subject_ids:
        logger.info(f"  - {sid}")
    
    # Create batch output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_output_dir = Path('./output_batch') / f'batch_{timestamp}'
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nOutput directory: {batch_output_dir}")
    
    timeout_sec = max(1, int(float(timeout_hours) * 3600.0))

    # Process each subject
    results = []
    for idx, subject_id in enumerate(subject_ids, 1):
        logger.info(f"\n[{idx}/{len(subject_ids)}] {subject_id}")
        result = process_subject(subject_id, batch_output_dir, timeout_sec=timeout_sec)
        results.append(result)

        runtime_sec = result.get('runtime_sec')
        runtime_str = f"{float(runtime_sec):.1f}s" if isinstance(runtime_sec, (int, float)) else "n/a"

        wps_str = "n/a"
        windows_str = "n/a"
        try:
            subject_dir = batch_output_dir / subject_id
            prop_file = subject_dir / 'propulsion_hr_metrics.csv'
            rest_file = subject_dir / 'resting_hr_metrics.csv'
            n_windows = 0
            if prop_file.exists():
                n_windows += len(pd.read_csv(prop_file))
            if rest_file.exists():
                n_windows += len(pd.read_csv(rest_file))

            windows_str = str(int(n_windows))
            if (
                result.get('status') == 'SUCCESS'
                and isinstance(runtime_sec, (int, float))
                and float(runtime_sec) > 0
                and n_windows > 0
            ):
                wps = float(n_windows) / float(runtime_sec)
                if wps < 0.1:
                    wps_str = "<0.1 w/s"
                else:
                    wps_str = f"{wps:.2f} w/s"
            elif result.get('status') != 'SUCCESS':
                wps_str = "n/a (incomplete run)"
        except Exception:
            wps_str = "n/a"
            windows_str = "n/a"

        logger.info(
            "  Subject summary: status=%s | runtime=%s | windows=%s | throughput=%s",
            result.get('status', 'UNKNOWN'),
            runtime_str,
            windows_str,
            wps_str,
        )
    
    # Generate summary report
    logger.info("\n" + "=" * 80)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 80)
    
    summary_df = pd.DataFrame(results)
    logger.info(f"\nTotal subjects: {len(summary_df)}")
    logger.info(f"Successful: {len(summary_df[summary_df['status'] == 'SUCCESS'])}")
    logger.info(f"Failed: {len(summary_df[summary_df['status'] != 'SUCCESS'])}")
    
    # Status breakdown
    logger.info("\nStatus breakdown:")
    status_counts = summary_df['status'].value_counts()
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")
    
    # Save summary
    summary_path = batch_output_dir / 'batch_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\n✓ Summary saved: {summary_path}")
    
    # Generate comparative analysis for successful subjects
    successful_df = summary_df[summary_df['status'] == 'SUCCESS']
    if len(successful_df) > 0:
        logger.info("\n" + "-" * 80)
        logger.info("COMPARATIVE METRICS")
        logger.info("-" * 80)
        
        # HR comparison by subject type
        logger.info("\nMean HR by Subject Type:")
        for subject_type in ['simulated', 'real']:
            subset = successful_df[successful_df['subject_type'] == subject_type]
            if len(subset) > 0:
                logger.info(f"\n  {subject_type.upper()}:")
                if 'propulsion_mean_hr' in subset.columns:
                    prop_hrs = subset['propulsion_mean_hr'].dropna()
                    if len(prop_hrs) > 0:
                        logger.info(f"    Propulsion HR: {prop_hrs.mean():.1f} ± {prop_hrs.std():.1f} bpm")
                
                if 'resting_mean_hr' in subset.columns:
                    rest_hrs = subset['resting_mean_hr'].dropna()
                    if len(rest_hrs) > 0:
                        logger.info(f"    Resting HR: {rest_hrs.mean():.1f} ± {rest_hrs.std():.1f} bpm")
                
                if 'stress_index_delta' in subset.columns:
                    deltas = subset['stress_index_delta'].dropna()
                    if len(deltas) > 0:
                        logger.info(f"    HR Delta: {deltas.mean():.1f} ± {deltas.std():.1f} bpm")
    
    logger.info("\n" + "=" * 80)
    logger.info("Batch processing completed!")
    logger.info("=" * 80)
    
    return batch_output_dir, summary_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process multiple subjects')
    parser.add_argument('--subjects', nargs='+', default=None,
                       help='Specific subject IDs to process')
    parser.add_argument('--subject-type', choices=['simulated', 'real'], default=None,
                       help='Process only simulated or real subjects')
    parser.add_argument('--max-workers', type=int, default=1,
                       help='Number of parallel workers')
    parser.add_argument('--timeout-hours', type=float, default=4.0,
                       help='Per-subject timeout in hours (default: 4.0)')
    
    args = parser.parse_args()
    
    # Filter subjects if needed
    subject_ids = args.subjects
    if args.subject_type == 'simulated':
        subject_ids = [s for s in SUBJECTS if s.startswith('sim_')]
    elif args.subject_type == 'real':
        subject_ids = [s for s in SUBJECTS if not s.startswith('sim_')]
    
    # Run batch processing
    batch_dir, summary = main(subject_ids, args.max_workers, args.timeout_hours)
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(summary.to_string())
    print(f"\nOutput directory: {batch_dir}")
