# Multi-Subject Data Processing Pipeline

## Overview

The pipeline now supports batch processing of multiple participants from the SCAI-NCGG dataset. Process all 13 subjects automatically and generate comparative analysis across participants.

## Available Subjects

**13 Total Subjects:**
- **Simulated (6)**: sim_elderly_1, sim_elderly_2, sim_healthy_1, sim_healthy_2, sim_severe_1, sim_severe_2
- **Real (7)**: sub_0103, sub_0301, sub_0301_2, sub_0302, sub_0303, sub_0304, sub_0305

**Data Location**: Set `DATA_BASE_PATH` to your dataset root.

## Quick Start

### 1. Process All Subjects (Batch Processing)

```bash
# Process all subjects
python batch_process_subjects.py

# Process only simulated subjects
python batch_process_subjects.py --subject-type simulated

# Process only real subjects
python batch_process_subjects.py --subject-type real

# Process specific subjects
python batch_process_subjects.py --subjects sim_elderly_1 sub_0103 sub_0301
```

**Outputs:**
- `output_batch/batch_YYYYMMDD_HHMMSS/` - Results for each subject
- `output_batch/batch_YYYYMMDD_HHMMSS/batch_summary.csv` - Summary statistics
- Each subject gets individual CSV outputs (propulsion_hr_metrics.csv, resting_hr_metrics.csv, etc.)

### 2. Inspect Results in Jupyter

```bash
# Launch multi-subject analysis notebook
jupyter notebook multi_subject_analysis.ipynb
```

The notebook loads all batch results and generates:
- ✓ Data coverage report (which subjects have data)
- ✓ Activity summary (propulsion/resting count per subject)
- ✓ HR metrics comparison across subjects
- ✓ Comparative analysis dashboard (9-panel visualization)
- ✓ Statistical comparison tables
- ✓ Export reports (CSV, PNG, TXT)

## Processing Scripts

### `batch_process_subjects.py`

**Purpose**: Automate processing of multiple subjects

**Key Features:**
- Checks data availability for each subject
- Creates subject-specific config files automatically
- Runs individual pipelines sequentially (or parallel with --max-workers)
- Aggregates results and generates summary
- Handles timeouts and errors gracefully

**Command-line Options:**
```
--subjects SUBJECT_IDS      Process specific subjects (space-separated)
--subject-type {simulated, real}  Filter by subject type
--max-workers N             Number of parallel workers (default: 1)
```

**Example:**
```bash
# Process all, show summary
python batch_process_subjects.py

# Process real subjects only
python batch_process_subjects.py --subject-type real

# Process 3 specific subjects
python batch_process_subjects.py --subjects sub_0103 sub_0301 sub_0301_2
```

## Notebook: `multi_subject_analysis.ipynb`

### Section Breakdown

1. **Import Libraries** - Core dependencies (pandas, matplotlib, seaborn)

2. **Define Paths** - Set up data locations and subject list

3. **Load Batch Results** - Automatically finds and loads latest batch results

4. **Load Individual Metrics** - Loads HR metrics CSV files for all subjects

5. **Data Quality Report** - Coverage analysis and summary statistics

6. **Visualization Dashboard** - 9-panel comparative analysis including:
   - Activity counts per subject
   - Mean HR comparison (propulsion vs resting)
   - HR distribution histograms
   - RMSSD (HRV) comparison
   - Stress Index comparison
   - HR delta (activity response)
   - Processing status breakdown
   - Subject type comparison
   - Summary statistics table

7. **Export Report** - Generates text and CSV reports for further analysis

## Output Structure

```
output_batch/
├── batch_YYYYMMDD_HHMMSS/          # Latest batch run
│   ├── batch_summary.csv             # Summary of all subjects
│   ├── MULTI_SUBJECT_REPORT.txt      # Comprehensive report
│   ├── multi_subject_summary.csv     # Per-subject metrics table
│   ├── multi_subject_dashboard.png   # Visualization dashboard
│   ├── sim_elderly_1/                # Individual subject results
│   │   ├── propulsion_hr_metrics.csv
│   │   ├── resting_hr_metrics.csv
│   │   ├── baseline_activity_comparisons.csv
│   │   └── ... (other outputs)
│   ├── sim_elderly_2/
│   ├── sub_0103/
│   └── ... (other subjects)
```

## Typical Workflow

### Step 1: Initial Setup
```bash
# Check available subjects
python batch_process_subjects.py --subjects sim_elderly_1
# This tests one subject to verify pipeline works
```

### Step 2: Full Batch Processing
```bash
# Run batch processing for all subjects
# This will take ~5-10 minutes depending on data size
python batch_process_subjects.py

# Output will include:
# - Processing log with status for each subject
# - Batch summary with statistics
# - Per-subject results in separate directories
```

### Step 3: Analysis and Visualization
```bash
# Open Jupyter and run multi_subject_analysis.ipynb
jupyter notebook multi_subject_analysis.ipynb

# Or run it headlessly:
# jupyter nbconvert --to notebook --execute multi_subject_analysis.ipynb
```

### Step 4: Review Results
- Check `MULTI_SUBJECT_REPORT.txt` for key findings
- Review `multi_subject_dashboard.png` for visual insights
- Examine `multi_subject_summary.csv` for per-subject metrics

## Configuration

### Auto-generated Subject Configs
- Each subject gets a temporary config file: `config_SUBJECT_ID.yaml`
- Paths are auto-detected from the data directory structure
- Activity types (propulsion/resting) are consistent across subjects
- Time offsets are auto-estimated for each subject

### Customizing Subject Configs
Edit `batch_process_subjects.py` function `create_subject_config()` to modify:
- Activity type definitions (currently: RUL/LUL for propulsion, RLL/LLL/SUP for resting)
- Output directory structure
- Preprocessing parameters

## Key Metrics Explained

### Mean HR (bpm)
- Heart rate during activity
- Propulsion typically higher than resting
- Range: ~60-180 bpm depending on activity intensity

### RMSSD (ms)
- Root Mean Square of Successive Differences in RR intervals
- Measure of heart rate variability (parasympathetic tone)
- Higher = more variable, lower = more steady

### Stress Index
- Baevsky Stress Index
- Low values = more organized rhythm (better parasympathetic tone)
- Can be paradoxically low during exertion (coordinated effort)

### HR Delta
- Difference in mean HR between activity and baseline
- Positive = HR increase during activity (expected)
- Useful for comparing individual responses to activity

## Troubleshooting

### Issue: "No batch results found"
**Solution**: Run `batch_process_subjects.py` first to generate results

### Issue: "Missing data for subject X"
**Solution**: Verify the subject directory exists under `DATA_BASE_PATH` for the given subject ID

### Issue: Memory errors with large datasets
**Solution**: Process subjects in smaller groups:
```bash
python batch_process_subjects.py --subjects sim_elderly_1 sim_elderly_2
python batch_process_subjects.py --subjects sim_healthy_1 sim_healthy_2
# Then manually combine results
```

### Issue: Notebook cells fail
**Solution**: Run cells sequentially; clear kernel and restart if issues persist

## Performance Notes

- **Processing time**: ~5-10 minutes for all 13 subjects (sequential)
- **Disk space**: ~100 MB per subject for output files
- **Memory**: ~500 MB typical usage
- **Bottleneck**: ECG peak detection (most time-consuming)

### Optimization Options
1. Use `--max-workers 4` for parallel processing (experimental)
2. Process subject subgroups separately
3. Skip preprocessing for faster initial analysis (set `preprocess=False` in hr_metrics.py)

## Advanced Usage

### Extract Specific Metrics
```python
import pandas as pd

# Load results from latest batch
batch_df = pd.read_csv('./output_batch/batch_*/batch_summary.csv')
prop_df = pd.read_csv('./output_batch/batch_*/*/propulsion_hr_metrics.csv')

# Filter by subject type
simulated = batch_df[batch_df['subject_type'] == 'simulated']
real = batch_df[batch_df['subject_type'] == 'real']

# Statistical comparison
print(prop_df.groupby('subject_id')['mean_hr'].describe())
```

### Generate Custom Reports
Edit `multi_subject_analysis.ipynb` sections 5-7 to:
- Add new visualizations
- Filter subjects by criteria
- Export custom metrics
- Generate statistical tests

## Next Steps

1. **Run initial batch**: `python batch_process_subjects.py`
2. **Review results**: `jupyter notebook multi_subject_analysis.ipynb`
3. **Inspect individual subjects**: Check `output_batch/batch_*/SUBJECT_ID/` directories
4. **Analyze patterns**: Look for differences between subject types and activity types
5. **Generate reports**: Use the export functions to create publishable outputs

## Support

For issues or questions about:
- **Data processing**: Check `run_inspection.py` and individual subject logs
- **Visualization**: See `multi_subject_analysis.ipynb` cells 6-7
- **Preprocessing**: See `PREPROCESSING_SUMMARY.md`
- **Individual subject analysis**: See original `activity_hr_visualization.ipynb`

---

**Status**: ✅ Production-ready for multi-subject studies  
**Last Updated**: 2024  
**Python Version**: 3.10+
