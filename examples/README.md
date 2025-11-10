# Lacuna Examples

This directory contains example scripts and notebooks demonstrating how to use the Lacuna.

## Batch Processing Example

### Standalone Script (Recommended for Production)

`batch_processing_example.py` - Production-ready script for batch processing multiple lesion masks with parallel processing.

**Advantages over Jupyter notebooks:**
- ✅ No pickling/serialization issues with parallel processing
- ✅ Full multiprocessing support leverages all CPU cores
- ✅ Better performance and reliability
- ✅ Easy to integrate into shell scripts and pipelines
- ✅ Command-line interface with flexible options

**Basic Usage:**

```bash
# Process all lesion files in a directory
python examples/batch_processing_example.py \
    --lesion-dir /path/to/lesions \
    --output-dir /path/to/results

# Use specific file pattern
python examples/batch_processing_example.py \
    --lesion-dir /path/to/lesions \
    --pattern "sub-*_lesion.nii.gz" \
    --output-dir ~/results

# Control parallelization
python examples/batch_processing_example.py \
    --lesion-dir /path/to/lesions \
    --n-jobs -1  # Use all cores (default)
    # --n-jobs 4   # Use 4 cores
    # --n-jobs 1   # Sequential (for debugging)

# Process limited subset (for testing)
python examples/batch_processing_example.py \
    --lesion-dir /path/to/lesions \
    --limit 10  # Only first 10 subjects
```

**Available Options:**

```
--lesion-dir PATH       Directory containing lesion NIfTI files (required)
--pattern PATTERN       Glob pattern for lesion files (default: *.nii.gz)
--output-dir PATH       Output directory (default: ~/batch_results)
--n-jobs N              Number of parallel workers (-1=all cores, 1=sequential)
--limit N               Limit number of subjects (for testing)
--analysis TYPE         Analysis type: 'regional' or 'atlas' (default: regional)
```

**Example Workflow:**

```bash
# 1. Test with small subset
python examples/batch_processing_example.py \
    --lesion-dir /data/lesions \
    --output-dir ~/test_results \
    --limit 5 \
    --n-jobs 1

# 2. Process full dataset with parallel processing
python examples/batch_processing_example.py \
    --lesion-dir /data/lesions \
    --output-dir ~/full_results \
    --n-jobs -1

# 3. Results are automatically exported to CSV/TSV
ls ~/full_results/
# batch_regional_results.csv
# batch_regional_results.tsv
```

**Output:**

The script automatically:
- Loads all lesion files from the specified directory
- Shows timing comparison (sequential vs parallel)
- Calculates speedup metrics
- Exports results to CSV and TSV formats
- Reports processing statistics

**Example Output:**

```
======================================================================
BATCH PROCESSING CONFIGURATION
======================================================================
Lesion directory: /data/lesions
Pattern: *.nii.gz
Found: 100 lesion files
Analysis: regional
Parallel workers: -1 (all cores)
Output directory: ~/results
======================================================================

Loading lesion files...
  ✓ sub-001
  ✓ sub-002
  ...

Successfully loaded: 100 subjects

----------------------------------------------------------------------
BASELINE: Sequential Processing (n_jobs=1)
----------------------------------------------------------------------
[Progress bar...]
✓ Sequential processing complete
  Processed: 100/100 subjects
  Time: 245.32s (2.45s/subject)

----------------------------------------------------------------------
PARALLEL PROCESSING (n_jobs=-1)
----------------------------------------------------------------------
[Progress bar...]
✓ Batch processing complete
  Processed: 100/100 subjects
  Time: 42.18s (0.42s/subject)
  🚀 Speedup: 5.82x faster than sequential
  ⚡ Time saved: 203.14s

----------------------------------------------------------------------
EXPORTING RESULTS
----------------------------------------------------------------------
✓ CSV exported: ~/results/batch_regional_results.csv
  Size: 245.3 KB
✓ TSV exported: ~/results/batch_regional_results.tsv
  Size: 245.3 KB

======================================================================
✅ BATCH PROCESSING COMPLETE
======================================================================
```

### Jupyter Notebook (Interactive Exploration)

`../notebooks/test_batch_processing.ipynb` - Interactive notebook for exploring batch processing features.

**Best for:**
- Learning and exploring the API
- Visualizing results
- Interactive data analysis
- Prototyping workflows

**Limitations:**
- May encounter pickling issues with parallel processing (use n_jobs=1 as workaround)
- Standalone script recommended for actual production use

**To use:**
1. Open the notebook in Jupyter
2. Update the lesion paths in cell 2
3. Run cells sequentially
4. If parallel processing fails, use n_jobs=1 or run the standalone script

## Integration with Your Workflow

### In Python Scripts

```python
from lacuna import LesionData, batch_process
from lacuna.analysis import RegionalDamage
from lacuna.io import batch_export_to_csv

# Load lesions
lesions = [LesionData.from_nifti(path) for path in lesion_paths]

# Run batch analysis
analysis = RegionalDamage()
results = batch_process(lesions, analysis, n_jobs=-1)

# Export
batch_export_to_csv(results, "results.csv")
```

### In Shell Scripts

```bash
#!/bin/bash
# Process multiple cohorts

for cohort in cohort1 cohort2 cohort3; do
    echo "Processing $cohort..."
    python examples/batch_processing_example.py \
        --lesion-dir "/data/$cohort/lesions" \
        --output-dir "/results/$cohort" \
        --n-jobs -1
done
```

### With BIDS Datasets

```bash
# Process BIDS derivatives
python examples/batch_processing_example.py \
    --lesion-dir /data/bids/derivatives/lesion_masks \
    --pattern "sub-*_space-MNI152NLin2009cAsym_lesion.nii.gz" \
    --output-dir /data/bids/derivatives/ldk_results
```

## Troubleshooting

### Pickling Errors in Jupyter

**Error:** `PicklingError: Could not pickle the task to send it to the workers`

**Solution:** This is a Jupyter notebook limitation. Options:
1. Use `n_jobs=1` (sequential mode) in the notebook
2. Use the standalone script instead (no pickling issues)
3. Save LesionData objects to disk and reload in a fresh Python process

### Out of Memory Errors

**Error:** `MemoryError` or killed process during parallel processing

**Solution:**
- Reduce number of parallel workers: `--n-jobs 4` instead of `-1`
- Process in batches: `--limit 50` to process 50 at a time
- Use sequential mode: `--n-jobs 1`

### Performance Lower Than Expected

**Check:**
- Are you using `-1` for n_jobs? (uses all cores)
- Is your analysis I/O bound? (parallel helps less for disk-heavy operations)
- Are you processing very few subjects? (overhead dominates for <10 subjects)

## Next Steps

- Modify the scripts for your specific workflow
- Integrate with your existing pipelines
- Add custom analyses by subclassing `BaseAnalysis`
- Chain multiple analyses in sequence
- Add custom export formats for your tools (R, SPSS, etc.)
