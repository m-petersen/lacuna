# Batch Process Multiple Subjects

This guide shows how to efficiently process multiple subjects in parallel using Lacuna's batch processing utilities.

## Goal

Analyze multiple lesion masks simultaneously, leveraging parallel processing for faster execution.

## Prerequisites

- Lacuna installed ([Installation Guide](installation.md))
- Multiple lesion masks in MNI space
- Sufficient RAM for parallel processing

## Step-by-step instructions

### 1. Load multiple subjects

```python
import nibabel as nib
from lacuna import SubjectData
from pathlib import Path

# Load all masks from a directory
mask_dir = Path("path/to/masks")
subjects = []

for mask_file in mask_dir.glob("sub-*_mask.nii.gz"):
    subject_id = mask_file.stem.replace("_mask", "")
    mask_img = nib.load(mask_file)
    subject = SubjectData(
        mask_img=mask_img,
        space="MNI152NLin6Asym",
        resolution=2.0,
        metadata={"subject_id": subject_id}
    )
    subjects.append(subject)

print(f"Loaded {len(subjects)} subjects")
```

### 2. Configure the analysis

```python
from lacuna.analysis import RegionalDamage

analysis = RegionalDamage(
    parcel_names=["Schaefer2018_100Parcels7Networks"],
    threshold=0.5
)
```

### 3. Run batch processing

```python
from lacuna import batch_process

results = batch_process(
    inputs=subjects,
    analysis=analysis,
    n_jobs=-1,          # Use all available CPUs
    show_progress=True  # Display progress bar
)

print(f"Processed {len(results)} subjects")
```

#### n_jobs options

| Value | Behavior |
|-------|----------|
| `-1` | Use all available CPU cores |
| `1` | Sequential processing (no parallelism) |
| `4` | Use 4 CPU cores |

### 4. Extract results as DataFrame

```python
from lacuna.batch import extract_parcel_table

df = extract_parcel_table(results, "RegionalDamage")
print(df.head())
```

Output format:

| subject_id | 7Networks_LH_Vis_1 | 7Networks_LH_Vis_2 | ... |
|------------|-------------------|-------------------|-----|
| sub-001 | 0.82 | 0.45 | ... |
| sub-002 | 0.10 | 0.00 | ... |

### 5. Save group results

```python
df.to_csv("group_regional_damage.csv", index=False)
```

## Using BIDS datasets

For BIDS-formatted data, use the built-in loader:

```python
from lacuna.io import load_bids_dataset

dataset = load_bids_dataset("path/to/bids_dataset")
subjects = list(dataset.values())

results = batch_process(
    inputs=subjects,
    analysis=analysis,
    n_jobs=-1,
    show_progress=True
)
```

## Memory considerations

!!! warning "Memory usage"
    
    Parallel processing multiplies memory usage. If you run out of memory:
    
    ```python
    # Reduce parallelism
    results = batch_process(
        inputs=subjects,
        analysis=analysis,
        n_jobs=2,  # Limit to 2 parallel jobs
        show_progress=True
    )
    ```

## Error handling

Individual subject failures don't stop the batch:

```python
# Check for failed subjects
for i, result in enumerate(results):
    if result is None:
        print(f"Subject {subjects[i].metadata['subject_id']} failed")
```

## Performance tips

!!! tip "Optimal n_jobs"
    
    For memory-intensive analyses (fLNM), use fewer jobs:
    
    ```python
    import os
    n_jobs = max(1, os.cpu_count() // 2)  # Use half the CPUs
    ```

!!! tip "Chunked processing"
    
    For very large datasets, process in chunks:
    
    ```python
    chunk_size = 50
    all_results = []
    
    for i in range(0, len(subjects), chunk_size):
        chunk = subjects[i:i + chunk_size]
        chunk_results = batch_process(chunk, analysis, n_jobs=-1)
        all_results.extend(chunk_results)
    ```

## Troubleshooting

??? question "Progress bar not showing"
    
    Install tqdm:
    
    ```bash
    pip install tqdm
    ```

??? question "Parallel processing slower than sequential"
    
    For small datasets or quick analyses, parallel overhead can exceed
    benefits. Use `n_jobs=1` for fewer than 10 subjects.
