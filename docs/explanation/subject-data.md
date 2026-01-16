# SubjectData: Immutable Data Containers

Understanding the design and purpose of `SubjectData` in Lacuna.

## Overview

`SubjectData` is Lacuna's core container for holding all data associated with
a single subject throughout the analysis pipeline. It follows an **immutable**
design pattern where operations return new instances rather than modifying
existing ones.

## The Problem SubjectData Solves

Neuroimaging analyses often involve multiple transformation steps:

1. Load raw lesion data
2. Resample to a standard space
3. Compute network connectivity
4. Extract regional statistics
5. Save results

Without a structured container, you'd have to track many loose variables:

```python
# Without SubjectData (error-prone)
lesion_mask = load_nifti(path)
resampled = resample(lesion_mask, target_space)
connectivity = compute_flnm(resampled, connectome)
regional = extract_regions(resampled, atlas)
# Which variables go together? Easy to mix up!
```

`SubjectData` bundles everything together:

```python
# With SubjectData (clear provenance)
subject = SubjectData.from_nifti(path)
subject = subject.with_mask_resampled(target_space)
subject = analyze(subject, analysis="flnm")
# All data stays together with clear history
```

## Immutability: Why It Matters

### What Is Immutability?

An immutable object cannot be changed after creation. Instead of modifying
an existing object, operations create a new object with the changes applied.

```python
# SubjectData is immutable
original = SubjectData.from_nifti(path)
resampled = original.with_mask_resampled(target)

# original is unchanged!
assert original.mask.affine != resampled.mask.affine
```

### Benefits of Immutability

1. **Reproducibility**: The same input always produces the same output
2. **Debugging**: You can inspect any intermediate state
3. **Parallelization**: Safe to share across threads/processes
4. **History tracking**: Each step creates a new, traceable object

### The `with_*` Pattern

Immutable operations follow the `with_*` naming convention:

```python
subject = subject.with_mask_resampled(target_space)
subject = subject.with_analysis_result("flnm", result)
subject = subject.with_metadata({"acquisition_date": "2024-01-15"})
```

This pattern makes it clear that a new object is returned.

## Structure of SubjectData

A `SubjectData` instance contains:

```
SubjectData
├── subject_id: str          # Unique identifier
├── session_id: str | None   # Optional session
├── mask: MaskData           # Lesion mask
├── metadata: dict           # Subject metadata
└── results: dict            # Analysis results
```

### Accessing Data

```python
# Identity
print(subject.subject_id)  # "sub-001"
print(subject.session_id)  # "ses-01" or None

# Mask data
print(subject.mask.shape)  # (91, 109, 91)
print(subject.mask.affine) # 4x4 transformation matrix

# Results (after analysis)
print(subject.results["flnm"])  # fLNM connectivity map
```

### Results Storage

Analysis results are stored in a dictionary keyed by analysis name:

```python
subject = analyze(subject, analysis="flnm")
subject = analyze(subject, analysis="regional")

# Access results
flnm_map = subject.results["flnm"]
regional_stats = subject.results["regional"]
```

## Creating SubjectData

### From NIfTI Files

```python
from lacuna.core import SubjectData

subject = SubjectData.from_nifti(
    "/path/to/lesion_mask.nii.gz",
    subject_id="sub-001",
    session_id="ses-01"  # optional
)
```

### From NumPy Arrays

```python
import numpy as np

mask_data = np.array([...])  # 3D binary array
affine = np.eye(4)  # 4x4 transformation matrix

subject = SubjectData.from_array(
    mask_data,
    affine,
    subject_id="sub-001"
)
```

## Common Patterns

### Pipeline Pattern

```python
# Build a processing pipeline
subject = (
    SubjectData.from_nifti(path, subject_id="sub-001")
    .with_mask_resampled(target_space)
    .with_mask_validated()
)

# Run analyses
subject = analyze(subject, analysis="flnm")
subject = analyze(subject, analysis="regional")
```

### Batch Processing

```python
from concurrent.futures import ProcessPoolExecutor

def process_subject(path):
    subject = SubjectData.from_nifti(path, subject_id=path.stem)
    return analyze(subject, analysis="flnm")

# Safe parallel processing due to immutability
with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_subject, paths))
```

## Design Decisions

### Why Not Mutable?

Mutable state causes subtle bugs in neuroimaging pipelines:

```python
# Dangerous mutable pattern (NOT how Lacuna works)
subject.resample(target)  # Modifies in-place
subject.analyze()         # Uses resampled data
# Later...
subject.analyze()         # Wait, which version of the mask?
```

### Why Bundle Everything?

Keeping data together ensures:

- **No orphaned data**: Results stay with their source mask
- **Clear provenance**: You know what input produced what output
- **Easy serialization**: Save/load complete analysis state

## See Also

- [MaskData Validation](mask-data.md) — Understanding mask requirements
- [The analyze() Function](analyze.md) — How analyses are orchestrated
- API Reference: `lacuna.core.SubjectData`
