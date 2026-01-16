# The analyze() Function

Understanding Lacuna's central orchestration function and how analyses are executed.

## Overview

The `analyze()` function is Lacuna's primary interface for running analyses.
It orchestrates the entire workflow: loading data, validating inputs,
executing analyses, and returning results. This document explains its design
and how to use it effectively.

## Basic Usage

```python
from lacuna import analyze
from lacuna.core import SubjectData

# Load subject data
subject = SubjectData.from_nifti("lesion.nii.gz", subject_id="sub-001")

# Run analysis
subject = analyze(subject, analysis="flnm")

# Access results
flnm_map = subject.results["flnm"]
```

## Function Signature

```python
def analyze(
    subject: SubjectData,
    analysis: str | list[str],
    *,
    connectome: str | None = None,
    atlas: str | None = None,
    **kwargs
) -> SubjectData:
    """Run one or more analyses on subject data.
    
    Parameters
    ----------
    subject : SubjectData
        Subject data container with lesion mask.
    analysis : str or list[str]
        Analysis type(s) to run: "flnm", "slnm", "regional", or "all".
    connectome : str, optional
        Connectome name for network analyses.
    atlas : str, optional
        Atlas name for regional analysis.
    **kwargs
        Additional analysis-specific parameters.
        
    Returns
    -------
    SubjectData
        Updated subject with results added.
    """
```

## Supported Analyses

### Functional LNM (`flnm`)

Maps functional connectivity between the lesion and the rest of the brain:

```python
subject = analyze(
    subject,
    analysis="flnm",
    connectome="HCP_S1200"  # Optional, uses default if omitted
)
```

**Output**: 3D NIfTI-like map where each voxel contains the correlation
between that voxel's connectivity and the lesion pattern.

### Structural LNM (`slnm`)

Maps structural disconnection using tractography:

```python
subject = analyze(
    subject,
    analysis="slnm",
    connectome="HCP_Tractogram"
)
```

**Output**: 3D map where each voxel contains the proportion of streamlines
passing through both that voxel and the lesion.

### Regional Damage (`regional`)

Quantifies lesion overlap with atlas regions:

```python
subject = analyze(
    subject,
    analysis="regional",
    atlas="AAL"
)
```

**Output**: Dictionary mapping region names to damage metrics (volume,
percentage affected).

### Multiple Analyses

Run several analyses in one call:

```python
subject = analyze(
    subject,
    analysis=["flnm", "slnm", "regional"],
    connectome="HCP_S1200",
    atlas="AAL"
)
```

Or run all available analyses:

```python
subject = analyze(subject, analysis="all")
```

## How analyze() Works

### 1. Input Validation

The function first validates inputs:

```python
# Pseudocode
def analyze(subject, analysis, ...):
    # Validate subject
    if not isinstance(subject, SubjectData):
        raise TypeError("Expected SubjectData")
    
    # Validate mask
    if not subject.mask.is_binary():
        raise ValidationError("Mask must be binary")
    
    # Validate analysis type
    if analysis not in analysis_registry:
        raise ValueError(f"Unknown analysis: {analysis}")
```

### 2. Resource Loading

Required resources are loaded from registries:

```python
# Load connectome if needed
if analysis in ["flnm", "slnm"]:
    connectome_data = connectome_registry.get(connectome).load()

# Load atlas if needed
if analysis == "regional":
    atlas_data = atlas_registry.get(atlas).load()
```

### 3. Space Alignment

The mask is resampled to match the target space:

```python
# Resample mask to connectome space
mask_resampled = resample_to_target(
    subject.mask,
    target_space=connectome_data.affine,
    target_shape=connectome_data.shape
)
```

### 4. Analysis Execution

The specific analysis algorithm runs:

```python
# Get analysis implementation
analysis_impl = analysis_registry.get(analysis)

# Run analysis
result = analysis_impl.run(
    mask=mask_resampled,
    connectome=connectome_data
)
```

### 5. Result Storage

Results are added to the subject (immutably):

```python
# Return new SubjectData with results
return subject.with_analysis_result(analysis, result)
```

## Immutable Returns

`analyze()` follows Lacuna's immutability principle:

```python
# Original is unchanged
original = SubjectData.from_nifti("lesion.nii.gz", subject_id="sub-001")
analyzed = analyze(original, analysis="flnm")

# Different objects
assert original is not analyzed
assert "flnm" not in original.results
assert "flnm" in analyzed.results
```

This enables:

- **Pipeline debugging**: Inspect any intermediate state
- **Safe parallelization**: Share subjects across processes
- **Reproducibility**: Same input always produces same output

## Error Handling

`analyze()` raises specific exceptions:

```python
from lacuna.core.exceptions import (
    ValidationError,
    ResourceNotFoundError,
    AnalysisError
)

try:
    result = analyze(subject, analysis="flnm")
except ValidationError as e:
    # Invalid input (non-binary mask, wrong space, etc.)
    print(f"Validation failed: {e}")
except ResourceNotFoundError as e:
    # Missing connectome, atlas, or template
    print(f"Resource not found: {e}")
except AnalysisError as e:
    # Analysis computation failed
    print(f"Analysis failed: {e}")
```

## Advanced Usage

### Custom Parameters

Pass analysis-specific parameters via kwargs:

```python
# Custom threshold for fLNM
subject = analyze(
    subject,
    analysis="flnm",
    threshold=0.1,  # Correlation threshold
    method="pearson"  # Correlation method
)

# Custom regional metrics
subject = analyze(
    subject,
    analysis="regional",
    atlas="Schaefer200",
    metrics=["volume", "percentage", "centroid"]
)
```

### Progress Tracking

For batch processing, use callbacks:

```python
def progress_callback(step, total, message):
    print(f"[{step}/{total}] {message}")

subject = analyze(
    subject,
    analysis="flnm",
    progress_callback=progress_callback
)
```

### Caching

Results are cached to avoid recomputation:

```python
# First call computes
subject = analyze(subject, analysis="flnm")

# Second call returns cached result
subject = analyze(subject, analysis="flnm")  # Instant
```

## Design Rationale

### Why a Single Entry Point?

A unified `analyze()` function provides:

1. **Consistency**: Same interface for all analyses
2. **Discoverability**: One function to learn
3. **Orchestration**: Handles all setup/teardown
4. **Future-proofing**: New analyses "just work"

### Why Keyword Arguments?

Required positional args plus optional kwargs:

```python
# Clear what's required vs optional
analyze(subject, analysis="flnm")  # Minimal
analyze(subject, analysis="flnm", connectome="HCP_S1200")  # Explicit
```

### Why Return SubjectData?

Returning updated `SubjectData` enables:

```python
# Chainable operations
subject = (
    analyze(subject, analysis="flnm")
    .with_metadata({"processed": True})
)

# Natural pipeline flow
subject = analyze(subject, "flnm")
subject = analyze(subject, "regional")
export_results(subject)
```

## See Also

- [SubjectData Design](subject-data.md) — Understanding the data container
- [Registry Pattern](registries.md) — How analyses are registered
- How-to: [Run Functional LNM](../how-to/run-flnm.md) — Practical usage guide
- How-to: [Batch Processing](../how-to/batch-processing.md) — Processing multiple subjects
