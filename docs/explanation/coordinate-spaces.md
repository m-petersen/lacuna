# Coordinate Spaces in Neuroimaging

Understanding MNI spaces and why spatial normalization matters for Lacuna.

## Overview

Lesion network mapping requires that lesion masks and normative connectomes
share the same **coordinate space**. This document explains what coordinate
spaces are, why they matter, and how Lacuna handles them.

## What Is a Coordinate Space?

A coordinate space defines how 3D voxel indices (i, j, k) map to physical
locations in the brain (x, y, z in millimeters).

```
Voxel (45, 54, 45) → Physical coordinates (0, 0, 0) mm
```

This mapping is defined by the **affine transformation matrix**:

```python
import nibabel as nib

img = nib.load("brain.nii.gz")
print(img.affine)
# [[ 2.  0.  0. -90.]
#  [ 0.  2.  0. -126.]
#  [ 0.  0.  2. -72.]
#  [ 0.  0.  0.  1.]]
```

## Why Spaces Matter for Lacuna

Lacuna compares your lesion mask against normative connectomes derived from
healthy subjects. For this comparison to be meaningful:

1. **Same template**: Both must be aligned to the same reference brain
2. **Same resolution**: Voxel grids must match (or be resampled)
3. **Same orientation**: Left-right, anterior-posterior must agree

If spaces don't match, a lesion at (45, 54, 45) in your mask would correspond
to a different anatomical location in the connectome.

## The MNI Standard

The Montreal Neurological Institute (MNI) template is the most common
standard space in neuroimaging. However, there are **multiple MNI variants**:

### MNI152NLin6Asym

- **Used by**: FSL, HCP pipelines
- **Resolution**: Typically 2mm isotropic
- **Orientation**: RAS (Right-Anterior-Superior)
- **Lacuna default** for functional connectomes

### MNI152NLin2009cAsym

- **Used by**: fMRIPrep, TemplateFlow
- **Resolution**: Various (0.5mm to 2mm)
- **Orientation**: RAS
- **Lacuna default** for structural connectomes

### Why Multiple MNI Spaces?

Different MNI variants use different:

- Registration algorithms
- Number of subjects averaged
- Tissue segmentation methods
- Skull stripping approaches

The differences are subtle (1-2mm) but can affect voxel-level analyses.

## Checking Your Space

### From File Headers

NIfTI files often include space information:

```python
import nibabel as nib

img = nib.load("lesion.nii.gz")

# Check sform/qform codes
print(f"sform code: {img.header['sform_code']}")
print(f"qform code: {img.header['qform_code']}")

# Check dimensions
print(f"Shape: {img.shape}")
print(f"Voxel size: {img.header.get_zooms()}")
```

### Common Indicators

| Property | MNI152NLin6Asym | MNI152NLin2009cAsym |
|----------|-----------------|---------------------|
| Dimensions | 91×109×91 | 193×229×193 |
| Voxel size | 2mm | 1mm |
| Origin | (-90, -126, -72) | (-96, -132, -78) |

## Space Requirements in Lacuna

### Functional LNM

The functional connectome (HCP-derived) uses **MNI152NLin6Asym** at 2mm:

```python
# Your mask must match or be resampled to:
# - MNI152NLin6Asym space
# - 2mm isotropic resolution
# - 91×109×91 voxels
```

### Structural LNM

The structural connectome uses **MNI152NLin2009cAsym**:

```python
# Your mask must match or be resampled to:
# - MNI152NLin2009cAsym space
# - Resolution depends on connectome (typically 2mm)
```

## Resampling in Lacuna

Lacuna can automatically resample masks to match the target connectome:

```python
from lacuna.core import SubjectData
from lacuna.spatial import resample_to_connectome

subject = SubjectData.from_nifti("lesion.nii.gz", subject_id="sub-001")

# Resample to functional connectome space
subject = resample_to_connectome(subject, connectome="functional")
```

### What Resampling Does

1. **Spatial transformation**: Applies affine to map between spaces
2. **Interpolation**: Fills in voxel values at new grid points
3. **Rounding**: For binary masks, rounds to 0 or 1

### Interpolation Methods

| Method | Use Case |
|--------|----------|
| Nearest neighbor | Binary masks (preserves 0/1 values) |
| Linear | Continuous data (T1, fMRI) |
| Cubic | High-quality continuous resampling |

Lacuna uses **nearest neighbor** for lesion masks to preserve binary values.

## TemplateFlow Integration

Lacuna uses [TemplateFlow](https://templateflow.org) to manage template
spaces:

```python
# TemplateFlow provides standardized templates
from templateflow import api as tflow

# Get MNI152NLin6Asym T1w template at 2mm
t1_template = tflow.get(
    "MNI152NLin6Asym",
    resolution=2,
    desc="brain",
    suffix="T1w"
)
```

### Automatic Template Selection

When you specify a connectome, Lacuna automatically uses the matching
template:

```python
# Uses MNI152NLin6Asym template internally
result = analyze(subject, analysis="flnm", connectome="HCP_S1200")
```

## Common Issues

### Mismatched Spaces

**Symptom**: Results don't make anatomical sense

**Solution**: Verify your mask is in the expected space:

```python
# Check before analysis
from lacuna.spatial import check_space_compatibility

compatible = check_space_compatibility(subject.mask, connectome)
if not compatible:
    subject = resample_to_connectome(subject, connectome)
```

### Wrong Resolution

**Symptom**: Shape mismatch errors

**Solution**: Resample to target resolution:

```python
# Lacuna handles this automatically during analysis
# But you can do it explicitly:
subject = subject.with_mask_resampled(target_shape=(91, 109, 91))
```

### Left-Right Confusion

**Symptom**: Results appear flipped

**Solution**: Check orientation in header:

```python
import nibabel as nib

img = nib.load("lesion.nii.gz")
print(nib.aff2axcodes(img.affine))  # Should be ('R', 'A', 'S')
```

## Best Practices

### 1. Know Your Source Space

Document what space your preprocessing used:

```python
metadata = {
    "source_space": "MNI152NLin6Asym",
    "source_resolution": "2mm",
    "preprocessing": "fmriprep-21.0.0"
}
```

### 2. Let Lacuna Handle Resampling

Don't manually resample unless necessary:

```python
# Preferred: Let analyze() handle it
result = analyze(subject, analysis="flnm")

# Manual resampling only if needed for inspection
subject = resample_to_connectome(subject, "functional")
```

### 3. Verify Alignment Visually

Overlay your mask on the template to check alignment:

```python
from nilearn import plotting
import nibabel as nib

mask = nib.load("lesion.nii.gz")
template = nib.load("MNI152_T1_2mm.nii.gz")

plotting.plot_roi(mask, bg_img=template)
```

## See Also

- [TemplateFlow Configuration](../how-to/templateflow.md) — Setting up templates
- [Fetch Connectomes](../how-to/fetch-connectomes.md) — Getting connectome data
- [MaskData Validation](mask-data.md) — Mask requirements
