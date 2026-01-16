# MaskData: Binary Mask Validation

Understanding Lacuna's requirements for lesion masks and why validation matters.

## Overview

`MaskData` is Lacuna's container for lesion masks. It enforces **strict binary
validation** to ensure analysis integrity. This document explains why binary
masks are required and how validation works.

## The Binary Requirement

### What Is a Binary Mask?

A binary mask contains only two values:

- **0**: No lesion present (background)
- **1**: Lesion present (foreground)

```python
# Valid binary mask
[0, 0, 1, 1, 0]  # ✓

# Invalid: continuous values
[0.0, 0.5, 0.8, 1.0, 0.2]  # ✗

# Invalid: multi-label
[0, 1, 2, 3, 0]  # ✗
```

### Why Binary?

Lesion network mapping algorithms require definitive lesion boundaries:

1. **Functional LNM** computes correlations between seed regions and the rest
   of the brain. Partial values would create ambiguous correlations.

2. **Structural LNM** traces streamlines through lesioned voxels. There's no
   meaningful way to "partially" trace through a voxel.

3. **Regional analysis** counts affected voxels per region. Fractional voxels
   would produce misleading statistics.

## Common Non-Binary Inputs

### Probability Maps

Many lesion segmentation tools output probability maps:

```
Voxel value = P(lesion | signal) ∈ [0, 1]
```

**Solution**: Threshold before using with Lacuna:

```python
import nibabel as nib
import numpy as np

# Load probability map
prob_img = nib.load("lesion_probability.nii.gz")
prob_data = prob_img.get_fdata()

# Threshold at 50%
binary_data = (prob_data >= 0.5).astype(np.int8)

# Save binary mask
binary_img = nib.Nifti1Image(binary_data, prob_img.affine)
nib.save(binary_img, "lesion_mask.nii.gz")
```

### Multi-Label Masks

Some workflows produce multi-label segmentations:

```
0 = background
1 = lesion type A
2 = lesion type B
```

**Solution**: Convert to binary or process separately:

```python
# Option 1: Combine all lesion types
binary_data = (multi_label_data > 0).astype(np.int8)

# Option 2: Analyze specific type
type_a_mask = (multi_label_data == 1).astype(np.int8)
type_b_mask = (multi_label_data == 2).astype(np.int8)
```

### Partial Volume Effects

High-resolution acquisitions may have voxels partially containing lesion:

**Solution**: Choose a threshold based on your research question:

- **Conservative** (≥0.75): Only include mostly-lesioned voxels
- **Liberal** (≥0.25): Include any voxel with lesion presence
- **Balanced** (≥0.50): Standard 50% threshold

## Validation in Practice

### Automatic Validation

MaskData validates on creation:

```python
from lacuna.core import MaskData

# This will raise ValidationError if not binary
mask = MaskData.from_nifti("lesion.nii.gz")
```

### Handling Validation Errors

```python
from lacuna.core import MaskData
from lacuna.core.exceptions import ValidationError

try:
    mask = MaskData.from_nifti("lesion.nii.gz")
except ValidationError as e:
    print(f"Mask validation failed: {e}")
    # Handle: threshold, binarize, or investigate
```

### Checking Before Loading

You can check a file without loading it fully:

```python
import nibabel as nib
import numpy as np

def is_binary_mask(path):
    """Check if a NIfTI file contains a binary mask."""
    img = nib.load(path)
    data = img.get_fdata()
    unique_vals = np.unique(data)
    return set(unique_vals).issubset({0, 1})
```

## What Validation Checks

MaskData validation ensures:

| Check | Description |
|-------|-------------|
| Binary values | Only 0 and 1 present |
| Non-empty | At least one lesioned voxel |
| 3D volume | Correct dimensionality |
| Valid affine | Proper spatial transformation |
| Finite values | No NaN or Inf values |

## Edge Cases

### All-Zero Masks

A mask with no lesioned voxels is technically binary but semantically
meaningless:

```python
# Raises ValidationError: no lesioned voxels
mask = MaskData.from_array(np.zeros((10, 10, 10)), affine)
```

### Floating-Point 0.0 and 1.0

Due to floating-point precision, values like 0.99999999 or 1.00000001 may
occur after resampling:

```python
# MaskData handles near-binary values with tolerance
# Values within 1e-6 of 0 or 1 are accepted and rounded
```

### NaN Values

NaN (Not a Number) values indicate missing data and are not allowed:

```python
# Raises ValidationError: contains NaN
data = np.array([0, 1, np.nan, 1, 0])
mask = MaskData.from_array(data, affine)
```

## Best Practices

### 1. Prepare Masks Before Lacuna

Do preprocessing (thresholding, cleaning) before loading:

```python
# Good: Prepare beforehand
binary = threshold_and_clean(probability_map)
nib.save(binary, "lesion_mask.nii.gz")

subject = SubjectData.from_nifti("lesion_mask.nii.gz", ...)
```

### 2. Document Your Thresholds

Record the threshold used for reproducibility:

```python
metadata = {
    "threshold": 0.5,
    "source": "lesion_probability.nii.gz",
    "method": "U-Net segmentation"
}
subject = SubjectData.from_nifti(path, metadata=metadata, ...)
```

### 3. Quality Check Visually

Always visually inspect masks before batch processing:

```python
import nibabel as nib
from nilearn import plotting

mask = nib.load("lesion_mask.nii.gz")
plotting.plot_roi(mask)
```

## See Also

- [SubjectData Design](subject-data.md) — How masks fit in the data model
- [Coordinate Spaces](coordinate-spaces.md) — Spatial requirements for masks
- Tutorial: [Getting Started](../tutorials/getting-started.md) — Loading your first mask
