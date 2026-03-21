# Coordinate Spaces in Lacuna

Understanding MNI spaces and how Lacuna handles spatial alignment automatically.

## Overview

Lesion network mapping requires that lesion masks and normative connectomes share the same coordinate space. Instead of requiring you to manually align all data to a single space, Lacuna uses [TemplateFlow](https://www.templateflow.org/) to handle transformations internally. You only need to provide lesion masks in one of the two supported MNI spaces — Lacuna ensures everything else is brought into alignment automatically.

For how to get your masks into MNI space, see the [Spatial Normalization](../how-to/spatial-normalization.md) how-to guide.

## What Is a Coordinate Space?

A coordinate space defines how 3D voxel indices (i, j, k) map to physical
locations in the brain (x, y, z in millimeters).

```
Voxel (45, 54, 45) → Physical coordinates (0, 0, 0) mm
```

This mapping is defined by the **affine transformation matrix** stored in the NIfTI header:

```python
import nibabel as nib

img = nib.load("brain.nii.gz")
print(img.affine)
# [[ 2.  0.  0. -90.]
#  [ 0.  2.  0. -126.]
#  [ 0.  0.  2. -72.]
#  [ 0.  0.  0.  1.]]
```

## Why Spaces Matter

Lacuna compares your lesion mask against normative connectomes derived from
healthy subjects. For this comparison to be meaningful:

1. **Same template**: Both must be aligned to the same reference brain
2. **Same resolution**: Voxel grids must match (or be resampled)
3. **Same orientation**: Left-right, anterior-posterior must agree

If spaces don't match, a lesion at voxel (45, 54, 45) in your mask would correspond
to a different anatomical location in the connectome.

## Supported MNI Spaces

The Montreal Neurological Institute (MNI) template is the standard space in neuroimaging. There are multiple MNI variants — Lacuna supports two as user input:

### MNI152NLin6Asym

- **Used by**: FSL, HCP pipelines
- **Resolution**: Typically 2mm isotropic
- **Orientation**: RAS (Right-Anterior-Superior)
- **Lacuna use**: Functional connectomes (GSP1000, HCP1065)

### MNI152NLin2009cAsym

- **Used by**: fMRIPrep, TemplateFlow
- **Resolution**: Various (0.5mm to 2mm)
- **Orientation**: RAS
- **Lacuna use**: Structural connectomes, TemplateFlow canonical

### Quick Reference

| Property | MNI152NLin6Asym (2mm) | MNI152NLin2009cAsym (1mm) |
|----------|----------------------|--------------------------|
| Dimensions | 91 x 109 x 91 | 193 x 229 x 193 |
| Voxel size | 2mm | 1mm |
| Affine origin | (-90, -126, -72) | (-96, -132, -78) |

### Why Multiple MNI Spaces?

Different MNI variants use different registration algorithms, numbers of subjects averaged, and tissue segmentation methods. The differences are subtle (1-2mm) but can affect voxel-level analyses.

## How Lacuna Detects Space

Lacuna auto-detects the coordinate space of your mask through two methods:

1. **BIDS filename parsing**: Extracts space and resolution from entities like `space-MNI152NLin6Asym_res-2` in the filename
2. **NIfTI affine matching**: Compares the image affine matrix against known reference affines for each supported space

If both are available, Lacuna validates that they agree. You can also explicitly specify the space via the `--space` flag in the CLI or the `space` parameter in the API.

## Automatic Transformation

When you run an analysis, Lacuna automatically transforms your mask to match the target data. This happens transparently before the analysis begins:

1. **Determine target space**: Each analysis type sets its target based on the connectome or atlas being used
2. **Compare spaces**: Lacuna checks if the mask is already in the target space and resolution
3. **Transform if needed**: Applies the appropriate transformation strategy

### Transformation Strategies

| Source → Target | Method | Description |
|----------------|--------|-------------|
| Same space, same resolution | None | No transformation needed |
| Same space, different resolution | Resample | Regrid to target resolution |
| 2009b ↔ 2009c | Regrid | Affine-aware regridding (same MNI world coordinates, different voxel grids) |
| NLin6 ↔ 2009c | Warp | Nonlinear warp transform from TemplateFlow |
| NLin6 ↔ 2009b | Chain | NLin6 → 2009c (warp) → 2009b (regrid) |

### Interpolation

Lacuna automatically selects the interpolation method based on the image content:

| Image Type | Interpolation | Reason |
|------------|--------------|--------|
| Binary masks (0/1) | Nearest neighbor | Preserves binary values |
| Integer label maps | Nearest neighbor | Preserves discrete labels |
| Continuous data | Cubic B-spline | Smooth resampling |

### MNI152NLin2009b (Internal)

The dTOR985 structural connectome is defined in MNI152NLin2009bAsym space. This space is not accepted as user input — Lacuna converts between 2009c and 2009b internally using affine-aware regridding (no nonlinear warp needed, since the 2009 variants share MNI world coordinates).

## TemplateFlow Integration

Lacuna uses [TemplateFlow](https://templateflow.org) to manage templates and spatial transforms. Templates and warp fields are downloaded automatically on first use and cached locally.

When you run an analysis, Lacuna:

1. Identifies the required template space from the connectome or atlas
2. Downloads the matching template and transform from TemplateFlow (if not cached)
3. Applies the transform to align your mask with the target data

No manual template management is required.

## Troubleshooting

### Results don't make anatomical sense

Your mask may be in a different space than expected. Verify with:

```python
import nibabel as nib

img = nib.load("lesion.nii.gz")
print(f"Shape: {img.shape}")
print(f"Voxel size: {img.header.get_zooms()[:3]}")
print(f"Affine origin: {img.affine[:3, 3]}")
```

Compare against the Quick Reference table above.

### Results appear left-right flipped

Check the orientation:

```python
import nibabel as nib

img = nib.load("lesion.nii.gz")
print(nib.aff2axcodes(img.affine))  # Should be ('R', 'A', 'S')
```

### Visual verification

Overlay your mask on the MNI template to confirm alignment:

```python
from nilearn import plotting, datasets

template = datasets.load_mni152_template(resolution=2)
plotting.plot_roi("lesion.nii.gz", bg_img=template, title="Lesion in MNI space")
plotting.show()
```

## See Also

- [Spatial Normalization](../how-to/spatial-normalization.md) — How to get your masks into MNI space
