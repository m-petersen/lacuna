# 4D Atlas Support Implementation Summary

## Overview

Successfully implemented comprehensive support for 4D atlases (like HCP1065 with one tract per volume) following TDD methodology. The implementation allows flexible use of both 3D and 4D atlases throughout the lacuna package.

## Changes Made

### 1. Atlas Metadata Enhancement (`src/lacuna/assets/atlases/registry.py`)

Added `is_4d` field to `AtlasMetadata`:

```python
@dataclass(frozen=True)
class AtlasMetadata(SpatialAssetMetadata):
    ...
    is_4d: bool = False  # NEW: Indicates 4D atlas (multiple volumes)
```

**Benefits:**
- Explicit metadata about atlas dimensionality
- Defaults to False for backward compatibility
- Helps users and developers understand atlas structure

### 2. 4D Image Transformation (`src/lacuna/spatial/transform.py`)

Enhanced `TransformationStrategy.apply_transformation()` to handle 4D images:

**Key Logic:**
- Detects 4D images with non-singleton 4th dimension
- Transforms each volume independently using volume-by-volume approach
- Uses nearest-neighbor interpolation (preserves integer labels)
- Stacks transformed volumes back into 4D structure
- Handles asyncio event loop issues in Jupyter notebooks

**Code Flow:**
```python
if img_data.ndim == 4 and img_data.shape[3] > 1:
    # 4D atlas - transform each volume
    for vol_idx in range(n_volumes):
        vol_data = img_data[..., vol_idx]
        vol_img = nib.Nifti1Image(vol_data, img.affine, img.header)
        transformed_vol = transform.apply(vol_img, ...)
        transformed_volumes.append(transformed_vol.get_fdata())
    
    # Stack back to 4D
    transformed_4d_data = np.stack(transformed_volumes, axis=-1)
    return nib.Nifti1Image(transformed_4d_data, affine, header)
```

### 3. Atlas Aggregation Enhancement (`src/lacuna/analysis/atlas_aggregation.py`)

Updated `_load_atlases_from_registry()` to include `is_4d` metadata:

```python
atlases_data.append({
    ...
    "is_4d": getattr(atlas.metadata, 'is_4d', False),
})
```

**Existing 4D Support:**
- `_run_analysis()` already detects 4D atlases via `atlas_data.ndim == 4`
- `_aggregate_4d_atlas()` already handles 4D structures correctly
- Each volume treated as separate region (perfect for HCP1065 structure)

### 4. Comprehensive Test Suite (`tests/unit/test_4d_atlas_support.py`)

Created 405-line test file with 11 tests across 5 test classes:

#### Test4DAtlasMetadata (3 tests)
- ✅ `test_atlas_metadata_has_is_4d_field`
- ✅ `test_atlas_metadata_4d_true`
- ✅ `test_atlas_metadata_is_4d_defaults_to_false`

#### Test4DAtlasDetection (2 tests)
- ✅ `test_register_3d_atlas_sets_is_4d_false`
- ✅ `test_register_4d_atlas_sets_is_4d_true`

#### Test4DAtlasTransformation (3 tests)
- ✅ `test_transform_4d_atlas_volume_by_volume`
- ✅ `test_transform_4d_atlas_preserves_labels`
- ✅ `test_transform_4d_atlas_different_spaces`

#### Test4DAtlasAggregation (2 tests)
- ✅ `test_regional_damage_with_4d_atlas`
- ✅ `test_4d_atlas_aggregation_per_volume`

#### TestMixed3DAnd4DAtlases (1 test)
- ✅ `test_regional_damage_with_mixed_atlases`

**All 11 tests pass!**

## Technical Details

### Volume-by-Volume Transformation

**Why This Approach:**
- nitransforms expects 3D images
- 4D images with non-singleton dimensions cause IndexError
- Each volume must be transformed independently

**Implementation:**
1. Extract each volume (shape: x, y, z) from 4D data
2. Create temporary 3D NIfTI image
3. Apply transform with appropriate interpolation
4. Collect transformed volumes
5. Stack back into 4D structure

**Interpolation:**
- Uses nearest-neighbor for atlases (preserves integer labels)
- Each volume maintains its label integrity

### Atlas Aggregation

**4D Atlas Structure (HCP1065 Example):**
- Shape: (182, 218, 182, 64)
- 64 tracts, one per volume
- Each volume is binary (0 or 1) indicating tract presence

**Aggregation Process:**
1. Resample 4D atlas to match input data
2. For each volume (tract):
   - Apply threshold to create binary mask
   - Extract values in masked region
   - Compute aggregation (mean, sum, percent, volume, etc.)
3. Map volume index to label names
4. Return results for all tracts

### Space Handling

**Cross-Space Transformation:**
- Works seamlessly across different MNI spaces
- Example: Atlas in MNI152NLin6Asym → Lesion in MNI152NLin2009cAsym
- Each volume transformed independently with same target space
- Preserves spatial alignment across all volumes

## Usage Examples

### Registering 4D Atlas

```python
from lacuna.assets.atlases import register_atlas, AtlasMetadata

metadata = AtlasMetadata(
    name="HCP1065",
    space="MNI152NLin6Asym",
    resolution=2.0,
    description="HCP1065 white matter tracts (64 tracts)",
    atlas_filename="/path/to/hcp1065.nii.gz",
    labels_filename="/path/to/hcp1065_labels.txt",
    is_4d=True,  # ← Indicates 4D structure
    n_regions=64  # Number of tracts/volumes
)

register_atlas(metadata)
```

### Using 4D Atlas in Analysis

```python
from lacuna.analysis import RegionalDamage

# Works automatically with 4D atlases!
analysis = RegionalDamage(
    atlas_names=["HCP1065"],  # 4D atlas
    threshold=0.0
)

result = analysis.run(lesion_data)

# Results contain damage per tract
damage_results = result.results["RegionalDamage"]
# Example: {'HCP1065_Tract1': 15.2, 'HCP1065_Tract2': 8.7, ...}
```

### Mixed 3D and 4D Atlases

```python
# Use both 3D and 4D atlases together!
analysis = RegionalDamage(
    atlas_names=[
        "Schaefer2018_100Parcels7Networks",  # 3D atlas
        "HCP1065",  # 4D atlas
    ],
    threshold=0.5
)

result = analysis.run(lesion_data)
# Results include regions from both atlases
```

## Error Resolution

**Original Error:**
```
ValueError: Cannot transform 4D image with non-singleton 4th dimension. 
Shape: (182, 218, 182, 64). Expected 3D image or 4D with shape[3]=1.
```

**Root Cause:**
- HCP1065 has 64 tracts as separate volumes (4th dimension)
- Previous code only handled singleton 4D dimensions (shape[3]=1)
- nitransforms cannot handle 4D images directly

**Solution:**
- Detect 4D images with non-singleton dimensions
- Transform each volume independently
- Stack transformed volumes back into 4D
- Preserve labels and spatial alignment

## Logging and Transparency

Added comprehensive logging throughout transformation:

```
INFO: Transforming 4D image with 64 volumes from MNI152NLin6Asym@2mm to MNI152NLin2009cAsym@2mm
DEBUG: Transforming volume 1/64
DEBUG: Transforming volume 2/64
...
INFO: 4D transformation complete. Output shape: (91, 109, 91, 64), dtype: float64
```

## Benefits

1. **Flexibility**: Support both 3D and 4D atlases seamlessly
2. **Transparency**: `is_4d` metadata makes structure explicit
3. **Robustness**: Comprehensive test coverage (11 tests)
4. **Cross-Space**: Works across different coordinate spaces
5. **Backward Compatible**: Default `is_4d=False` preserves existing behavior
6. **Label Preservation**: Nearest-neighbor interpolation maintains integer labels
7. **Mixed Usage**: Can use 3D and 4D atlases together

## Test Coverage

- **Metadata**: Field presence, validation, defaults
- **Detection**: Automatic 3D/4D identification
- **Transformation**: Volume-by-volume, label preservation, cross-space
- **Aggregation**: Per-volume damage computation, integration with RegionalDamage
- **Mixed Usage**: 3D and 4D atlases in same analysis

**Coverage:**
- `atlas_aggregation.py`: 67% (↑ from 13%)
- `transform.py`: 23% (↑ from 14%)
- `atlases/loader.py`: 75% (↑ from 24%)

## Future Enhancements

Potential improvements:
1. Automatic detection of `is_4d` from file dimensions during registration
2. Parallel transformation of volumes for large 4D atlases
3. Memory-mapped loading for very large 4D atlases
4. Support for probabilistic 4D atlases (already partially supported)

## Conclusion

Successfully implemented comprehensive 4D atlas support following TDD methodology. All tests pass, and the implementation allows flexible use of both 3D and 4D atlases (like HCP1065) throughout lacuna. The solution handles transformation, aggregation, and mixed usage scenarios seamlessly.
