# Migration Guide: 003-package-optimization

This guide helps you migrate code from pre-0.5.0 versions to the optimized v0.5.0+ API.

## Breaking Changes Summary

1. **Unified Data Type Containers** - Old `*Result` classes removed
2. **MaskData Replaces LesionData** - More general masking support  
3. **Explicit Space Handling** - Space and resolution now required
4. **Simplified Analysis API** - Consistent patterns across all analyses
5. **snake_case Result Keys** - Analysis result keys now use snake_case (v0.6.0+)

---

## 0. snake_case Result Keys (v0.6.0+)

### What Changed

All analysis result keys have been changed from PascalCase to snake_case for Python convention compliance. This affects `FunctionalNetworkMapping` and `StructuralNetworkMapping` result dictionaries.

### Key Mappings

| Old Key (v0.5.x)    | New Key (v0.6.0+)     | Analysis                      |
|---------------------|----------------------|-------------------------------|
| `CorrelationMap`    | `correlation_map`    | FunctionalNetworkMapping      |
| `ZMap`              | `z_map`              | FunctionalNetworkMapping      |
| `TMap`              | `t_map`              | FunctionalNetworkMapping      |
| `TThresholdMap`     | `t_threshold_map`    | FunctionalNetworkMapping      |
| `DisconnectionMap`  | `disconnection_map`  | StructuralNetworkMapping      |
| `LesionTractogram`  | `lesion_tractogram`  | StructuralNetworkMapping      |
| `LesionTDI`         | `lesion_tdi`         | StructuralNetworkMapping      |

### Migration Examples

**Before (v0.5.x):**
```python
result = fnm.run(mask_data)
corr_map = result.results["FunctionalNetworkMapping"]["CorrelationMap"]
z_map = result.results["FunctionalNetworkMapping"]["ZMap"]
t_map = result.results["FunctionalNetworkMapping"]["TMap"]

result = snm.run(mask_data)
disconn = result.results["StructuralNetworkMapping"]["DisconnectionMap"]
```

**After (v0.6.0+):**
```python
result = fnm.run(mask_data)
corr_map = result.results["FunctionalNetworkMapping"]["correlation_map"]
z_map = result.results["FunctionalNetworkMapping"]["z_map"]
t_map = result.results["FunctionalNetworkMapping"]["t_map"]

result = snm.run(mask_data)
disconn = result.results["StructuralNetworkMapping"]["disconnection_map"]
```

### Find-and-Replace Pattern

Run this to update your codebase:

```bash
# FunctionalNetworkMapping keys
sed -i 's/\["CorrelationMap"\]/["correlation_map"]/g' *.py
sed -i 's/\["ZMap"\]/["z_map"]/g' *.py
sed -i 's/\["TMap"\]/["t_map"]/g' *.py
sed -i 's/\["TThresholdMap"\]/["t_threshold_map"]/g' *.py

# StructuralNetworkMapping keys
sed -i 's/\["DisconnectionMap"\]/["disconnection_map"]/g' *.py
sed -i 's/\["LesionTractogram"\]/["lesion_tractogram"]/g' *.py
sed -i 's/\["LesionTDI"\]/["lesion_tdi"]/g' *.py
```

---

## 1. Unified Data Type Containers

### What Changed

Old analysis result classes have been **removed** and replaced with unified container classes that work as both inputs and outputs.

### Class Name Mapping

| Old Class (REMOVED)           | New Class          | Purpose                          |
|-------------------------------|-------------------|----------------------------------|
| `VoxelMapResult`              | `VoxelMap`        | 3D/4D brain maps                 |
| `AtlasAggregationResult`      | `ParcelData`      | Region-level aggregated data     |
| `ROIResult`                   | `ParcelData`      | Region-level aggregated data     |
| `ConnectivityMatrixResult`    | `ConnectivityMatrix` | Connectivity matrices         |
| `TractogramResult`            | `Tractogram`      | Tractography streamlines         |
| `SurfaceResult`               | `SurfaceMesh`     | Surface meshes                   |
| `MiscResult`                  | `ScalarMetric`    | Summary stats & scalars          |

### Migration Examples

**Before (v0.4.x):**
```python
from lacuna.core.output import VoxelMapResult, AtlasAggregationResult

# Results were separate output-only classes
result = analysis.run(mask_data)
voxel_map = result.results["Analysis"]["map"]  # VoxelMapResult
```

**After (v0.5.0+):**
```python
from lacuna.core.data_types import VoxelMap, ParcelData

# Results use unified containers
result = analysis.run(mask_data)
voxel_map = result.results["Analysis"]["map"]  # VoxelMap
```

### Import Updates

**Before:**
```python
from lacuna.core.output import (
    VoxelMapResult,
    AtlasAggregationResult,
    ConnectivityMatrixResult,
    MiscResult,
)
```

**After:**
```python
from lacuna.core.data_types import (
    VoxelMap,
    ParcelData,
    ConnectivityMatrix,
    ScalarMetric,
)
```

---

## 2. LesionData → MaskData

### What Changed

`LesionData` has been renamed to `MaskData` to support general binary masking beyond lesions.

### Migration

**Before:**
```python
from lacuna import LesionData

lesion = LesionData.from_nifti("lesion.nii.gz", space="MNI152NLin6Asym", resolution=2.0)
```

**After:**
```python
from lacuna import MaskData

mask = MaskData.from_nifti("mask.nii.gz", space="MNI152NLin6Asym", resolution=2.0)
```

### Parameter Changes

- `lesion_img` → `mask_img` (attribute name)
- `anatomical_img` parameter **removed** (was unused)

**Before:**
```python
lesion = LesionData.from_nifti(
    "lesion.nii.gz",
    anatomical_img="T1w.nii.gz",  # REMOVED
    space="MNI152NLin6Asym",
    resolution=2.0
)
```

**After:**
```python
mask = MaskData.from_nifti(
    "mask.nii.gz",  # Just the mask, no anatomical needed
    space="MNI152NLin6Asym",
    resolution=2.0
)
```

---

## 3. Explicit Space Handling

### What Changed

Space and resolution are now **required** (no defaults). This prevents silent errors from ambiguous coordinate spaces.

### Migration

**Before (implicit defaults):**
```python
# Dangerous: space defaulted to "MNI152NLin6Asym"
lesion = LesionData.from_nifti("lesion.nii.gz")
```

**After (explicit required):**
```python
# Safe: must specify space and resolution
mask = MaskData.from_nifti(
    "mask.nii.gz",
    space="MNI152NLin6Asym",  # Required
    resolution=2.0            # Required
)
```

### Supported Spaces

Only these standard spaces are supported:
- `MNI152NLin6Asym`
- `MNI152NLin2009aAsym`
- `MNI152NLin2009cAsym`

Custom/native spaces must be transformed to standard space before analysis.

---

## 4. Analysis API Changes

### AtlasAggregation → ParcelAggregation

**Before:**
```python
from lacuna.analysis import AtlasAggregation  # Deprecated

analysis = AtlasAggregation(aggregation="percent")
```

**After:**
```python
from lacuna.analysis import ParcelAggregation

analysis = ParcelAggregation(aggregation="percent")
```

### Result Access Patterns

Result storage structure has been simplified and made consistent.

**Before (mixed formats):**
```python
# Inconsistent access patterns
results = mask_data.results["ParcelAggregation"][0]  # List-based
results = mask_data.results["Analysis"]["key"]        # Dict-based
```

**After (consistent dict pattern):**
```python
# All analyses use dict-based storage
results = mask_data.results["ParcelAggregation"]  # Dict of containers
roi_data = results["Schaefer2018_100Parcels"]     # ParcelData container
```

### Threshold Parameter Relaxed

The `threshold` parameter in `ParcelAggregation` now accepts any float value (not just 0-1).

**Before (restricted):**
```python
# Only values 0.0-1.0 allowed
analysis = ParcelAggregation(threshold=0.5)
```

**After (flexible):**
```python
# Any float value accepted (e.g., z-scores)
analysis = ParcelAggregation(threshold=-2.5)  # Negative z-score
analysis = ParcelAggregation(threshold=3.1)   # Positive z-score
```

---

## 5. Logging and Error Messages

### Log Level Parameter

All analysis classes now accept a `log_level` parameter:

```python
from lacuna.analysis import ParcelAggregation

# Control verbosity: 0 (silent), 1 (standard), 2 (verbose)
analysis = ParcelAggregation(
    aggregation="percent",
    log_level=2  # Verbose logging
)
```

### Improved Error Messages

Error messages now follow a consistent pattern:
1. Clear problem statement
2. Context (what was found vs. expected)
3. Actionable guidance

**Example:**
```
ValueError: Atlas 'InvalidAtlas' not found in registry.
Available atlases: Schaefer2018_100Parcels, Schaefer2018_200Parcels...
Use list_atlases() to see all options.
```

---

## 6. Nibabel Input Support

`ParcelAggregation` now accepts nibabel images directly, not just `MaskData`.

**New capability:**
```python
import nibabel as nib
from lacuna.analysis import ParcelAggregation

# Direct nibabel input
img = nib.load("data.nii.gz")
analysis = ParcelAggregation(aggregation="mean")
result = analysis.run(img)  # Returns ParcelData directly
```

**Batch processing:**
```python
images = [nib.load(f"sub-{i}.nii.gz") for i in range(10)]
results = analysis.run(images)  # Returns list[ParcelData]
```

---

## Complete Migration Checklist

- [ ] Replace all `LesionData` with `MaskData`
- [ ] Update `lesion_img` to `mask_img` in code
- [ ] Remove `anatomical_img` parameters
- [ ] Add explicit `space` and `resolution` to all `MaskData.from_nifti()` calls
- [ ] Replace old `*Result` imports with new unified container names:
  - `VoxelMapResult` → `VoxelMap`
  - `AtlasAggregationResult` → `ParcelData`
  - `ConnectivityMatrixResult` → `ConnectivityMatrix`
  - `MiscResult` → `ScalarMetric`
  - `TractogramResult` → `Tractogram`
  - `SurfaceResult` → `SurfaceMesh`
  - `ROIResult` → `ParcelData`
- [ ] Replace `AtlasAggregation` with `ParcelAggregation`
- [ ] Update result access to use consistent dict pattern
- [ ] Test with explicit `log_level` parameters
- [ ] Verify space/resolution metadata is present in all data

---

## Getting Help

- **Documentation**: See `docs/unified_containers.md` for unified container guide
- **Examples**: Check `examples/` directory for updated usage patterns
- **Issues**: Report migration problems on GitHub issue tracker

## Version Support

- **v0.4.x and earlier**: Old API (deprecated)
- **v0.5.0+**: New unified API (current)
- **v0.6.0**: Old classes fully removed (future)
