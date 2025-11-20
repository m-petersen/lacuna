# Asset Management System Summary

## Overview

The Lacuna asset management system provides unified registration and loading for all neuroimaging assets needed for lesion network mapping analyses.

## Asset Types

### 1. Atlases
**Status**: ✅ Complete and tested

**Purpose**: Brain parcellations for regional damage quantification

**Location**: `lacuna.assets.atlases`

**Features**:
- 8 bundled atlases (Schaefer, Tian, HCP1065, dTOR)
- User registration supported
- Automatic validation (space, resolution, n_regions)

**Usage**:
```python
from lacuna.assets.atlases import list_atlases, load_atlas

# List available
atlases = list_atlases(space="MNI152NLin2009cAsym")

# Load atlas
atlas = load_atlas("Schaefer2018_100Parcels7Networks")
```

### 2. Templates
**Status**: ✅ Complete, needs testing

**Purpose**: Reference brain templates for spatial normalization

**Location**: `lacuna.assets.templates`

**Features**:
- Automatic downloading from TemplateFlow
- Caching to avoid re-downloads
- Supports MNI152NLin6Asym and MNI152NLin2009cAsym

**Usage**:
```python
from lacuna.assets.templates import list_templates, load_template

# List available
templates = list_templates(space="MNI152NLin2009cAsym", resolution=1.0)

# Load template (auto-downloads if needed)
template_path = load_template("MNI152NLin2009cAsym_res-1")
```

### 3. Transforms
**Status**: ✅ Complete, needs testing

**Purpose**: Spatial transformations between template spaces

**Location**: `lacuna.assets.transforms`

**Features**:
- Automatic downloading from TemplateFlow
- Bidirectional transforms (NLin6 ↔ NLin2009c)
- HDF5 format (.h5 files)

**Usage**:
```python
from lacuna.assets.transforms import list_transforms, load_transform

# List available
transforms = list_transforms(from_space="MNI152NLin6Asym")

# Load transform (auto-downloads if needed)
transform_path = load_transform("MNI152NLin6Asym_to_MNI152NLin2009cAsym_res-1")
```

### 4. Structural Connectomes
**Status**: ✅ Complete, needs testing

**Purpose**: Tractography data for structural lesion network mapping (sLNM)

**Location**: `lacuna.assets.connectomes`

**Requirements**:
- Tractogram file (.tck from MRtrix3)
- Whole-brain track density image (TDI, .nii.gz)
- Optional: template image for output grid

**Features**:
- User registration only (files too large to bundle)
- Validation of file existence and formats
- Space and resolution metadata

**Usage**:
```python
from lacuna.assets.connectomes import (
    register_structural_connectome,
    load_structural_connectome
)
from lacuna.analysis import StructuralNetworkMapping

# Register connectome
register_structural_connectome(
    name="HCP842_dTOR",
    space="MNI152NLin2009cAsym",
    resolution=1.0,
    tractogram_path="/data/dtor/hcp842_tractogram.tck",
    tdi_path="/data/dtor/hcp842_tdi_1mm.nii.gz",
    n_subjects=842,
    description="HCP dTOR tractogram"
)

# Load and use in analysis
connectome = load_structural_connectome("HCP842_dTOR")
analysis = StructuralNetworkMapping(
    tractogram_path=connectome.tractogram_path,
    whole_brain_tdi=connectome.tdi_path,
    template=connectome.template_path
)
```

### 5. Functional Connectomes
**Status**: ✅ Complete, needs testing

**Purpose**: Voxel-wise timeseries data for functional lesion network mapping (fLNM)

**Location**: `lacuna.assets.connectomes`

**Requirements**:
- HDF5 file(s) with voxel-wise BOLD timeseries
- NOT parcellated connectivity matrices
- Required HDF5 structure:
  - `timeseries`: (n_subjects, n_timepoints, n_voxels)
  - `mask_indices`: (3, n_voxels) or (n_voxels, 3)
  - `mask_affine`: (4, 4)
  - `mask_shape`: attribute tuple

**Features**:
- Supports single file or directory with batch files
- User registration only
- Validation of HDF5 structure
- Memory-efficient batch processing

**Usage**:
```python
from lacuna.assets.connectomes import (
    register_functional_connectome,
    load_functional_connectome
)
from lacuna.analysis import FunctionalNetworkMapping

# Register single file
register_functional_connectome(
    name="GSP1000",
    space="MNI152NLin6Asym",
    resolution=2.0,
    data_path="/data/gsp/gsp1000_connectome.h5",
    n_subjects=1000,
    description="GSP1000 voxel-wise connectome"
)

# Register batched directory
register_functional_connectome(
    name="GSP1000_batched",
    space="MNI152NLin6Asym",
    resolution=2.0,
    data_path="/data/gsp/batches/",
    n_subjects=1000,
    description="GSP1000 batched for memory efficiency"
)

# Load and use in analysis
connectome = load_functional_connectome("GSP1000")
analysis = FunctionalNetworkMapping(
    connectome_path=connectome.data_path,
    method="boes"
)
```

## Unified API Pattern

All asset types follow the same pattern:

```python
# List available assets
list_*()              # Returns list of metadata objects

# Load assets
load_*(name)          # Returns asset with paths/data

# Register assets (atlases and connectomes)
register_*(...)       # Add user assets to registry

# Unregister assets
unregister_*(name)    # Remove from registry

# Check cache (templates and transforms)
is_*_cached(name)     # Check if already downloaded
```

## Design Principles

1. **Consistency**: Same API pattern across all asset types
2. **Validation**: All registrations validate files and metadata
3. **Lazy Loading**: Templates/transforms downloaded only when needed
4. **Memory Efficiency**: Support for batched processing of large connectomes
5. **Backward Compatibility**: Analyses still accept direct file paths
6. **Discoverability**: `list_*()` functions with filtering options

## Testing Status

- ✅ Atlases: 15/15 contract tests passing
- ⏳ Templates: Contract tests needed
- ⏳ Transforms: Contract tests needed
- ⏳ Structural Connectomes: Contract tests needed
- ⏳ Functional Connectomes: Contract tests needed
- ⏳ Integration: Multi-asset workflow tests needed

## Next Steps

1. Write contract tests for templates, transforms, and connectomes
2. Write integration tests for network mapping with registries
3. Update analysis documentation with registry examples
4. Fix remaining test failures in non-asset modules
5. Add examples demonstrating the full asset management system
