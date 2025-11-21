# Unified Data Type Containers

**Version**: 0.4.0+ | **Feature**: 003-package-optimization-US6

## Overview

Lacuna uses **unified data type containers** - classes that serve as both inputs and outputs for analyses. This design enables clean analysis chaining, automatic metadata preservation, and a more intuitive API.

### Key Benefits

1. **Simplified API**: Same class for input and output eliminates confusion
2. **Analysis Chaining**: Output from one analysis becomes input to another seamlessly  
3. **Metadata Preservation**: Spatial information (space, resolution) travels with data automatically
4. **Type Safety**: Clear type hints and validation at boundaries

## Core Data Types

### VoxelMap

Container for voxel-level brain maps (3D/4D images).

**Use Cases**:
- Functional connectivity maps (z-maps, correlation maps)
- Structural disconnection maps  
- Probability maps
- Any volumetric brain data

**Example - Output from Analysis**:
```python
from lacuna.analysis import FunctionalNetworkMapping

fnm = FunctionalNetworkMapping(connectome="HCP1200")
result = fnm.run(mask_data)

# Get VoxelMap from analysis output
network_map = result.results["FunctionalNetworkMapping"]["network_map"]
assert isinstance(network_map, VoxelMap)
```

**Example - Input to Analysis**:
```python
from lacuna.core import VoxelMap
from lacuna.analysis import ParcelAggregation

# Load external z-map as VoxelMap
zmap = VoxelMap.from_nifti(
    "external_zmap.nii.gz",
    space="MNI152NLin6Asym",
    resolution=2.0,
    name="functional_connectivity"
)

# Use as input to parcel aggregation
agg = ParcelAggregation(parcel_names=["Schaefer2018_400Parcels7Networks"])
result = agg.run(zmap)  # Returns ParcelData directly
```

**Attributes**:
- `data`: nibabel.Nifti1Image - The brain image
- `space`: str - Coordinate space (e.g., 'MNI152NLin6Asym')
- `resolution`: float - Voxel resolution in mm (e.g., 2.0)
- `name`: str | None - Descriptive name
- `metadata`: dict - Additional metadata

---

### ParcelData

Container for parcel-level aggregated data (ROI statistics).

**Use Cases**:
- Atlas-based aggregation (Schaefer, AAL, DKT, etc.)
- Region-level statistics
- Summary metrics by brain parcel

**Example**:
```python
from lacuna.analysis import ParcelAggregation

agg = ParcelAggregation(
    parcel_names=["Schaefer2018_400Parcels7Networks"],
    aggregation="mean"
)

# From MaskData
result = agg.run(mask_data)
parcel_data = result.results["ParcelAggregation"]["Schaefer400_from_mask_img"]

# From VoxelMap (external data)
zmap = VoxelMap.from_nifti("zmap.nii.gz", space="MNI152NLin6Asym", resolution=2.0)
parcel_data = agg.run(zmap)  # Returns ParcelData directly

# Access data
top_regions = parcel_data.get_top_regions(n=5)
print(top_regions)
```

**Attributes**:
- `data`: dict[str, float] - Mapping of region labels → aggregated values
- `parcel_names`: list[str] - Names of parcellations used
- `aggregation_method`: str - Aggregation method (mean, sum, percent, etc.)
- `name`: str | None - Descriptive name
- `metadata`: dict - Additional metadata (source, threshold, n_regions, etc.)

---

### ConnectivityMatrix

Container for connectivity matrices (structural, functional, effective).

**Use Cases**:
- Structural connectivity matrices
- Functional connectivity matrices
- Effective connectivity matrices

**Example**:
```python
from lacuna.analysis import StructuralNetworkMapping

snm = StructuralNetworkMapping(
    tractogram_path="whole_brain.tck",
    parcellation_names=["Schaefer2018_400Parcels7Networks"]
)

result = snm.run(mask_data)
conn_matrix = result.results["StructuralNetworkMapping"]["connectivity_matrix"]

# Access data
matrix_data = conn_matrix.get_data()  # numpy array
region_labels = conn_matrix.region_labels  # Actual parcel names
```

**Attributes**:
- `matrix`: np.ndarray - N x N connectivity matrix
- `region_labels`: list[str] - Region labels for rows/columns
- `matrix_type`: str - Type ('structural', 'functional', 'effective')
- `name`: str | None - Descriptive name
- `metadata`: dict - Additional metadata

---

### Tractogram

Container for tractography streamline data (path-based storage).

**Example**:
```python
result = snm.run(mask_data)
tractogram = result.results["StructuralNetworkMapping"]["lesion_tractogram"]

# Tractogram uses on-demand loading
streamlines = tractogram.get_data()  # Loads from .tck file
```

**Attributes**:
- `tractogram_path`: Path - Path to .tck file
- `streamlines`: Any | None - Optional in-memory streamlines (caching)
- `name`: str | None - Descriptive name
- `metadata`: dict - Additional metadata

---

### SurfaceMesh

Container for surface-based brain data (vertices, faces, vertex data).

**Attributes**:
- `vertices`: np.ndarray - N x 3 array of vertex coordinates
- `faces`: np.ndarray - M x 3 array of triangle face indices
- `vertex_data`: np.ndarray | None - Optional per-vertex data values
- `name`: str | None - Descriptive name
- `metadata`: dict - Additional metadata

---

### ScalarMetric

Container for scalar values, summary statistics, and non-spatial data.

**Example**:
```python
from lacuna.analysis import FunctionalNetworkMapping

fnm = FunctionalNetworkMapping(connectome="HCP1200")
result = fnm.run(mask_data)

mean_conn = result.results["FunctionalNetworkMapping"]["mean_connectivity"]
assert isinstance(mean_conn, ScalarMetric)
print(mean_conn.data)  # e.g., 0.42
```

**Attributes**:
- `data`: Any - Scalar value, dict, list, or other data
- `data_type`: str - Type description (e.g., 'scalar', 'dict', 'summary')
- `name`: str | None - Descriptive name
- `metadata`: dict - Additional metadata

---

## Migration from Old Names

Deprecated aliases are provided for backward compatibility (will be removed in v0.6.0):

| Old Name | New Name | Type |
|----------|----------|------|
| `VoxelMapResult` | `VoxelMap` | Unified container |
| `AtlasAggregationResult` | `ParcelData` | Unified container |
| `ConnectivityMatrixResult` | `ConnectivityMatrix` | Unified container |
| `TractogramResult` | `Tractogram` | Unified container |
| `SurfaceResult` | `SurfaceMesh` | Unified container |
| `MiscResult` | `ScalarMetric` | Unified container |
| `ROIResult` | `ParcelData` | Unified container |
| `AtlasAggregation` | `ParcelAggregation` | Analysis class |
| `atlas_names` | `parcel_names` | Parameter/attribute |

**Migration Example**:
```python
# Old (deprecated, emits warning)
from lacuna.core.data_types import VoxelMap, ParcelData
from lacuna.analysis.atlas_aggregation import AtlasAggregation

# New (recommended)
from lacuna.core.data_types import VoxelMap, ParcelData
from lacuna.analysis.parcel_aggregation import ParcelAggregation
```

---

## Analysis Chaining Pattern

The unified container design enables clean analysis chaining:

```python
from lacuna import MaskData
from lacuna.analysis import FunctionalNetworkMapping, ParcelAggregation

# Step 1: Load mask
mask = MaskData.from_nifti(
    "lesion.nii.gz",
    metadata={"space": "MNI152NLin6Asym", "resolution": 2}
)

# Step 2: Run functional network mapping
fnm = FunctionalNetworkMapping(connectome="HCP1200")
result1 = fnm.run(mask)

# Step 3: Get network map (VoxelMap)
network_map = result1.results["FunctionalNetworkMapping"]["network_map"]

# Step 4: Aggregate to parcels (VoxelMap → ParcelData)
agg = ParcelAggregation(parcel_names=["Schaefer2018_400Parcels7Networks"])
parcel_data = agg.run(network_map)  # Clean chaining!

# Access parcel-level results
top_regions = parcel_data.get_top_regions(n=5)
```

---

## Design Rationale

### Why Unified Containers?

**Before (Result suffix)**:
- Separate classes for inputs and outputs created confusion
- Users had to wrap external data before using it
- Analysis chaining required manual result extraction and wrapping

**After (Unified)**:
- Single class for both input and output is intuitive
- External data can be loaded directly as the container type
- Analysis chaining is seamless - output → input directly
- Metadata preservation is automatic

### Terminology Changes

**"Atlas" → "Parcel"**:
- "Atlas" is ambiguous (brain parcellations vs neurotransmitter atlases)
- "Parcel" is more precise for brain region subdivisions
- Aligns with neuroimaging terminology (parcellation, parcel-based analysis)

**File Rename: `output.py` → `data_types.py`**:
- Avoids confusion with Docker "containers" terminology
- More descriptive of actual purpose (data type definitions)
- Clarifies that these are data structures, not just outputs

---

## Best Practices

### 1. Use Type Hints

```python
from lacuna.core.data_types import VoxelMap, ParcelData

def process_voxel_data(vmap: VoxelMap) -> ParcelData:
    """Process voxel-level data to parcel-level."""
    agg = ParcelAggregation(parcel_names=["Schaefer2018_400Parcels7Networks"])
    return agg.run(vmap)
```

### 2. Preserve Metadata

```python
# When creating containers, include relevant metadata
vmap = VoxelMap(
    name="z_map",
    data=image_data,
    space="MNI152NLin6Asym",
    resolution=2.0,
    metadata={
        "source_analysis": "group_analysis",
        "threshold": 2.3,
        "n_subjects": 50
    }
)
```

### 3. Leverage get_data() for Downstream Use

```python
# Extract underlying data when needed
vmap = result.results["FunctionalNetworkMapping"]["network_map"]
nibabel_image = vmap.get_data()

# Now use with nilearn, nibabel, etc.
from nilearn.plotting import plot_stat_map
plot_stat_map(nibabel_image, ...)
```

---

## See Also

- [Data Model Specification](../specs/003-package-optimization/data-model.md)
- [Unified Containers Contract](../specs/003-package-optimization/contracts/unified-data-types.yaml)
- [Migration Guide](migration_guide.md)
