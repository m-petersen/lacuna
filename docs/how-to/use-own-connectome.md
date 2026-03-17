# Use Your Own Connectome

This guide shows how to use your own functional or structural connectome with Lacuna.

## Goal

Run network mapping analyses using connectome data that was not fetched via `lacuna fetch`.

## Prerequisites

- Lacuna installed ([Installation](installation.md))
- For functional connectomes: an HDF5 file with voxel-wise timeseries data
- For structural connectomes: a `.tck` tractogram file (MRtrix3 format)
- Data in MNI space

## Functional connectome

### Required HDF5 format

Your functional connectome must be an HDF5 file (`.h5`) containing:

| Dataset/Attribute | Shape | Description |
|-------------------|-------|-------------|
| `timeseries` | `(n_subjects, n_timepoints, n_voxels)` | Resting-state fMRI timeseries |
| `mask_indices` | `(3, n_voxels)` or `(n_voxels, 3)` | Voxel coordinates in volume space |
| `mask_affine` | `(4, 4)` | Affine transformation matrix |
| `mask_shape` (attribute) | tuple | Volume dimensions, e.g., `(91, 109, 91)` |

### Via the CLI

Point `--connectome-path` to your HDF5 file or a directory of batch files:

```bash
# Single HDF5 file
lacuna run fnm /bids /output \
    --connectome-path /data/my_connectome.h5

# Directory of batch files
lacuna run fnm /bids /output \
    --connectome-path /data/my_connectome_batches/
```

### Via the Python API

```python
from lacuna.assets.connectomes import register_functional_connectome
from lacuna.analysis import FunctionalNetworkMapping

# Register your connectome
register_functional_connectome(
    name="MyConnectome",
    space="MNI152NLin6Asym",
    resolution=2.0,
    data_path="/data/my_connectome.h5",
    n_subjects=500,
    description="My custom functional connectome",
)

# Use in analysis
fnm = FunctionalNetworkMapping(
    connectome_name="MyConnectome",
    method="boes",
)
result = fnm.run(subject_data)
```

For a directory of batch files (lower memory usage):

```python
register_functional_connectome(
    name="MyConnectome_batched",
    space="MNI152NLin6Asym",
    resolution=2.0,
    data_path="/data/my_connectome_batches/",  # directory with .h5 files
    n_subjects=500,
)
```

## Structural connectome

### Required format

Your structural connectome must be a `.tck` file (MRtrix3 streamlines format) in MNI space.

!!! warning "MRtrix3 required"
    Structural network mapping requires MRtrix3 to be installed. See [Setup MRtrix3](setup-mrtrix3.md).

### Via the CLI

```bash
lacuna run snm /bids /output \
    --connectome-path /data/my_tractogram.tck
```

### Via the Python API

```python
from lacuna.assets.connectomes import register_structural_connectome
from lacuna.analysis import StructuralNetworkMapping

# Register your tractogram
register_structural_connectome(
    name="MyTractogram",
    space="MNI152NLin2009cAsym",
    tractogram_path="/data/my_tractogram.tck",
    description="My custom whole-brain tractogram",
)

# Use in analysis
snm = StructuralNetworkMapping(connectome_name="MyTractogram")
result = snm.run(subject_data)
```

## List registered connectomes

```python
from lacuna.assets.connectomes import (
    list_functional_connectomes,
    list_structural_connectomes,
)

for c in list_functional_connectomes():
    print(f"{c.name}: {c.space} @ {c.resolution}mm ({c.n_subjects} subjects)")

for c in list_structural_connectomes():
    print(f"{c.name}: {c.space}")
```

Or from the CLI:

```bash
lacuna info connectomes
```

## Tips

!!! tip "Coordinate space matters"
    Ensure your connectome is in the correct MNI space. Functional connectomes typically use `MNI152NLin6Asym`, while structural connectomes often use `MNI152NLin2009cAsym`. If there's a space mismatch with your input masks, Lacuna will apply spatial transforms automatically.

!!! tip "Batching for large functional connectomes"
    If your functional connectome is too large for available RAM, split it into multiple HDF5 files in a directory and register the directory path. Each batch file is loaded and processed independently.

## See also

- [Fetch Connectomes](fetch-connectomes.md) — Downloading bundled connectomes
- [Run Functional LNM](run-flnm.md) — Functional network mapping workflow
- [Run Structural LNM](run-slnm.md) — Structural network mapping workflow
- [Coordinate Spaces](../explanation/coordinate-spaces.md) — Understanding MNI spaces
