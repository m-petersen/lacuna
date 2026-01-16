# Run Functional Lesion Network Mapping

This guide shows how to perform functional lesion network mapping (fLNM) analysis with Lacuna.

## Goal

Map functional connectivity patterns associated with a lesion location using a normative resting-state connectome.

## Prerequisites

- Lacuna installed ([Installation Guide](installation.md))
- Binary lesion mask in MNI space (NIfTI format)
- Functional connectome downloaded (see below)

## Step-by-step instructions

### 1. Load your lesion mask

```python
import nibabel as nib
from lacuna import SubjectData

mask_img = nib.load("path/to/lesion.nii.gz")
subject = SubjectData(
    mask_img=mask_img,
    space="MNI152NLin6Asym",
    resolution=2.0,
    metadata={"subject_id": "sub-001"}
)
```

### 2. Register a functional connectome

```python
from lacuna.assets.connectomes import register_functional_connectome

register_functional_connectome(
    name="HCP_S1200",
    space="MNI152NLin6Asym",
    resolution=2.0,
    data_path="/path/to/connectome/directory",
    n_subjects=1000
)
```

### 3. Configure the analysis

```python
from lacuna.analysis import FunctionalNetworkMapping

fnm = FunctionalNetworkMapping(
    connectome_name="HCP_S1200",
    method="boes"  # Default correlation method
)
```

#### Available methods

| Method | Description |
|--------|-------------|
| `boes` | Boes et al. correlation method (default) |
| `fischer` | Fischer z-transformed correlations |

### 4. Run the analysis

```python
result = fnm.run(subject)
```

### 5. Access results

```python
# Get the correlation map
conn_map = result.results["FunctionalNetworkMapping"]["correlation_map"]

# View statistics
data = conn_map.get_data()
print(f"Shape: {data.shape}")
print(f"Range: [{data.min():.3f}, {data.max():.3f}]")
```

### 6. Save results

```python
# Save as NIfTI
conn_nifti = conn_map.to_nifti()
conn_nifti.to_filename("sub-001_flnm.nii.gz")
```

## Expected results

The output is a 3D correlation map where each voxel contains:

- **Positive values**: Regions positively connected to the lesion
- **Negative values**: Regions negatively connected to the lesion
- **Near-zero values**: Regions with weak or no connectivity

## Tips

!!! tip "Memory usage"
    
    Functional connectomes can be large. For lower memory usage, use a
    4mm resolution connectome instead of 2mm.

!!! tip "Visualization"
    
    Open the output NIfTI in FSLeyes or MRIcron with a red-blue colormap
    to visualize positive and negative connectivity.

## Troubleshooting

??? question "Error: Connectome not registered"
    
    Ensure you called `register_functional_connectome()` before creating
    the `FunctionalNetworkMapping` object.

??? question "Spatial mismatch error"
    
    Your lesion mask space must match the connectome space. Check that both
    use the same template (e.g., MNI152NLin6Asym) and resolution.
