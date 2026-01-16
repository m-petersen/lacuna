# Run Structural Lesion Network Mapping

This guide shows how to perform structural lesion network mapping (sLNM) analysis using tractography-based disconnection mapping.

## Goal

Identify white matter pathways disconnected by a lesion using a normative structural connectome derived from diffusion MRI tractography.

## Prerequisites

- Lacuna installed ([Installation Guide](installation.md))
- Binary lesion mask in MNI space (NIfTI format)
- **MRtrix3** installed and accessible from the command line
- Structural tractogram downloaded

## Step-by-step instructions

### 1. Verify MRtrix3 installation

```bash
tckinfo --version
```

If this fails, install MRtrix3 first.

### 2. Load your lesion mask

```python
import nibabel as nib
from lacuna import SubjectData

mask_img = nib.load("path/to/lesion.nii.gz")
subject = SubjectData(
    mask_img=mask_img,
    space="MNI152NLin2009cAsym",
    resolution=2.0,
    metadata={"subject_id": "sub-001"}
)
```

!!! note "Coordinate space"
    
    Structural connectomes are typically in MNI152NLin2009cAsym space.
    Ensure your lesion mask matches this space.

### 3. Register a structural connectome

```python
from lacuna.assets.connectomes import register_structural_connectome

register_structural_connectome(
    name="HCP_Tractogram",
    space="MNI152NLin2009cAsym",
    resolution=2.0,
    tractogram_path="/path/to/tractogram.tck",
    n_subjects=985
)
```

### 4. Configure the analysis

```python
from lacuna.analysis import StructuralNetworkMapping

snm = StructuralNetworkMapping(
    connectome_name="HCP_Tractogram",
    parcellation_name="Schaefer2018_100Parcels7Networks"
)
```

#### Available parcellations

| Parcellation | Regions | Description |
|-------------|---------|-------------|
| `Schaefer2018_100Parcels7Networks` | 100 | Schaefer 7-network cortical parcellation |
| `Schaefer2018_200Parcels7Networks` | 200 | Higher resolution version |
| `Schaefer2018_400Parcels7Networks` | 400 | Highest resolution version |

### 5. Run the analysis

```python
result = snm.run(subject)
```

This step may take several minutes depending on tractogram size.

### 6. Access results

```python
# Get disconnection map
disconn_map = result.results["StructuralNetworkMapping"]["disconnection_map"]

# Get parcel-level disconnection
parcel_disconn = result.results["StructuralNetworkMapping"]["parcel_disconnection"]

print(f"Disconnection map shape: {disconn_map.get_data().shape}")
```

### 7. Save results

```python
# Save disconnection map
disconn_nifti = disconn_map.to_nifti()
disconn_nifti.to_filename("sub-001_slnm.nii.gz")
```

## Expected results

The output includes:

| Result | Description |
|--------|-------------|
| `disconnection_map` | Voxelwise probability of streamline disconnection |
| `parcel_disconnection` | Disconnection probability per parcellation region |

## Tips

!!! tip "Processing time"
    
    Structural analysis is computationally intensive. For faster results:
    
    - Use a tractogram with fewer streamlines
    - Use a coarser parcellation (100 vs 400 parcels)

!!! tip "Interpretation"
    
    Higher disconnection values indicate stronger disruption of
    white matter pathways passing through the lesion.

## Troubleshooting

??? question "MRtrix3 not found"
    
    Ensure MRtrix3 is installed and the `tck*` commands are in your PATH:
    
    ```bash
    export PATH=/path/to/mrtrix3/bin:$PATH
    ```

??? question "Memory error"
    
    Large tractograms require significant RAM. Try:
    
    - Using a downsampled tractogram
    - Running on a machine with more memory
