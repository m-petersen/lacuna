# Export BIDS Derivatives

This guide shows how to export Lacuna analysis results as BIDS-compliant derivatives.

## Goal

Save analysis outputs in standardized BIDS derivatives format for reproducibility and sharing.

## Prerequisites

- Lacuna installed ([Installation Guide](installation.md))
- Completed analysis with results

## Step-by-step instructions

### 1. Run an analysis

```python
import nibabel as nib
from lacuna import SubjectData
from lacuna.analysis import FunctionalNetworkMapping

# Load mask and run analysis
mask_img = nib.load("path/to/lesion.nii.gz")
subject = SubjectData(
    mask_img=mask_img,
    space="MNI152NLin6Asym",
    resolution=2.0,
    metadata={
        "subject_id": "sub-001",
        "session": "ses-01"
    }
)

fnm = FunctionalNetworkMapping(connectome_name="HCP_S1200")
result = fnm.run(subject)
```

### 2. Export to BIDS derivatives

```python
from lacuna.io.bids import export_bids_derivatives

export_bids_derivatives(
    subject_data=result,
    output_dir="derivatives/lacuna"
)
```

### 3. Verify output structure

The output follows BIDS derivatives conventions:

```
derivatives/lacuna/
├── dataset_description.json
├── sub-001/
│   └── ses-01/
│       └── func/
│           ├── sub-001_ses-01_space-MNI152NLin6Asym_desc-flnm_connectivity.nii.gz
│           └── sub-001_ses-01_space-MNI152NLin6Asym_desc-flnm_connectivity.json
```

## Output files

### dataset_description.json

Contains pipeline metadata:

```json
{
    "Name": "Lacuna Lesion Network Mapping",
    "BIDSVersion": "1.8.0",
    "PipelineDescription": {
        "Name": "lacuna",
        "Version": "0.1.0"
    }
}
```

### Sidecar JSON files

Each NIfTI output has a companion JSON with provenance:

```json
{
    "Sources": ["sub-001_ses-01_mask.nii.gz"],
    "RawSources": ["sub-001_ses-01_T1w.nii.gz"],
    "SpatialReference": "MNI152NLin6Asym",
    "Resolution": 2.0,
    "Analysis": "FunctionalNetworkMapping",
    "Connectome": "HCP_S1200"
}
```

## Customizing output

### Custom pipeline name

```python
export_bids_derivatives(
    subject_data=result,
    output_dir="derivatives/my-pipeline",
    pipeline_name="my-custom-analysis"
)
```

### Selective export

```python
export_bids_derivatives(
    subject_data=result,
    output_dir="derivatives/lacuna",
    include_intermediates=False,  # Skip intermediate files
    overwrite=True                # Replace existing files
)
```

## Batch export

For multiple subjects:

```python
for result in results:
    export_bids_derivatives(
        subject_data=result,
        output_dir="derivatives/lacuna"
    )
```

## BIDS validation

Validate your derivatives with the BIDS validator:

```bash
# Install bids-validator
npm install -g bids-validator

# Validate derivatives
bids-validator derivatives/lacuna --ignoreNiftiHeaders
```

## Tips

!!! tip "Version tracking"
    
    The dataset_description.json automatically includes the Lacuna version
    used for analysis, ensuring reproducibility.

!!! tip "Integration with other tools"
    
    BIDS derivatives can be easily read by tools like fMRIPrep, Nilearn,
    and other BIDS-aware neuroimaging software.

## Troubleshooting

??? question "Missing metadata in sidecar"
    
    Ensure your SubjectData includes complete metadata:
    
    ```python
    subject = SubjectData(
        mask_img=mask_img,
        space="MNI152NLin6Asym",
        resolution=2.0,
        metadata={
            "subject_id": "sub-001",
            "session": "ses-01",
            "task": "rest"
        }
    )
    ```

??? question "File already exists error"
    
    Use `overwrite=True`:
    
    ```python
    export_bids_derivatives(result, output_dir, overwrite=True)
    ```
