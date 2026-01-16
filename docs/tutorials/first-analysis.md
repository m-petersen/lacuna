# Your First Analysis

In this tutorial, you'll run your first lesion network mapping analysis with Lacuna.

**Time required**: ~20 minutes

**What you'll learn**:

- Run functional lesion network mapping (fLNM)
- Interpret the output connectivity map
- Save results to disk

## Prerequisites

- Completed the [Getting Started](getting-started.md) tutorial
- A binary lesion mask in MNI space
- Basic Python knowledge

## Overview

Lesion Network Mapping (LNM) identifies brain networks that are functionally or structurally connected to a lesion location. This helps researchers understand:

- Which brain regions are affected by a lesion
- Network-level consequences of focal brain damage
- Potential relationships between lesion location and symptoms

## Step 1: Set up your analysis

First, let's import the necessary modules and load a lesion mask:

```python
import nibabel as nib
from lacuna import SubjectData
from lacuna.analysis import FunctionalNetworkMapping

# Load your lesion mask
mask_img = nib.load("path/to/lesion.nii.gz")
subject = SubjectData(
    mask_img=mask_img,
    space="MNI152NLin6Asym",
    resolution=2.0,
    metadata={"subject_id": "sub-001"}
)

print(f"Loaded lesion for {subject.metadata['subject_id']}")
print(f"Lesion volume: {subject.get_volume_mm3():.1f} mm³")
```

## Step 2: Register a connectome

Functional LNM requires a normative functional connectome. You'll need to register one before running the analysis:

```python
from lacuna.assets.connectomes import register_functional_connectome

# Register your connectome (download instructions in How-to guides)
register_functional_connectome(
    name="HCP_S1200",
    space="MNI152NLin6Asym",
    resolution=2.0,
    data_path="/path/to/connectome/directory",
    n_subjects=1000
)
```

!!! note "Getting connectomes"
    
    See the How-to Guides section for the Fetch Connectomes guide with
    instructions on downloading pre-built normative connectomes.

## Step 3: Run functional network mapping

Now let's run the analysis:

```python
# Initialize the analysis
fnm = FunctionalNetworkMapping(
    connectome_name="HCP_S1200",
    method="boes"  # Default method
)

# Run analysis
result = fnm.run(subject)
print("Analysis complete!")
```

The analysis computes functional connectivity between your lesion and every voxel in the brain using the normative connectome.

## Step 4: Access the results

Results are stored in the `results` dictionary:

```python
# Get the connectivity map
conn_map = result.results["FunctionalNetworkMapping"]["correlation_map"]

# Check the output
print(f"Result type: {type(conn_map)}")
print(f"Data shape: {conn_map.get_data().shape}")
print(f"Value range: [{conn_map.get_data().min():.3f}, {conn_map.get_data().max():.3f}]")
```

The connectivity map shows correlation values:

- **Positive values**: Regions positively correlated with the lesion location
- **Negative values**: Regions negatively correlated with the lesion location
- **Near-zero values**: Regions with little functional connection to the lesion

## Step 5: Save your results

Save the connectivity map as a NIfTI file:

```python
# Get the underlying NIfTI image
conn_nifti = conn_map.to_nifti()

# Save to disk
conn_nifti.to_filename("output/sub-001_connectivity.nii.gz")
print("Saved connectivity map!")
```

You can view this file in any neuroimaging viewer (FSLeyes, MRIcron, etc.).

## Step 6: Export BIDS derivatives (optional)

For reproducibility, export results in BIDS format:

```python
from lacuna.io.bids import export_bids_derivatives

export_bids_derivatives(
    subject_data=result,
    output_dir="derivatives/lacuna"
)
print("Exported BIDS derivatives!")
```

This creates standardized output files with proper naming conventions.

## Understanding the output

### Interpreting connectivity maps

The connectivity map represents "functional connectivity fingerprint" of your lesion:

| Value Range | Interpretation |
|-------------|----------------|
| 0.5 to 1.0  | Strong positive connectivity |
| 0.1 to 0.5  | Moderate positive connectivity |
| -0.1 to 0.1 | Weak or no connectivity |
| -0.5 to -0.1 | Moderate negative connectivity |
| -1.0 to -0.5 | Strong negative connectivity |

### Provenance tracking

Lacuna automatically tracks all processing steps:

```python
# View processing history
for step in result.provenance:
    print(f"- {step['operation']}: {step.get('details', '')}")
```

## What's next?

Now that you've run your first analysis, explore more in the How-to Guides:

- **Run Structural LNM** — Tractography-based network mapping
- **Regional Damage Analysis** — Quantify damage by brain region
- **Batch Processing** — Process multiple subjects

## Troubleshooting

??? question "Error: Connectome not found"
    
    Make sure you've registered your connectome before running the analysis.
    See the How-to Guides section for the Fetch Connectomes guide.

??? question "My connectivity values seem wrong"
    
    Check that your lesion mask is:
    
    1. Binary (0s and 1s only)
    2. In the correct coordinate space (MNI152NLin6Asym)
    3. At the correct resolution (matching the connectome)

??? question "Analysis is very slow"
    
    Functional LNM can be memory-intensive. Try:
    
    1. Using a lower resolution (4mm instead of 2mm)
    2. Closing other applications to free RAM
    3. Running on a machine with more memory
