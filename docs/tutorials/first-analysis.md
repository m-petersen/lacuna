# Your First Analysis

In this tutorial, you'll run your first lesion network mapping analysis with Lacuna.

**Time required**: ~20 minutes

**What you'll learn**:

- Load subjects from the tutorial dataset
- Run functional lesion network mapping (fLNM)
- Interpret the output connectivity map
- Save results to disk

## Prerequisites

- Completed the [Getting Started](getting-started.ipynb) tutorial
- Basic Python knowledge

## Overview

Lesion Network Mapping (LNM) identifies brain networks that are functionally or structurally connected to a lesion location. This helps researchers understand:

- Which brain regions are affected by a lesion
- Network-level consequences of focal brain damage
- Potential relationships between lesion location and symptoms

## Step 1: Load the tutorial data

Let's start by loading a subject from the bundled tutorial dataset:

```python
from lacuna.core import SubjectData
from lacuna.data import get_subject_mask_path, get_tutorial_subjects
import numpy as np

# See available subjects
subjects = get_tutorial_subjects()
print(f"Available subjects: {subjects}")

# Load the first subject
mask_path = get_subject_mask_path("sub-01")
subject = SubjectData.from_nifti(
    mask_path,
    space="MNI152NLin6Asym",
    metadata={"subject_id": "sub-01", "session_id": "ses-01"}
)

lesion_voxels = int((subject.mask_img.get_fdata() > 0).sum())
print(f"Loaded: {subject.metadata['subject_id']}")
print(f"Lesion voxels: {lesion_voxels}")
```

## Step 2: Fetch a connectome

Functional LNM requires a normative functional connectome. Let's fetch the HCP connectome:

```python
from lacuna.cli.fetch_cmd import fetch_connectome

# Download the HCP functional connectome (first time only)
fetch_connectome(connectome_type="functional", name="HCP_S1200")
```

!!! note "First-time download"
    
    The connectome download is ~2GB and only needs to be done once. Subsequent
    runs will use the cached data. See the [Fetch Connectomes](../how-to/fetch-connectomes.md)
    guide for more options.

## Step 3: Run functional network mapping

## Step 3: Run functional network mapping

Now let's run the analysis using Lacuna's `analyze` function:

```python
from lacuna import analyze

# Run functional LNM
subject = analyze(subject, analysis="flnm", connectome="HCP_S1200")
print("Analysis complete!")

# Check results
print(f"Results available: {list(subject.results.keys())}")
```

The analysis computes functional connectivity between your lesion and every voxel in the brain using the normative connectome.

## Step 4: Access the results

Results are stored in the subject's `results` dictionary:

```python
# Get the connectivity map
flnm_result = subject.results["flnm"]

# Check the output
print(f"Result type: {type(flnm_result)}")
print(f"Data shape: {flnm_result.shape}")

# Get value range
import numpy as np
data = flnm_result.get_fdata()
print(f"Value range: [{data.min():.3f}, {data.max():.3f}]")
```

The connectivity map shows correlation values:

- **Positive values**: Regions positively correlated with the lesion location
- **Negative values**: Regions negatively correlated with the lesion location
- **Near-zero values**: Regions with little functional connection to the lesion

## Step 5: Save your results

Save the connectivity map as a NIfTI file:

```python
import nibabel as nib
from pathlib import Path

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Save the connectivity map
output_path = output_dir / f"{subject.subject_id}_desc-flnm_connectivity.nii.gz"
nib.save(flnm_result, output_path)
print(f"Saved: {output_path}")
```

You can view this file in any neuroimaging viewer (FSLeyes, MRIcron, etc.).

## Step 6: Process multiple subjects

Let's run the analysis on all tutorial subjects:

```python
from lacuna.core import SubjectData
from lacuna.data import get_tutorial_subjects, get_subject_mask_path
from lacuna import analyze
import numpy as np

results = []
for subject_id in get_tutorial_subjects():
    # Load subject
    mask_path = get_subject_mask_path(subject_id)
    subject = SubjectData.from_nifti(
        mask_path,
        space="MNI152NLin6Asym",
        metadata={"subject_id": subject_id}
    )
    
    # Run analysis
    subject = analyze(subject, analysis="flnm", connectome="HCP_S1200")
    results.append(subject)
    
    print(f"Completed: {subject_id}")

print(f"\nProcessed {len(results)} subjects")
```

## Understanding the output

### Interpreting connectivity maps

The connectivity map represents the "functional connectivity fingerprint" of your lesion:

| Value Range | Interpretation |
|-------------|----------------|
| 0.5 to 1.0  | Strong positive connectivity |
| 0.1 to 0.5  | Moderate positive connectivity |
| -0.1 to 0.1 | Weak or no connectivity |
| -0.5 to -0.1 | Moderate negative connectivity |
| -1.0 to -0.5 | Strong negative connectivity |

### What the results mean

High connectivity values indicate brain regions that are functionally "wired" to your lesion location in the normative population. These regions may show:

- Similar activation patterns during rest
- Co-activation during tasks
- Potential remote effects of the lesion

## What's next?

Now that you've run your first analysis, explore more in the How-to Guides:

- [Run Structural LNM](../how-to/run-slnm.md) — Tractography-based network mapping
- [Regional Damage Analysis](../how-to/regional-damage.md) — Quantify damage by brain region
- [Batch Processing](../how-to/batch-processing.md) — Process many subjects efficiently

## Troubleshooting

??? question "Error: Connectome not found"
    
    Make sure you've fetched the connectome before running the analysis:
    
    ```bash
    lacuna fetch --connectome functional --name HCP_S1200
    ```

??? question "My connectivity values seem wrong"
    
    Check that your lesion mask is:
    
    1. Binary (0s and 1s only)
    2. In the correct coordinate space (MNI152NLin6)
    3. Contains valid lesion voxels

??? question "Analysis is very slow"
    
    Functional LNM can be memory-intensive. Try:
    
    1. Running on a machine with more RAM
    2. Processing fewer subjects at once
    3. Using Docker/Apptainer on an HPC cluster
