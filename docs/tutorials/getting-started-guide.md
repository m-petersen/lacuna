# Getting Started

This tutorial guides you through installing Lacuna and running your first lesion network mapping analysis.

**Time required**: ~15 minutes

**What you'll learn**:

- Install Lacuna and its dependencies
- Load a lesion mask from a NIfTI file
- Verify your installation is working correctly

## Prerequisites

- Python 3.10 or higher
- pip package manager
- A terminal or command-line interface

## Step 1: Install Lacuna

### Option A: Install from PyPI (recommended)

```bash
pip install lacuna
```

### Option B: Install from source

```bash
git clone https://github.com/lacuna/lacuna.git
cd lacuna
pip install -e .
```

### Verify installation

```bash
python -c "import lacuna; print(f'Lacuna {lacuna.__version__} installed successfully!')"
```

## Step 2: Get the tutorial data

Lacuna includes a synthetic BIDS dataset with lesion masks for learning. You can access it directly or copy it to your working directory:

=== "Direct access (recommended)"

    ```python
    from lacuna.data import get_tutorial_bids_dir, get_tutorial_subjects
    
    # Get path to bundled tutorial data
    bids_dir = get_tutorial_bids_dir()
    print(f"Tutorial data: {bids_dir}")
    
    # List available subjects
    subjects = get_tutorial_subjects()
    print(f"Subjects: {subjects}")
    ```

=== "Copy to working directory"

    ```python
    from lacuna.data import setup_tutorial_data
    
    # Copy tutorial data to your own directory
    tutorial_dir = setup_tutorial_data("./my_tutorial_data")
    print(f"Tutorial data copied to: {tutorial_dir}")
    ```

!!! info "About the tutorial data"
    
    The synthetic dataset includes 3 subjects with binary lesion masks in
    MNI152NLin6 space (1mm resolution). These are artificial lesions designed
    for learning purposes only.

## Step 3: Load your first mask

Now let's load a lesion mask from the tutorial data:

```python
from lacuna.data import get_subject_mask_path
from lacuna.core import SubjectData

# Get path to a subject's mask
mask_path = get_subject_mask_path("sub-01")
print(f"Loading: {mask_path}")

# Load as SubjectData (validates binary mask automatically)
subject = SubjectData.from_nifti(
    mask_path,
    space="MNI152NLin6Asym",
    metadata={"subject_id": "sub-01", "session_id": "ses-01"}
)
print(f"Loaded successfully!")
print(f"Subject ID: {subject.metadata['subject_id']}")
```

!!! tip "What's SubjectData?"
    
    `SubjectData` is Lacuna's core container for lesion data. It automatically
    validates that your mask is binary (0 and 1 values only) and tracks
    metadata like coordinate space and subject ID.

## Step 4: Inspect your data

Let's explore the mask properties:

```python
import numpy as np

# Get the mask data
mask_data = subject.mask_img.get_fdata()

# Basic properties
print(f"Shape: {mask_data.shape}")
print(f"Total voxels: {np.prod(mask_data.shape)}")
print(f"Lesion voxels: {int((mask_data > 0).sum())}")

# Calculate volume in mm³
voxel_sizes = subject.mask_img.header.get_zooms()
voxel_volume = np.prod(voxel_sizes)
lesion_volume = (mask_data > 0).sum() * voxel_volume
print(f"Lesion volume: {lesion_volume:.1f} mm³")
```

## Step 5: Load all tutorial subjects

Let's load all three subjects from the tutorial dataset:

```python
from lacuna.data import get_tutorial_subjects, get_subject_mask_path
from lacuna.core import SubjectData

subjects = []
for sub_id in get_tutorial_subjects():
    mask_path = get_subject_mask_path(sub_id)
    subject = SubjectData.from_nifti(
        mask_path,
        space="MNI152NLin6Asym",
        metadata={"subject_id": sub_id}
    )
    
    voxels = int((subject.mask_img.get_fdata() > 0).sum())
    print(f"{sub_id}: {voxels} lesion voxels")
    subjects.append(subject)

print(f"\nLoaded {len(subjects)} subjects")
```

!!! tip "SubjectData immutability"
    
    `SubjectData` is immutable — operations return new instances rather than
    modifying the original. This ensures reproducibility and safe parallel
    processing. See the [SubjectData Design](../explanation/subject-data.md)
    explanation for details.

## What's next?

Now that you can load data, you're ready to run your first analysis!

Continue to [Your First Analysis](first-analysis.md) to learn how to run
functional lesion network mapping.

## Troubleshooting

??? question "ImportError: No module named 'lacuna'"
    
    Make sure you installed Lacuna in the same Python environment you're using:
    
    ```bash
    pip install --upgrade lacuna
    ```

??? question "My mask has the wrong coordinate space"
    
    Lacuna requires masks in MNI space. If your mask is in native space, you'll
    need to transform it first using tools like ANTs or FSL.

??? question "How do I check my Python version?"
    
    ```bash
    python --version
    ```
    
    Lacuna requires Python 3.10 or higher.
