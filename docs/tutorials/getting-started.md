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

## Step 2: Load your first mask

Lacuna works with binary brain masks in NIfTI format. Let's load one:

```python
import nibabel as nib
from lacuna import SubjectData

# Load a NIfTI file (replace with your file path)
mask_img = nib.load("path/to/your/mask.nii.gz")

# Create a SubjectData object
subject = SubjectData(
    mask_img=mask_img,
    space="MNI152NLin6Asym",  # Coordinate space
    resolution=2.0,           # Resolution in mm
    metadata={"subject_id": "sub-001"}
)
```

 pkill -f "mkdocs serve" 2>/dev/null; echo "Server stopped"! tip "What's SubjectData?"
    
    `SubjectData` is Lacuna's core data container. It holds your brain mask along
    with metadata about coordinate space, resolution, and processing history.

## Step 3: Inspect your data

Now let's verify the data loaded correctly:

```python
# Check basic properties
print(f"Subject ID: {subject.metadata.get('subject_id', 'unknown')}")
print(f"Coordinate space: {subject.space}")
print(f"Resolution: {subject.resolution} mm")

# Calculate mask statistics
volume_mm3 = subject.get_volume_mm3()
print(f"Mask volume: {volume_mm3:.1f} mm³")

# Get the underlying NumPy array
mask_array = subject.mask_data.get_data()
print(f"Data shape: {mask_array.shape}")
print(f"Non-zero voxels: {(mask_array > 0).sum()}")
```

## Step 4: Validate your mask

Lacuna expects binary masks (values of 0 and 1 only). Let's verify:

```python
import numpy as np

mask_data = subject.mask_data.get_data()
unique_values = np.unique(mask_data)
print(f"Unique values: {unique_values}")

# Check if binary
is_binary = set(unique_values).issubset({0, 1})
print(f"Is binary mask: {is_binary}")
```

 pkill -f "mkdocs serve" 2>/dev/null; echo "Server stopped"! warning "Non-binary masks"
    
    If your mask contains values other than 0 and 1, you'll need to binarize it
    before running analyses. This typically happens with probability maps or
    lesion segmentation outputs.

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
