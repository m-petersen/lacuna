# Using Bundled MNI152 Templates

The lacuna package includes bundled MNI152 reference templates for reproducibility and offline usage.

## Quick Start

```python
from lacuna.data import get_mni_template

# Load 2mm template (default)
template_2mm = get_mni_template(resolution=2)
print(template_2mm.shape)  # (91, 109, 91)

# Load 1mm template
template_1mm = get_mni_template(resolution=1)
print(template_1mm.shape)  # (182, 218, 182)
```

## Installation

### Option 1: Automatic Download (Recommended)

Run the included download script:

```bash
python download_templates.py
```

This script will:
1. Try to copy templates from your FSL installation (if available)
2. Fall back to downloading via nilearn
3. Place templates in `src/lacuna/data/templates/`

### Option 2: Manual Installation

If you have FSL installed:

```bash
# Copy from FSL
cp $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz src/lacuna/data/templates/
cp $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz src/lacuna/data/templates/
```

Or download directly:
1. Get templates from http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009
2. Place in `src/lacuna/data/templates/`

## API Reference

### `get_mni_template(resolution=2)`

Load MNI152 T1-weighted template at specified resolution.

**Parameters:**
- `resolution` (int): 1 or 2 (mm). Default: 2

**Returns:**
- `nibabel.Nifti1Image`: Brain template

**Raises:**
- `ValueError`: If resolution not 1 or 2
- `FileNotFoundError`: If template file missing

**Example:**
```python
from lacuna.data import get_mni_template

# Get default 2mm template
template = get_mni_template()

# Use with analysis
from lacuna.analysis import StructuralNetworkMapping
analysis = StructuralNetworkMapping(
    tractogram_path="tractogram.tck",
    whole_brain_tdi="tdi.nii.gz"
    # template auto-loaded based on lesion resolution
)
```

### `get_template_path(resolution=2)`

Get path to template file without loading it.

**Parameters:**
- `resolution` (int): 1 or 2 (mm). Default: 2

**Returns:**
- `Path`: Absolute path to template file

**Example:**
```python
from lacuna.data import get_template_path

path = get_template_path(resolution=2)
print(path)  # /path/to/ldk/data/templates/MNI152_T1_2mm.nii.gz
print(path.exists())  # True
```

### `list_templates()`

List all available templates with metadata.

**Returns:**
- `dict`: Template information

**Example:**
```python
from lacuna.data import list_templates

templates = list_templates()
for res, info in templates.items():
    print(f"{res}:")
    print(f"  Path: {info['path']}")
    print(f"  Shape: {info['shape']}")
    print(f"  Exists: {info['exists']}")
```

## Template Specifications

### MNI152_T1_1mm.nii.gz
- **Dimensions:** 182 × 218 × 182 voxels
- **Resolution:** 1mm isotropic
- **Size:** ~15 MB compressed
- **Use case:** High-resolution analyses, 1mm lesions

### MNI152_T1_2mm.nii.gz
- **Dimensions:** 91 × 109 × 91 voxels
- **Resolution:** 2mm isotropic
- **Size:** ~2 MB compressed
- **Use case:** Standard resolution analyses, 2mm lesions

## Integration with Analysis Modules

The templates are automatically used by analysis modules:

```python
from lacuna import LesionData
from lacuna.analysis import StructuralNetworkMapping

# Load lesion (must specify space)
lesion = LesionData.from_nifti(
    "lesion.nii.gz",
    metadata={"subject_id": "sub-001", "space": "MNI152_2mm"}
)

# Create analysis - template auto-loaded
analysis = StructuralNetworkMapping(
    tractogram_path="tractogram.tck",
    whole_brain_tdi="tdi_2mm.nii.gz"
    # No template needed! Auto-detects from lesion.metadata["space"]
)

# Run analysis
result = analysis.run(lesion)
```

The template resolution is automatically determined from `lesion.metadata["space"]`:
- `"MNI152_1mm"` → loads 1mm template
- `"MNI152_2mm"` → loads 2mm template

## Benefits of Bundled Templates

✅ **Reproducibility**: Everyone uses exact same templates
✅ **Offline Usage**: No internet required after installation
✅ **Speed**: No download delays during analysis
✅ **Version Control**: Templates versioned with package
✅ **No Dependencies**: Removes nilearn dependency for template loading

## Package Distribution

Templates are included in:
1. **Source distributions** (sdist) via MANIFEST.in
2. **Wheel distributions** via package_data in pyproject.toml

This ensures templates are available after:
```bash
pip install lesion-decoding-toolkit
```

## Testing Template Installation

```bash
# Run test script
python test_templates.py
```

This verifies:
- Templates exist and are loadable
- Correct dimensions
- Proper error handling
- Path resolution works

## Troubleshooting

### Templates Not Found

**Error:**
```
FileNotFoundError: MNI152 template not found at: .../templates/MNI152_T1_2mm.nii.gz
```

**Solution:**
```bash
python download_templates.py
```

### Import Error

**Error:**
```
ImportError: cannot import name 'get_mni_template' from 'lacuna.data'
```

**Solution:**
Reinstall package in development mode:
```bash
pip install -e .
```

### Wrong Dimensions

If template has unexpected dimensions, verify files:
```python
import nibabel as nib
from lacuna.data import get_template_path

path = get_template_path(resolution=2)
img = nib.load(path)
print(img.shape)  # Should be (91, 109, 91) for 2mm
```

## Source Information

Templates are from FSL's MNI152 standard space:
- **Full name**: ICBM 152 Nonlinear 6th Generation Symmetric Average Brain Stereotaxic Registration Model
- **Based on**: 152 T1-weighted MRI scans
- **Reference**: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases
- **License**: FSL license (free for research)
