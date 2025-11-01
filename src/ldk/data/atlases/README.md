# Bundled Reference Atlases

This directory contains lightweight neuroimaging atlases bundled with the Lesion Decoding Toolkit for zero-configuration usage.

## Purpose

These atlases enable:
- **Immediate usage**: No download or configuration required
- **Reproducibility**: Version-controlled reference data
- **Testing**: Reliable test fixtures without external dependencies
- **Documentation**: Working examples out-of-the-box

## Atlas Files

Each atlas consists of two files:
- **`<atlas_name>.nii.gz`**: NIfTI image file with region labels or probabilities
- **`<atlas_name>_labels.txt`**: Text file mapping region IDs to anatomical names

## Included Atlases

### Schaefer 2018 Atlas - 100 Parcels (7 Networks)
- **File**: `schaefer2018-100parcels-7networks.nii.gz`
- **Space**: MNI152 1mm
- **Type**: Discrete labels (3D)
- **Coverage**: Cerebral cortex
- **Networks**: 7 networks (Visual, Somatomotor, Dorsal Attention, Ventral Attention, Limbic, Frontoparietal, Default)
- **Citation**: Schaefer et al. (2018), Cerebral Cortex, 28(9), 3095-3114
- **License**: Free for research use

### Schaefer 2018 Atlas - 400 Parcels (7 Networks)
- **File**: `schaefer2018-400parcels-7networks.nii.gz`
- **Space**: MNI152 1mm
- **Type**: Discrete labels (3D)
- **Coverage**: Cerebral cortex (finer parcellation)
- **Networks**: 7 networks (same as 100 parcels version)
- **Citation**: Schaefer et al. (2018), Cerebral Cortex, 28(9), 3095-3114
- **License**: Free for research use

## Usage

### Automatic (Default)
```python
from ldk.analysis import RegionalDamage

# Uses bundled atlases automatically
analysis = RegionalDamage()
result = analysis.run(lesion_data)
```

### List Available Atlases
```python
from ldk.data import list_bundled_atlases

atlases = list_bundled_atlases()
print(atlases)
# ['schaefer2018-100parcels-7networks', ...]
```

### Get Atlas Path
```python
from ldk.data import get_bundled_atlas

img_path, labels_path = get_bundled_atlas('aal3')
print(img_path)  # Full path to aal3.nii.gz
print(labels_path)  # Full path to aal3_labels.txt
```

### Get Citation
```python
from ldk.data import get_atlas_citation

citation = get_atlas_citation('aal3')
print(citation)
# Prints full citation information
```

## Custom Atlases

You can still use your own atlas directory:

```python
from ldk.analysis import RegionalDamage

# Use custom directory
analysis = RegionalDamage(atlas_dir="/path/to/my/atlases")

# Or mix bundled + custom
analysis = RegionalDamage(
    atlas_dir="/path/to/my/atlases",
    include_bundled=True
)
```

## Adding More Atlases

To add your own atlas to this directory:

1. Place NIfTI file: `<name>.nii.gz`
2. Create labels file: `<name>_labels.txt` with format:
   ```
   0 Background
   1 Region_Name_1
   2 Region_Name_2
   ...
   ```
3. The atlas will be automatically discovered

## File Format Requirements

### NIfTI Images
- **Format**: `.nii.gz` (compressed NIfTI)
- **Space**: Any standard space (MNI152 recommended)
- **3D atlases**: Integer labels (1, 2, 3, ...)
- **4D atlases**: Probabilistic values (0.0-1.0), one volume per region

### Labels Files
- **Format**: Plain text, one region per line
- **Structure**: `<id> <name>`
  - `<id>`: Integer region ID (matches atlas values)
  - `<name>`: Region name (can contain spaces)
- **Example**:
  ```
  0 Background
  1 Left_Precentral_Gyrus
  2 Right_Precentral_Gyrus
  ```

## Licenses and Attribution

All bundled atlases are available for non-commercial research use. If you use these atlases in a publication, please cite the original papers listed above.

For commercial use, please check the license terms of each atlas:
- **Harvard-Oxford**: Check FSL license
- **AAL3**: Contact atlas authors
- **Schaefer 2018**: Contact atlas authors

## References

1. **Schaefer 2018**: Schaefer, A., Kong, R., Gordon, E.M., et al. (2018). Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI. *Cerebral Cortex*, 28(9), 3095-3114. https://doi.org/10.1093/cercor/bhx179

## Size Information

- **Total size**: ~3-4 MB (compressed)
- **Schaefer-100**: ~600 KB
- **Schaefer-400**: ~1 MB

This is negligible compared to typical Python package sizes and enables a much better user experience.
