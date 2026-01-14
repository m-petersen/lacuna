# Bundled Reference Atlases

This directory contains lightweight neuroimaging atlases bundled with the Lacuna for zero-configuration usage.

## Purpose

These atlases enable:
- **Immediate usage**: No download or configuration required
- **Reproducibility**: Version-controlled reference data
- **Testing**: Reliable test fixtures without external dependencies
- **Documentation**: Working examples out-of-the-box

## Atlas Files

Each atlas uses BIDS-style naming:
- **`tpl-<space>_res-<resolution>_atlas-<name>_desc-<description>_dseg.nii.gz`**: NIfTI image file with discrete region labels
- **`tpl-<space>_res-<resolution>_atlas-<name>_desc-<description>_probseg.nii.gz`**: NIfTI image file with probabilistic values (4D)
- **`..._labels.txt`**: Text file mapping region IDs to anatomical names

## Included Atlases

### HCP1065 White Matter Tracts (Atlas)
- **File**: `tpl-MNI152Nlin2009aAsym_res-01_atlas-HCP1065_desc-thr0p1_probseg.nii.gz`
- **Space**: MNI152NLin2009aAsym (1mm)
- **Type**: Probabilistic (4D)
- **Coverage**: White matter tracts
- **Threshold**: 0.1 (10% probability threshold applied)
- **Source**: Human Connectome Project
- **Citation**: Yeh et al. (2022), Nature Communications, 22;13(1):4933
- **License**: CC BY-SA 4.0

### Schaefer 2018 Atlas - 100 Parcels (7 Networks)
- **File**: `tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-100Parcels7Networks_dseg.nii.gz`
- **Space**: MNI152NLin6Asym (1mm)
- **Type**: Discrete labels (3D)
- **Coverage**: Cerebral cortex
- **Networks**: 7 networks (Visual, Somatomotor, Dorsal Attention, Ventral Attention, Limbic, Frontoparietal, Default)
- **Citation**: Schaefer et al. (2018), Cerebral Cortex, 28(9), 3095-3114
- **License**: MIT

### Schaefer 2018 Atlas - 200 Parcels (7 Networks)
- **File**: `tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.nii.gz`
- **Space**: MNI152NLin6Asym (1mm)
- **Type**: Discrete labels (3D)
- **Coverage**: Cerebral cortex (medium parcellation)
- **Networks**: 7 networks
- **Citation**: Schaefer et al. (2018), Cerebral Cortex, 28(9), 3095-3114
- **License**: MIT

### Schaefer 2018 Atlas - 400 Parcels (7 Networks)
- **File**: `tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-400Parcels7Networks_dseg.nii.gz`
- **Space**: MNI152NLin6Asym (1mm)
- **Type**: Discrete labels (3D)
- **Coverage**: Cerebral cortex (fine parcellation)
- **Networks**: 7 networks
- **Citation**: Schaefer et al. (2018), Cerebral Cortex, 28(9), 3095-3114
- **License**: MIT

### Schaefer 2018 Atlas - 1000 Parcels (7 Networks)
- **File**: `tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-1000Parcels7Networks_dseg.nii.gz`
- **Space**: MNI152NLin6Asym (1mm)
- **Type**: Discrete labels (3D)
- **Coverage**: Cerebral cortex (very fine parcellation)
- **Networks**: 7 networks
- **Citation**: Schaefer et al. (2018), Cerebral Cortex, 28(9), 3095-3114
- **License**: MIT

### Tian Subcortical Atlas - Scale 1
- **File**: `tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS1_dseg.nii.gz`
- **Space**: MNI152NLin6Asym (1mm)
- **Type**: Discrete labels (3D)
- **Coverage**: Subcortical structures (coarse parcellation)
- **Regions**: 16 regions
- **Citation**: Tian et al. (2020), Nature Neuroscience, 23, 1516-1528
- **License**: Permissive with attribution

### Tian Subcortical Atlas - Scale 2
- **File**: `tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS2_dseg.nii.gz`
- **Space**: MNI152NLin6Asym (1mm)
- **Type**: Discrete labels (3D)
- **Coverage**: Subcortical structures (medium parcellation)
- **Regions**: 32 regions
- **Citation**: Tian et al. (2020), Nature Neuroscience, 23, 1516-1528
- **License**: Permissive with attribution

### Tian Subcortical Atlas - Scale 3
- **File**: `tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS3_dseg.nii.gz`
- **Space**: MNI152NLin6Asym (1mm)
- **Type**: Discrete labels (3D)
- **Coverage**: Subcortical structures (fine parcellation)
- **Regions**: 54 regions
- **Citation**: Tian et al. (2020), Nature Neuroscience, 23, 1516-1528
- **License**: Permissive with attribution

## Usage

### Automatic (Default)
```python
from lacuna.analysis import RegionalDamage

# Uses bundled atlases automatically
analysis = RegionalDamage()
result = analysis.run(lesion_data)
```

### List Available Atlases
```python
from lacuna.data import list_bundled_atlases

atlases = list_bundled_atlases()
print(atlases)
# ['schaefer2018-100parcels-7networks', ...]
```

### Get Atlas Path
```python
from lacuna.data import get_bundled_atlas

img_path, labels_path = get_bundled_atlas('schaefer2018-100parcels-7networks')
print(img_path)  # Full path to schaefer2018-100parcels-7networks.nii.gz
print(labels_path)  # Full path to schaefer2018-100parcels-7networks_labels.txt
```

### Get Citation
```python
from lacuna.data import get_atlas_citation

citation = get_atlas_citation('schaefer2018-100parcels-7networks')
print(citation)
# Prints full citation information
```

## Custom Atlases

You can still use your own atlas directory:

```python
from lacuna.analysis import RegionalDamage

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

## References

1. **Schaefer 2018**: Schaefer, A., Kong, R., Gordon, E.M., et al. (2018). Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI. *Cerebral Cortex*, 28(9), 3095-3114. https://doi.org/10.1093/cercor/bhx179

2. **Tian Subcortical Atlas**: Tian, Y., Margulies, D.S., Breakspear, M., & Zalesky, A. (2020). Topographic organization of the human subcortex unveiled with functional connectivity gradients. *Nature Neuroscience*, 23, 1516-1528. https://doi.org/10.1038/s41593-020-00711-6

3. **HCP1065**: Yeh, F.-C., (2022). Population-based tract-to-region connectome of the human brain and its hierarchical topology. *Nature communications*, 22;13(1):4933. https://doi.org/10.1038/s41467-022-32595-4. Data were provided by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research.

## Size Information

- **Total size**: ~2.3 MB (compressed)
- **HCP1065**: ~1.1 MB
- **Schaefer-100**: ~229 KB
- **Schaefer-200**: ~253 KB
- **Schaefer-400**: ~289 KB
- **Schaefer-1000**: ~353 KB
- **Tian S1-S3**: ~27 KB total

This is negligible compared to typical Python package sizes and enables a much better user experience.
