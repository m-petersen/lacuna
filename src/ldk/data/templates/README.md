# MNI152 Reference Templates

This directory contains reference MNI152 T1-weighted brain templates used throughout the lesion_decoding_toolkit package.

## Templates

### MNI152_T1_1mm.nii.gz
- **Resolution**: 1mm isotropic
- **Dimensions**: 182 x 218 x 182
- **Source**: FSL MNI152 T1 1mm template
- **Use**: Reference for 1mm resolution analyses

### MNI152_T1_2mm.nii.gz
- **Resolution**: 2mm isotropic
- **Dimensions**: 91 x 109 x 91
- **Source**: FSL MNI152 T1 2mm template
- **Use**: Reference for 2mm resolution analyses

## Usage

Templates are automatically loaded by the toolkit:

```python
from ldk.data import get_mni_template

# Get 2mm template
template_2mm = get_mni_template(resolution=2)

# Get 1mm template
template_1mm = get_mni_template(resolution=1)
```

## Source

These templates are from the FSL MNI152 standard space:
- ICBM 152 Nonlinear 6th Generation Symmetric Average Brain Stereotaxic Registration Model
- Based on 152 T1-weighted MRI scans from the ICBM database
- More info: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases

## License

These templates are distributed under the FSL license and are freely available for research purposes.
