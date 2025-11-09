# MNI Templates - Implementation Complete ✅

## Summary

Successfully integrated FSL MNI152 templates into the package as bundled data.

## What Changed

### Files Modified
- **`src/ldk/data/__init__.py`**: Added `get_mni_template()`, `get_template_path()`, `list_templates()` functions
- **`src/ldk/analysis/structural_network_mapping.py`**: 
  - Removed `from nilearn.datasets import load_mni152_template`
  - Added `from ldk.data import get_mni_template`
  - Updated `_validate_inputs()` to load templates as nibabel images
- **`pyproject.toml`**: Added templates to package_data
- **`MANIFEST.in`**: Added templates to source distribution

### Templates Included
- ✅ `src/ldk/data/templates/MNI152_T1_1mm.nii.gz` (182×218×182)
- ✅ `src/ldk/data/templates/MNI152_T1_2mm.nii.gz` (91×109×91)
- ✅ `src/ldk/data/templates/README.md`

## Testing Results

✅ Template loading works:
```python
from ldk.data import get_mni_template
template = get_mni_template(resolution=2)  # (91, 109, 91)
```

✅ StructuralNetworkMapping auto-loads templates:
```python
analysis = StructuralNetworkMapping(
    tractogram_path="...",
    whole_brain_tdi="..."
    # template automatically loaded based on lesion space
)
```

## Benefits

1. **No External Downloads**: Templates bundled with package
2. **Reproducibility**: Everyone uses same templates
3. **Offline Usage**: No internet required
4. **Faster**: No download delays
5. **Distribution**: Included in pip installs

## Package Size

- 1mm template: ~15 MB
- 2mm template: ~2 MB
- **Total added**: ~17 MB

## API

```python
from ldk.data import get_mni_template, list_templates

# Load template
template_2mm = get_mni_template(resolution=2)
template_1mm = get_mni_template(resolution=1)

# List available
templates = list_templates()
# {'1mm': {'path': ..., 'shape': (182,218,182), 'exists': True}, 
#  '2mm': {'path': ..., 'shape': (91,109,91), 'exists': True}}
```

## Note on nilearn

nilearn is still required for:
- Image resampling (`nilearn.image.resample_to_img`)
- Atlas maskers (`nilearn.maskers.NiftiLabelsMasker`)

Only template loading was replaced with bundled data.

## Complete ✅

The implementation is complete and tested. Templates are ready for distribution with the package.
