# Bundled MNI Templates Implementation Summary

## What Was Done

Successfully implemented a bundled template system replacing nilearn dependency for MNI152 template loading.

## Files Created

### 1. **src/ldk/data/templates/** (directory)
   - Storage location for MNI152 templates
   - Location: Inside package data structure for distribution

### 2. **src/ldk/data/templates/README.md**
   - Documentation about included templates
   - Source information, dimensions, usage

### 3. **download_templates.py** (root directory)
   - Automatic template download script
   - Tries FSL first, falls back to nilearn
   - Places files in correct location

### 4. **test_templates.py** (root directory)
   - Test script to verify template loading
   - Tests all API functions
   - Validates dimensions and error handling

### 5. **docs/templates.md**
   - Comprehensive user documentation
   - API reference
   - Installation instructions
   - Troubleshooting guide

## Files Modified

### 1. **src/ldk/data/__init__.py**
   - Added `get_mni_template(resolution=2)` function
   - Added `get_template_path(resolution=2)` function
   - Added `list_templates()` function
   - Added to `__all__` exports

### 2. **src/ldk/analysis/structural_network_mapping.py**
   - Replaced `from nilearn.datasets import load_mni152_template`
   - With: `from ldk.data import get_mni_template`
   - Updated template loading logic (line ~178)
   - Updated docstring to reflect bundled templates

### 3. **pyproject.toml**
   - Updated `[tool.setuptools.package-data]` section
   - Added `"templates/*.nii.gz"` and `"templates/*.md"`
   - Ensures templates included in wheel distributions

### 4. **MANIFEST.in**
   - Added explicit includes for template files
   - Ensures templates included in source distributions

## Directory Structure

```
lesion_decoding_toolkit/
├── download_templates.py          # NEW: Download script
├── test_templates.py              # NEW: Test script
├── MANIFEST.in                    # MODIFIED: Added templates
├── pyproject.toml                 # MODIFIED: Added package data
├── docs/
│   └── templates.md               # NEW: Documentation
└── src/
    └── ldk/
        ├── data/
        │   ├── __init__.py        # MODIFIED: Added template functions
        │   └── templates/         # NEW: Template storage
        │       ├── README.md      # NEW: Template info
        │       ├── MNI152_T1_1mm.nii.gz   # NEEDS DOWNLOAD
        │       └── MNI152_T1_2mm.nii.gz   # NEEDS DOWNLOAD
        └── analysis/
            └── structural_network_mapping.py  # MODIFIED: Use bundled templates
```

## Next Steps (Required)

### 1. Download Templates
Run the download script to fetch the actual template files:

```bash
cd /home/marvin/projects/lesion_decoding_toolkit
python download_templates.py
```

This will:
- Try to copy from your FSL installation (you have FSL, right?)
- Or download via nilearn
- Place files in `src/ldk/data/templates/`

### 2. Test Template Loading
Verify everything works:

```bash
python test_templates.py
```

Should show:
- ✓ Both templates found
- ✓ Correct dimensions
- ✓ Proper loading

### 3. Update Notebook
The structural network mapping notebook should work without changes since we maintained API compatibility. But you may want to update comments:

**Old comment:**
```python
# Template is automatically loaded from nilearn (1mm or 2mm based on lesion)
```

**New comment:**
```python
# Template is automatically loaded from bundled package data (1mm or 2mm based on lesion)
```

### 4. Optional: Remove nilearn Dependency
If nilearn is only used for template loading, you can now remove it from dependencies:

**pyproject.toml:**
```toml
dependencies = [
    "nibabel>=5.0",
    "numpy>=1.24",
    "scipy>=1.10",
    # "nilearn>=0.10",  # No longer needed for templates!
    "pandas>=2.0",
    ...
]
```

**Note:** Only do this if nilearn is not used elsewhere in the package!

## API Usage Examples

### Before (nilearn):
```python
from nilearn.datasets import load_mni152_template
template = load_mni152_template(resolution=2)
```

### After (bundled):
```python
from ldk.data import get_mni_template
template = get_mni_template(resolution=2)
```

### In StructuralNetworkMapping:
```python
# Automatic - no changes needed!
analysis = StructuralNetworkMapping(
    tractogram_path="tractogram.tck",
    whole_brain_tdi="tdi.nii.gz"
    # template auto-loaded from bundled data
)
```

## Benefits

✅ **No External Dependencies**: Templates bundled with package
✅ **Reproducibility**: Everyone uses exact same templates
✅ **Offline Usage**: No internet required
✅ **Faster**: No download delays during analysis
✅ **Version Control**: Templates versioned with package
✅ **Distribution**: Included in pip installs automatically

## File Sizes

- **MNI152_T1_1mm.nii.gz**: ~15 MB
- **MNI152_T1_2mm.nii.gz**: ~2 MB
- **Total**: ~17 MB added to package

This is acceptable for a neuroimaging package where data files are expected.

## Testing Checklist

After downloading templates, test:

- [ ] `python test_templates.py` - all tests pass
- [ ] Import works: `from ldk.data import get_mni_template`
- [ ] Templates load: `template = get_mni_template(resolution=2)`
- [ ] Correct shapes: 1mm=(182,218,182), 2mm=(91,109,91)
- [ ] StructuralNetworkMapping works without template parameter
- [ ] Notebook runs successfully with new code

## Rollback Plan

If issues arise, revert these commits:
1. `git checkout HEAD~1 src/ldk/data/__init__.py`
2. `git checkout HEAD~1 src/ldk/analysis/structural_network_mapping.py`
3. `git checkout HEAD~1 pyproject.toml MANIFEST.in`
4. Restore nilearn import in structural_network_mapping.py

## Documentation Updates Needed

- [ ] Update README.md to mention bundled templates
- [ ] Update installation instructions (mention template download)
- [ ] Add to FAQ: "Where are the MNI templates stored?"
- [ ] Update contributing guide (how to update templates)

## Future Enhancements

Consider:
1. **More templates**: Add other standard spaces (Talairach, etc.)
2. **Atlases**: Bundle common atlases similarly
3. **Checksums**: Add MD5/SHA verification for template integrity
4. **Lazy loading**: Only load when needed to save memory
5. **Caching**: Cache loaded templates in memory for repeat use
