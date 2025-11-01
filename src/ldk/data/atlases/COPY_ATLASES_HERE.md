# Instructions: Copy Your Atlas Files Here

## Directory
Copy your atlas files to: `src/ldk/data/atlases/`

## Required File Format

For each atlas, you need TWO files:

### 1. NIfTI Image
- **Name**: `<atlas_name>.nii.gz` (must be compressed)
- **Format**: Standard NIfTI format
- **Space**: Any standard space (MNI152 recommended)
- **Type**: 
  - 3D: Integer labels (1, 2, 3, ...) for discrete atlases
  - 4D: Probabilistic values (0.0-1.0) for probabilistic atlases

### 2. Labels File
- **Name**: `<atlas_name>_labels.txt` (preferred) or `<atlas_name>.txt`
- **Format**: Plain text, one region per line
- **Structure**: `<region_id> <region_name>`

**Example labels file:**
```
0 Background
1 Left_Precentral_Gyrus
2 Right_Precentral_Gyrus
3 Left_Superior_Frontal_Gyrus
4 Right_Superior_Frontal_Gyrus
```

## Naming Convention

Use descriptive names without spaces:
- ✅ Good: `aal3.nii.gz`
- ✅ Good: `harvard-oxford-cortical.nii.gz`
- ✅ Good: `schaefer2018-400parcels-7networks.nii.gz`
- ❌ Bad: `my atlas.nii.gz` (contains space)
- ❌ Bad: `AAL 3.nii.gz` (contains space)

## Recommended Atlases to Include

Based on common usage:

1. **Harvard-Oxford Cortical** (48 regions)
   - `harvard-oxford-cortical.nii.gz`
   - `harvard-oxford-cortical_labels.txt`

2. **AAL3** (170 regions)
   - `aal3.nii.gz`
   - `aal3_labels.txt`

3. **Schaefer 2018 - 100 Parcels**
   - `schaefer2018-100parcels-7networks.nii.gz`
   - `schaefer2018-100parcels-7networks_labels.txt`

4. **Schaefer 2018 - 400 Parcels**
   - `schaefer2018-400parcels-7networks.nii.gz`
   - `schaefer2018-400parcels-7networks_labels.txt`

5. **HCP1065** (white matter tracts - if you have it)
   - `HCP1065_thr0p1.nii.gz`
   - `HCP1065_thr0p1_labels.txt`

## After Copying

1. **Verify files are detected:**
   ```bash
   python -c "from ldk.data import list_bundled_atlases; print(list_bundled_atlases())"
   ```

2. **Test with a specific atlas:**
   ```bash
   python -c "from ldk.data import get_bundled_atlas; print(get_bundled_atlas('aal3'))"
   ```

3. **Run the tests:**
   ```bash
   pytest tests/unit/test_bundled_atlases.py -v
   ```

4. **Try zero-config usage:**
   ```python
   from ldk.analysis import RegionalDamage
   from ldk import LesionData
   
   # Load your lesion
   lesion = LesionData.from_nifti("path/to/lesion.nii.gz")
   
   # Run analysis without specifying atlas_dir!
   analysis = RegionalDamage()
   result = analysis.run(lesion)
   ```

## Size Recommendations

Keep bundled atlases lightweight:
- ✅ Under 1MB per atlas: Excellent
- ✅ 1-2MB per atlas: Good
- ⚠️ 2-5MB per atlas: Acceptable
- ❌ Over 5MB per atlas: Consider making it optional/downloadable

Total recommended size: **3-10 MB** for all bundled atlases

## What to Copy From

If you have a directory like `/data/atlases/` with your atlas files, just copy them:

```bash
# Example command (adjust paths as needed)
cp /path/to/your/atlases/*.nii.gz src/ldk/data/atlases/
cp /path/to/your/atlases/*_labels.txt src/ldk/data/atlases/
```

The system will automatically discover them!
