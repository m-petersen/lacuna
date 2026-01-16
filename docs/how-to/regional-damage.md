# Quantify Regional Damage

This guide shows how to quantify lesion damage across brain atlas regions using Lacuna's Regional Damage analysis.

## Goal

Calculate the proportion of each brain region (parcel) affected by a lesion, enabling comparison across subjects and regions.

## Prerequisites

- Lacuna installed ([Installation Guide](installation.md))
- Binary lesion mask in MNI space (NIfTI format)

## Step-by-step instructions

### 1. Load your lesion mask

```python
import nibabel as nib
from lacuna import SubjectData

mask_img = nib.load("path/to/lesion.nii.gz")
subject = SubjectData(
    mask_img=mask_img,
    space="MNI152NLin6Asym",
    resolution=2.0,
    metadata={"subject_id": "sub-001"}
)
```

### 2. Configure the analysis

```python
from lacuna.analysis import RegionalDamage

damage = RegionalDamage(
    parcel_names=["Schaefer2018_100Parcels7Networks"],
    threshold=0.5  # Minimum overlap fraction to count as damaged
)
```

#### Available parcellations

| Parcellation | Regions | Type |
|-------------|---------|------|
| `Schaefer2018_100Parcels7Networks` | 100 | Cortical |
| `Schaefer2018_200Parcels7Networks` | 200 | Cortical |
| `AAL` | 116 | Cortical + Subcortical |
| `Harvard-Oxford` | 48 | Cortical |

### 3. Run the analysis

```python
result = damage.run(subject)
```

### 4. Access results

```python
# Get damage values per region
damage_values = result.results["RegionalDamage"]["parcel_damage"]

# Print top damaged regions
import pandas as pd

df = pd.DataFrame({
    "region": list(damage_values.keys()),
    "damage": list(damage_values.values())
})
df = df.sort_values("damage", ascending=False)
print(df.head(10))
```

### 5. Export as table

```python
# Save to CSV
df.to_csv("sub-001_regional_damage.csv", index=False)
```

## Expected results

The output is a dictionary mapping region names to damage proportions (0.0 to 1.0):

| Region | Damage |
|--------|--------|
| 7Networks_LH_Vis_1 | 0.82 |
| 7Networks_LH_Vis_2 | 0.45 |
| 7Networks_LH_SomMot_1 | 0.12 |

## Threshold parameter

The `threshold` parameter controls which regions are considered damaged:

| Threshold | Behavior |
|-----------|----------|
| `0.0` | Any overlap counts as damage |
| `0.5` | At least 50% overlap required (default) |
| `1.0` | Complete overlap required |

## Analyzing multiple parcellations

```python
damage = RegionalDamage(
    parcel_names=[
        "Schaefer2018_100Parcels7Networks",
        "AAL"
    ]
)

result = damage.run(subject)

# Access each parcellation's results
schaefer_damage = result.results["RegionalDamage"]["Schaefer2018_100Parcels7Networks"]
aal_damage = result.results["RegionalDamage"]["AAL"]
```

## Tips

!!! tip "Choosing a parcellation"
    
    - Use **Schaefer** for network-level analysis
    - Use **AAL** for traditional anatomical regions
    - Use higher resolution (200, 400) for fine-grained analysis

!!! tip "Batch analysis"
    
    See the [Batch Processing](batch-processing.md) guide for analyzing
    multiple subjects and extracting a group-level damage table.

## Troubleshooting

??? question "All damage values are zero"
    
    Check that your lesion mask:
    
    1. Is in the correct coordinate space
    2. Contains non-zero voxels
    3. Overlaps with the parcellation volume

??? question "Region names look wrong"
    
    Region names depend on the parcellation. Use `damage_values.keys()`
    to see all available region names.
