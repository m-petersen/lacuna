# Use a Custom Parcellation

This guide shows how to use your own brain parcellation atlas with Lacuna.

## Goal

Run regional damage analysis using a custom parcellation atlas that is not bundled with Lacuna.

## Prerequisites

- Lacuna installed ([Installation](installation.md))
- A parcellation NIfTI file (3D integer-labeled or 4D probabilistic)
- A labels file mapping region IDs to names
- Both files in MNI space (MNI152NLin6Asym or MNI152NLin2009cAsym)

## File format requirements

### Parcellation NIfTI

A 3D NIfTI where each voxel contains an integer region ID:

```
Voxel value 0 → background (not a region)
Voxel value 1 → Region 1
Voxel value 2 → Region 2
...
```

4D probabilistic atlases are also supported, where each volume represents a region with probability values between 0 and 1.

### Labels file

A text file mapping region IDs to names. Two formats are supported:

=== "ID-Name format"

    ```text
    # Lines starting with # are comments
    1 Left-Visual-Cortex
    2 Right-Visual-Cortex
    3 Left-Motor-Area
    ```

=== "Name-per-line format"

    ```text
    # Region IDs assigned automatically (1, 2, 3, ...)
    Left-Visual-Cortex
    Right-Visual-Cortex
    Left-Motor-Area
    ```

## Via the CLI

The `--custom-parcellation` flag takes two arguments: the NIfTI path and the labels path.

```bash
lacuna run rd /bids /output \
    --custom-parcellation /path/to/my_atlas.nii.gz /path/to/my_labels.txt
```

You can combine custom parcellations with built-in atlases:

```bash
lacuna run rd /bids /output \
    --parcel-atlases Schaefer2018_100Parcels7Networks \
    --custom-parcellation /path/to/my_atlas.nii.gz /path/to/my_labels.txt
```

You can also specify multiple custom parcellations:

```bash
lacuna run rd /bids /output \
    --custom-parcellation /path/to/atlas_A.nii.gz /path/to/labels_A.txt \
    --custom-parcellation /path/to/atlas_B.nii.gz /path/to/labels_B.txt
```

## Via the Python API

### Register and use

```python
from lacuna.assets.parcellations import register_parcellation_from_files
from lacuna.analysis import RegionalDamage

# Register your parcellation (once per session)
register_parcellation_from_files(
    name="MyAtlas",
    parcellation_path="/path/to/my_atlas.nii.gz",
    labels_path="/path/to/my_labels.txt",
    space="MNI152NLin6Asym",
    resolution=2,
    description="My custom brain parcellation",
)

# Use in analysis
analysis = RegionalDamage(parcel_names=["MyAtlas"])
result = analysis.run(subject_data)
```

### Register from a directory

If you have multiple parcellations in a directory, register them all at once:

```python
from lacuna.assets.parcellations import register_parcellations_from_directory

registered = register_parcellations_from_directory(
    directory="/data/my_parcellations",
    space="MNI152NLin6Asym",
    resolution=2,
)
print(f"Registered: {registered}")
```

Files must follow the naming pattern: `*atlas.nii.gz` with a matching `*atlas_labels.txt`.

### Optional metadata

You can provide additional metadata when registering:

```python
register_parcellation_from_files(
    name="MyAtlas",
    parcellation_path="/path/to/my_atlas.nii.gz",
    labels_path="/path/to/my_labels.txt",
    space="MNI152NLin6Asym",
    resolution=2,
    description="My custom brain parcellation",
    citation="Smith et al. 2024",
    networks=["Visual", "Motor", "Default"],
    n_regions=100,
)
```

## Verify registration

List all available parcellations to confirm your atlas was registered:

```python
from lacuna.assets.parcellations import list_parcellations

for p in list_parcellations():
    print(f"{p.name}: {p.space} @ {p.resolution}mm")
```

Or from the CLI:

```bash
lacuna info atlases
```

## Tips

!!! tip "Use absolute paths"
    Use absolute paths when registering parcellations. Relative paths are auto-resolved, but absolute paths prevent ambiguity.

!!! tip "Match coordinate space"
    Ensure your parcellation is in the same MNI space as your input masks. If there's a mismatch, Lacuna will attempt spatial transformation, which may introduce interpolation artifacts in integer-labeled atlases.

## See also

- [Regional Damage](regional-damage.md) — Running regional damage analysis
- [Coordinate Spaces](../explanation/coordinate-spaces.md) — Understanding MNI spaces
- [Registry Pattern](../explanation/registries.md) — How the registry system works
