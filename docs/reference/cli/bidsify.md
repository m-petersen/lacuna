# Bidsify command

The `lacuna bidsify` command converts a directory of NIfTI mask files to BIDS format.

## Synopsis

```
lacuna bidsify <input_dir> <output_dir> --space <SPACE> [options]
```

## Description

Converts loose NIfTI mask files into a BIDS-compliant dataset structure. Input filenames become subject IDs (special characters are removed). For example: `patient_001.nii.gz` becomes `sub-patient001/`.

## Arguments

### Required arguments

| Argument | Description |
|----------|-------------|
| `input_dir` | Directory containing NIfTI mask files (`.nii` or `.nii.gz`) |
| `output_dir` | Output directory for BIDS dataset |
| `--space`, `-s` | Coordinate space of the masks: `MNI152NLin6Asym` or `MNI152NLin2009cAsym` |

### Optional BIDS entities

| Option | Description |
|--------|-------------|
| `--session LABEL`, `-ses LABEL` | Session label (e.g., `01`, `baseline`). Creates `ses-<label>` subdirectory. |
| `--label NAME`, `-l NAME` | Label for the mask entity (e.g., `lesion`, `tumor`) |

### Other options

| Option | Description |
|--------|-------------|
| `-v`, `--verbose` | Print progress messages |

## Examples

```bash
# Basic conversion
lacuna bidsify /raw/masks /bids --space MNI152NLin6Asym

# With session and label
lacuna bidsify /raw /bids --space MNI152NLin6Asym --session 01 --label lesion

# Different MNI space
lacuna bidsify ./masks ./bids_masks --space MNI152NLin2009cAsym
```

## See Also

- [How-to Guides](../../how-to/index.md)
