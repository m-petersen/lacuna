# Check command

The `lacuna check` command validates input data and checks output completeness.

## Synopsis

```
lacuna check input <bids_dir> [options]
lacuna check <rd|fnm|snm> <bids_dir> <output_dir> [options]
```

## Description

`lacuna check` has two modes:

- **Input check** (`lacuna check input`) — Validate mask files *before* running analyses. Catches common issues (non-binary masks, empty masks, space mismatches) that would otherwise cause mid-batch failures.
- **Output check** (`lacuna check rd|fnm|snm`) — Check which subjects have complete outputs *after* a run. Useful for identifying failed subjects and generating rerun commands.

---

## `lacuna check input` — Validate input masks

Checks each mask file for issues that would cause analysis failures:

| Check | Level | Description |
|-------|-------|-------------|
| File loadable | ERROR | nibabel can open the NIfTI file |
| 3D image | ERROR | Mask is exactly 3D (not 4D or 2D) |
| Binary | ERROR | Only values 0 and 1 |
| Non-empty | ERROR | At least 1 non-zero voxel |
| Space detectable | ERROR | `space-` in filename or affine matches a known template |
| Space consistency | ERROR | Filename `space-` entity matches the detected affine space |
| Small lesion | WARNING | Fewer than 10 non-zero voxels |
| Consistent dimensions | WARNING | Shape differs from majority of masks in dataset |
| Consistent spaces | WARNING | Detected space differs from majority of masks in dataset |

### Arguments

| Argument | Description |
|----------|-------------|
| `bids_dir` | Root folder of BIDS dataset |

### Options

| Option | Description |
|--------|-------------|
| `--participant-label LABEL [...]` | Subject IDs to check (without `sub-` prefix) |
| `--session-id SESSION [...]` | Session IDs to check (without `ses-` prefix) |
| `--pattern GLOB` | Glob pattern to filter mask files |
| `--mask-space SPACE` | Suppress space detection errors for files without `space-` in filename |
| `--output-file PATH` | Write error file paths to a file (one per line) |
| `--quiet`, `-q` | Print only error file paths (one per line) |

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | No errors (warnings are ok) |
| 1 | At least one error |

### Examples

```bash
# Validate all masks
lacuna check input /bids

# Provide space when not in filenames
lacuna check input /bids --mask-space MNI152NLin6Asym

# Save problematic file paths
lacuna check input /bids --output-file problems.txt
```

---

## `lacuna check rd|fnm|snm` — Check output completeness

Enumerates input subjects and checks whether output files exist in the derivatives directory.

### Sentinel files checked

| Analysis | Alias | Sentinel file |
|----------|-------|---------------|
| `rd` | `regionaldamage` | `*method-rd*parcelstats.tsv` |
| `fnm` | `functionalnetworkmapping` | `*method-fnm*desc-rmap*.nii.gz` |
| `snm` | `structuralnetworkmapping` | `*method-snm*desc-disconnectionpct*.nii.gz` |

### Arguments

| Argument | Description |
|----------|-------------|
| `bids_dir` | Root folder of BIDS dataset |
| `output_dir` | Derivatives directory to check |

### Options

| Option | Description |
|--------|-------------|
| `--participant-label LABEL [...]` | Subject IDs to check (without `sub-` prefix) |
| `--session-id SESSION [...]` | Session IDs to check |
| `--pattern GLOB` | Glob pattern to filter mask files |
| `--mask-space SPACE` | Coordinate space of input masks |
| `--output-file PATH` | Write missing subject IDs to a file (one per line) |
| `--quiet`, `-q` | Print only missing subject IDs (one per line) |

### `lacuna check rd` options

| Option | Description |
|--------|-------------|
| `--parcel-atlases ATLAS [...]` | Check each named atlas individually. If omitted, any parcelstats TSV counts as complete. |

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | All subjects complete |
| 1 | One or more subjects missing |

### Examples

```bash
# Check all subjects for RD outputs
lacuna check rd /bids /output

# Check specific atlas
lacuna check rd /bids /output --parcel-atlases schaefer2018parcels400networks7

# Write missing subjects and use for rerun
lacuna check rd /bids /output --output-file missing.txt
lacuna run rd /bids /output --participant-label $(cat missing.txt | tr '\n' ' ')

# Quiet mode for scripting
lacuna check fnm /bids /output --quiet

# Check SNM for specific subjects
lacuna check snm /bids /output --participant-label 001 002 003
```

## See Also

- [`lacuna run`](run.md) — Run analyses
- [`lacuna collect`](collect.md) — Aggregate results across subjects
