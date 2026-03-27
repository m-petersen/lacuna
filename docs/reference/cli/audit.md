# Audit command

The `lacuna audit` command checks which subjects have complete outputs in a derivatives directory.

## Synopsis

```
lacuna audit <analysis> <bids_dir> <output_dir> [options]
```

## Description

Reads the same BIDS input directory as `lacuna run` to enumerate expected subjects, then checks whether output files exist in the derivatives directory. Prints a per-subject status table, a summary, and a ready-to-use `--participant-label` snippet for rerunning missing subjects.

Available analyses:

| Analysis | Alias | Sentinel file checked |
|----------|-------|-----------------------|
| `rd` | `regionaldamage` | `*source-regionaldamage*parcelstats.tsv` |
| `fnm` | `functionalnetworkmapping` | `*desc-fnm_rmap.nii.gz` |
| `snm` | `structuralnetworkmapping` | `*desc-snm_disconnectionpct.nii.gz` |

## Arguments

### Required arguments

| Argument | Description |
|----------|-------------|
| `bids_dir` | Root folder of BIDS dataset (`sub-XXXXX` folders at top level) |
| `output_dir` | Derivatives directory to audit for existing results |

### BIDS filtering options

| Option | Description |
|--------|-------------|
| `--participant-label LABEL [...]` | Subject IDs to check (without `sub-` prefix) |
| `--session-id SESSION [...]` | Session IDs to check (without `ses-` prefix) |
| `--pattern GLOB` | Glob pattern to filter mask files (e.g., `*label-WMH*`) |
| `--mask-space SPACE` | Coordinate space of input masks. Required if not detectable from filename or sidecar JSON. |

### Output options

| Option | Description |
|--------|-------------|
| `--output-file PATH` | Write missing subject IDs to a file (one per line) |
| `--quiet`, `-q` | Print only missing subject IDs to stdout (one per line). Useful for shell scripting. |

### `lacuna audit rd` options

| Option | Description |
|--------|-------------|
| `--parcel-atlases ATLAS [...]` | Check each named atlas individually. If omitted, any parcelstats TSV counts as complete. Use `lacuna info atlases` to list. |

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | All subjects complete |
| 1 | One or more subjects missing |
| 2 | Invalid arguments |
| 64 | BIDS input could not be read |

## Examples

```bash
# Basic audit — check all subjects
lacuna audit rd /bids /output

# Check that a specific atlas is present for each subject
lacuna audit rd /bids /output --parcel-atlases Schaefer2018_400Parcels7Networks

# Write missing subject IDs to a file
lacuna audit rd /bids /output --output-file missing.txt

# Use the file to rerun only missing subjects
lacuna run rd /bids /output --participant-label $(cat missing.txt | tr '\n' ' ')

# Quiet mode — print missing IDs only (suitable for piping)
lacuna audit rd /bids /output --quiet

# Audit FNM outputs
lacuna audit fnm /bids /output

# Audit SNM outputs, check only specific subjects
lacuna audit snm /bids /output --participant-label 001 002 003
```

## See Also

- [`lacuna run`](run.md) — Run analyses
- [`lacuna collect`](collect.md) — Aggregate results across subjects
