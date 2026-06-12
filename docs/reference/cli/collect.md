# Collect command

The `lacuna collect` command aggregates subject-level parcelstats into group-level tables.

## Synopsis

```
lacuna collect <derivatives_dir> [options]
```

## Description

Scans a derivatives directory (the output directory of a `lacuna run`) for
`*_parcelstats.tsv` files and combines them into group-level TSV files. Run this
after participant-level analyses to create summary tables across subjects.

## Arguments

### Required argument

| Argument | Description |
|----------|-------------|
| `derivatives_dir` | Directory of per-subject derivatives to aggregate (a `lacuna run` output dir) |

### Filtering options

| Option | Description |
|--------|-------------|
| `--pattern GLOB` | Glob pattern to filter parcelstats files (e.g., `*acuteinfarct*`, `*schaefer2018*`) |

### Output options

| Option | Description |
|--------|-------------|
| `--output-dir DIR` | Where to write the group-level tables (default: the derivatives directory) |
| `--overwrite` | Overwrite existing group files |
| `-v`, `--verbose` | Increase verbosity (`-v`=INFO, `-vv`=DEBUG) |

## Examples

```bash
# Aggregate all results in place
lacuna collect /output

# Write group tables elsewhere, filtering by pattern
lacuna collect /output --output-dir /group --pattern '*schaefer2018*'

# Overwrite existing group files
lacuna collect /output --overwrite
```
