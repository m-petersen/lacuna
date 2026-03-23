# Collect Command

The `lacuna collect` command aggregates subject-level parcelstats into group-level tables.

## Synopsis

```
lacuna collect <bids_dir> <output_dir> [options]
```

## Description

Scans the output directory for `*_parcelstats.tsv` files and combines them into group-level TSV files. Run this after participant-level analyses to create summary tables across subjects.

## Arguments

### Required arguments

| Argument | Description |
|----------|-------------|
| `bids_dir` | Root folder of BIDS dataset (for metadata) |
| `output_dir` | Output directory containing derivatives to aggregate |

### Filtering options

| Option | Description |
|--------|-------------|
| `--pattern GLOB` | Glob pattern to filter parcelstats files (e.g., `*acuteinfarct*`, `*lesion*`) |

### Output options

| Option | Description |
|--------|-------------|
| `--overwrite` | Overwrite existing group files |
| `-v`, `--verbose` | Increase verbosity (`-v`=INFO, `-vv`=DEBUG) |

## Examples

```bash
# Aggregate all results
lacuna collect /bids /output

# Filter by pattern
lacuna collect /bids /output --pattern '*acuteinfarct*'

# Overwrite existing group files
lacuna collect /bids /output --overwrite
```

## See Also

- [Batch Processing](../../how-to/batch-processing.md) — Processing multiple subjects
- [Use Your Own Connectome](../../how-to/use-own-connectome.md) — Using custom connectomes
