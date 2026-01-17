# BIDS-Apps Interface

The Lacuna command-line interface follows the BIDS-Apps specification for standardized neuroimaging workflows.

## Synopsis

```
lacuna <bids_dir> <output_dir> {participant,group} [options]
```

## Arguments

### Required arguments

| Argument | Description |
|----------|-------------|
| `bids_dir` | Root folder of BIDS dataset (sub-XXXXX folders at top level), OR path to a single NIfTI mask file for quick analysis |
| `output_dir` | Output directory for derivatives (will be created if needed) |
| `{participant,group}` | Processing level: `participant` runs per-subject analysis, `group` aggregates subject-level parcelstats into group TSV files |

### Configuration options

| Option | Description |
|--------|-------------|
| `-c YAML`, `--config YAML` | Path to YAML configuration file. Use `lacuna --generate-config` to create a template. |
| `--generate-config` | Print a template configuration file to stdout and exit |

### BIDS filtering options

| Option | Description |
|--------|-------------|
| `--participant-label LABEL [LABEL ...]` | Subject IDs to process (without `sub-` prefix) |
| `--session-id SESSION [SESSION ...]` | Session IDs to process (without `ses-` prefix) |
| `--pattern GLOB` | Glob pattern to filter mask files (e.g., `*label-WMH*`) |

### Mask space options

| Option | Description |
|--------|-------------|
| `--mask-space SPACE` | Coordinate space of input masks (e.g., `MNI152NLin6Asym`). Required if not detectable from filename or sidecar JSON. |

### Analysis options

| Option | Description |
|--------|-------------|
| `--functional-connectome PATH` | Path to functional connectome directory or HDF5 file. Enables FunctionalNetworkMapping analysis. |
| `--structural-tractogram PATH` | Path to whole-brain tractogram (.tck). Enables StructuralNetworkMapping analysis. Requires MRtrix3. |
| `--structural-tdi PATH` | Path to pre-computed whole-brain TDI NIfTI (optional, speeds up processing) |
| `--parcel-atlases ATLAS [ATLAS ...]` | Atlas names for RegionalDamage analysis. Use `lacuna list-parcellations` to see available atlases. |
| `--custom-parcellation NIFTI LABELS` | Custom parcellation: NIfTI file path and labels file path. Can be specified multiple times. |
| `--skip-regional-damage` | Skip RegionalDamage analysis (enabled by default) |

### Performance options

| Option | Description |
|--------|-------------|
| `--nprocs N` | Number of parallel processes (-1 for all CPUs, default: -1) |
| `--batch-size N` | Number of masks to process together per batch. Controls memory usage. Use -1 for all masks at once (fastest). |
| `-w PATH`, `--tmp-dir PATH` | Temporary directory for intermediate files (default: `$LACUNA_TMP_DIR` or `./tmp`) |

### Other options

| Option | Description |
|--------|-------------|
| `--overwrite` | Overwrite existing output files |
| `--version` | Show program version number and exit |
| `-v`, `--verbose` | Increase verbosity (`-v`=INFO, `-vv`=DEBUG) |
| `-h`, `--help` | Show help message and exit |

## Examples

### Run all analyses on all participants

```bash
lacuna /data/my_study /output participant
```

### Run on specific participants

```bash
lacuna /data/my_study /output participant \
    --participant-label 001 002
```

### Run with functional network mapping

```bash
lacuna /data/my_study /output participant \
    --functional-connectome /data/connectomes/GSP1000.h5
```

### Run with structural network mapping

```bash
lacuna /data/my_study /output participant \
    --structural-tractogram /data/tractograms/wholebrain.tck
```

### Run with specific atlases

```bash
lacuna /data/my_study /output participant \
    --parcel-atlases Schaefer2018_100Parcels7Networks Tian_Subcortex_S1
```

### Run with parallel processing

```bash
lacuna /data/my_study /output participant \
    --nprocs 8
```

### Group-level analysis

```bash
# First run participant-level
lacuna /data/my_study /output participant

# Then run group-level to aggregate results
lacuna /data/my_study /output group
```

### Quick analysis with single mask file

```bash
lacuna /path/to/mask.nii.gz /output participant \
    --mask-space MNI152NLin6Asym
```

### Use a configuration file

```bash
# Generate template
lacuna --generate-config > config.yaml

# Edit config.yaml, then run
lacuna /data/my_study /output participant -c config.yaml
```

## Input requirements

### BIDS dataset structure

```
my_study/
├── dataset_description.json
├── participants.tsv
├── sub-001/
│   └── ses-01/
│       └── anat/
│           └── sub-001_ses-01_space-MNI152NLin6Asym_mask.nii.gz
└── sub-002/
    └── ses-01/
        └── anat/
            └── sub-002_ses-01_space-MNI152NLin6Asym_mask.nii.gz
```

### Required files

| File | Description |
|------|-------------|
| `dataset_description.json` | BIDS dataset metadata |
| `*_mask.nii.gz` | Binary mask in MNI space |

## Output structure

```
output/
└── lacuna/
    ├── dataset_description.json
    ├── sub-001/
    │   └── ses-01/
    │       └── func/
    │           ├── sub-001_ses-01_desc-flnm_connectivity.nii.gz
    │           └── sub-001_ses-01_desc-flnm_connectivity.json
    └── logs/
        └── lacuna_20240101_120000.log
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Input validation failed |
| 4 | Analysis error |

## Environment variables

| Variable | Description |
|----------|-------------|
| `LACUNA_CACHE` | Custom cache directory for connectomes |
| `LACUNA_TMP_DIR` | Custom temporary directory |
| `TEMPLATEFLOW_HOME` | TemplateFlow cache directory |

## See also

- [Installation](../../how-to/installation.md)
- [Docker usage](../../how-to/docker.md)
- [Apptainer usage](../../how-to/apptainer.md)
