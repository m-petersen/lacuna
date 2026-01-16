# BIDS-Apps Interface

The Lacuna command-line interface follows the BIDS-Apps specification for standardized neuroimaging workflows.

## Synopsis

```
lacuna <bids_dir> <output_dir> <analysis_level> [options]
```

## Arguments

### Required arguments

| Argument | Description |
|----------|-------------|
| `bids_dir` | Path to BIDS-formatted input dataset |
| `output_dir` | Path for output files (will be created if needed) |
| `analysis_level` | Level of analysis: `participant` or `group` |

### Optional arguments

#### Participant selection

| Option | Description |
|--------|-------------|
| `--participant_label LABEL [LABEL ...]` | Process only specified participants (e.g., `sub-001 sub-002`) |
| `--session_label LABEL [LABEL ...]` | Process only specified sessions |

#### Analysis selection

| Option | Description |
|--------|-------------|
| `--analysis {flnm,slnm,regional,all}` | Analysis type(s) to run. Default: `all` |
| `--connectome NAME` | Connectome to use for LNM analyses |
| `--parcellation NAME` | Parcellation for regional analysis |

#### Processing options

| Option | Description |
|--------|-------------|
| `--n_jobs N` | Number of parallel jobs. Default: 1. Use -1 for all CPUs |
| `--verbose` | Enable verbose output |
| `--skip_validation` | Skip BIDS validation (faster but less safe) |

#### Output options

| Option | Description |
|--------|-------------|
| `--derivatives_name NAME` | Name for derivatives folder. Default: `lacuna` |
| `--overwrite` | Overwrite existing outputs |

## Examples

### Run all analyses on all participants

```bash
lacuna /data/my_study /output participant
```

### Run fLNM on specific participants

```bash
lacuna /data/my_study /output participant \
    --participant_label sub-001 sub-002 \
    --analysis flnm
```

### Run with custom connectome

```bash
lacuna /data/my_study /output participant \
    --analysis flnm \
    --connectome HCP_S1200
```

### Run with parallel processing

```bash
lacuna /data/my_study /output participant \
    --n_jobs 8 \
    --analysis regional
```

### Group-level analysis

```bash
# First run participant-level
lacuna /data/my_study /output participant

# Then run group-level
lacuna /data/my_study /output group
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
| `TEMPLATEFLOW_HOME` | TemplateFlow cache directory |
| `OMP_NUM_THREADS` | OpenMP thread limit |

## See also

- [Installation](../../how-to/installation.md)
- [Docker usage](../../how-to/docker.md)
- [Apptainer usage](../../how-to/apptainer.md)
