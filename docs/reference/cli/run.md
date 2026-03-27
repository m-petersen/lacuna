# Run command

The `lacuna run` command runs lesion network mapping analyses on derivative lesion mask datasets organized with BIDS-style naming conventions.

## Synopsis

```
lacuna run <analysis> <bids_dir> <output_dir> [options]
```

## Available Analyses

| Analysis | Alias | Description |
|----------|-------|-------------|
| `rd` | `regionaldamage` | Lesion overlap with brain parcellations |
| `fnm` | `functionalnetworkmapping` | Functional connectivity disruption mapping |
| `snm` | `structuralnetworkmapping` | White matter disconnection mapping |

## Shared Options

All `lacuna run` subcommands share these options:

### Required arguments

| Argument | Description |
|----------|-------------|
| `bids_dir` | Root folder of dataset using BIDS-style naming (`sub-XXXXX` folders at top level), OR path to a single NIfTI mask file for quick analysis |
| `output_dir` | Output directory for derivatives |

### BIDS filtering options

| Option | Description |
|--------|-------------|
| `--participant-label LABEL [...]` | Subject IDs to process (without `sub-` prefix) |
| `--session-id SESSION [...]` | Session IDs to process (without `ses-` prefix) |
| `--pattern GLOB` | Glob pattern to filter mask files (e.g., `*label-WMH*`) |

### Mask space options

| Option | Description |
|--------|-------------|
| `--mask-space SPACE` | Coordinate space of input masks (e.g., `MNI152NLin6Asym`). Required if not detectable from filename or sidecar JSON. |

### Performance options

| Option | Description |
|--------|-------------|
| `--nprocs N` | Number of parallel processes (-1 for all CPUs, default: -1). Also used for MRtrix3 thread count in SNM. |
| `--batch-size N` | Number of masks to process together per batch. Use -1 for all masks at once (fastest). Lower values reduce peak memory. |
| `-w PATH`, `--tmp-dir PATH` | Temporary directory for intermediate files (default: `$LACUNA_TMP_DIR` or `./tmp`) |

### Other options

| Option | Description |
|--------|-------------|
| `--overwrite` | Overwrite existing output files |
| `--keep-intermediate` | Keep intermediate results in output |
| `-v`, `--verbose` | Increase verbosity (`-v`=INFO, `-vv`=DEBUG) |

---

## `lacuna run rd` — Regional Damage

Computes lesion overlap with brain parcellations (atlases). For each parcel, calculates the percentage of voxels overlapping with the lesion mask.

### RegionalDamage options

| Option | Description |
|--------|-------------|
| `--parcel-atlases ATLAS [...]` | Atlas names to use. Use `lacuna info atlases` to list available atlases. |
| `--custom-parcellation NAME NIFTI LABELS SPACE` | Custom parcellation: a short name for output labelling, NIfTI file path, labels file path, and coordinate space (e.g., `MNI152NLin6Asym`). Can be specified multiple times. |

!!! note
    At least one of `--parcel-atlases` or `--custom-parcellation` must be provided.

### Examples

```bash
# Basic regional damage with one atlas
lacuna run rd /bids /output --parcel-atlases Schaefer2018_100Parcels7Networks

# Multiple atlases
lacuna run rd /bids /output \
    --parcel-atlases Schaefer2018_400Parcels17Networks TianSubcortex_3TS2

# With custom parcellation
lacuna run rd /bids /output \
    --custom-parcellation MyAtlas /path/to/my_atlas.nii.gz /path/to/labels.txt MNI152NLin6Asym

# Mix bundled and custom atlases
lacuna run rd /bids /output \
    --parcel-atlases Schaefer2018_100Parcels7Networks \
    --custom-parcellation MyAtlas /path/to/my_atlas.nii.gz /path/to/labels.txt MNI152NLin6Asym
```

---

## `lacuna run fnm` — Functional Network Mapping

Computes functional connectivity disruption using a normative functional connectome. Generates correlation, z-score, t-score, and p-value maps.

### FunctionalNetworkMapping options

| Option | Description |
|--------|-------------|
| `--connectome-path PATH` | **(required)** Path to HDF5 connectome file or directory of batch files (from `lacuna fetch gsp1000`) |
| `--method {boes,pini}` | Timeseries extraction method (default: `boes`). `boes` = mean timeseries across lesion voxels. `pini` = PCA-based selection of representative voxels. |
| `--pini-percentile N` | For PINI method: PC1 loading percentile threshold (default: 20) |
| `--no-p-map` | Disable p-value map computation (enabled by default) |
| `--fdr-alpha ALPHA` | FDR correction alpha (default: 0.05, use 0 to disable) |
| `--t-threshold VALUE` | Create binary mask for \|t\| > threshold |
| `--output-resolution {1,2}` | Output resolution in mm (default: match input) |
| `--no-return-input-space` | Keep outputs in connectome space (default: transform to input space) |

### Parcel aggregation options

| Option | Description |
|--------|-------------|
| `--parcel-atlases ATLAS [...]` | Aggregate FNM outputs to these atlases. Use `lacuna info atlases` to list. |
| `--custom-parcellation NAME NIFTI LABELS SPACE` | Custom parcellation for aggregation: a short name for output labelling, NIfTI file path, labels file path, and coordinate space. Can be specified multiple times. |

### Examples

```bash
# Basic functional network mapping
lacuna run fnm /bids /output \
    --connectome-path ~/.cache/lacuna/connectomes/gsp1000/

# With PINI method and atlas aggregation
lacuna run fnm /bids /output \
    --connectome-path /data/gsp1000_batches \
    --method pini \
    --parcel-atlases Schaefer2018_100Parcels7Networks

# Specific participants with t-threshold
lacuna run fnm /bids /output \
    --connectome-path /data/gsp1000_batches \
    --participant-label 001 002 \
    --t-threshold 3.0
```

---

## `lacuna run snm` — Structural Network Mapping

Computes white matter disconnection using tractography. Generates disconnection maps showing regions affected by streamline interruption through lesioned tissue.

!!! warning "MRtrix3 required"
    This analysis requires MRtrix3 to be installed and in PATH.

### StructuralNetworkMapping options

| Option | Description |
|--------|-------------|
| `--connectome-path PATH` | **(required)** Path to `.tck` tractogram file (from `lacuna fetch dtor985`) |
| `--parcel-atlas NAME` | Atlas for connectivity matrices. Use `lacuna info atlases` to list. |
| `--custom-parcellation NAME NIFTI LABELS SPACE` | Custom parcellation: a short name for output labelling, NIfTI file path, labels file path, and coordinate space. Can be specified multiple times. |
| `--compute-disconnectivity-matrix` | Compute disconnectivity matrices (requires `--parcel-atlases` or `--custom-parcellation`) |
| `--compute-roi-disconnection` | Compute per-ROI disconnection values (requires `--parcel-atlases` or `--custom-parcellation`) |
| `--output-resolution {1,2}` | Output resolution in mm (default: 2) |
| `--no-cache-tdi` | Disable TDI caching (enabled by default) |
| `--no-return-input-space` | Keep outputs in connectome space (default: transform to input space) |
| `--show-mrtrix-output` | Display MRtrix3 command output |

### Examples

```bash
# Basic structural network mapping
lacuna run snm /bids /output \
    --connectome-path ~/.cache/lacuna/connectomes/dtor985/dtor985.tck

# With ROI disconnection and 4 threads
lacuna run snm /bids /output \
    --connectome-path /data/dtor985.tck \
    --parcel-atlas Schaefer2018_100Parcels7Networks \
    --compute-roi-disconnection \
    --nprocs 4
```

## Input Requirements

### Expected directory structure

```
my_study/
├── dataset_description.json
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
| `dataset_description.json` | Dataset metadata (BIDS-style) |
| `*_mask.nii.gz` | Binary mask in MNI space |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LACUNA_TMP_DIR` | Custom temporary directory (default: `./tmp`) |
| `TEMPLATEFLOW_HOME` | TemplateFlow cache directory |

