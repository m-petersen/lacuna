# Running Lacuna on an HPC cluster

This guide walks you through setting up and running Lacuna analyses on a high-performance computing (HPC) cluster using SLURM and Apptainer.

## Overview

The typical HPC workflow is:

1. Pull the Lacuna container image once
2. Fetch needed data like connectomes
3. Prepare a batch script for the analysis you want to run
4. Submit it as a SLURM job array that distributes subjects across nodes

Ready-to-use scripts for FNM and SNM are provided in the [`hpc_scripts/`](https://github.com/m-petersen/lacuna/tree/main/docs/how-to/hpc_scripts) directory.

Focal damage analysis is fast enough to run locally and does not require HPC resources.

## Prerequisites

- SLURM workload manager
- Apptainer (or Singularity) available as a module
- BIDS-formatted dataset on a shared filesystem
- Connectome data for the analysis you want to run

## Pull the container

Pull the Lacuna image once and store the resulting `.sif` file on the shared filesystem:

```bash
module load apptainer
apptainer pull lacuna.sif docker://ghcr.io/m-petersen/lacuna:latest 
```

This creates `lacuna.sif` in the current directory.

## Batch scripts

Each batch script follows the same structure:

1. **Initialization** — load Apptainer, create log directory
2. **Configuration** — define paths to BIDS data, outputs, connectomes, and the SIF image
3. **Subject slicing** — discover all subjects and assign a batch to the current array task
4. **Execution** — run Lacuna inside the container with bind mounts

### Functional network mapping

```bash title="lacuna_fnm.batch"
--8<-- "docs/how-to/hpc_scripts/lacuna_fnm.batch"
```

### Structural network mapping

SNM is the most resource-intensive analysis. The script copies the tractogram to node-local storage (`$TMPDIR`) for faster I/O and uses a smaller batch size:

```bash title="lacuna_snm.batch"
--8<-- "docs/how-to/hpc_scripts/lacuna_snm.batch"
```

## Submit scripts

Each submit script counts the subjects, calculates how many array tasks are needed, and submits the corresponding batch script. You can pass specific subject names as arguments to process a subset; if none are given, all subjects in `BIDS_DIR` are submitted:

```bash title="submit_fnm_jobs.sh"
--8<-- "docs/how-to/hpc_scripts/submit_fnm_jobs.sh"
```

Usage:

```bash
# Edit BIDS_DIR in the submit script, then:

# Submit all subjects
bash submit_fnm_jobs.sh

# Submit specific subjects only
bash submit_fnm_jobs.sh sub-001 sub-002 sub-003
```

The SNM submit script (`submit_snm_jobs.sh`) follows the same pattern.

## Adapting the scripts

Before running, update the placeholder paths in both the batch and submit scripts:

| Variable | Description | Example |
|----------|-------------|---------|
| `BIDS_DIR` | BIDS dataset on shared storage | `/data/projects/my_study/bids` |
| `OUTPUT_DIR` | Output directory (read-write) | `/scratch/$USER/lacuna_output` |
| `CONNECTOMES_DIR` | Connectome files | `/data/connectomes` |
| `SIF_IMAGE` | Path to the `.sif` container | `/containers/lacuna_latest.sif` |
| `CACHE_ROOT` | Per-job cache directory | `/scratch/$USER/.cache_slurm` |

### Resource requirements

| Analysis | CPU | Memory | Time (per batch) | Batch size |
|----------|-----|--------|-------------------|------------|
| Functional network mapping | 16 | 64 GB | ~4 h | 200 |
| Structural network mapping | 16 | 64 GB | ~4 h | 20 |

Adjust `--cpus-per-task`, `--mem`, and `--time` to match your cluster's constraints and dataset size.

### Subject slicing

The scripts use SLURM job arrays to distribute subjects across nodes. Each array task processes a slice of `BATCH_SIZE` subjects:

- Array task 0 processes subjects 0–199
- Array task 1 processes subjects 200–399
- etc.

The submit script calculates the required number of array tasks automatically.

### Filtering masks

Use `--pattern` to select specific masks within each subject directory:

```bash
--pattern "*label-acuteinfarct*"
```

This is useful when subjects have multiple lesion masks (e.g., acute vs. chronic infarct).

## Caching

Each array task gets its own cache directory to prevent write conflicts between parallel jobs. The cache is cleaned up after each job completes.

For SNM, the tractogram is copied to node-local storage (`$TMPDIR`) before processing. This avoids repeated reads from the shared filesystem and significantly improves I/O performance.

## Monitoring jobs

```bash
# Check job status
squeue -u $USER

# View logs for a specific array task
cat logs/lacuna_<job_id>_<array_id>.out

# Cancel all jobs in an array
scancel <job_id>
```

## Collecting results

After all jobs complete, aggregate results into group-level tables:

```bash
apptainer run \
    --bind /path/to/output:/output \
    lacuna_latest.sif \
    collect /output --output-dir /output/group
```
