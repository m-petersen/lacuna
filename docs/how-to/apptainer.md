# Use Apptainer/Singularity

This guide shows how to run Lacuna analyses using Apptainer (formerly Singularity) containers, commonly used on HPC clusters.

## Goal

Run Lacuna on HPC systems or restricted environments where Docker is not available.

## Prerequisites

- Apptainer or Singularity installed (check with `apptainer --version` or `singularity --version`)
- Access to container images
- Input data on a shared filesystem

## Quick start

```bash
# Pull the container image
apptainer pull docker://ghcr.io/lacuna/lacuna:latest

# Run an analysis
apptainer run lacuna_latest.sif \
    lacuna /path/to/bids /path/to/output participant
```

## Converting from Docker

Apptainer can directly use Docker images:

```bash
# Pull and convert Docker image to SIF
apptainer pull docker://ghcr.io/lacuna/lacuna:latest

# Result: lacuna_latest.sif
```

### Using the definition file

For custom builds:

```bash
apptainer build lacuna.sif lacuna.def
```

## BIDS-Apps usage

```bash
apptainer run lacuna_latest.sif \
    lacuna /bids /output participant \
    --participant_label sub-001 \
    --analysis flnm
```

## Bind paths

Apptainer requires explicit bind mounts for paths outside the container:

```bash
apptainer run \
    --bind /data/study:/bids:ro \
    --bind /results:/output \
    lacuna_latest.sif \
    lacuna /bids /output participant
```

### Default bind paths

Apptainer automatically binds:

- `$HOME`
- `$PWD`
- `/tmp`

### Custom bind paths

```bash
# Multiple binds
apptainer run \
    --bind /data:/data:ro \
    --bind /scratch:/scratch \
    --bind /projects/connectomes:/connectomes:ro \
    lacuna_latest.sif \
    lacuna /data/bids /scratch/output participant
```

## HPC job scripts

### SLURM example

```bash
#!/bin/bash
#SBATCH --job-name=lacuna
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

module load apptainer  # or singularity

apptainer run \
    --bind /data/my_study:/bids:ro \
    --bind /scratch/$USER/output:/output \
    /containers/lacuna_latest.sif \
    lacuna /bids /output participant \
    --participant_label ${SLURM_ARRAY_TASK_ID} \
    --n_jobs ${SLURM_CPUS_PER_TASK}
```

### PBS/Torque example

```bash
#!/bin/bash
#PBS -N lacuna
#PBS -l nodes=1:ppn=8
#PBS -l mem=32gb
#PBS -l walltime=4:00:00

module load singularity

singularity run \
    --bind /data/my_study:/bids:ro \
    --bind /scratch/$USER/output:/output \
    /containers/lacuna_latest.sif \
    lacuna /bids /output participant
```

## Environment variables

Pass environment variables to the container:

```bash
apptainer run \
    --env LACUNA_CACHE=/connectomes \
    --bind /shared/connectomes:/connectomes:ro \
    lacuna_latest.sif \
    lacuna /bids /output participant
```

## Writable containers

For development or saving cache:

```bash
# Create a writable overlay
apptainer run \
    --overlay my_overlay.img \
    lacuna_latest.sif \
    lacuna /bids /output participant
```

## Comparison with Docker

| Feature | Docker | Apptainer |
|---------|--------|-----------|
| Root required | Yes (daemon) | No |
| HPC support | Limited | Full |
| Security | Root in container | User in container |
| Bind syntax | `-v host:container` | `--bind host:container` |

## Troubleshooting

??? question "Cannot find libraries"
    
    Ensure you're using the containerized commands:
    
    ```bash
    # Wrong: calling system Python
    python -c "import lacuna"
    
    # Right: calling container Python
    apptainer exec lacuna_latest.sif python -c "import lacuna"
    ```

??? question "Permission denied"
    
    Check if the bind paths are accessible:
    
    ```bash
    ls -la /path/to/data
    # Ensure read permissions for your user
    ```

??? question "Out of memory in /tmp"
    
    Set a different temporary directory:
    
    ```bash
    export APPTAINER_TMPDIR=/scratch/tmp
    apptainer run lacuna_latest.sif ...
    ```

??? question "Singularity vs Apptainer"
    
    Apptainer is the new name for Singularity. Most commands are compatible:
    
    ```bash
    # Singularity syntax
    singularity run lacuna_latest.sif ...
    
    # Apptainer syntax
    apptainer run lacuna_latest.sif ...
    ```

## Resources

- [Apptainer Documentation](https://apptainer.org/docs/)
- [Singularity Hub](https://singularity-hub.org/)
- [BIDS-Apps Specification](https://bids-apps.neuroimaging.io/)
