# Use Docker

This guide shows how to run Lacuna analyses using Docker containers.

## Goal

Run Lacuna without local installation, using a pre-built Docker container with all dependencies included.

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Basic familiarity with Docker commands
- Input data accessible to Docker

## Quick start

```bash
# Pull the Lacuna image
docker pull ghcr.io/lacuna/lacuna:latest

# Run a basic analysis
docker run --rm \
    -v /path/to/data:/data \
    -v /path/to/output:/output \
    ghcr.io/lacuna/lacuna:latest \
    lacuna /data /output participant --analysis flnm
```

## BIDS-Apps interface

Lacuna follows the BIDS-Apps specification:

```bash
docker run --rm \
    -v /path/to/bids:/bids:ro \
    -v /path/to/output:/output \
    ghcr.io/lacuna/lacuna:latest \
    lacuna /bids /output participant
```

### Required arguments

| Argument | Description |
|----------|-------------|
| `bids_dir` | Path to BIDS dataset (inside container) |
| `output_dir` | Path for outputs (inside container) |
| `analysis_level` | `participant` or `group` |

### Analysis options

```bash
docker run --rm \
    -v /path/to/bids:/bids:ro \
    -v /path/to/output:/output \
    ghcr.io/lacuna/lacuna:latest \
    lacuna /bids /output participant \
    --participant_label sub-001 sub-002 \
    --analysis flnm slnm \
    --n_jobs 4
```

## Volume mounts

Docker requires explicit volume mounts for data access:

```bash
-v /host/path:/container/path:ro  # Read-only mount
-v /host/path:/container/path     # Read-write mount
```

### Common mounts

| Purpose | Host path | Container path |
|---------|-----------|----------------|
| BIDS data | `/data/my_study` | `/bids` |
| Output | `/results/lacuna` | `/output` |
| Connectomes | `~/.cache/lacuna` | `/home/lacuna/.cache/lacuna` |

## Using cached connectomes

Mount your local connectome cache:

```bash
docker run --rm \
    -v ~/.cache/lacuna:/home/lacuna/.cache/lacuna \
    -v /path/to/bids:/bids:ro \
    -v /path/to/output:/output \
    ghcr.io/lacuna/lacuna:latest \
    lacuna /bids /output participant --analysis flnm
```

## Resource limits

Control CPU and memory usage:

```bash
docker run --rm \
    --cpus="4" \
    --memory="16g" \
    -v /path/to/bids:/bids:ro \
    -v /path/to/output:/output \
    ghcr.io/lacuna/lacuna:latest \
    lacuna /bids /output participant
```

## Interactive mode

For debugging or development:

```bash
docker run --rm -it \
    -v /path/to/data:/data \
    ghcr.io/lacuna/lacuna:latest \
    bash
```

Then run Python interactively:

```bash
python
>>> from lacuna import SubjectData
>>> # ... your code here
```

## Available tags

| Tag | Description |
|-----|-------------|
| `latest` | Most recent stable release |
| `v0.1.0` | Specific version |
| `dev` | Development version (unstable) |

## Troubleshooting

??? question "Permission denied on output directory"
    
    Ensure the output directory is writable:
    
    ```bash
    chmod 777 /path/to/output
    # Or run with user mapping
    docker run --rm -u $(id -u):$(id -g) ...
    ```

??? question "Container exits immediately"
    
    Check logs for errors:
    
    ```bash
    docker run ghcr.io/lacuna/lacuna:latest lacuna --help
    ```

??? question "Cannot access GPU"
    
    GPU support requires nvidia-docker:
    
    ```bash
    docker run --gpus all ...
    ```

## Building locally

To build the image from source:

```bash
git clone https://github.com/lacuna/lacuna.git
cd lacuna
docker build -t lacuna:local .
```
