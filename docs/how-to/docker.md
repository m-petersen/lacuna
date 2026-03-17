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

# Run a regional damage analysis
docker run --rm \
    -v /path/to/bids:/bids:ro \
    -v /path/to/output:/output \
    ghcr.io/lacuna/lacuna:latest \
    run rd /bids /output --parcel-atlases Schaefer2018_100Parcels7Networks
```

## Running analyses

The container entrypoint is `lacuna`, so you pass subcommands directly:

```bash
docker run --rm \
    -v /path/to/bids:/bids:ro \
    -v /path/to/output:/output \
    ghcr.io/lacuna/lacuna:latest \
    run fnm /bids /output --connectome-path /home/lacuna/.cache/lacuna/connectomes/gsp1000
```

### Analysis examples

```bash
# Regional damage
docker run --rm \
    -v /path/to/bids:/bids:ro \
    -v /path/to/output:/output \
    ghcr.io/lacuna/lacuna:latest \
    run rd /bids /output \
    --participant-label 001 002 \
    --parcel-atlases Schaefer2018_100Parcels7Networks

# Functional network mapping
docker run --rm \
    -v /path/to/bids:/bids:ro \
    -v /path/to/output:/output \
    -v ~/.cache/lacuna:/home/lacuna/.cache/lacuna \
    ghcr.io/lacuna/lacuna:latest \
    run fnm /bids /output \
    --connectome-path /home/lacuna/.cache/lacuna/connectomes/gsp1000 \
    --nprocs 4

# Structural network mapping
docker run --rm \
    -v /path/to/bids:/bids:ro \
    -v /path/to/output:/output \
    -v /path/to/tractogram.tck:/connectomes/tractogram.tck:ro \
    ghcr.io/lacuna/lacuna:latest \
    run snm /bids /output \
    --connectome-path /connectomes/tractogram.tck
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
    run fnm /bids /output \
    --connectome-path /home/lacuna/.cache/lacuna/connectomes/gsp1000
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
    run rd /bids /output --parcel-atlases Schaefer2018_100Parcels7Networks
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
git clone https://github.com/m-petersen/lacuna.git
cd lacuna
docker build -t lacuna:local .
```
