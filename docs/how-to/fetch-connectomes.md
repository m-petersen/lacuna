# Fetch Connectomes

This guide shows how to download pre-built normative connectomes for lesion network mapping.

## Goal

Download and register functional and structural connectomes for use with Lacuna analyses.

## Available connectomes

Lacuna supports connectomes from various sources:

| Type | Source | Space | Description |
|------|--------|-------|-------------|
| Functional | HCP S1200 | MNI152NLin6Asym | Resting-state connectivity from 1000 subjects |
| Structural | HCP Tractogram | MNI152NLin2009cAsym | Group tractography from 985 subjects |

## Fetching functional connectomes

### Using the CLI

```bash
lacuna fetch --connectome functional --name HCP_S1200
```

### Using Python

```python
from lacuna.assets.fetcher import fetch_functional_connectome

# Download to default cache directory
path = fetch_functional_connectome(
    name="HCP_S1200",
    space="MNI152NLin6Asym",
    resolution=2
)
print(f"Downloaded to: {path}")
```

### Register the connectome

After downloading, register for use in analyses:

```python
from lacuna.assets.connectomes import register_functional_connectome

register_functional_connectome(
    name="HCP_S1200",
    space="MNI152NLin6Asym",
    resolution=2.0,
    data_path=path,
    n_subjects=1000
)
```

## Fetching structural connectomes

### Using the CLI

```bash
lacuna fetch --connectome structural --name HCP_Tractogram
```

### Using Python

```python
from lacuna.assets.fetcher import fetch_structural_connectome

path = fetch_structural_connectome(
    name="HCP_Tractogram",
    space="MNI152NLin2009cAsym"
)
print(f"Downloaded to: {path}")
```

### Register the connectome

```python
from lacuna.assets.connectomes import register_structural_connectome

register_structural_connectome(
    name="HCP_Tractogram",
    space="MNI152NLin2009cAsym",
    resolution=2.0,
    tractogram_path=path,
    n_subjects=985
)
```

## Cache location

By default, connectomes are cached in:

| OS | Path |
|----|------|
| Linux | `~/.cache/lacuna/` |
| macOS | `~/Library/Caches/lacuna/` |
| Windows | `%LOCALAPPDATA%\lacuna\cache\` |

### Custom cache directory

```python
import os
os.environ["LACUNA_CACHE"] = "/path/to/custom/cache"

# Now fetch will use this directory
from lacuna.assets.fetcher import fetch_functional_connectome
path = fetch_functional_connectome("HCP_S1200")
```

## Using custom connectomes

You can register your own connectomes:

```python
register_functional_connectome(
    name="MyConnectome",
    space="MNI152NLin6Asym",
    resolution=2.0,
    data_path="/path/to/my/connectome",
    n_subjects=500  # Number of subjects in your connectome
)
```

### Required format

Functional connectomes should be in HDF5 format with:

- Correlation matrix per voxel
- Matching brain mask

Structural connectomes should be in TCK format (MRtrix tractography).

## List registered connectomes

```python
from lacuna.assets.connectomes import list_connectomes

# List all registered connectomes
connectomes = list_connectomes()
for c in connectomes:
    print(f"{c['name']}: {c['type']} ({c['space']})")
```

## Troubleshooting

??? question "Download fails with network error"
    
    Try downloading manually from the source and placing in the cache directory,
    then register using `data_path`.

??? question "Checksum mismatch"
    
    Delete the partial download and try again:
    
    ```bash
    rm -rf ~/.cache/lacuna/HCP_S1200*
    lacuna fetch --connectome functional --name HCP_S1200
    ```

??? question "Not enough disk space"
    
    Functional connectomes can be 10-50 GB. Free up space or use an external drive:
    
    ```bash
    export LACUNA_CACHE=/external/drive/lacuna_cache
    ```
