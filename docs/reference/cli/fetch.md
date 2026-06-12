# Fetch command

The `lacuna fetch` command downloads and sets up connectomes for lesion network mapping analysis.

## Synopsis

```
lacuna fetch [connectome] [options]
lacuna fetch --list
lacuna fetch --interactive
```

## Description

The fetch command downloads, processes, and registers normative connectomes required for lesion network mapping analyses. Downloaded and processed files are cached in the Lacuna cache directory (`~/.cache/lacuna/connectomes/`).

## Available Connectomes

| Name | Type | Description | Size | API key |
|------|------|-------------|------|--------|
| `gsp1000` | Functional | GSP1000 functional connectome | ~200GB | Dataverse |
| `dtor985` | Structural | dTOR985 structural tractogram (full) | ~11GB | Figshare |
| `dtor985_10pct` | Structural | dTOR985 10% subsample, ~1.2M streamlines | ~1.3GB | None |
| `dtor985_25pct` | Structural | dTOR985 25% subsample, ~3M streamlines | ~3.1GB | None |
| `hcp1065` | Structural | HCP1065 structural tractogram | ~1.5GB | None |

The `dtor985_10pct` / `dtor985_25pct` subsamples (CC0) and `hcp1065` download as ready-to-use `.tck` files and need no API key — a good starting point for structural network mapping.

## Options

### Positional argument

| Argument | Description |
|----------|-------------|
| `connectome` | Connectome to fetch: `gsp1000`, `dtor985`, `dtor985_10pct`, `dtor985_25pct`, or `hcp1065` |

### Display options

| Option | Description |
|--------|-------------|
| `--list` | List available connectomes without downloading |
| `--interactive` | Interactive guided setup wizard |

### Output options

| Option | Description |
|--------|-------------|
| `--output-dir PATH` | Output directory for processed files (default: `~/.cache/lacuna/connectomes/<name>`) |

### Common options

| Option | Description |
|--------|-------------|
| `--api-key KEY` | API key for authenticated downloads (or use env vars) |
| `--force` | Overwrite existing files |
| `--clean` | Remove cached data for a specific connectome |

### GSP1000-specific options

| Option | Description |
|--------|-------------|
| `--batches N` | Number of HDF5 batch files to create (default: 10). More batches = lower RAM usage. Recommendations: 16GB RAM → 100, 32GB+ RAM → 50 |

## Examples

### List available connectomes

```bash
lacuna fetch --list
```

### Interactive setup wizard

```bash
lacuna fetch --interactive
```

### Download GSP1000 functional connectome

```bash
# Using environment variable (recommended)
export DATAVERSE_API_KEY="your-key-here"
lacuna fetch gsp1000

# Or pass key directly
lacuna fetch gsp1000 --api-key YOUR_DATAVERSE_KEY
```

### Download with optimized batching for 16GB RAM

```bash
lacuna fetch gsp1000 --api-key YOUR_KEY --batches 100
```

### Download dTOR985 structural tractogram

```bash
export FIGSHARE_API_KEY="your-key-here"
lacuna fetch dtor985

# Or pass key directly
lacuna fetch dtor985 --api-key YOUR_FIGSHARE_KEY
```

### Download a dTOR985 subsample or HCP1065 (no API key)

```bash
# Smaller dTOR985 subsamples — ready-to-use .tck, no key required
lacuna fetch dtor985_10pct
lacuna fetch dtor985_25pct

# HCP1065 structural tractogram
lacuna fetch hcp1065
```

### Download to custom directory

```bash
lacuna fetch gsp1000 --api-key YOUR_KEY --output-dir /data/connectomes
```

### Force re-download

```bash
lacuna fetch gsp1000 --api-key YOUR_KEY --force
```

### Clean cached data

```bash
lacuna fetch gsp1000 --clean
```

## API Keys

### Dataverse (GSP1000)

The GSP1000 functional connectome is hosted on Harvard Dataverse and requires an API key:

1. Create account at [Harvard Dataverse](https://dataverse.harvard.edu/)
2. Go to your account settings → API Token
3. Generate or copy your API token

Set via environment variable (recommended):

```bash
export DATAVERSE_API_KEY="your-token-here"
```

### Figshare (dTOR985)

The dTOR985 structural tractogram is hosted on Figshare and requires an API key:

1. Go to [Figshare Applications](https://figshare.com/account/applications)
2. Create a new personal token
3. Copy the token

Set via environment variable (recommended):

```bash
export FIGSHARE_API_KEY="your-token-here"
```

## Cache Location

Downloaded and processed files are stored under `~/.cache/lacuna/connectomes/`. The
functional GSP1000 is converted into chunked HDF5 files under a `processed/`
subdirectory; the structural tractograms are stored as `.tck` files. For example:

```
~/.cache/lacuna/connectomes/
├── gsp1000/
│   └── processed/
│       ├── gsp1000_chunk_000.h5
│       ├── gsp1000_chunk_001.h5
│       └── ...
└── hcp1065/
    └── hcp1065.tck
```

When running an analysis, point `--connectome-path` at the directory that contains
the `.h5` files (e.g. `~/.cache/lacuna/connectomes/gsp1000/processed/`) or at a
`.tck` file directly.

Override the base cache directory with the `LACUNA_CACHE_DIR` environment variable:

```bash
export LACUNA_CACHE_DIR=/path/to/custom/cache
lacuna fetch gsp1000 --api-key YOUR_KEY
```
