# Fetch Command

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

| Name | Type | Description | Size | Source |
|------|------|-------------|------|--------|
| `gsp1000` | Functional | GSP1000 functional connectome | ~100GB | Dataverse |
| `dtor985` | Structural | dTOR985 structural tractogram | ~10GB | Figshare |

## Options

### Positional argument

| Argument | Description |
|----------|-------------|
| `connectome` | Connectome to fetch: `gsp1000` or `dtor985` |

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

Downloaded and processed files are stored in:

```
~/.cache/lacuna/connectomes/
├── gsp1000/
│   ├── GSP1000_batch_0.h5
│   ├── GSP1000_batch_1.h5
│   └── ...
└── dtor985/
    └── dtor985.tck
```

Override the base cache directory with the `LACUNA_CACHE` environment variable:

```bash
export LACUNA_CACHE=/path/to/custom/cache
lacuna fetch gsp1000 --api-key YOUR_KEY
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Network error |
| 4 | Verification failed |
| 5 | Disk space insufficient |

## See also

- [Fetch Connectomes guide](../../how-to/fetch-connectomes.md)
- [Run Functional LNM](../../how-to/run-flnm.md)
- [Run Structural LNM](../../how-to/run-slnm.md)
