# Fetch Command

The `lacuna fetch` command downloads pre-built connectomes and other assets.

## Synopsis

```
lacuna fetch [--connectome TYPE] [--name NAME] [options]
```

## Description

The fetch command downloads and caches normative connectomes required for lesion network mapping analyses. Downloaded files are stored in the Lacuna cache directory.

## Options

### Asset selection

| Option | Description |
|--------|-------------|
| `--connectome {functional,structural}` | Type of connectome to download |
| `--name NAME` | Name of the specific connectome |
| `--list` | List available connectomes without downloading |

### Download options

| Option | Description |
|--------|-------------|
| `--output DIR` | Custom output directory (default: cache) |
| `--force` | Re-download even if cached |
| `--verify` | Verify checksums after download |

### Display options

| Option | Description |
|--------|-------------|
| `--verbose` | Show detailed progress |
| `--quiet` | Suppress all output except errors |

## Examples

### List available connectomes

```bash
lacuna fetch --list
```

Output:

```
Available connectomes:

Functional:
  - HCP_S1200 (MNI152NLin6Asym, 2mm, 1000 subjects)
  
Structural:
  - HCP_Tractogram (MNI152NLin2009cAsym, 2mm, 985 subjects)
```

### Download functional connectome

```bash
lacuna fetch --connectome functional --name HCP_S1200
```

### Download structural connectome

```bash
lacuna fetch --connectome structural --name HCP_Tractogram
```

### Download to custom directory

```bash
lacuna fetch --connectome functional --name HCP_S1200 \
    --output /data/connectomes
```

### Force re-download

```bash
lacuna fetch --connectome functional --name HCP_S1200 --force
```

## Cache location

Downloaded files are stored in:

| OS | Default path |
|----|--------------|
| Linux | `~/.cache/lacuna/` |
| macOS | `~/Library/Caches/lacuna/` |
| Windows | `%LOCALAPPDATA%\lacuna\cache\` |

Override with the `LACUNA_CACHE` environment variable:

```bash
export LACUNA_CACHE=/path/to/custom/cache
lacuna fetch --connectome functional --name HCP_S1200
```

## File structure

After download:

```
~/.cache/lacuna/
├── functional/
│   └── HCP_S1200/
│       ├── connectivity.h5
│       ├── mask.nii.gz
│       └── metadata.json
└── structural/
    └── HCP_Tractogram/
        ├── tractogram.tck
        └── metadata.json
```

## Verification

Downloaded files are verified against known checksums:

```bash
lacuna fetch --connectome functional --name HCP_S1200 --verify
```

If verification fails, use `--force` to re-download.

## Proxy configuration

For environments behind a proxy:

```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
lacuna fetch --connectome functional --name HCP_S1200
```

## Exit codes

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
