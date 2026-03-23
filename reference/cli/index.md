# CLI Reference

Command-line interface documentation for Lacuna.

## Overview

Lacuna provides a subcommand-based CLI for running lesion network mapping analyses, managing connectomes, and working with BIDS datasets.

```
lacuna <command> [options]
```

## Commands

<div class="grid cards" markdown>

-   **Run**

    ---

    Run lesion analyses (regional damage, functional and structural network mapping).

    [Run](run.md)

-   **Fetch**

    ---

    Download normative connectomes and other assets to the local cache.

    [Fetch](fetch.md)

-   **Collect**

    ---

    Aggregate subject-level parcelstats into group-level tables.

    [Collect](collect.md)

-   **Info**

    ---

    Display available resources (atlases, connectomes).

    [Info](info.md)

-   **Bidsify**

    ---

    Convert a directory of NIfTI mask files to BIDS format.

    [Bidsify](bidsify.md)

-   **Tutorial**

    ---

    Setup tutorial data for learning Lacuna.

    [Tutorial](tutorial.md)

</div>

## Quick Usage

```bash
# Setup tutorial data
lacuna tutorial ./my_tutorial

# Fetch a connectome
lacuna fetch gsp1000 --api-key $DATAVERSE_API_KEY

# Run regional damage analysis
lacuna run rd /bids /output --parcel-atlases Schaefer2018_100Parcels7Networks

# Run functional network mapping
lacuna run fnm /bids /output --connectome-path /path/to/gsp1000_batches

# Aggregate results across subjects
lacuna collect /bids /output

# List available atlases
lacuna info atlases
```

## Getting Help

```bash
# General help
lacuna --help

# Command-specific help
lacuna run --help
lacuna run fnm --help
lacuna fetch --help
```

