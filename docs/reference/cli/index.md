# CLI

Command-line interface documentation for Lacuna.

## Overview

Lacuna provides a subcommand-based CLI for running lesion network mapping analyses, managing connectomes, and organizing data using BIDS-style naming conventions.

```
lacuna <command> [options]
```

## Commands

<div class="grid cards" markdown>

-   [**Audit**](audit.md)

    ---

    Check which subjects have complete outputs and identify missing runs.

-   [**Bidsify**](bidsify.md)

    ---

    Convert a directory of NIfTI mask files to BIDS format.

-   [**Collect**](collect.md)

    ---

    Aggregate subject-level parcelstats into group-level tables.

-   [**Fetch**](fetch.md)

    ---

    Download normative connectomes and other assets to the local cache.

-   [**Info**](info.md)

    ---

    Display available resources (atlases, connectomes).

-   [**Run**](run.md)

    ---

    Run lesion analyses (regional damage, functional and structural network mapping).

-   [**Tutorial**](tutorial.md)

    ---

    Setup tutorial data for learning Lacuna.

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

# Check for missing outputs after a batch run
lacuna audit rd /bids /output

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

