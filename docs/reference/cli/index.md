# CLI

Command-line interface documentation for Lacuna.

## Overview

Lacuna provides a subcommand-based CLI for running lesion network mapping analyses, managing connectomes, and organizing data using BIDS-style naming conventions.

```
lacuna <command> [options]
```

## Commands

<div class="grid cards" markdown>

-   [**Check**](check.md)

    ---

    Validate input masks and check output completeness.

-   [**Bidsify**](bidsify.md)

    ---

    Convert a directory of NIfTI mask files to BIDS format.

-   [**Collect**](collect.md)

    ---

    Aggregate subject-level parcelstats into group-level tables.

-   [**Fetch**](fetch.md)

    ---

    Download normative connectomes and other assets to the local cache.

-   [**Prepare**](prepare.md)

    ---

    Precompute connectome-derived data products — currently `prepare afnm`, which builds the parcel-level connectivity matrix that accelerated FNM consumes.

-   [**Info**](info.md)

    ---

    Display available resources (atlases, connectomes).

-   [**Run**](run.md)

    ---

    Run lesion analyses (focal damage, functional, accelerated functional, and structural network mapping).

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

# Run focal damage analysis
lacuna run fd /bids /output --parcel-atlases schaefer2018parcels100networks7

# Run functional network mapping
lacuna run fnm /bids /output --connectome-path /path/to/gsp1000_batches

# Aggregate results across subjects
lacuna collect /output

# Check for missing outputs after a batch run
lacuna check fd /bids /output

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

