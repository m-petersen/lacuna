# CLI

Command-line interface documentation for Lacuna.

## Overview

Lacuna provides a subcommand-based CLI for running lesion network mapping analyses, managing connectomes, and organizing data using BIDS-style naming conventions.

```
lacuna <command> [options]
```

## Commands

<div class="grid cards" markdown>

-   [**Tutorial**](01-tutorial.md)

    ---

    Set up the bundled tutorial data for learning Lacuna.

-   [**Bidsify**](02-bidsify.md)

    ---

    Convert a directory of NIfTI mask files to BIDS format.

-   [**Fetch**](03-fetch.md)

    ---

    Download normative connectomes and other assets to the local cache.

-   [**Prepare**](04-prepare.md)

    ---

    Precompute connectome-derived data products — currently `prepare afnm`, which builds the parcel-level connectivity matrix that accelerated FNM consumes.

-   [**Run**](05-run.md)

    ---

    Run lesion analyses (focal damage, functional, accelerated functional, and structural network mapping).

-   [**Check**](06-check.md)

    ---

    Validate input masks and check output completeness.

-   [**Collect**](07-collect.md)

    ---

    Aggregate subject-level parcelstats into group-level tables.

-   [**Info**](08-info.md)

    ---

    Display available resources (atlases, connectomes).

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

