# CLI Reference

Command-line interface documentation for Lacuna.

## Overview

Lacuna provides a subcommand-based CLI for running lesion network mapping analyses, managing connectomes, and working with BIDS datasets.

```
lacuna <command> [options]
```

## Commands

<div class="grid cards" markdown>

-   :material-play:{ .lg .middle } **Run**

    ---

    Run lesion analyses (regional damage, functional and structural network mapping).

    [:octicons-arrow-right-24: Run](run.md)

-   :material-download:{ .lg .middle } **Fetch**

    ---

    Download normative connectomes and other assets to the local cache.

    [:octicons-arrow-right-24: Fetch](fetch.md)

-   :material-table:{ .lg .middle } **Collect**

    ---

    Aggregate subject-level parcelstats into group-level tables.

    [:octicons-arrow-right-24: Collect](collect.md)

-   :material-information:{ .lg .middle } **Info**

    ---

    Display available resources (atlases, connectomes).

    [:octicons-arrow-right-24: Info](info.md)

-   :material-file-swap:{ .lg .middle } **Bidsify**

    ---

    Convert a directory of NIfTI mask files to BIDS format.

    [:octicons-arrow-right-24: Bidsify](bidsify.md)

-   :material-school:{ .lg .middle } **Tutorial**

    ---

    Setup tutorial data for learning Lacuna.

    [:octicons-arrow-right-24: Tutorial](tutorial.md)

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

## See Also

- [Installation Guide](../../how-to/installation.md) — Setting up Lacuna
- [Docker Guide](../../how-to/docker.md) — Running via containers
- [Apptainer Guide](../../how-to/apptainer.md) — Running on HPC clusters
