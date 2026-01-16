# CLI Reference

Command-line interface documentation for Lacuna.

## Overview

Lacuna provides a BIDS-Apps compatible command-line interface for running
lesion network mapping analyses directly from the terminal.

## Commands

<div class="grid cards" markdown>

-   :material-application-braces:{ .lg .middle } **BIDS-Apps Interface**

    ---

    Main entry point for participant and group analyses following BIDS-Apps
    conventions.

    [:octicons-arrow-right-24: BIDS-Apps](bids-apps.md)

-   :material-download:{ .lg .middle } **Fetch Command**

    ---

    Download normative connectomes and other assets to the local cache.

    [:octicons-arrow-right-24: Fetch](fetch.md)

</div>

## Quick Usage

```bash
# Basic BIDS-Apps usage
lacuna /bids/input /output participant

# Run specific analysis
lacuna /bids/input /output participant --analysis flnm

# Fetch connectomes
lacuna fetch --list
lacuna fetch --connectome functional --name HCP_S1200
```

## Getting Help

```bash
# General help
lacuna --help

# Command-specific help
lacuna fetch --help
```

## See Also

- [Installation Guide](../../how-to/installation.md) — Setting up Lacuna
- [Docker Guide](../../how-to/docker.md) — Running via containers
