# Lacuna

**Lacuna** is a Python package for lesion-network mapping in neuroimaging research.

## What is Lacuna?

Lacuna provides tools for analyzing how brain lesions affect neural networks. It supports:

- **Functional Lesion Network Mapping (fLNM)** — Map lesion effects using resting-state connectivity
- **Structural Lesion Network Mapping (sLNM)** — Map lesion effects using diffusion tractography
- **Regional Damage Quantification** — Measure damage across brain atlas regions

## Quick Start

```bash
# Install Lacuna
pip install lacuna

# Run a basic analysis
lacuna /bids/input /output participant --analysis flnm
```

## Documentation Structure

This documentation follows the [Diátaxis](https://diataxis.fr/) framework:

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Learning-oriented guides for newcomers

    [:octicons-arrow-right-24: Start learning](tutorials/index.md)

-   :material-tools:{ .lg .middle } **How-to Guides**

    ---

    Task-oriented recipes for specific goals

    [:octicons-arrow-right-24: Find a guide](how-to/index.md)

-   :material-book-open-page-variant:{ .lg .middle } **Reference**

    ---

    Technical descriptions of the API and CLI

    [:octicons-arrow-right-24: Browse reference](reference/index.md)

-   :material-lightbulb:{ .lg .middle } **Explanation**

    ---

    Understanding concepts and design decisions

    [:octicons-arrow-right-24: Learn concepts](explanation/index.md)

</div>

## Features

- **BIDS-compatible** — Works with standard BIDS-formatted datasets
- **Reproducible** — Containerized workflows with Docker and Apptainer
- **Extensible** — Plugin architecture for custom analyses and atlases
- **Fast** — Optimized for parallel processing of multiple subjects

## Getting Help

- [GitHub Issues](https://github.com/lacuna/lacuna/issues) — Bug reports and feature requests
- [Discussions](https://github.com/lacuna/lacuna/discussions) — Questions and community support
