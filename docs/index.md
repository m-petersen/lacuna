---
hide:
  - navigation
---

# Lacuna

A Python package for advanced brain lesion analysis.

Lacuna bridges the gap between individual lesion masks and normative brain data.
It provides a reproducible, BIDS-compatible workflow for lesion network mapping
and regional damage quantification.

!!! warning "This project is under active development and has not yet been fully validated. Use with caution."

## Install

```bash
pip install git+https://github.com/m-petersen/lacuna
```

## Analyses

<div class="grid cards" markdown>

-   **Functional network mapping**

    ---

    Map the functional brain circuitry linked to a lesion using resting-state functional connectivity.

-   **Structural network mapping**

    ---

    Map the structural disconnectivity of a lesion using normative tractogram data.

-   **Regional damage**

    ---

    Quantify regional damage by measuring lesion overlap with standard brain parcellation atlases.

</div>

## Usage

```bash
# Create a tutorial dataset with synthetic lesion masks
lacuna tutorial my_dataset

# Fetch the HCP1065 structural tractogram
lacuna fetch hcp1065 --output-dir connectomes

# Run structural network mapping
lacuna run snm my_dataset output \
    --connectome-path connectomes/hcp1065.tck \
    --mask-space MNI152NLin6Asym
```

For the full walkthrough, see the [Getting started](tutorials/getting-started.ipynb) tutorial. Note that Lacuna expects lesion masks to be in MNI space.

## Documentation

<div class="grid cards" markdown>

-   [**Tutorials**](tutorials/index.md)

    ---

    Step-by-step Jupyter notebook tutorials covering each analysis type.

-   [**How-to guides**](how-to/index.md)

    ---

    Practical guides for specific tasks beyond core analysis workflows.

-   [**Reference**](reference/index.md)

    ---

    CLI commands, options, and auto-generated API documentation.

-   [**Explanation**](explanation/index.md)

    ---

    Background knowledge for using the package.

</div>
