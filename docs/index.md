---
hide:
  - navigation
---

# Lacuna

A scientific Python package for advanced brain lesion analysis.

Lacuna bridges the gap between individual lesion masks and normative brain data.
It provides a reproducible, BIDS-compatible workflow for lesion network mapping
and regional damage quantification.

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
lacuna tutorial my_dataset --raw

# Convert a directory of NIfTI masks into a BIDS dataset
lacuna bidsify my_dataset my_dataset_bids

# Run a regional damage analysis
lacuna run rd my_dataset_bids output --parcel-atlases Schaefer2018_100Parcels7Networks

# Collect results into group-level tables
lacuna collect output --pattern "*schaefer2018*" --output-dir /tmp/group_outputs()
```

For the full walkthrough, see the [Getting Started](tutorials/getting-started.ipynb) tutorial. Note that Lacuna expects lesion masks to be in MNI space.

## Documentation

<div class="grid cards" markdown>

-   **Tutorials**

    ---

    Step-by-step Jupyter notebook tutorials covering each analysis type.

    [Start learning](tutorials/index.md)

-   **How-to Guides**

    ---

    Practical guides for specific tasks like spatial normalization.

    [Find a guide](how-to/index.md)

-   **Reference**

    ---

    CLI commands, options, and auto-generated API documentation.

    [Browse reference](reference/index.md)

-   **Explanation**

    ---

    Background on coordinate spaces and Lacuna's design.

    [Learn concepts](explanation/index.md)

</div>
