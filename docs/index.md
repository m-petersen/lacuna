---
hide:
  - navigation
---

## Lacuna: a scientific Python package for neuroimaging lesion analysis.

Lacuna bridges the gap between individual lesion masks and normative brain data, e.g. connectomes. It provides a reproducible, BIDS-compatible workflow that currently covers the following primary modes of analysis:

<div class="grid cards" markdown>

-   **Functional Lesion Network Mapping**

    ---

    Perform **fLNM** to map the functional brain circuitry linked to a lesion using resting-state functional connectivity.

-   **Structural Lesion Network Mapping**

    ---

    Perform **sLNM** to map the structural disconnectivity of a lesion using normative tractogram data.

-   **Regional Damage**

    ---

    Quantify regional damage by measuring lesion overlap with standard brain parcellation atlases.

</div>

## Quick Start

Get up and running in minutes.

=== "1. Install"

    Lacuna is available via the github repository.
    
    ```bash
    pip install git+https://github.com/m-petersen/lacuna
    ```

=== "2. Setup tutorial data"

    Setup tutorial dataset with 3 synthetic lesion masks in MNI space.

    ```bash
    lacuna tutorial bids_tutorial
    ```

=== "3. Fetch data"

    Download necessary connectome. You will need an API key from [Figshare](https://figshare.com/account/login) to automatically download the [dTOR985 connectome](https://springernature.figshare.com/articles/dataset/dTOR-985_structural_connectome_full_tractogram_trk_file/25209947?file=44515847).

    ```bash
    lacuna fetch dtor985 \
        --output-dir conn \
        --api-key <YOUR_FIGSHARE_TOKEN>
    ```

=== "4. Run analysis"

    Run a standard Structural Network Mapping analysis on the tutorial dataset.

    ```bash
    lacuna run snm \
        bids_tutorial \
        lacuna_output \
        --connectome-path conn/dTOR_full_tractogram.tck 
    ```

## Documentation

This documentation is organized by the [Diátaxis](https://diataxis.fr/) framework:

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Step-by-step lessons to introduce you to the package.

    [:octicons-arrow-right-24: Start learning](tutorials/index.md)

-   :material-tools:{ .lg .middle } **How-to Guides**

    ---

    Solutions for specific analysis goals.

    [:octicons-arrow-right-24: Find a guide](how-to/index.md)

-   :material-book-open-page-variant:{ .lg .middle } **Reference**

    ---

    Technical descriptions of the API, CLI, and config.

    [:octicons-arrow-right-24: Browse reference](reference/index.md)

-   :material-lightbulb:{ .lg .middle } **Explanation**

    ---

    Discussion of background concepts and design.

    [:octicons-arrow-right-24: Learn concepts](explanation/index.md)

</div>

## Key Features

| Feature | Description |
| :--- | :--- |
| **BIDS-native** | Designed to work seamlessly with BIDS-formatted datasets out of the box. |
| **Reproducible** | Fully containerized workflows available via Docker and Apptainer. |
| **Efficient** | Implementations optimized for fast analyses. |
| **Extensible** | Plugin architecture allows for custom analyses and atlas integration. |
