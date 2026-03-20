---
hide:
  - navigation
---

## Lacuna: a scientific Python package for advanced brain lesion analysis.

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

<div class="grid cards" markdown>

-   **Tutorials**

    ---

    Learn how to use Lacuna with hands-on Jupyter notebook tutorials.

    [Start learning](tutorials/index.md)

-   **Reference**

    ---

    CLI commands, options, and auto-generated API documentation.

    [Browse reference](reference/index.md)

</div>

## Key Features

| Feature | Description |
| :--- | :--- |
| **BIDS-native** | Designed to work seamlessly with BIDS-formatted datasets out of the box. |
| **Reproducible** | Fully containerized workflows available via Docker and Apptainer. |
| **Efficient** | Implementations optimized for fast analyses. |
| **Extensible** | Plugin architecture allows for custom analyses and atlas integration. |
