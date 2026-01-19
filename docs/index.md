---
hide:
  - navigation
---

# Lacuna

<div align="center">

**A scientific Python package for neuroimaging lesion analysis.**

[Get Started](tutorials/getting-started.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/m-petersen/lacuna){ .md-button }

</div>

---

## What is Lacuna?

Lacuna bridges the gap between individual lesion masks and normative brain atlases. It provides a reproducible, BIDS-compatible workflow that currently covers three primary modes of analysis:

<div class="grid cards" markdown>

-   **Functional Network Mapping**

    ---

    Perform **fLNM** to map the functional brain circuitry linked to a lesion using resting-state functional connectivity.

-   **Structural Network Mapping**

    ---

    Perform **sLNM** to map the structural disconnectivity of a lesion using normative tractogram data.

-   **Regional Quantification**

    ---

    Quantify regional damage by measuring lesion overlap with standard brain parcellation atlases.

</div>

## Quick Start

Get up and running in minutes.

=== "1. Install"

    Lacuna is available via the github repository.
    
    ```bash
    pip install git+[https://github.com/m-petersen/lacuna](https://github.com/m-petersen/lacuna)
    ```

=== "2. Fetch Data"

    Download necessary connectomes. You will need an API key from [Figshare](https://figshare.com/account/login) to automatically download the [dTOR985 connectome](https://springernature.figshare.com/articles/dataset/dTOR-985_structural_connectome_full_tractogram_trk_file/25209947?file=44515847).

    ```bash
    lacuna fetch dtor985 \
        --output-dir conn \
        --api-key <YOUR_FIGSHARE_TOKEN>
    ```

=== "3. Run Analysis"

    Run a standard Structural Network Mapping analysis on a BIDS dataset.

    ```bash
    lacuna run snm \
        /path/to/bids_input \
        /path/to/output_dir \
        --connectome-path conn/dTOR_full_tractogram.tck 
    ```

## Documentation

This documentation is organized by the [Diátaxis](https://diataxis.fr/) framework:

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Step-by-step lessons to help you start learning.

    [:octicons-arrow-right-24: Start learning](tutorials/index.md)

-   :material-tools:{ .lg .middle } **How-to Guides**

    ---

    Recipes and solutions for specific analysis goals.

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
| **BIDS-Native** | Designed to work seamlessly with BIDS-formatted datasets out of the box. |
| **Reproducible** | Fully containerized workflows available via Docker and Apptainer. |
| **Extensible** | Plugin architecture allows for custom analyses and atlas integration. |
| **Parallelized** | Optimized for speed with parallel processing of multiple subjects. |

---

<div align="center" markdown>

**Need help?**
[Report a Bug](https://github.com/m-petersen/lacuna/issues){ .md-button }

</div>