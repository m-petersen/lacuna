# Reference

Technical reference documentation for Lacuna.

## Documentation Sections

<div class="grid cards" markdown>

-   :material-console:{ .lg .middle } **Command-Line Interface**

    ---

    Complete CLI documentation with commands, options, and examples.

    [:octicons-arrow-right-24: CLI Reference](cli/index.md)

-   :material-api:{ .lg .middle } **Python API**

    ---

    Auto-generated API reference from source code docstrings.

    [:octicons-arrow-right-24: API Reference](api/lacuna/index.md)

</div>

## Quick Links

### Core Modules

- `lacuna.core` — Core data structures (`SubjectData`, `MaskData`)
- `lacuna.analysis` — Analysis modules (fLNM, sLNM, RegionalDamage)
- `lacuna.spatial` — Spatial operations and transformations

### Assets

- `lacuna.assets.connectomes` — Connectome management and registration
- `lacuna.assets.parcellations` — Brain parcellation atlases

### I/O & Processing

- `lacuna.io` — BIDS dataset loading and export
- `lacuna.batch` — Parallel batch processing

## About This Reference

This reference documentation is auto-generated from NumPy-style docstrings
in the source code using [mkdocstrings](https://mkdocstrings.github.io/).
For conceptual explanations, see the [Explanation](../explanation/index.md)
section.
