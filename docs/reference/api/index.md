# API Reference

This section contains automatically generated API documentation for all public modules in Lacuna.

## Core API

The main entry points for using Lacuna programmatically:

| Symbol | Description |
|--------|-------------|
| [`lacuna.SubjectData`](lacuna/core/subject_data.md) | Immutable container for subject data and analysis results |
| [`lacuna.Pipeline`](lacuna/core/pipeline.md) | Chainable analysis pipeline builder |
| [`lacuna.analyze`](lacuna/core/pipeline.md) | One-liner for standard analysis workflows |
| [`lacuna.batch_process`](lacuna/batch/api.md) | Parallel processing of multiple subjects |
| [`lacuna.data`](lacuna/data/index.md) | Access bundled example data |

## Analysis Modules

| Module | Description |
|--------|-------------|
| [`FunctionalNetworkMapping`](lacuna/analysis/functional_network_mapping.md) | Functional lesion network mapping |
| [`StructuralNetworkMapping`](lacuna/analysis/structural_network_mapping.md) | Structural disconnection analysis |
| [`RegionalDamage`](lacuna/analysis/regional_damage.md) | Atlas-based regional damage quantification |

## I/O and Assets

| Module | Description |
|--------|-------------|
| [`lacuna.io`](lacuna/io/index.md) | BIDS loading/export, format conversion |
| [`lacuna.assets`](lacuna/assets/index.md) | Connectomes, parcellations, templates |

## Full Module Index

See the sidebar for the complete module hierarchy, or browse:

- [lacuna](lacuna/index.md) - Package root