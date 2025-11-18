# Lacuna

A scientific Python package for neuroimaging lesion analysis.

## Overview

Lacuna provides researchers with an end-to-end pipeline for:
- Loading lesion data from NIfTI files and BIDS-compliant datasets
- Performing spatial preprocessing and normalization
- Conducting lesion network mapping analyses
- Exporting results in standard formats

## Features

- **Standardized Data Handling**: Work with a consistent `LesionData` API across all pipeline stages
- **BIDS Compliance**: First-class support for BIDS dataset organization
- **Provenance Tracking**: Automatic recording of all transformations for reproducibility
- **Spatial Correctness**: Built on validated neuroimaging libraries (nibabel, nilearn)
- **Modular Architecture**: Easy to extend with new analysis modules

## Installation

### Basic Installation

```bash
pip install lacuna
```

### With Optional Dependencies

```bash
# For visualization features
pip install lacuna[viz]

# For BIDS support
pip install lacuna[bids]

# For development
pip install lacuna[dev]

# Install everything
pip install lacuna[all]
```

### From Source

```bash
git clone https://github.com/lacuna/lacuna.git
cd lacuna
pip install -e ".[dev]"
```

## Quick Start

### Load a Single Lesion Mask

```python
from lacuna import LesionData

# Load a NIfTI lesion mask
lesion = LesionData.from_nifti("path/to/lesion_mask.nii.gz")

# Inspect the data
print(f"Subject ID: {lesion.metadata['subject_id']}")
print(f"Lesion volume: {lesion.get_volume_mm3()} mm³")
print(f"Coordinate space: {lesion.get_coordinate_space()}")
```

### Load a BIDS Dataset

```python
from lacuna.io import load_bids_dataset

# Load all subjects with lesion masks
dataset = load_bids_dataset("path/to/bids_dataset")

# Process each subject
for subject_id, lesion in dataset.items():
    print(f"Processing {subject_id}: {lesion.get_volume_mm3()} mm³")
```

### Spatial Normalization

```python
from lacuna import LesionData
from lacuna.preprocess import normalize_to_mni

# Load lesion in native space
lesion = LesionData.from_nifti("lesion_native.nii.gz")

# Normalize to MNI152 space
lesion_mni = normalize_to_mni(lesion, template="MNI152_2mm")

# Check provenance
print(f"Transformations applied: {len(lesion_mni.provenance)}")
```

### Lesion Network Mapping

```python
from lacuna.analysis import LesionNetworkMapping

# Initialize analyzer
lnm = LesionNetworkMapping(
    connectome="HCP1200",
    atlas="Schaefer2018_400Parcels"
)

# Run analysis (requires MNI152 space)
results = lnm.fit(lesion_mni)

# Access network disruption scores
for network, score in results['network_disruption_scores'].items():
    print(f"{network}: {score:.3f}")
```

### Save Results

```python
from lacuna.io import export_bids_derivatives

# Export to BIDS derivatives format
export_bids_derivatives(
    lesion_mni,
    output_dir="derivatives/lacuna-v0.1.0",
    include_images=True,
    include_results=True
)
```

## Requirements

- Python 3.10 or higher
- nibabel >= 5.0
- nilearn >= 0.10
- numpy >= 1.24
- scipy >= 1.10
- pandas >= 2.0

## Documentation

Full documentation is available at: https://lacuna.readthedocs.io

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=lacuna --cov-report=html

# Run specific test categories
pytest -m contract  # Contract tests only
pytest -m integration  # Integration tests only
```

### Code Quality

```bash
# Format code
black src tests

# Lint code
ruff check src tests

# Type check
mypy src
```

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{lacuna,
  title = {Lacuna},
  author = {Lacuna Contributors},
  year = {2025},
  url = {https://github.com/lacuna/lacuna}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! See [DEVELOPMENT.md](DEVELOPMENT.md) for:
- Setting up your development environment
- Running tests locally with `act`
- Code formatting and linting
- Versioning and release workflow

Quick start for contributors:
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install act (task runner)
brew install act  # macOS
# or: curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run tests
act -j test

# Format code
act -j format -W .github/workflows/local-format.yml
```

## Support

- Issue Tracker: https://github.com/lacuna/lacuna/issues
- Documentation: https://lacuna.readthedocs.io
