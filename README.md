# Lacuna

A scientific Python package for neuroimaging lesion analysis.

## Overview

Lacuna provides researchers with an end-to-end pipeline for:
- Loading binary mask data from NIfTI files and BIDS-compliant datasets
- Performing spatial preprocessing
- Conducting functional and structural network mapping analyses
- Exporting results in standard formats

## Features

- **Standardized Data Handling**: Work with a consistent `MaskData` API across all pipeline stages
- **BIDS Compliance**: First-class support for BIDS dataset organization and derivatives export
- **Provenance Tracking**: Automatic recording of all transformations for reproducibility
- **Spatial Correctness**: Built on validated neuroimaging libraries (nibabel, nilearn, templateflow)
- **Modular Architecture**: Easy to extend with new analysis modules
- **Network Mapping**: Functional and structural connectivity-based lesion network mapping
- **Registry System**: Pre-configured atlases, parcellations, and connectomes for reproducible analyses

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

### Load a Single Mask

```python
import nibabel as nib
from lacuna import MaskData

# Load a binary mask (e.g., lesion, ROI)
mask_img = nib.load("path/to/mask.nii.gz")
mask = MaskData(
    mask_img=mask_img,
    space="MNI152NLin6Asym",
    resolution=2.0,
    metadata={"subject_id": "sub-001"}
)

# Inspect the data
print(f"Space: {mask.space}")
print(f"Resolution: {mask.resolution}mm")
print(f"Volume: {mask.get_volume_mm3():.1f} mm³")
```

### Load a BIDS Dataset

```python
from lacuna.io import load_bids_dataset

# Load all subjects with masks from a BIDS dataset
dataset = load_bids_dataset("path/to/bids_dataset")

# Process each subject
for subject_id, mask in dataset.items():
    print(f"Processing {subject_id}: space={mask.space}, resolution={mask.resolution}mm")
```

### Functional Network Mapping

```python
from lacuna.analysis import FunctionalNetworkMapping
from lacuna.assets.connectomes import register_functional_connectome

# Register your functional connectome
register_functional_connectome(
    name="MyConnectome",
    space="MNI152NLin6Asym",
    resolution=2.0,
    data_path="/path/to/connectome",
    n_subjects=1000
)

# Initialize analyzer with the registered connectome
fnm = FunctionalNetworkMapping(
    connectome_name="MyConnectome",
    method="boes"
)

# Run analysis
result = fnm.run(mask)

# Access results
corr_map = result.results['FunctionalNetworkMapping']['correlation_map']
print(f"Correlation map shape: {corr_map.get_data().shape}")
```

### Structural Network Mapping

```python
from lacuna.analysis import StructuralNetworkMapping
from lacuna.assets.connectomes import register_structural_connectome

# Register your structural connectome
register_structural_connectome(
    name="MyTractogram",
    space="MNI152NLin2009cAsym",
    resolution=2.0,
    tractogram_path="/path/to/tractogram.tck",
    n_subjects=985
)

# Initialize with the registered connectome
snm = StructuralNetworkMapping(
    connectome_name="MyTractogram",
    parcellation_name="Schaefer2018_100Parcels7Networks"
)

# Run analysis (requires MRtrix3)
result = snm.run(mask)
```

### Batch Processing

```python
from lacuna import batch_process
from lacuna.analysis import RegionalDamage

# Process multiple masks in parallel
masks = [mask1, mask2, mask3]
analysis = RegionalDamage(
    parcel_names=["Schaefer2018_100Parcels7Networks"],
    threshold=0.5
)

results = batch_process(
    inputs=masks,
    analysis=analysis,
    n_jobs=-1,
    show_progress=True
)

# Extract results as DataFrame
from lacuna.batch import extract_parcel_table
df = extract_parcel_table(results, "RegionalDamage")
print(df.head())
```

### Save Results

```python
from lacuna.io.bids import export_bids_derivatives

# Export single subject results to BIDS derivatives format
export_bids_derivatives(
    mask_data=result,
    output_dir="derivatives/lacuna",
    export_lesion_mask=True,
    export_voxelmaps=True,
    export_parcel_data=True
)
```

## Requirements

- Python 3.10 or higher
- nibabel >= 5.0
- nilearn >= 0.10
- numpy >= 1.24
- scipy >= 1.10
- pandas >= 2.0
- templateflow >= 24.0

## Documentation

Full documentation and API reference: See `notebooks/comprehensive_api_test.ipynb` for examples.

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run fast tests (unit + contract, ~30s)
make test-fast

# Run full CI suite (~2 min)
make ci-native

# Run specific test categories
make test-unit          # Unit tests only (~15s)
make test-contract      # Contract tests only (~15s)
make test-integration   # Integration tests (~1 min)

# Run with coverage
make test-coverage
```

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Type check
make typecheck

# Run all checks (format + lint + typecheck)
make check
```

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{lacuna,
  title = {Lacuna},
  author = {Petersen, M},
  year = {2025},
  url = {https://github.com/lacuna/lacuna}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! See [DEVELOPMENT.md](DEVELOPMENT.md) for:
- Setting up your development environment
- Running tests locally (native and Docker)
- Code formatting and linting
- Versioning and release workflow

Quick start for contributors:
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests before committing
make test-fast

# Run full CI before pushing
make ci-native

# Format code
make format
```

## Support

- Issue Tracker: https://github.com/m-petersen/lacuna/issues
- Examples: See `notebooks/comprehensive_api_test.ipynb` and `examples/`
