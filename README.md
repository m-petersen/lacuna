# Lacuna

![Status: Alpha](https://img.shields.io/badge/status-alpha-orange)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A scientific Python package for neuroimaging lesion analysis.

> ⚠️ **Alpha Status**: This package is under active development. APIs may change. Documentation is coming soon.

## Overview

Lacuna provides researchers with an end-to-end pipeline for:
- Loading binary mask data from NIfTI files and BIDS-compliant datasets
- Performing spatial preprocessing
- Conducting functional and structural network mapping analyses
- Exporting results in standard formats

## Features

**Implemented and Tested:**
- ✅ **Functional Lesion Network Mapping (fLNM)**: Connectivity-based lesion analysis using normative connectomes
- ✅ **Structural Lesion Network Mapping (sLNM)**: Tractography-based disconnection analysis (requires MRtrix3)
- ✅ **Regional Damage Analysis**: Quantify lesion overlap with brain parcellations
- ✅ **Parcel Aggregation**: Aggregate voxel-wise results to standard atlases
- ✅ **BIDS Support**: Load BIDS datasets and export BIDS-compliant derivatives
- ✅ **Batch Processing**: Parallel processing with progress tracking

**Architecture:**
- **Flexible Usage**: Full-featured CLI for batch processing + Python API for custom workflows
- **Standardized Data Handling**: Consistent `SubjectData` API across all pipeline stages
- **Provenance Tracking**: Automatic recording of all transformations for reproducibility
- **Spatial Correctness**: Built on validated neuroimaging libraries (nibabel, nilearn, templateflow)
- **Modular Architecture**: Easy to extend with new analysis modules (auto-discovery)
- **Registry System**: Pre-configured atlases, parcellations, and connectomes

## Installation

### Basic Installation

```bash
pip install git+https://github.com/m-petersen/lacuna
```

### From Source

```bash
git clone https://github.com/lacuna/lacuna.git
cd lacuna
pip install -e .
```

## Docker & Singularity coming soon


## Quick Start

Lacuna can be used via the **command-line interface (CLI)** for standard workflows or the **Python API** for programmatic control.

### Command-Line Interface

```bash
# 1. Setup tutorial data (3 synthetic lesions in MNI space)
lacuna tutorial bids_tutorial

# 2. Download a normative connectome (requires Figshare API key)
lacuna fetch dtor985 --output-dir conn --api-key YOUR_KEY

# 3. Run structural network mapping
lacuna run snm bids_tutorial output --connectome-path conn/dTOR_full_tractogram.tck

# 4. Run functional network mapping  
lacuna run fnm bids_tutorial output --connectome-path conn/gsp1000

# 5. Run regional damage analysis
lacuna run rd bids_tutorial output --atlas Schaefer2018_100Parcels7Networks
```

See `lacuna --help` for all available commands.

### Python API

#### Load a Single Mask

```python
import nibabel as nib
from lacuna import SubjectData

# Load a binary mask (e.g., lesion, ROI)
mask_img = nib.load("path/to/mask.nii.gz")
subject = SubjectData(
    mask_img=mask_img,
    space="MNI152NLin6Asym",
    resolution=2.0,
    metadata={"subject_id": "sub-001"}
)

# Inspect the data
print(f"Space: {subject.space}")
print(f"Resolution: {subject.resolution}mm")
print(f"Volume: {subject.get_volume_mm3():.1f} mm³")
```

#### Load a BIDS Dataset

```python
from lacuna.io import load_bids_dataset

# Load all subjects with masks from a BIDS dataset
dataset = load_bids_dataset("path/to/bids_dataset")

# Process each subject
for subject_id, mask in dataset.items():
    print(f"Processing {subject_id}: space={mask.space}, resolution={mask.resolution}mm")
```

#### Functional Network Mapping

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

#### Structural Network Mapping

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

#### Batch Processing

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

#### Save Results

```python
from lacuna.io.bids import export_bids_derivatives

# Export single subject results to BIDS derivatives format
export_bids_derivatives(
    subject_data=result,
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

## Documentation coming soon

## License

MIT License - see LICENSE file for details
