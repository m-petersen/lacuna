# Lacuna

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A scientific Python package for advanced brain lesion analysis. Lacuna bridges the gap between individual lesion masks and normative brain data, providing a reproducible, BIDS-compatible workflow for lesion network mapping and regional damage quantification.

## Install

```bash
pip install git+https://github.com/m-petersen/lacuna
```

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

Lacuna expects lesion masks in MNI space (`MNI152NLin6Asym` or `MNI152NLin2009cAsym`).

## Documentation

Full documentation including tutorials, how-to guides, and API reference: **[m-petersen.github.io/lacuna](https://m-petersen.github.io/lacuna)**