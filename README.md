<p align="center">
  <img src="docs/assets/logo.svg" alt="Lacuna" width="200">
</p>

<h1 align="center">Lacuna</h1>

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-m--petersen.github.io%2Flacuna-blue)](https://m-petersen.github.io/lacuna)
[![status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/m-petersen/lacuna)

A scientific Python package for advanced brain lesion analysis. Lacuna bridges the gap between individual lesion masks and normative brain data, providing a reproducible, BIDS-compatible workflow for lesion network mapping and regional damage quantification.

> **Warning**
> Lacuna is under active development and not yet stable. APIs may change without notice. Use in research at your own discretion.

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