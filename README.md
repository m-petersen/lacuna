<p align="center">
  <img src="docs/assets/logo.svg" alt="Lacuna" width="200">
</p>

<h1 align="center">Lacuna</h1>

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://readthedocs.org/projects/lacuna-py/badge/?version=latest)](https://lacuna-py.readthedocs.io/en/latest/)
[![status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/m-petersen/lacuna)

A scientific Python package for advanced brain lesion analysis. Lacuna bridges the gap between individual lesion masks and normative brain data.

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

Full documentation including tutorials, how-to guides, and API reference: **[lacuna-py.readthedocs.io](https://lacuna-py.readthedocs.io)**