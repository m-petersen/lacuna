<p align="center">
  <img src="docs/assets/logo.svg" alt="Lacuna" width="200">
</p>

<h1 align="center">Lacuna</h1>

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://readthedocs.org/projects/lacuna-py/badge/?version=latest)](https://lacuna-py.readthedocs.io/en/latest/)
[![status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/m-petersen/lacuna)

When a stroke, tumor, or other focal injury damages brain tissue, understanding its consequences requires placing the lesion in the broader context of brain organization.

Lacuna is an open-source toolbox designed to facilitate this process. Using normative reference data from healthy individuals, including functional and structural connectivity datasets and anatomical atlases, it generates a range of measures that characterize a lesion beyond its anatomical location alone.

Lacuna is built to be easy to adopt: you provide lesion masks in MNI space, and it takes care of the rest — fetching the normative reference data, aligning coordinate spaces, and writing BIDS-organized outputs through a command line interface. Developed for multicenter research, it standardizes inputs and outputs, scales from a single mask to entire cohorts with batch processing, supports deployment on high-performance computing (HPC) systems through containerized environments and records traceable provenance for every result, all within a modular architecture designed to grow toward further lesion-characterization methods.

<p align="center">
  <img src="docs/assets/method_figures/pipeline.png" alt="Lacuna analysis pipeline" width="820">
</p>

## What Lacuna does

| Analysis | What it answers | Needs |
|---|---|---|
| **Focal damage** | Direct damage at the lesion site — how much does it overlap each region of a brain atlas? | Bundled atlases |
| **Functional network mapping** | Which functional circuit is connected to the lesion? | Normative functional connectome |
| **Structural network mapping** | Which white-matter tracts does the lesion disconnect? | Normative tractogram + MRtrix3 |
| **Accelerated functional mapping** | The functional map, via a fast matrix-based method | Parcellated functional connectome |

Lacuna treats every reference — connectomes, atlases, templates — as a uniform, extensible asset type, so its characterization toolkit is designed to grow toward further normative data and modalities.

## Install

Install the latest version from GitHub:

```bash
pip install git+https://github.com/m-petersen/lacuna
```

Requires Python ≥ 3.10. We recommend a fresh virtual environment (`venv` or `conda`).

**Additional requirements by analysis:**

- **Structural network mapping** needs [MRtrix3](https://www.mrtrix.org/) available on your `PATH`.
- **Normative connectomes** are downloaded on demand and are large — the functional GSP1000 is ~200 GB, the structural dTOR985 ~11 GB, and HCP1065 ~1.5 GB. Make sure you have the disk space before fetching. Focal damage needs none of these.

## Quickstart

```bash
# 1. Create a tutorial dataset with synthetic lesion masks (BIDS-organized)
lacuna tutorial my_dataset

# 2. Fetch the HCP1065 tractogram (~1.5 GB)
lacuna fetch hcp1065 --output-dir connectomes

# 3. Run structural network mapping
lacuna run snm my_dataset output \
    --connectome-path connectomes/hcp1065.tck \
    --mask-space MNI152NLin6Asym
```

Lacuna expects lesion masks in MNI space (`MNI152NLin6Asym` or `MNI152NLin2009cAsym`). To get masks there, see the [spatial normalization guide](https://lacuna-py.readthedocs.io/en/latest/how-to/spatial-normalization/).

## Documentation

Documentation — tutorials, how-to guides, conceptual background, and API reference — is at **[lacuna-py.readthedocs.io](https://lacuna-py.readthedocs.io)**.

> **Note**
> Lacuna is under active development and has not yet been fully validated. APIs may change without notice. Use in research at your own discretion.

## Citation

If you use Lacuna in your research, please cite it. A citation entry is provided in [`CITATION.cff`](CITATION.cff). <!-- TODO: add DOI / preprint reference once available -->

## Contributing

Bug reports, feature requests, and pull requests are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) and the [issue tracker](https://github.com/m-petersen/lacuna/issues).

## Acknowledgements

Lacuna is developed as part of the [Meta VCI Map Consortium](https://metavcimap.org/), an international collaboration advancing multicenter lesion analysis in vascular cognitive impairment.

## License

MIT — see [LICENSE](LICENSE).
