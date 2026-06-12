# Changelog

All notable changes to Lacuna are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While Lacuna is in alpha, minor releases may include breaking changes.

## [Unreleased]

## [0.1.0] - 2026-06-12

First public alpha release.

### Added

- **Analyses**: focal damage, functional lesion network mapping (original and accelerated) structural  lesion network mapping.
- **Command-line interface**: `tutorial`,
  `bidsify`, `fetch`, `run`, `check`, `collect`, `prepare`, and `info`.
- **Python API**: `SubjectData`, `Pipeline`, `analyze`, and `batch_process`.
- **Normative connectome fetching** for GSP1000 (functional), dTOR985 and
  HCP1065 (structural) from DOI-backed public sources.
- **Automatic spatial alignment** between supported MNI spaces via nonlinear warps downloaded on first use.
- **Bundled atlases** Schaefer / Tian variants.
- **Batch processing** across cohorts with sequential and vectorized
  strategies.
- **Provenance tracking** and asset envelopes for reproducible outputs.
- **Docker image** bundling MRtrix3 for reproducible runs.

[Unreleased]: https://github.com/m-petersen/lacuna/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/m-petersen/lacuna/releases/tag/v0.1.0
