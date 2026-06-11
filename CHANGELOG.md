# Changelog

All notable changes to Lacuna are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While Lacuna is in alpha (`0.x`), minor releases may include breaking changes.

## [Unreleased]

## [0.1.0] - 2026-06-10

First public alpha release.

### Added

- **Analyses**: regional damage, functional lesion network mapping, structural
  lesion network mapping, and an accelerated matrix-based functional mapping
  implementation.
- **Command-line interface** with a BIDS-App-style workflow: `tutorial`,
  `bidsify`, `fetch`, `run`, `check`, `collect`, `parcellate`, and `info`.
- **Python API**: `SubjectData`, `Pipeline`, `analyze`, and `batch_process`.
- **Normative connectome fetching** for GSP1000 (functional), dTOR985 and
  HCP1065 (structural) from persistent, DOI-backed public sources.
- **Automatic spatial alignment** between supported MNI spaces via TemplateFlow.
- **Bundled atlases** (Schaefer / Tian variants) for regional damage with no
  download required.
- **Batch processing** across cohorts with sequential and vectorized
  strategies.
- **Provenance tracking** and asset envelopes for reproducible outputs.
- **Docker image** bundling MRtrix3 and TemplateFlow for reproducible runs.

[Unreleased]: https://github.com/m-petersen/lacuna/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/m-petersen/lacuna/releases/tag/v0.1.0
