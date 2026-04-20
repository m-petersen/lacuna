# Lesion Neurotransmitter Mapping

**Spec ID**: 009-neurotransmitter-mapping
**Date**: 2026-04-20
**Status**: Draft

## Overview

Extend lacuna with lesion neurotransmitter mapping: derive per-patient NT biomarkers by scoring binary lesion masks against PET receptor/transporter density maps, using three complementary spatial approaches (local, structural, functional). Includes REACT-enriched variants that replace static PET maps with functionally-informed NT representations.

## Design Principles

- **Three independent analyses**: `lntm` (local), `sntm` (structural), `fntm` (functional) — each a `BaseAnalysis` subclass, each independently runnable
- **Shared atlas engine**: a new `src/lacuna/atlas/` module that handles atlas loading, averaging, z-scoring, caching, and scoring. NT mapping is the first consumer; the engine's internals are representation-agnostic to support future atlas domains (metabolic, transcriptomic)
- **Global scores as primary output**: one scalar per NT target per patient per analysis. Regional (per-parcel) scores available when the user specifies a parcellation atlas
- **Two-level configuration**: map selection at prepare time (which raw maps per target), target subsetting at run time (which targets to score)
- **Static and REACT-enriched modes**: all three analyses support `--enriched` to use REACT-derived NT maps instead of static PET averages

## Module Layout

```
src/lacuna/
  atlas/                                        # NEW: shared atlas engine
    __init__.py
    types.py                                    # VoxelAtlas (+ParcelAtlas, SurfaceAtlas stubs)
    store.py                                    # load, average, z-score, cache, fetch
    scoring.py                                  # scoring functions
    config.py                                   # NT presets, grouping, map selection
  analysis/
    local_neurotransmitter_mapping.py           # NEW
    structural_neurotransmitter_mapping.py       # NEW
    functional_neurotransmitter_mapping.py       # NEW
    local_damage.py                             # RENAMED from regional_damage.py
    structural_network_mapping.py               # MODIFIED: add keep_filtered_tractogram
  cli/
    prepare.py                                  # NEW: lacuna prepare subcommand
```

## Atlas Engine (`src/lacuna/atlas/`)

### `types.py` — Atlas Representations

**`VoxelAtlas`**: a collection of named 3D brain maps in a common space.

- `maps: dict[str, nib.Nifti1Image]` — one z-scored, averaged map per target (e.g., `{"5HT1a": <img>, "D1": <img>, ...}`)
- `space: str` — coordinate space (e.g., `MNI152NLin6Asym`)
- `resolution: float` — voxel size in mm
- `domain: str` — atlas domain identifier (e.g., `"neurotransmitter"`)
- `metadata: dict` — source info, map selection config, creation timestamp
- `targets: list[str]` — ordered list of target names
- `to_matrix(mask) -> np.ndarray` — extract (n_targets, n_voxels_masked) matrix
- `get_map(target) -> nib.Nifti1Image` — single target map
- `subset(targets) -> VoxelAtlas` — return atlas with only the specified targets
- `resample_to(space, resolution) -> VoxelAtlas` — resample all maps to a target space/resolution

**`ParcelAtlas`** and **`SurfaceAtlas`**: defined as empty classes with matching interface signatures. Not implemented in v1.

### `store.py` — Atlas Lifecycle

**`build_nt_atlas(source_dir, map_config=None) -> VoxelAtlas`**:

1. Scan `source_dir` for NIfTI files matching `target-{TARGET}_tracer-{TRACER}_..._pub-{PUB}_...` naming
2. Group files by target (parsed from `target-{X}`)
3. Apply map selection config if provided (specific tracers per target, exclusions)
4. For each target: load all selected maps, average voxelwise (excluding zeros from the average — zeros indicate outside-coverage voxels), z-score the result
5. Return a `VoxelAtlas` with one map per target

**`save_atlas(atlas, cache_dir)`** / **`load_atlas(cache_dir) -> VoxelAtlas`**: serialize/deserialize to disk. Store maps as NIfTI, metadata as JSON manifest.

**Cache location**: `~/.lacuna/cache/atlases/{domain}/` (configurable via env var `LACUNA_CACHE_DIR` or config file).

### `scoring.py` — Scoring Functions

**`score_focal(atlas, lesion_mask, aggregation="mean") -> dict[str, float]`**:
- For each target: extract atlas values within the binary lesion mask, exclude zeros, compute mean (or sum)
- Returns `{target_name: score}`

**`score_structural_endpoints(atlas, streamline_endpoints, intersecting_ids, aggregation="mean") -> tuple[dict[str, float], int]`**:
- For each intersecting streamline: compute mean of the two endpoint NT values (per target)
- Sum these per-streamline values across all intersecting streamlines → one score per target
- Also returns the count of intersecting streamlines (covariate, independent of NT)
- Endpoint combination rule: mean (hardcoded for v1)

**`score_functional_overlap(atlas, connectivity_map, aggregation="mean") -> dict[str, float]`**:
- Threshold connectivity map to positive values only (default)
- For each target: weighted mean of atlas values x positive connectivity values, zeros excluded
- Returns `{target_name: score}`

**`score_react_temporal(nt_timeseries, lesion_timeseries) -> dict[str, float]`**:
- Correlate lesion BOLD timeseries with each NT target's REACT stage 1 timeseries
- Returns `{target_name: correlation}`

All scoring functions accept an optional `parcel_mask` parameter. When provided, scoring is restricted to voxels within that parcel — enabling regional scoring with the same functions.

### `config.py` — NT Configuration

**Target grouping**: parsed from PET map filenames. The `target-{X}` field defines groups:
- `5HT1a`, `5HT1b`, `5HT2a`, `5HT4`, `5HT6`, `5HTT` (serotonin)
- `D1`, `D23`, `DAT`, `FDOPA` (dopamine)
- `VAChT`, `M1`, `A4B2` (cholinergic)
- `NET` (norepinephrine)
- `GABAa`, `GABAa5` (GABA)
- `CB1` (cannabinoid)
- `MOR`, `KOR` (opioid)
- `H3` (histamine)
- `mGluR5`, `NMDA` (glutamate)
- `VMAT2` (vesicular monoamine)

**Presets** (shipped as YAML config):
- `all` — all available targets
- `dopaminergic` — D1, D23, DAT
- `serotonergic` — 5HT1a, 5HT1b, 5HT2a, 5HT4, 5HT6, 5HTT
- `cholinergic` — VAChT, M1, A4B2
- `monoaminergic` — serotonergic + dopaminergic + NET
- User-specified subsets via list of target names

**Map selection config** (YAML, optional):

```yaml
targets:
  5HT1a: beliveau2017          # use one specific tracer study
  5HT1b: [savli2012, gallezot2010]  # use two, average them
  D1: all                      # use all available (default)
  DAT: exclude                 # skip entirely
  # unlisted targets: use all available
```

Selection specified by publication key parsed from `pub-{KEY}` in filenames.

## Three Analysis Classes

### Common to All Three

- Inherit `BaseAnalysis`
- Auto-discovered by `AnalysisRegistry`
- Accept `targets` parameter: preset name or list of target names, default `"all"`
- Accept `enriched` parameter: `bool`, default `False`. When `True`, use REACT-enriched NT atlas
- Accept `parcel_atlases` and `custom_parcellation` parameters for regional scoring (matching existing `--parcel-atlases` and `--custom-parcellation` CLI flags)
- `TARGET_SPACE = None` — adapt atlas to mask space/resolution (resample the atlas, not the mask)
- `batch_strategy = "sequential"`
- Output keys: flat `{target_name}` pattern per result namespace (e.g., `results["LocalNeurotransmitterMapping"]["5HT1a"]`)

### `LocalNeurotransmitterMapping` (`lntm`)

**Requires**: lesion mask (from `SubjectData`), prepared NT atlas (from `prepare lntm`)

**Computation**:
1. Load NT atlas, resample to mask space/resolution
2. For each target: score_focal(atlas, lesion_mask)
3. If parcellation requested: for each parcel, score_focal restricted to parcel voxels

**Outputs**:
- `ScalarMetric` per target — global local NT score
- `ParcelData` per target per atlas — regional scores (when parcellation specified)

**No dependencies on other analyses.**

### `StructuralNeurotransmitterMapping` (`sntm`)

**Requires**: lesion mask, prepared NT atlas, tractogram, precomputed endpoint NT weights (from `prepare sntm`)

**Resilience chain for streamline data**:
1. Check `SubjectData.results` for sLNM filtered tractogram (`StructuralNetworkMapping` with `keep_filtered_tractogram=True`)
2. Check output directory for previously written sLNM outputs
3. Compute streamline filtering itself (using MRtrix `tckedit`)

**Falls back to on-the-fly endpoint sampling** if precomputed endpoint weight matrix is unavailable.

**Computation**:
1. Obtain lesion-intersecting streamline IDs (via resilience chain)
2. For each target: score_structural_endpoints(atlas, endpoints, intersecting_ids)
3. If parcellation requested: for each parcel, restrict to streamlines with at least one endpoint in the parcel

**Outputs**:
- `ScalarMetric` per target — global structural NT score
- `ScalarMetric`: `streamline_count` — number of intersecting streamlines (covariate, not per-NT)
- `ParcelData` per target per atlas — regional scores (when parcellation specified)

### `FunctionalNeurotransmitterMapping` (`fntm`)

**Requires**: lesion mask, prepared NT atlas, normative fMRI connectome data

**Computes functional connectivity internally** — does not depend on a prior `FunctionalNetworkMapping` run.

**Static mode** (`enriched=False`):
1. Extract lesion BOLD timeseries from normative fMRI (mean signal within lesion mask per timepoint per subject)
2. Correlate with each voxel's timeseries across subjects → z-map (same method as existing `FunctionalNetworkMapping`)
3. For each target: score_functional_overlap(atlas, z_map)
4. Regional: restrict weighted mean to parcel voxels

**REACT-enriched mode** (`enriched=True`):
- **Global score**: extract lesion BOLD timeseries, correlate with REACT stage 1 NT timeseries → one correlation per target (`score_react_temporal`)
- **Regional score**: REACT stage 2 enriched atlas x z-map (computed internally by fntm, same as static mode) → weighted mean per parcel per target (`score_functional_overlap` with REACT atlas)

**Outputs**:
- `ScalarMetric` per target — global functional NT score
- `ParcelData` per target per atlas — regional scores (when parcellation specified)

## REACT Implementation

### Overview

REACT (Receptor-Enriched Analysis of functional Connectivity by Targets) derives functionally-informed NT representations from normative fMRI data using PET maps as spatial priors. Two stages:

- **Stage 1**: for each fMRI subject, regress BOLD spatial patterns onto NT atlas maps → NT-weighted timeseries
- **Stage 2**: for each subject, regress BOLD timeseries onto NT timeseries → enriched spatial maps. Fisher-z average across subjects.

### Computation Details

**Input**: prepared NT atlas (averaged, z-scored maps from `prepare lntm`), normative fMRI connectome (e.g., GSP1000 in HDF5)

**Stage 1** (per subject):
```
x = atlas_matrix[mask_stage1, :]           # (n_voxels_s1, n_targets) — demeaned
y = bold_data[mask_stage1, :]              # (n_voxels_s1, n_timepoints) — demeaned
model = LinearRegression(fit_intercept=True).fit(x, y)
beta1 = model.coef_                        # (n_timepoints, n_targets) — NT timeseries
```

**Stage 2** (per subject):
```
x = standardize(beta1)                     # (n_timepoints, n_targets) — standardized
y = bold_data.T[:, mask_stage2]            # (n_timepoints, n_voxels_s2) — demeaned
model = LinearRegression(fit_intercept=True).fit(x, y)
beta2 = model.coef_                        # (n_voxels_s2, n_targets) — enriched maps
```

Fisher-z transform beta2, accumulate across subjects, average.

**Implementation**: uses `sklearn.LinearRegression` for consistency with the reference REACT implementation.

### Masking

- **Stage 1 mask**: intersection of non-zero voxels across all NT atlas maps — ensures regression only runs where NT data exists
- **Stage 2 mask**: whole-brain mask (GM + WM) at 2mm in MNI152NLin6Asym — captures both gray and white matter signal

### Collinearity Handling

- Uses pre-averaged NT atlas maps (one per target) as regressors — eliminates within-target collinearity from multiple tracer studies
- Checks condition number of the regressor matrix before inversion
- If condition number exceeds threshold: warns and falls back to grouped regression (per NT system family). Each group gets a separate regression; results are concatenated.

### Outputs of `prepare react`

- **Stage 1 timeseries**: per-subject NT timeseries matrix `(n_timepoints, n_targets)`. Stored per subject. Used by fntm REACT-enriched global scoring.
- **Stage 2 atlas**: Fisher-z averaged maps across subjects → one `VoxelAtlas` with enriched maps per target. Used by all three analyses in `--enriched` mode.

## CLI

### `lacuna fetch ntatlas`

Downloads raw PET maps from the `lacuna-data` repository to `~/.lacuna/cache/atlases/neurotransmitter/raw/`.

### `lacuna prepare` (new top-level subcommand)

**`lacuna prepare lntm [--map-config PATH] [--source-dir PATH] [--cache-dir PATH]`**
- Downloads raw PET maps if not present (auto-triggers `fetch ntatlas`)
- Groups by target, applies map selection config, averages (excluding zeros), z-scores
- Caches `VoxelAtlas` to `~/.lacuna/cache/atlases/neurotransmitter/`

**`lacuna prepare sntm [--connectome-path PATH] [--cache-dir PATH]`**
- Requires: prepared NT atlas (auto-triggers `prepare lntm` if missing)
- Requires: tractogram (must be fetched separately via `lacuna fetch`)
- Samples NT values at all streamline endpoints, computes endpoint means
- Caches float16 matrix `(n_targets, n_streamlines)` to `~/.lacuna/cache/sntm/{connectome}/`

**`lacuna prepare react [--connectome-name NAME] [--cache-dir PATH]`**
- Requires: prepared NT atlas (auto-triggers `prepare lntm` if missing)
- Requires: normative fMRI connectome
- Runs REACT stage 1+2 on all normative subjects
- Caches stage 1 timeseries and stage 2 enriched atlas

### `lacuna run` (existing subcommand, new analysis types)

```
lacuna run lntm <bids_dir> <output_dir> [options]
lacuna run sntm <bids_dir> <output_dir> [options]
lacuna run fntm <bids_dir> <output_dir> [options]
```

**Common options** (all three):
- `--targets PRESET_OR_LIST` — target subset, default `all`
- `--enriched` — use REACT-enriched NT atlas instead of static
- `--parcel-atlases ATLAS [ATLAS ...]` — aggregate to these atlases (use `lacuna info atlases` to list)
- `--custom-parcellation NAME NIFTI LABELS SPACE` — custom parcellation (can be specified multiple times)
- Standard lacuna options: `--participant-label`, `--session-id`, `--n-procs`, etc.

**sntm-specific**:
- `--connectome-path PATH` — tractogram path (matching existing snm convention)

**fntm-specific**:
- `--connectome-name NAME` — normative fMRI connectome (matching existing fnm convention)

### Dependency chain

```
lacuna fetch ntatlas
    ↓
lacuna prepare lntm [--map-config]
    ↓                   ↓
lacuna prepare sntm     lacuna prepare react
    ↓                   ↓
lacuna run sntm     lacuna run {lntm,sntm,fntm} --enriched
lacuna run lntm
lacuna run fntm
```

Each `prepare` step auto-triggers its dependencies if they haven't been run.

### Error Handling

- If `--targets` at run time references a target excluded at prepare time: raise with message naming the conflicting target and advising to re-run `prepare lntm`
- If `--enriched` is specified but `prepare react` has not been run: raise with instructions to run `prepare react` first
- If atlas cache is missing or corrupt: raise with instructions to run the appropriate `prepare` command
- If sntm cannot find filtered tractogram or precomputed weights: warn and fall back to on-the-fly computation (not an error)

## Scoring Conventions

- All NT maps are z-scored after averaging (per-target, across brain voxels, excluding zeros)
- Zeros are always excluded from scoring (partial brain coverage maps should not contribute zeros as data)
- Functional scoring thresholds connectivity map to positive values only by default
- Endpoint combination rule for structural scoring: mean of two endpoint values (v1, not configurable)
- Aggregation default: mean. Sum available as alternative.

## Output Format

**Result keys** (flat, per target):
- `lntm_5HT1a`, `lntm_D1`, etc. — local NT scores
- `sntm_5HT1a`, `sntm_D1`, etc. — structural NT scores
- `sntm_streamline_count` — covariate (one value, not per-NT)
- `fntm_5HT1a`, `fntm_D1`, etc. — functional NT scores

**Result types**:
- Global scores: `ScalarMetric` (one per target per analysis)
- Regional scores: `ParcelData` (per target per atlas, when parcellation requested)

**Export**: compatible with existing `export_results_to_csv` / `export_results_to_tsv`. The flat key structure ensures clean column names in tabular output.

## Other Changes

### `RegionalDamage` renamed to `LocalDamage`

Consistent naming convention across the toolbox:
- **local** = at the lesion site (tissue-level)
- **structural** = via structural connectivity (network-level)
- **functional** = via functional connectivity (network-level)

The class, module file, CLI identifier, tests, and documentation all update. The old name is removed (no backward-compatibility shim).

### `StructuralNetworkMapping` — new option

`keep_filtered_tractogram: bool = False` — when `True`, retains the lesion-intersecting tractogram as a `Tractogram` DataContainer in the results. Consumed by `sntm` when available.

## Data

### `lacuna-data` repository (GitHub or OSF)

External repository hosting atlas data, fetched by lacuna on demand.

```
lacuna-data/
  neurotransmitter/
    pet_maps/             # raw PET NIfTIs (current pet_atlas_raw/ contents)
    metadata.json         # target grouping, citations, map-to-target mapping
  README.md
```

One directory per atlas domain. Future additions (metabolic, transcriptomic) get their own directories.

### Docker

The lacuna Docker image pre-fetches `lacuna-data` contents during container build, so `fetch ntatlas` and `prepare` steps are not needed inside the container.

## Future Extensions (Not in v1, Designed For)

- **ParcelAtlas** and **SurfaceAtlas** representations in `atlas/types.py` — stubs defined, interface matches `VoxelAtlas`
- **Additional atlas domains**: metabolic, transcriptomic, gene expression — new `prepare`/`run` subcommands using the same engine
- **REACT as cross-cutting enrichment**: `prepare react` works with any spatial-prior atlas, not just NT
- **Configurable endpoint combination**: expose product/sum/max alongside mean for structural scoring
- **Laterality decomposition** (ipsi/contra) and **pre/post-synaptic decomposition** — out of scope, API does not preclude later addition
