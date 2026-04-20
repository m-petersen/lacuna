# Lesion Neurotransmitter Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement three lesion neurotransmitter mapping analyses (local, structural, functional) with a shared atlas engine, REACT enrichment, and CLI integration.

**Architecture:** A new `src/lacuna/atlas/` module provides the shared engine for loading, averaging, z-scoring, and scoring voxel atlases. Three `BaseAnalysis` subclasses (`LocalNeurotransmitterMapping`, `StructuralNeurotransmitterMapping`, `FunctionalNeurotransmitterMapping`) consume the engine. A `lacuna prepare` CLI subcommand handles precomputation. REACT enrichment is implemented as a cross-cutting atlas-level enhancement.

**Tech Stack:** Python 3.10+, nibabel, numpy, scipy, scikit-learn (REACT), nilearn, h5py, MRtrix3 (structural), existing lacuna infrastructure (BaseAnalysis, SubjectData, DataContainer hierarchy, spatial transforms, CLI parser).

**Spec:** `specs/009-neurotransmitter-mapping/design.md`

---

## File Structure

### New files

```
src/lacuna/atlas/__init__.py          — Package init, public exports
src/lacuna/atlas/types.py             — VoxelAtlas dataclass (+ParcelAtlas, SurfaceAtlas stubs)
src/lacuna/atlas/store.py             — build_nt_atlas, save_atlas, load_atlas, fetch
src/lacuna/atlas/scoring.py           — score_focal, score_structural_endpoints, score_functional_overlap, score_react_temporal
src/lacuna/atlas/config.py            — NT target presets, grouping, map selection config
src/lacuna/atlas/react.py             — REACT stage 1+2 implementation
src/lacuna/analysis/local_neurotransmitter_mapping.py    — lntm analysis
src/lacuna/analysis/structural_neurotransmitter_mapping.py — sntm analysis
src/lacuna/analysis/functional_neurotransmitter_mapping.py — fntm analysis
src/lacuna/analysis/local_damage.py   — renamed from regional_damage.py
src/lacuna/cli/prepare.py             — lacuna prepare subcommand
tests/unit/atlas/test_types.py        — VoxelAtlas unit tests
tests/unit/atlas/test_store.py        — Atlas building/caching tests
tests/unit/atlas/test_scoring.py      — Scoring function tests
tests/unit/atlas/test_config.py       — Config/preset tests
tests/unit/atlas/test_react.py        — REACT implementation tests
tests/unit/analysis/test_local_ntm.py — lntm analysis tests
tests/unit/analysis/test_structural_ntm.py — sntm analysis tests
tests/unit/analysis/test_functional_ntm.py — fntm analysis tests
tests/unit/analysis/test_local_damage_rename.py — Rename verification tests
```

### Modified files

```
src/lacuna/analysis/__init__.py       — Add new analysis imports, update RegionalDamage→LocalDamage
src/lacuna/analysis/regional_damage.py — DELETED (replaced by local_damage.py)
src/lacuna/analysis/structural_network_mapping.py — Add keep_filtered_tractogram option
src/lacuna/cli/parser.py              — Add prepare subcommand, lntm/sntm/fntm run parsers
src/lacuna/cli/main.py                — Add prepare dispatch, analysis aliases for lntm/sntm/fntm
```

---

### Task 1: Atlas Types — `VoxelAtlas` Dataclass

**Files:**
- Create: `src/lacuna/atlas/__init__.py`
- Create: `src/lacuna/atlas/types.py`
- Test: `tests/unit/atlas/test_types.py`

- [ ] **Step 1: Write failing tests for VoxelAtlas**

```python
# tests/unit/atlas/test_types.py
"""Tests for atlas type representations."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.atlas.types import VoxelAtlas


@pytest.fixture
def sample_maps():
    """Create minimal test NT maps."""
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    shape = (91, 109, 91)
    maps = {}
    for target in ["5HT1a", "D1", "DAT"]:
        data = np.random.default_rng(42).standard_normal(shape).astype(np.float32)
        maps[target] = nib.Nifti1Image(data, affine)
    return maps


class TestVoxelAtlasConstruction:
    def test_basic_construction(self, sample_maps):
        atlas = VoxelAtlas(
            maps=sample_maps,
            space="MNI152NLin6Asym",
            resolution=2.0,
            domain="neurotransmitter",
        )
        assert atlas.space == "MNI152NLin6Asym"
        assert atlas.resolution == 2.0
        assert atlas.domain == "neurotransmitter"
        assert atlas.targets == sorted(sample_maps.keys())

    def test_targets_are_sorted(self, sample_maps):
        atlas = VoxelAtlas(
            maps=sample_maps,
            space="MNI152NLin6Asym",
            resolution=2.0,
            domain="neurotransmitter",
        )
        assert atlas.targets == ["5HT1a", "D1", "DAT"]

    def test_empty_maps_raises(self):
        with pytest.raises(ValueError, match="at least one map"):
            VoxelAtlas(
                maps={},
                space="MNI152NLin6Asym",
                resolution=2.0,
                domain="neurotransmitter",
            )


class TestVoxelAtlasGetMap:
    def test_get_existing_map(self, sample_maps):
        atlas = VoxelAtlas(
            maps=sample_maps,
            space="MNI152NLin6Asym",
            resolution=2.0,
            domain="neurotransmitter",
        )
        img = atlas.get_map("D1")
        assert isinstance(img, nib.Nifti1Image)
        assert img.shape == (91, 109, 91)

    def test_get_nonexistent_map_raises(self, sample_maps):
        atlas = VoxelAtlas(
            maps=sample_maps,
            space="MNI152NLin6Asym",
            resolution=2.0,
            domain="neurotransmitter",
        )
        with pytest.raises(KeyError, match="GABA"):
            atlas.get_map("GABA")


class TestVoxelAtlasSubset:
    def test_subset_returns_new_atlas(self, sample_maps):
        atlas = VoxelAtlas(
            maps=sample_maps,
            space="MNI152NLin6Asym",
            resolution=2.0,
            domain="neurotransmitter",
        )
        sub = atlas.subset(["D1", "DAT"])
        assert sub.targets == ["D1", "DAT"]
        assert atlas.targets == ["5HT1a", "D1", "DAT"]  # original unchanged

    def test_subset_invalid_target_raises(self, sample_maps):
        atlas = VoxelAtlas(
            maps=sample_maps,
            space="MNI152NLin6Asym",
            resolution=2.0,
            domain="neurotransmitter",
        )
        with pytest.raises(KeyError, match="GABA"):
            atlas.subset(["D1", "GABA"])


class TestVoxelAtlasToMatrix:
    def test_to_matrix_shape(self, sample_maps):
        atlas = VoxelAtlas(
            maps=sample_maps,
            space="MNI152NLin6Asym",
            resolution=2.0,
            domain="neurotransmitter",
        )
        mask = np.zeros((91, 109, 91), dtype=bool)
        mask[40:50, 50:60, 40:50] = True
        n_voxels = mask.sum()
        matrix = atlas.to_matrix(mask)
        assert matrix.shape == (3, n_voxels)  # 3 targets x n_voxels

    def test_to_matrix_values_match_maps(self, sample_maps):
        atlas = VoxelAtlas(
            maps=sample_maps,
            space="MNI152NLin6Asym",
            resolution=2.0,
            domain="neurotransmitter",
        )
        mask = np.zeros((91, 109, 91), dtype=bool)
        mask[45, 55, 45] = True
        matrix = atlas.to_matrix(mask)
        for i, target in enumerate(atlas.targets):
            expected = sample_maps[target].get_fdata()[45, 55, 45]
            assert np.isclose(matrix[i, 0], expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/atlas/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lacuna.atlas'`

- [ ] **Step 3: Implement VoxelAtlas**

```python
# src/lacuna/atlas/__init__.py
"""
Atlas engine for scoring lesion footprints against spatial atlas data.

Provides loading, caching, and scoring of voxel-level atlas data
(neurotransmitter PET maps, metabolic maps, etc.) against lesion masks
and connectivity footprints.
"""

from lacuna.atlas.types import VoxelAtlas

__all__ = ["VoxelAtlas"]
```

```python
# src/lacuna/atlas/types.py
"""Atlas data representations.

VoxelAtlas is the primary type for v1. ParcelAtlas and SurfaceAtlas
are defined as extension points for future atlas domains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import nibabel as nib


@dataclass
class VoxelAtlas:
    """A collection of named 3D brain maps in a common coordinate space.

    Each map represents one target (e.g., a neurotransmitter receptor/transporter).
    Maps are z-scored and averaged per target during construction.

    Parameters
    ----------
    maps : dict[str, nib.Nifti1Image]
        One map per target. Keys are target names (e.g., "5HT1a", "D1").
    space : str
        Coordinate space identifier (e.g., "MNI152NLin6Asym").
    resolution : float
        Voxel size in mm.
    domain : str
        Atlas domain identifier (e.g., "neurotransmitter").
    metadata : dict
        Source info, map selection config, creation timestamp.
    """

    maps: dict[str, nib.Nifti1Image]
    space: str
    resolution: float
    domain: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.maps:
            raise ValueError("VoxelAtlas requires at least one map.")

    @property
    def targets(self) -> list[str]:
        """Sorted list of target names."""
        return sorted(self.maps.keys())

    def get_map(self, target: str) -> nib.Nifti1Image:
        """Get the map for a single target.

        Raises KeyError if target not found.
        """
        if target not in self.maps:
            raise KeyError(
                f"Target '{target}' not in atlas. "
                f"Available: {self.targets}"
            )
        return self.maps[target]

    def subset(self, targets: list[str]) -> VoxelAtlas:
        """Return a new atlas with only the specified targets.

        Raises KeyError if any target not found.
        """
        for t in targets:
            if t not in self.maps:
                raise KeyError(
                    f"Target '{t}' not in atlas. "
                    f"Available: {self.targets}"
                )
        return VoxelAtlas(
            maps={t: self.maps[t] for t in targets},
            space=self.space,
            resolution=self.resolution,
            domain=self.domain,
            metadata=self.metadata,
        )

    def to_matrix(self, mask: np.ndarray) -> np.ndarray:
        """Extract atlas values within a boolean mask as a 2D matrix.

        Parameters
        ----------
        mask : np.ndarray
            Boolean 3D mask. Shape must match atlas map shapes.

        Returns
        -------
        np.ndarray
            Shape (n_targets, n_masked_voxels), ordered by sorted target names.
        """
        n_voxels = mask.sum()
        matrix = np.empty((len(self.targets), n_voxels), dtype=np.float64)
        for i, target in enumerate(self.targets):
            data = self.maps[target].get_fdata()
            matrix[i] = data[mask]
        return matrix

    def resample_to(self, target_affine: np.ndarray, target_shape: tuple) -> VoxelAtlas:
        """Resample all maps to a target affine and shape.

        Uses nilearn for resampling with continuous interpolation.

        Parameters
        ----------
        target_affine : np.ndarray
            4x4 affine matrix of the target space.
        target_shape : tuple
            3D shape of the target space.

        Returns
        -------
        VoxelAtlas
            New atlas with resampled maps.
        """
        from nilearn.image import resample_img

        resampled = {}
        for target in self.targets:
            resampled[target] = resample_img(
                self.maps[target],
                target_affine=target_affine,
                target_shape=target_shape,
                interpolation="continuous",
            )

        # Detect resolution from affine
        voxel_sizes = np.sqrt(np.sum(target_affine[:3, :3] ** 2, axis=0))
        new_resolution = float(np.mean(voxel_sizes))

        return VoxelAtlas(
            maps=resampled,
            space=self.space,
            resolution=new_resolution,
            domain=self.domain,
            metadata={**self.metadata, "resampled": True},
        )


class ParcelAtlas:
    """Placeholder for parcel-level atlas data. Not implemented in v1."""

    pass


class SurfaceAtlas:
    """Placeholder for surface-level atlas data. Not implemented in v1."""

    pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/atlas/test_types.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna
git add src/lacuna/atlas/__init__.py src/lacuna/atlas/types.py tests/unit/atlas/test_types.py
git commit -m "feat(atlas): add VoxelAtlas dataclass with subset, to_matrix, resample_to"
```

---

### Task 2: Atlas Config — Target Grouping and Presets

**Files:**
- Create: `src/lacuna/atlas/config.py`
- Test: `tests/unit/atlas/test_config.py`

- [ ] **Step 1: Write failing tests for config**

```python
# tests/unit/atlas/test_config.py
"""Tests for NT atlas configuration and presets."""

import pytest
import yaml

from lacuna.atlas.config import (
    NT_TARGET_GROUPS,
    NT_PRESETS,
    resolve_targets,
    parse_map_selection,
    parse_target_from_filename,
    parse_publication_from_filename,
)


class TestFilenameParser:
    def test_parse_target(self):
        fname = "target-5HT1a_tracer-cumi101_n-8_dx-hc_pub-beliveau2017_space-MNI152NLin6Asym_desc-proc.nii.gz"
        assert parse_target_from_filename(fname) == "5HT1a"

    def test_parse_target_d23(self):
        fname = "target-D23_tracer-fallypride_n-49_dx-hc_pub-jaworska2020_space-MNI152NLin6Asym_desc-proc.nii.gz"
        assert parse_target_from_filename(fname) == "D23"

    def test_parse_publication(self):
        fname = "target-5HT1a_tracer-cumi101_n-8_dx-hc_pub-beliveau2017_space-MNI152NLin6Asym_desc-proc.nii.gz"
        assert parse_publication_from_filename(fname) == "beliveau2017"

    def test_parse_no_target_raises(self):
        with pytest.raises(ValueError, match="No target"):
            parse_target_from_filename("random_file.nii.gz")


class TestTargetGroups:
    def test_serotonergic_group_exists(self):
        assert "serotonergic" in NT_TARGET_GROUPS

    def test_dopaminergic_group_exists(self):
        assert "dopaminergic" in NT_TARGET_GROUPS
        assert "D1" in NT_TARGET_GROUPS["dopaminergic"]
        assert "D23" in NT_TARGET_GROUPS["dopaminergic"]
        assert "DAT" in NT_TARGET_GROUPS["dopaminergic"]


class TestPresets:
    def test_all_preset(self):
        assert "all" in NT_PRESETS

    def test_dopaminergic_preset(self):
        targets = NT_PRESETS["dopaminergic"]
        assert "D1" in targets
        assert "D23" in targets
        assert "DAT" in targets

    def test_serotonergic_preset(self):
        targets = NT_PRESETS["serotonergic"]
        assert "5HT1a" in targets
        assert "5HTT" in targets


class TestResolveTargets:
    def test_resolve_preset_name(self):
        available = ["5HT1a", "D1", "D23", "DAT", "5HTT"]
        result = resolve_targets("dopaminergic", available)
        assert set(result) == {"D1", "D23", "DAT"}

    def test_resolve_explicit_list(self):
        available = ["5HT1a", "D1", "D23", "DAT"]
        result = resolve_targets(["D1", "DAT"], available)
        assert set(result) == {"D1", "DAT"}

    def test_resolve_all(self):
        available = ["5HT1a", "D1", "DAT"]
        result = resolve_targets("all", available)
        assert set(result) == {"5HT1a", "D1", "DAT"}

    def test_resolve_unavailable_target_raises(self):
        available = ["D1", "DAT"]
        with pytest.raises(ValueError, match="GABA"):
            resolve_targets(["D1", "GABA"], available)


class TestMapSelection:
    def test_parse_default_config(self):
        config = parse_map_selection(None)
        assert config is None

    def test_parse_yaml_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "targets": {
                "5HT1a": "beliveau2017",
                "D1": "all",
                "DAT": "exclude",
            }
        }))
        config = parse_map_selection(config_file)
        assert config["5HT1a"] == ["beliveau2017"]
        assert config["D1"] == "all"
        assert config["DAT"] == "exclude"

    def test_parse_list_value(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "targets": {
                "5HT1b": ["savli2012", "gallezot2010"],
            }
        }))
        config = parse_map_selection(config_file)
        assert config["5HT1b"] == ["savli2012", "gallezot2010"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/atlas/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lacuna.atlas.config'`

- [ ] **Step 3: Implement atlas config**

```python
# src/lacuna/atlas/config.py
"""NT atlas configuration: target grouping, presets, and map selection.

Target names are parsed from PET map filenames following the pattern:
    target-{TARGET}_tracer-{TRACER}_..._pub-{PUB}_...
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# NT system groupings — maps target names to their neurotransmitter system
NT_TARGET_GROUPS: dict[str, list[str]] = {
    "serotonergic": ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"],
    "dopaminergic": ["D1", "D23", "DAT", "FDOPA"],
    "cholinergic": ["VAChT", "M1", "A4B2"],
    "noradrenergic": ["NET"],
    "gabaergic": ["GABAa", "GABAa5"],
    "cannabinoid": ["CB1"],
    "opioid": ["MOR", "KOR"],
    "histaminergic": ["H3"],
    "glutamatergic": ["mGluR5", "NMDA"],
    "vesicular": ["VMAT2"],
}

# All known targets (flat list derived from groups)
ALL_TARGETS: list[str] = sorted(
    t for targets in NT_TARGET_GROUPS.values() for t in targets
)

# Presets for target selection at run time
NT_PRESETS: dict[str, list[str]] = {
    "all": ALL_TARGETS,
    "dopaminergic": NT_TARGET_GROUPS["dopaminergic"],
    "serotonergic": NT_TARGET_GROUPS["serotonergic"],
    "cholinergic": NT_TARGET_GROUPS["cholinergic"],
    "monoaminergic": (
        NT_TARGET_GROUPS["serotonergic"]
        + NT_TARGET_GROUPS["dopaminergic"]
        + NT_TARGET_GROUPS["noradrenergic"]
    ),
}

# Regex for parsing BIDS-style PET atlas filenames
_TARGET_RE = re.compile(r"target-([A-Za-z0-9]+)")
_PUB_RE = re.compile(r"pub-([A-Za-z0-9]+)")


def parse_target_from_filename(filename: str) -> str:
    """Extract the target name from a PET atlas filename.

    Parameters
    ----------
    filename : str
        Filename like "target-5HT1a_tracer-cumi101_..._desc-proc.nii.gz"

    Returns
    -------
    str
        Target name (e.g., "5HT1a")

    Raises
    ------
    ValueError
        If no target- field found.
    """
    match = _TARGET_RE.search(filename)
    if not match:
        raise ValueError(f"No target found in filename: {filename}")
    return match.group(1)


def parse_publication_from_filename(filename: str) -> str:
    """Extract the publication key from a PET atlas filename.

    Parameters
    ----------
    filename : str
        Filename like "target-5HT1a_..._pub-beliveau2017_..."

    Returns
    -------
    str
        Publication key (e.g., "beliveau2017")

    Raises
    ------
    ValueError
        If no pub- field found.
    """
    match = _PUB_RE.search(filename)
    if not match:
        raise ValueError(f"No publication key found in filename: {filename}")
    return match.group(1)


def resolve_targets(
    targets: str | list[str],
    available: list[str],
) -> list[str]:
    """Resolve a target specification to a list of target names.

    Parameters
    ----------
    targets : str or list[str]
        Either a preset name ("all", "dopaminergic", etc.) or an explicit
        list of target names.
    available : list[str]
        Targets actually available in the prepared atlas.

    Returns
    -------
    list[str]
        Resolved target names (sorted).

    Raises
    ------
    ValueError
        If a requested target is not available.
    """
    if isinstance(targets, str):
        if targets in NT_PRESETS:
            requested = NT_PRESETS[targets]
        else:
            # Treat as a single target name
            requested = [targets]
    else:
        requested = list(targets)

    # For "all" preset, intersect with what's actually available
    if isinstance(targets, str) and targets == "all":
        return sorted(available)

    # Validate all requested targets are available
    available_set = set(available)
    for t in requested:
        if t not in available_set:
            raise ValueError(
                f"Target '{t}' not available in prepared atlas. "
                f"Available targets: {sorted(available)}"
            )
    return sorted(requested)


def parse_map_selection(config_path: Path | None) -> dict[str, Any] | None:
    """Parse a map selection YAML config file.

    The config specifies which raw PET maps to include per target:
        targets:
          5HT1a: beliveau2017         # single study
          5HT1b: [savli2012, gallezot2010]  # multiple studies
          D1: all                     # all available (default)
          DAT: exclude                # skip this target

    Parameters
    ----------
    config_path : Path or None
        Path to YAML config. None returns None (use defaults).

    Returns
    -------
    dict or None
        Parsed config with normalized values: single strings wrapped in lists,
        "all" and "exclude" as literals.
    """
    if config_path is None:
        return None

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if "targets" not in raw:
        raise ValueError(
            f"Map selection config must have a 'targets' key. "
            f"Got keys: {list(raw.keys())}"
        )

    config = {}
    for target, value in raw["targets"].items():
        if value == "all" or value == "exclude":
            config[target] = value
        elif isinstance(value, str):
            config[target] = [value]
        elif isinstance(value, list):
            config[target] = value
        else:
            raise ValueError(
                f"Invalid value for target '{target}': {value}. "
                f"Expected string, list, 'all', or 'exclude'."
            )
    return config
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/atlas/test_config.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna
git add src/lacuna/atlas/config.py tests/unit/atlas/test_config.py
git commit -m "feat(atlas): add NT target grouping, presets, and map selection config"
```

---

### Task 3: Atlas Store — Build, Save, Load NT Atlas

**Files:**
- Create: `src/lacuna/atlas/store.py`
- Test: `tests/unit/atlas/test_store.py`

- [ ] **Step 1: Write failing tests for atlas store**

```python
# tests/unit/atlas/test_store.py
"""Tests for atlas building, saving, and loading."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.atlas.store import build_nt_atlas, save_atlas, load_atlas
from lacuna.atlas.types import VoxelAtlas


def _create_pet_map(tmp_path, target, tracer, pub, shape=(91, 109, 91), add_zeros=False):
    """Create a fake PET map NIfTI file with standard naming."""
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    rng = np.random.default_rng(hash((target, tracer, pub)) % 2**32)
    data = rng.random(shape).astype(np.float32) + 0.1  # all positive
    if add_zeros:
        # Simulate partial brain coverage: zero out a quarter of the volume
        data[:, :, :shape[2] // 4] = 0.0
    fname = f"target-{target}_tracer-{tracer}_n-10_dx-hc_pub-{pub}_space-MNI152NLin6Asym_desc-proc.nii.gz"
    path = tmp_path / fname
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


@pytest.fixture
def pet_dir(tmp_path):
    """Create a directory with fake PET maps for multiple targets."""
    _create_pet_map(tmp_path, "5HT1a", "cumi101", "beliveau2017")
    _create_pet_map(tmp_path, "5HT1a", "way100635", "savli2012")
    _create_pet_map(tmp_path, "D1", "sch23390", "kaller2017")
    _create_pet_map(tmp_path, "DAT", "fpcit", "dukart2018", add_zeros=True)
    _create_pet_map(tmp_path, "DAT", "fepe2i", "sasaki2012", add_zeros=True)
    return tmp_path


class TestBuildNtAtlas:
    def test_groups_by_target(self, pet_dir):
        atlas = build_nt_atlas(pet_dir)
        assert set(atlas.targets) == {"5HT1a", "D1", "DAT"}

    def test_averages_multiple_maps_per_target(self, pet_dir):
        atlas = build_nt_atlas(pet_dir)
        # 5HT1a has 2 maps, D1 has 1 — both should produce single maps
        assert atlas.get_map("5HT1a").shape == (91, 109, 91)
        assert atlas.get_map("D1").shape == (91, 109, 91)

    def test_zeros_excluded_from_average(self, pet_dir):
        atlas = build_nt_atlas(pet_dir)
        dat_data = atlas.get_map("DAT").get_fdata()
        # Where both input maps had zeros, the average should still be zero
        # (no data to average). Where only one had data, it should be non-zero.
        # The z-scoring happens after averaging, so check the raw structure.
        # Actually after z-scoring: zeros in input → remain as the z-score of 0
        # which is negative (since mean of nonzero values is positive).
        # The key test: the atlas is z-scored (mean ~0, std ~1 across nonzero voxels)
        nonzero_mask = dat_data != 0
        nonzero_vals = dat_data[nonzero_mask]
        assert abs(np.mean(nonzero_vals)) < 0.1  # approximately zero mean
        assert abs(np.std(nonzero_vals) - 1.0) < 0.1  # approximately unit std

    def test_atlas_is_z_scored(self, pet_dir):
        atlas = build_nt_atlas(pet_dir)
        d1_data = atlas.get_map("D1").get_fdata()
        nonzero = d1_data[d1_data != 0]
        assert abs(np.mean(nonzero)) < 0.1
        assert abs(np.std(nonzero) - 1.0) < 0.1

    def test_map_config_exclude(self, pet_dir):
        config = {"DAT": "exclude"}
        atlas = build_nt_atlas(pet_dir, map_config=config)
        assert "DAT" not in atlas.targets
        assert "5HT1a" in atlas.targets

    def test_map_config_select_specific_pub(self, pet_dir):
        config = {"5HT1a": ["beliveau2017"]}
        atlas = build_nt_atlas(pet_dir, map_config=config)
        assert "5HT1a" in atlas.targets

    def test_space_and_domain(self, pet_dir):
        atlas = build_nt_atlas(pet_dir)
        assert atlas.space == "MNI152NLin6Asym"
        assert atlas.domain == "neurotransmitter"


class TestSaveLoadAtlas:
    def test_roundtrip(self, pet_dir, tmp_path):
        atlas = build_nt_atlas(pet_dir)
        cache_dir = tmp_path / "cache"
        save_atlas(atlas, cache_dir)
        loaded = load_atlas(cache_dir)
        assert set(loaded.targets) == set(atlas.targets)
        assert loaded.space == atlas.space
        assert loaded.resolution == atlas.resolution
        assert loaded.domain == atlas.domain
        for t in atlas.targets:
            np.testing.assert_array_almost_equal(
                loaded.get_map(t).get_fdata(),
                atlas.get_map(t).get_fdata(),
            )

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_atlas(tmp_path / "nonexistent")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/atlas/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lacuna.atlas.store'`

- [ ] **Step 3: Implement atlas store**

```python
# src/lacuna/atlas/store.py
"""Atlas lifecycle: build, save, load, and cache voxel atlases.

Handles grouping PET maps by target, averaging (excluding zeros),
z-scoring, and serialization to disk.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from lacuna.atlas.config import (
    parse_publication_from_filename,
    parse_target_from_filename,
)
from lacuna.atlas.types import VoxelAtlas

logger = logging.getLogger(__name__)


def build_nt_atlas(
    source_dir: Path,
    map_config: dict[str, Any] | None = None,
) -> VoxelAtlas:
    """Build a neurotransmitter VoxelAtlas from raw PET map NIfTIs.

    Groups maps by target, averages per target (excluding zeros),
    and z-scores the result.

    Parameters
    ----------
    source_dir : Path
        Directory containing PET NIfTI files with standard naming:
        target-{TARGET}_tracer-{TRACER}_..._pub-{PUB}_...nii.gz
    map_config : dict or None
        Per-target selection. Keys are target names, values are:
        - list of publication keys to include
        - "all" to include all available
        - "exclude" to skip the target
        Targets not in config default to "all".

    Returns
    -------
    VoxelAtlas
        Z-scored, averaged atlas with one map per target.
    """
    source_dir = Path(source_dir)

    # Discover and group NIfTI files by target
    file_groups: dict[str, list[Path]] = defaultdict(list)
    for nifti_path in sorted(source_dir.glob("*.nii.gz")):
        try:
            target = parse_target_from_filename(nifti_path.name)
        except ValueError:
            logger.warning("Skipping file with no target: %s", nifti_path.name)
            continue
        file_groups[target].append(nifti_path)

    if not file_groups:
        raise ValueError(f"No PET NIfTI files found in {source_dir}")

    # Apply map selection config
    if map_config:
        filtered_groups: dict[str, list[Path]] = {}
        for target, paths in file_groups.items():
            selection = map_config.get(target, "all")
            if selection == "exclude":
                logger.info("Excluding target: %s", target)
                continue
            if selection == "all":
                filtered_groups[target] = paths
            else:
                # Filter by publication key
                pubs_wanted = set(selection)
                selected = [
                    p for p in paths
                    if parse_publication_from_filename(p.name) in pubs_wanted
                ]
                if not selected:
                    logger.warning(
                        "No maps found for target '%s' with publications %s",
                        target, selection,
                    )
                    continue
                filtered_groups[target] = selected
        file_groups = filtered_groups

    # Build averaged, z-scored maps
    maps: dict[str, nib.Nifti1Image] = {}
    reference_affine = None
    reference_shape = None

    for target in sorted(file_groups):
        paths = file_groups[target]
        logger.info("Building %s from %d maps", target, len(paths))

        # Load all maps for this target
        loaded = [nib.load(str(p)) for p in paths]

        if reference_affine is None:
            reference_affine = loaded[0].affine
            reference_shape = loaded[0].shape[:3]

        # Average excluding zeros
        averaged = _average_excluding_zeros(
            [img.get_fdata() for img in loaded]
        )

        # Z-score (excluding zeros)
        z_scored = _zscore_excluding_zeros(averaged)

        maps[target] = nib.Nifti1Image(z_scored.astype(np.float32), reference_affine)

    # Detect space and resolution from affine
    voxel_sizes = np.sqrt(np.sum(reference_affine[:3, :3] ** 2, axis=0))
    resolution = float(np.mean(voxel_sizes))

    return VoxelAtlas(
        maps=maps,
        space="MNI152NLin6Asym",
        resolution=resolution,
        domain="neurotransmitter",
        metadata={
            "source_dir": str(source_dir),
            "map_config": map_config,
            "n_targets": len(maps),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _average_excluding_zeros(arrays: list[np.ndarray]) -> np.ndarray:
    """Average multiple 3D arrays, excluding zeros from the mean.

    Zeros indicate outside-coverage voxels (e.g., cortical-only tracers).
    Voxels where ALL arrays are zero remain zero.
    """
    stacked = np.stack(arrays, axis=0)  # (n_maps, x, y, z)
    nonzero_mask = stacked != 0
    count = nonzero_mask.sum(axis=0)  # (x, y, z)
    safe_count = np.maximum(count, 1)  # avoid division by zero
    total = np.where(nonzero_mask, stacked, 0).sum(axis=0)
    averaged = total / safe_count
    # Where all maps had zero, keep zero
    averaged[count == 0] = 0.0
    return averaged


def _zscore_excluding_zeros(data: np.ndarray) -> np.ndarray:
    """Z-score a 3D array, computing stats only on nonzero voxels.

    Zero voxels remain zero after z-scoring.
    """
    result = np.zeros_like(data)
    nonzero_mask = data != 0
    values = data[nonzero_mask]
    if len(values) == 0:
        return result
    mean = values.mean()
    std = values.std()
    if std == 0:
        return result
    result[nonzero_mask] = (values - mean) / std
    return result


def save_atlas(atlas: VoxelAtlas, cache_dir: Path) -> None:
    """Save a VoxelAtlas to disk.

    Stores each map as a NIfTI file and metadata as JSON.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save each map
    maps_dir = cache_dir / "maps"
    maps_dir.mkdir(exist_ok=True)
    for target in atlas.targets:
        nib.save(atlas.get_map(target), str(maps_dir / f"{target}.nii.gz"))

    # Save manifest
    manifest = {
        "targets": atlas.targets,
        "space": atlas.space,
        "resolution": atlas.resolution,
        "domain": atlas.domain,
        "metadata": atlas.metadata,
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def load_atlas(cache_dir: Path) -> VoxelAtlas:
    """Load a VoxelAtlas from disk.

    Raises FileNotFoundError if cache_dir or manifest does not exist.
    """
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No atlas found at {cache_dir}. "
            f"Run 'lacuna prepare lntm' to create one."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    maps = {}
    maps_dir = cache_dir / "maps"
    for target in manifest["targets"]:
        map_path = maps_dir / f"{target}.nii.gz"
        if not map_path.exists():
            raise FileNotFoundError(
                f"Map file missing for target '{target}': {map_path}"
            )
        maps[target] = nib.load(str(map_path))

    return VoxelAtlas(
        maps=maps,
        space=manifest["space"],
        resolution=manifest["resolution"],
        domain=manifest["domain"],
        metadata=manifest.get("metadata", {}),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/atlas/test_store.py -v`
Expected: All PASS

- [ ] **Step 5: Update atlas __init__.py exports**

```python
# Update src/lacuna/atlas/__init__.py
"""
Atlas engine for scoring lesion footprints against spatial atlas data.

Provides loading, caching, and scoring of voxel-level atlas data
(neurotransmitter PET maps, metabolic maps, etc.) against lesion masks
and connectivity footprints.
"""

from lacuna.atlas.config import NT_PRESETS, NT_TARGET_GROUPS, resolve_targets
from lacuna.atlas.store import build_nt_atlas, load_atlas, save_atlas
from lacuna.atlas.types import VoxelAtlas

__all__ = [
    "VoxelAtlas",
    "build_nt_atlas",
    "save_atlas",
    "load_atlas",
    "resolve_targets",
    "NT_PRESETS",
    "NT_TARGET_GROUPS",
]
```

- [ ] **Step 6: Commit**

```bash
cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna
git add src/lacuna/atlas/store.py src/lacuna/atlas/__init__.py tests/unit/atlas/test_store.py
git commit -m "feat(atlas): add atlas building (averaging, z-scoring) and save/load"
```

---

### Task 4: Atlas Scoring Functions

**Files:**
- Create: `src/lacuna/atlas/scoring.py`
- Test: `tests/unit/atlas/test_scoring.py`

- [ ] **Step 1: Write failing tests for scoring functions**

```python
# tests/unit/atlas/test_scoring.py
"""Tests for atlas scoring functions."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.atlas.scoring import (
    score_focal,
    score_structural_endpoints,
    score_functional_overlap,
    score_react_temporal,
)
from lacuna.atlas.types import VoxelAtlas


@pytest.fixture
def simple_atlas():
    """Atlas with known values for predictable scoring."""
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    shape = (10, 10, 10)

    # 5HT1a: value 2.0 everywhere except zeros at edges
    data_5ht = np.full(shape, 2.0, dtype=np.float32)
    data_5ht[0, :, :] = 0.0

    # D1: value 1.0 everywhere, no zeros
    data_d1 = np.full(shape, 1.0, dtype=np.float32)

    maps = {
        "5HT1a": nib.Nifti1Image(data_5ht, affine),
        "D1": nib.Nifti1Image(data_d1, affine),
    }
    return VoxelAtlas(maps=maps, space="MNI152NLin6Asym", resolution=2.0, domain="neurotransmitter")


class TestScoreFocal:
    def test_mean_score(self, simple_atlas):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5, 5, 5] = True
        mask[5, 5, 6] = True
        scores = score_focal(simple_atlas, mask)
        assert scores["D1"] == pytest.approx(1.0)
        assert scores["5HT1a"] == pytest.approx(2.0)

    def test_zeros_excluded(self, simple_atlas):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[0, 5, 5] = True  # this is a zero voxel in 5HT1a
        mask[5, 5, 5] = True  # this is 2.0 in 5HT1a
        scores = score_focal(simple_atlas, mask)
        # Zero excluded, so mean of [2.0] = 2.0
        assert scores["5HT1a"] == pytest.approx(2.0)
        # D1 has no zeros, mean of [1.0, 1.0] = 1.0
        assert scores["D1"] == pytest.approx(1.0)

    def test_sum_aggregation(self, simple_atlas):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5, 5, 5] = True
        mask[5, 5, 6] = True
        scores = score_focal(simple_atlas, mask, aggregation="sum")
        assert scores["D1"] == pytest.approx(2.0)
        assert scores["5HT1a"] == pytest.approx(4.0)

    def test_empty_mask_returns_nan(self, simple_atlas):
        mask = np.zeros((10, 10, 10), dtype=bool)
        scores = score_focal(simple_atlas, mask)
        assert np.isnan(scores["D1"])

    def test_parcel_mask_restricts_scoring(self, simple_atlas):
        lesion_mask = np.zeros((10, 10, 10), dtype=bool)
        lesion_mask[5, 5, 5] = True
        lesion_mask[5, 5, 6] = True
        parcel_mask = np.zeros((10, 10, 10), dtype=bool)
        parcel_mask[5, 5, 5] = True  # only one voxel overlaps
        scores = score_focal(simple_atlas, lesion_mask, parcel_mask=parcel_mask)
        assert scores["D1"] == pytest.approx(1.0)


class TestScoreStructuralEndpoints:
    def test_basic_scoring(self, simple_atlas):
        # 3 streamlines, each with 2 endpoints (start, end) in (x,y,z)
        endpoints_start = np.array([[5, 5, 5], [5, 5, 6], [5, 5, 7]], dtype=np.int32)
        endpoints_end = np.array([[5, 5, 8], [5, 5, 9], [5, 5, 5]], dtype=np.int32)
        intersecting_ids = np.array([0, 2])  # streamlines 0 and 2 intersect lesion

        scores, count = score_structural_endpoints(
            simple_atlas, endpoints_start, endpoints_end, intersecting_ids
        )
        assert count == 2
        # D1 is 1.0 everywhere → each streamline mean = 1.0 → sum of 2 = 2.0
        assert scores["D1"] == pytest.approx(2.0)

    def test_returns_streamline_count(self, simple_atlas):
        endpoints_start = np.array([[5, 5, 5]], dtype=np.int32)
        endpoints_end = np.array([[5, 5, 6]], dtype=np.int32)
        intersecting_ids = np.array([0])
        _, count = score_structural_endpoints(
            simple_atlas, endpoints_start, endpoints_end, intersecting_ids
        )
        assert count == 1

    def test_empty_intersecting_returns_zero(self, simple_atlas):
        endpoints_start = np.array([[5, 5, 5]], dtype=np.int32)
        endpoints_end = np.array([[5, 5, 6]], dtype=np.int32)
        intersecting_ids = np.array([], dtype=np.int32)
        scores, count = score_structural_endpoints(
            simple_atlas, endpoints_start, endpoints_end, intersecting_ids
        )
        assert count == 0
        assert scores["D1"] == 0.0


class TestScoreFunctionalOverlap:
    def test_positive_connectivity_only(self, simple_atlas):
        # z-map with positive and negative values
        z_data = np.zeros((10, 10, 10), dtype=np.float32)
        z_data[5, 5, 5] = 0.5   # positive
        z_data[5, 5, 6] = -0.3  # negative — should be excluded
        z_map = nib.Nifti1Image(z_data, np.eye(4) * 2)

        scores = score_functional_overlap(simple_atlas, z_map)
        # Only positive voxel [5,5,5] contributes
        # D1 at [5,5,5] = 1.0, z = 0.5 → weighted = 0.5 / 0.5 = 1.0 (normalized)
        assert scores["D1"] > 0
        # Negative z-values should not reduce the score
        assert scores["D1"] == pytest.approx(1.0)

    def test_all_negative_returns_nan(self, simple_atlas):
        z_data = np.full((10, 10, 10), -0.5, dtype=np.float32)
        z_map = nib.Nifti1Image(z_data, np.eye(4) * 2)
        scores = score_functional_overlap(simple_atlas, z_map)
        assert np.isnan(scores["D1"])


class TestScoreReactTemporal:
    def test_perfect_correlation(self):
        n_timepoints = 100
        nt_timeseries = {
            "D1": np.sin(np.linspace(0, 4 * np.pi, n_timepoints)),
            "5HT1a": np.cos(np.linspace(0, 4 * np.pi, n_timepoints)),
        }
        lesion_ts = nt_timeseries["D1"].copy()  # identical to D1
        scores = score_react_temporal(nt_timeseries, lesion_ts)
        assert scores["D1"] == pytest.approx(1.0, abs=0.01)
        assert abs(scores["5HT1a"]) < 0.3  # sin/cos are ~uncorrelated

    def test_returns_all_targets(self):
        n_timepoints = 50
        nt_timeseries = {
            "D1": np.random.default_rng(42).standard_normal(n_timepoints),
            "5HT1a": np.random.default_rng(43).standard_normal(n_timepoints),
        }
        lesion_ts = np.random.default_rng(44).standard_normal(n_timepoints)
        scores = score_react_temporal(nt_timeseries, lesion_ts)
        assert set(scores.keys()) == {"D1", "5HT1a"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/atlas/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lacuna.atlas.scoring'`

- [ ] **Step 3: Implement scoring functions**

```python
# src/lacuna/atlas/scoring.py
"""Scoring functions for atlas-lesion overlap computation.

All functions accept an optional parcel_mask parameter for regional scoring.
Zero atlas values are always excluded from scoring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import nibabel as nib

    from lacuna.atlas.types import VoxelAtlas


def score_focal(
    atlas: VoxelAtlas,
    lesion_mask: np.ndarray,
    aggregation: str = "mean",
    parcel_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Score NT atlas values within a binary lesion mask.

    Parameters
    ----------
    atlas : VoxelAtlas
        Z-scored NT atlas.
    lesion_mask : np.ndarray
        Binary 3D lesion mask. Shape must match atlas maps.
    aggregation : str
        "mean" or "sum". Zeros always excluded.
    parcel_mask : np.ndarray or None
        If provided, restrict scoring to voxels within this parcel.

    Returns
    -------
    dict[str, float]
        {target_name: score}. NaN if no valid voxels.
    """
    effective_mask = lesion_mask.astype(bool)
    if parcel_mask is not None:
        effective_mask = effective_mask & parcel_mask.astype(bool)

    scores = {}
    for target in atlas.targets:
        data = atlas.get_map(target).get_fdata()
        values = data[effective_mask]
        # Exclude zeros
        nonzero = values[values != 0]
        if len(nonzero) == 0:
            scores[target] = float("nan")
        elif aggregation == "mean":
            scores[target] = float(np.mean(nonzero))
        elif aggregation == "sum":
            scores[target] = float(np.sum(nonzero))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    return scores


def score_structural_endpoints(
    atlas: VoxelAtlas,
    endpoints_start: np.ndarray,
    endpoints_end: np.ndarray,
    intersecting_ids: np.ndarray,
    parcel_mask: np.ndarray | None = None,
) -> tuple[dict[str, float], int]:
    """Score NT values at streamline endpoints for lesion-intersecting streamlines.

    For each intersecting streamline, computes the mean of NT values at its
    two endpoints. Sums these per-streamline means across all intersecting
    streamlines.

    Parameters
    ----------
    atlas : VoxelAtlas
        Z-scored NT atlas.
    endpoints_start : np.ndarray
        Voxel coordinates of streamline start points. Shape (n_streamlines, 3).
    endpoints_end : np.ndarray
        Voxel coordinates of streamline end points. Shape (n_streamlines, 3).
    intersecting_ids : np.ndarray
        Indices of streamlines that intersect the lesion.
    parcel_mask : np.ndarray or None
        If provided, only include streamlines with at least one endpoint
        in this parcel.

    Returns
    -------
    tuple[dict[str, float], int]
        ({target_name: score}, streamline_count).
        Streamline count is the number of intersecting streamlines used.
    """
    if len(intersecting_ids) == 0:
        return {t: 0.0 for t in atlas.targets}, 0

    # Get endpoints for intersecting streamlines
    starts = endpoints_start[intersecting_ids]  # (n_intersect, 3)
    ends = endpoints_end[intersecting_ids]  # (n_intersect, 3)

    # Optional parcel filtering
    if parcel_mask is not None:
        parcel_bool = parcel_mask.astype(bool)
        in_parcel = (
            parcel_bool[starts[:, 0], starts[:, 1], starts[:, 2]]
            | parcel_bool[ends[:, 0], ends[:, 1], ends[:, 2]]
        )
        starts = starts[in_parcel]
        ends = ends[in_parcel]
        if len(starts) == 0:
            return {t: 0.0 for t in atlas.targets}, 0

    count = len(starts)
    scores = {}
    for target in atlas.targets:
        data = atlas.get_map(target).get_fdata()
        val_start = data[starts[:, 0], starts[:, 1], starts[:, 2]]
        val_end = data[ends[:, 0], ends[:, 1], ends[:, 2]]
        # Mean of two endpoints per streamline, then sum across streamlines
        per_streamline = (val_start + val_end) / 2.0
        scores[target] = float(np.sum(per_streamline))

    return scores, count


def score_functional_overlap(
    atlas: VoxelAtlas,
    connectivity_map: nib.Nifti1Image,
    aggregation: str = "mean",
    parcel_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Score NT atlas weighted by a functional connectivity map.

    Only positive connectivity values are used (default). The score is the
    weighted mean of atlas values, with connectivity values as weights.

    Parameters
    ----------
    atlas : VoxelAtlas
        Z-scored NT atlas.
    connectivity_map : nib.Nifti1Image
        Voxelwise connectivity z-map (e.g., from fLNM). Same shape as atlas.
    aggregation : str
        "mean" (weighted mean) or "sum" (weighted sum).
    parcel_mask : np.ndarray or None
        If provided, restrict scoring to this parcel.

    Returns
    -------
    dict[str, float]
        {target_name: score}. NaN if no valid voxels.
    """
    z_data = connectivity_map.get_fdata()

    # Threshold to positive values
    positive_mask = z_data > 0
    if parcel_mask is not None:
        positive_mask = positive_mask & parcel_mask.astype(bool)

    scores = {}
    for target in atlas.targets:
        nt_data = atlas.get_map(target).get_fdata()

        # Valid voxels: positive connectivity AND nonzero NT
        valid = positive_mask & (nt_data != 0)

        if not valid.any():
            scores[target] = float("nan")
            continue

        weights = z_data[valid]
        values = nt_data[valid]
        weighted_product = weights * values

        if aggregation == "mean":
            scores[target] = float(np.sum(weighted_product) / np.sum(weights))
        elif aggregation == "sum":
            scores[target] = float(np.sum(weighted_product))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    return scores


def score_react_temporal(
    nt_timeseries: dict[str, np.ndarray],
    lesion_timeseries: np.ndarray,
) -> dict[str, float]:
    """Correlate lesion BOLD timeseries with NT-weighted timeseries.

    Used for REACT-enriched global functional scoring.

    Parameters
    ----------
    nt_timeseries : dict[str, np.ndarray]
        Per-target NT timeseries from REACT stage 1. Each is 1D (n_timepoints,).
    lesion_timeseries : np.ndarray
        Mean BOLD timeseries within the lesion mask. 1D (n_timepoints,).

    Returns
    -------
    dict[str, float]
        {target_name: pearson_correlation}
    """
    scores = {}
    lesion_centered = lesion_timeseries - lesion_timeseries.mean()
    lesion_std = lesion_centered.std()

    for target, nt_ts in nt_timeseries.items():
        nt_centered = nt_ts - nt_ts.mean()
        nt_std = nt_centered.std()
        if lesion_std == 0 or nt_std == 0:
            scores[target] = 0.0
        else:
            r = np.dot(lesion_centered, nt_centered) / (
                len(lesion_centered) * lesion_std * nt_std
            )
            scores[target] = float(r)
    return scores
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/atlas/test_scoring.py -v`
Expected: All PASS

- [ ] **Step 5: Update atlas __init__.py**

Add to `src/lacuna/atlas/__init__.py`:
```python
from lacuna.atlas.scoring import (
    score_focal,
    score_functional_overlap,
    score_react_temporal,
    score_structural_endpoints,
)
```

And add to `__all__`:
```python
    "score_focal",
    "score_structural_endpoints",
    "score_functional_overlap",
    "score_react_temporal",
```

- [ ] **Step 6: Commit**

```bash
cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna
git add src/lacuna/atlas/scoring.py src/lacuna/atlas/__init__.py tests/unit/atlas/test_scoring.py
git commit -m "feat(atlas): add focal, structural endpoint, functional overlap, and REACT temporal scoring"
```

---

### Task 5: REACT Implementation

**Files:**
- Create: `src/lacuna/atlas/react.py`
- Test: `tests/unit/atlas/test_react.py`

- [ ] **Step 1: Write failing tests for REACT**

```python
# tests/unit/atlas/test_react.py
"""Tests for REACT stage 1 and stage 2 implementation."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.atlas.react import (
    react_stage1,
    react_stage2,
    compute_react_atlas,
    compute_stage1_mask,
)
from lacuna.atlas.types import VoxelAtlas


@pytest.fixture
def small_atlas():
    """Small atlas for fast REACT tests."""
    shape = (10, 10, 10)
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    rng = np.random.default_rng(42)
    maps = {}
    for target in ["D1", "5HT1a"]:
        data = rng.random(shape).astype(np.float32)
        maps[target] = nib.Nifti1Image(data, affine)
    return VoxelAtlas(maps=maps, space="MNI152NLin6Asym", resolution=2.0, domain="neurotransmitter")


@pytest.fixture
def fake_bold_subjects():
    """Two fake fMRI subjects: (n_timepoints, n_voxels_flat)."""
    rng = np.random.default_rng(42)
    n_timepoints = 50
    n_voxels = 1000  # flat voxel count within mask
    return [rng.standard_normal((n_timepoints, n_voxels)).astype(np.float32) for _ in range(2)]


class TestComputeStage1Mask:
    def test_intersection_of_nonzero(self, small_atlas):
        mask = compute_stage1_mask(small_atlas)
        # All maps have nonzero values everywhere (random 0-1)
        assert mask.shape == (10, 10, 10)
        assert mask.dtype == bool
        assert mask.sum() > 0


class TestReactStage1:
    def test_output_shape(self, small_atlas, fake_bold_subjects):
        n_timepoints = 50
        n_voxels = 1000
        stage1_mask = np.ones(n_voxels, dtype=bool)
        atlas_matrix = np.random.default_rng(42).standard_normal((2, n_voxels)).astype(np.float32)

        beta1 = react_stage1(fake_bold_subjects[0], atlas_matrix, stage1_mask)
        assert beta1.shape == (n_timepoints, 2)  # (timepoints, n_targets)

    def test_output_not_all_zero(self, small_atlas, fake_bold_subjects):
        n_voxels = 1000
        stage1_mask = np.ones(n_voxels, dtype=bool)
        atlas_matrix = np.random.default_rng(42).standard_normal((2, n_voxels)).astype(np.float32)
        beta1 = react_stage1(fake_bold_subjects[0], atlas_matrix, stage1_mask)
        assert not np.allclose(beta1, 0)


class TestReactStage2:
    def test_output_shape(self):
        n_timepoints = 50
        n_voxels = 1000
        n_targets = 2
        rng = np.random.default_rng(42)
        bold = rng.standard_normal((n_timepoints, n_voxels)).astype(np.float32)
        beta1 = rng.standard_normal((n_timepoints, n_targets)).astype(np.float32)
        stage2_mask = np.ones(n_voxels, dtype=bool)

        beta2 = react_stage2(bold, beta1, stage2_mask)
        assert beta2.shape == (n_voxels, n_targets)

    def test_output_not_all_zero(self):
        n_timepoints = 50
        n_voxels = 1000
        n_targets = 2
        rng = np.random.default_rng(42)
        bold = rng.standard_normal((n_timepoints, n_voxels)).astype(np.float32)
        beta1 = rng.standard_normal((n_timepoints, n_targets)).astype(np.float32)
        stage2_mask = np.ones(n_voxels, dtype=bool)
        beta2 = react_stage2(bold, beta1, stage2_mask)
        assert not np.allclose(beta2, 0)


class TestComputeReactAtlas:
    def test_produces_voxel_atlas(self, small_atlas):
        """Test full REACT pipeline with synthetic data."""
        shape = (10, 10, 10)
        n_voxels = np.prod(shape)
        n_timepoints = 50
        rng = np.random.default_rng(42)

        # Create synthetic subjects as list of (n_timepoints, n_voxels) arrays
        subjects_data = [
            rng.standard_normal((n_timepoints, n_voxels)).astype(np.float32)
            for _ in range(3)
        ]

        brain_mask = np.ones(shape, dtype=bool)

        result = compute_react_atlas(
            atlas=small_atlas,
            subjects_data=subjects_data,
            brain_mask=brain_mask,
            mask_shape=shape,
        )
        assert isinstance(result["stage2_atlas"], VoxelAtlas)
        assert set(result["stage2_atlas"].targets) == {"5HT1a", "D1"}
        assert "stage1_timeseries" in result
        assert len(result["stage1_timeseries"]) == 3  # one per subject
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/atlas/test_react.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lacuna.atlas.react'`

- [ ] **Step 3: Implement REACT**

```python
# src/lacuna/atlas/react.py
"""REACT (Receptor-Enriched Analysis of functional Connectivity by Targets).

Implements REACT stage 1 and stage 2 following the reference implementation
(Dipasquale et al., 2019, NeuroImage). Uses sklearn.LinearRegression for
consistency with the published method.

Stage 1: Regress BOLD spatial patterns onto NT atlas maps → NT timeseries
Stage 2: Regress BOLD timeseries onto NT timeseries → enriched spatial maps
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from lacuna.atlas.types import VoxelAtlas

logger = logging.getLogger(__name__)


def compute_stage1_mask(atlas: VoxelAtlas) -> np.ndarray:
    """Compute stage 1 mask: intersection of nonzero voxels across all atlas maps.

    Parameters
    ----------
    atlas : VoxelAtlas
        The NT atlas.

    Returns
    -------
    np.ndarray
        Boolean 3D mask where all atlas maps have nonzero values.
    """
    mask = None
    for target in atlas.targets:
        data = atlas.get_map(target).get_fdata()
        target_mask = data != 0
        if mask is None:
            mask = target_mask
        else:
            mask = mask & target_mask
    return mask


def react_stage1(
    bold_data: np.ndarray,
    atlas_matrix: np.ndarray,
    stage1_mask: np.ndarray,
) -> np.ndarray:
    """REACT Stage 1: extract NT-weighted timeseries from fMRI.

    Parameters
    ----------
    bold_data : np.ndarray
        fMRI data, shape (n_timepoints, n_voxels). Already masked to brain.
    atlas_matrix : np.ndarray
        NT atlas values, shape (n_targets, n_voxels). Same voxel ordering.
    stage1_mask : np.ndarray
        Boolean mask within the brain mask indicating voxels to use.
        Shape (n_voxels,).

    Returns
    -------
    np.ndarray
        NT timeseries, shape (n_timepoints, n_targets).
    """
    # Mask to stage1 voxels
    x = atlas_matrix[:, stage1_mask].T  # (n_voxels_s1, n_targets)
    y = bold_data[:, stage1_mask].T  # (n_voxels_s1, n_timepoints)

    # Demean
    scaler_x = StandardScaler(with_mean=True, with_std=False)
    x = scaler_x.fit_transform(x)
    scaler_y = StandardScaler(with_mean=True, with_std=False)
    y = scaler_y.fit_transform(y)

    # Regress: voxels as observations, PET maps as predictors, timepoints as DVs
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)
    beta1 = model.coef_  # (n_timepoints, n_targets)

    return beta1


def react_stage2(
    bold_data: np.ndarray,
    beta1: np.ndarray,
    stage2_mask: np.ndarray,
    normalize_data: bool = False,
) -> np.ndarray:
    """REACT Stage 2: project NT timeseries back to voxel space.

    Parameters
    ----------
    bold_data : np.ndarray
        fMRI data, shape (n_timepoints, n_voxels). Already masked to brain.
    beta1 : np.ndarray
        NT timeseries from stage 1, shape (n_timepoints, n_targets).
    stage2_mask : np.ndarray
        Boolean mask within the brain mask indicating voxels to use.
        Shape (n_voxels,).
    normalize_data : bool
        If True, normalize BOLD data to unit standard deviation (optional).

    Returns
    -------
    np.ndarray
        Enriched maps, shape (n_voxels, n_targets). Voxels outside stage2_mask
        are zero.
    """
    n_voxels = bold_data.shape[1]
    n_targets = beta1.shape[1]

    # Prepare predictors (NT timeseries): standardize
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    x = scaler_x.fit_transform(beta1)  # (n_timepoints, n_targets)

    # Prepare dependent variable (BOLD timeseries): demean, optionally normalize
    y = bold_data[:, stage2_mask]  # (n_timepoints, n_voxels_s2)
    scaler_y = StandardScaler(with_mean=True, with_std=normalize_data)
    y = scaler_y.fit_transform(y)

    # Regress: timepoints as observations, NT timeseries as predictors, voxels as DVs
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)

    beta2 = np.zeros((n_voxels, n_targets), dtype=np.float32)
    beta2[stage2_mask] = model.coef_  # (n_voxels_s2, n_targets)

    return beta2


def compute_react_atlas(
    atlas: VoxelAtlas,
    subjects_data: list[np.ndarray],
    brain_mask: np.ndarray,
    mask_shape: tuple[int, int, int],
) -> dict[str, Any]:
    """Run full REACT pipeline across normative subjects.

    Parameters
    ----------
    atlas : VoxelAtlas
        The prepared (averaged, z-scored) NT atlas.
    subjects_data : list[np.ndarray]
        Per-subject fMRI data. Each is (n_timepoints, n_voxels) where n_voxels
        corresponds to the flattened brain_mask.
    brain_mask : np.ndarray
        Boolean 3D brain mask. Shape must match atlas maps.
    mask_shape : tuple
        3D shape of the brain volume.

    Returns
    -------
    dict with keys:
        "stage2_atlas": VoxelAtlas — Fisher-z averaged enriched maps
        "stage1_timeseries": list[np.ndarray] — per-subject NT timeseries
    """
    import nibabel as nib

    n_subjects = len(subjects_data)
    n_voxels_brain = brain_mask.sum() if brain_mask.ndim == 3 else brain_mask.shape[0]
    n_targets = len(atlas.targets)

    # Build atlas matrix within brain mask
    flat_mask = brain_mask.ravel() if brain_mask.ndim == 3 else brain_mask
    atlas_matrix = np.empty((n_targets, int(flat_mask.sum())), dtype=np.float64)
    for i, target in enumerate(atlas.targets):
        data = atlas.get_map(target).get_fdata().ravel()
        atlas_matrix[i] = data[flat_mask]

    # Compute stage 1 mask within brain voxels
    stage1_nonzero = np.all(atlas_matrix != 0, axis=0)  # (n_voxels_brain,)

    # Stage 2 mask: all brain voxels
    stage2_mask = np.ones(int(flat_mask.sum()), dtype=bool)

    # Check collinearity
    x_check = atlas_matrix[:, stage1_nonzero].T
    cond = np.linalg.cond(x_check.T @ x_check)
    if cond > 1000:
        logger.warning(
            "High condition number (%.1f) in NT atlas regressor matrix. "
            "Consider grouped regression or reviewing map selection.",
            cond,
        )

    # Process each subject
    stage1_timeseries = []
    stage2_accumulator = np.zeros((int(flat_mask.sum()), n_targets), dtype=np.float64)

    for i, bold in enumerate(subjects_data):
        logger.info("REACT: processing subject %d/%d", i + 1, n_subjects)

        # Stage 1
        beta1 = react_stage1(bold, atlas_matrix, stage1_nonzero)
        stage1_timeseries.append(beta1)

        # Stage 2
        beta2 = react_stage2(bold, beta1, stage2_mask)

        # Fisher-z transform and accumulate
        # Clip to avoid arctanh(±1) = ±inf
        beta2_clipped = np.clip(beta2, -0.9999, 0.9999)
        beta2_z = np.arctanh(beta2_clipped)
        stage2_accumulator += beta2_z

    # Average across subjects
    mean_stage2_z = stage2_accumulator / n_subjects

    # Build enriched atlas maps
    reference_affine = atlas.get_map(atlas.targets[0]).affine
    enriched_maps = {}
    for j, target in enumerate(atlas.targets):
        vol = np.zeros(np.prod(mask_shape), dtype=np.float32)
        vol[flat_mask] = mean_stage2_z[:, j].astype(np.float32)
        vol = vol.reshape(mask_shape)
        enriched_maps[target] = nib.Nifti1Image(vol, reference_affine)

    stage2_atlas = VoxelAtlas(
        maps=enriched_maps,
        space=atlas.space,
        resolution=atlas.resolution,
        domain=atlas.domain,
        metadata={
            **atlas.metadata,
            "enriched": True,
            "method": "REACT",
            "n_subjects": n_subjects,
        },
    )

    return {
        "stage2_atlas": stage2_atlas,
        "stage1_timeseries": stage1_timeseries,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/atlas/test_react.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna
git add src/lacuna/atlas/react.py tests/unit/atlas/test_react.py
git commit -m "feat(atlas): implement REACT stage 1+2 with sklearn LinearRegression"
```

---

### Task 6: Rename RegionalDamage to LocalDamage

**Files:**
- Create: `src/lacuna/analysis/local_damage.py`
- Delete: `src/lacuna/analysis/regional_damage.py`
- Modify: `src/lacuna/analysis/__init__.py`
- Test: `tests/unit/analysis/test_local_damage_rename.py`

- [ ] **Step 1: Write failing test for LocalDamage import**

```python
# tests/unit/analysis/test_local_damage_rename.py
"""Tests verifying the RegionalDamage → LocalDamage rename."""

from lacuna.analysis import LocalDamage
from lacuna.analysis.base import BaseAnalysis


class TestLocalDamageRename:
    def test_local_damage_importable(self):
        assert LocalDamage is not None

    def test_is_base_analysis_subclass(self):
        assert issubclass(LocalDamage, BaseAnalysis)

    def test_regional_damage_removed(self):
        """RegionalDamage should no longer be importable from analysis."""
        import lacuna.analysis as analysis_module
        assert not hasattr(analysis_module, "RegionalDamage")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/analysis/test_local_damage_rename.py -v`
Expected: FAIL — `ImportError: cannot import name 'LocalDamage'`

- [ ] **Step 3: Copy regional_damage.py to local_damage.py and rename the class**

Copy `src/lacuna/analysis/regional_damage.py` to `src/lacuna/analysis/local_damage.py`. Then in `local_damage.py`:
- Rename class `RegionalDamage` → `LocalDamage`
- Update the docstring to say "Local damage analysis" instead of "Regional damage"
- Update `_run_analysis()` to use `source="LocalDamage"` in `build_result_key()` calls
- Remove the old `regional_damage.py` file

- [ ] **Step 4: Update `analysis/__init__.py`**

In `src/lacuna/analysis/__init__.py`:
- Replace `from lacuna.analysis.regional_damage import RegionalDamage` with `from lacuna.analysis.local_damage import LocalDamage`
- Update `__all__` to replace `"RegionalDamage"` with `"LocalDamage"`
- Update the module docstring

- [ ] **Step 5: Update CLI parser and main to use new name**

In `src/lacuna/cli/parser.py`: rename `_build_rd_parser` references from "rd" to "ld" (or keep "rd" as alias and add "ld"). Check exact usage.

In `src/lacuna/cli/main.py`: update the analysis alias mapping from `"rd"` → `"RegionalDamage"` to `"ld"` → `"LocalDamage"` (keep "rd" as backward-compatible alias).

- [ ] **Step 6: Update existing tests that reference RegionalDamage**

Search for `RegionalDamage` in all test files and update to `LocalDamage`. Update imports from `lacuna.analysis.regional_damage` to `lacuna.analysis.local_damage`.

- [ ] **Step 7: Run all tests to verify rename is clean**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/ -v --tb=short -q 2>&1 | head -100`
Expected: All previously passing tests still pass, new tests pass

- [ ] **Step 8: Commit**

```bash
cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna
git add -A
git commit -m "refactor: rename RegionalDamage to LocalDamage for consistent naming"
```

---

### Task 7: Extend StructuralNetworkMapping with keep_filtered_tractogram

**Files:**
- Modify: `src/lacuna/analysis/structural_network_mapping.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/analysis/test_snm_keep_tractogram.py
"""Test that StructuralNetworkMapping can expose the filtered tractogram."""

import pytest

from lacuna.analysis import StructuralNetworkMapping


class TestKeepFilteredTractogram:
    def test_parameter_accepted(self):
        """Verify the parameter is accepted without error."""
        snm = StructuralNetworkMapping(
            connectome_name="dTOR-985",
            keep_filtered_tractogram=True,
            check_dependencies=False,
        )
        assert snm.keep_filtered_tractogram is True

    def test_default_is_false(self):
        snm = StructuralNetworkMapping(
            connectome_name="dTOR-985",
            check_dependencies=False,
        )
        assert snm.keep_filtered_tractogram is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/analysis/test_snm_keep_tractogram.py -v`
Expected: FAIL — `TypeError: unexpected keyword argument 'keep_filtered_tractogram'`

- [ ] **Step 3: Add keep_filtered_tractogram parameter to SNM**

In `src/lacuna/analysis/structural_network_mapping.py`:

Add `keep_filtered_tractogram: bool = False` to `__init__` parameters.
Store as `self.keep_filtered_tractogram = keep_filtered_tractogram`.

In `_run_analysis()`, after the line that creates `mask_tractogram` as a `Tractogram` (around line 822-835 where `keep_intermediate` is checked), add:

```python
if self.keep_filtered_tractogram and mask_tck_path.exists():
    results["filtered_tractogram"] = Tractogram(
        name="filtered_tractogram",
        tractogram_path=mask_tck_path,
        metadata={"description": "Lesion-intersecting streamlines"},
    )
```

This stores the filtered tractogram as a regular result (not intermediate), so sntm can find it via `SubjectData.results["StructuralNetworkMapping"]["filtered_tractogram"]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/analysis/test_snm_keep_tractogram.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna
git add src/lacuna/analysis/structural_network_mapping.py tests/unit/analysis/test_snm_keep_tractogram.py
git commit -m "feat(snm): add keep_filtered_tractogram option for sntm consumption"
```

---

### Task 8: LocalNeurotransmitterMapping Analysis

**Files:**
- Create: `src/lacuna/analysis/local_neurotransmitter_mapping.py`
- Test: `tests/unit/analysis/test_local_ntm.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/analysis/test_local_ntm.py
"""Tests for LocalNeurotransmitterMapping analysis."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis.local_neurotransmitter_mapping import LocalNeurotransmitterMapping
from lacuna.atlas.store import build_nt_atlas, save_atlas
from lacuna.atlas.types import VoxelAtlas
from lacuna.core.data_types import ScalarMetric
from lacuna.core.subject_data import SubjectData


def _create_pet_map(tmp_path, target, tracer, pub, shape=(91, 109, 91)):
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    rng = np.random.default_rng(hash((target, tracer, pub)) % 2**32)
    data = rng.random(shape).astype(np.float32) + 0.1
    fname = f"target-{target}_tracer-{tracer}_n-10_dx-hc_pub-{pub}_space-MNI152NLin6Asym_desc-proc.nii.gz"
    path = tmp_path / fname
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


@pytest.fixture
def atlas_cache(tmp_path):
    """Build and cache a small NT atlas."""
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _create_pet_map(pet_dir, "D1", "sch23390", "kaller2017")
    _create_pet_map(pet_dir, "5HT1a", "cumi101", "beliveau2017")
    atlas = build_nt_atlas(pet_dir)
    cache_dir = tmp_path / "cache"
    save_atlas(atlas, cache_dir)
    return cache_dir


@pytest.fixture
def lesion_subject():
    """A SubjectData with a small lesion in MNI152NLin6Asym 2mm."""
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    mask_data = np.zeros((91, 109, 91), dtype=np.int8)
    mask_data[45:50, 55:60, 45:50] = 1  # small lesion
    mask_img = nib.Nifti1Image(mask_data, affine)
    return SubjectData(
        mask_img=mask_img,
        space="MNI152NLin6Asym",
        resolution=2.0,
        metadata={"subject_id": "sub-001"},
    )


class TestLocalNTMConstruction:
    def test_basic_construction(self, atlas_cache):
        lntm = LocalNeurotransmitterMapping(atlas_cache_dir=atlas_cache)
        assert lntm.TARGET_SPACE is None  # adapts to input

    def test_targets_parameter(self, atlas_cache):
        lntm = LocalNeurotransmitterMapping(
            atlas_cache_dir=atlas_cache,
            targets=["D1"],
        )
        assert lntm._target_spec == ["D1"]


class TestLocalNTMRun:
    def test_produces_scalar_metrics(self, atlas_cache, lesion_subject):
        lntm = LocalNeurotransmitterMapping(atlas_cache_dir=atlas_cache)
        result = lntm.run(lesion_subject)
        # Should have results for D1 and 5HT1a
        lntm_results = result.results["LocalNeurotransmitterMapping"]
        assert "D1" in lntm_results
        assert "5HT1a" in lntm_results
        assert isinstance(lntm_results["D1"], ScalarMetric)

    def test_scores_are_finite(self, atlas_cache, lesion_subject):
        lntm = LocalNeurotransmitterMapping(atlas_cache_dir=atlas_cache)
        result = lntm.run(lesion_subject)
        lntm_results = result.results["LocalNeurotransmitterMapping"]
        for target in ["D1", "5HT1a"]:
            score = lntm_results[target].get_data()
            assert np.isfinite(score)

    def test_target_subsetting(self, atlas_cache, lesion_subject):
        lntm = LocalNeurotransmitterMapping(
            atlas_cache_dir=atlas_cache,
            targets=["D1"],
        )
        result = lntm.run(lesion_subject)
        lntm_results = result.results["LocalNeurotransmitterMapping"]
        assert "D1" in lntm_results
        assert "5HT1a" not in lntm_results
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/analysis/test_local_ntm.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement LocalNeurotransmitterMapping**

```python
# src/lacuna/analysis/local_neurotransmitter_mapping.py
"""Local Neurotransmitter Mapping (lntm).

Scores NT atlas values directly within the lesion mask.
Answers: "what neurotransmitter landscape did the lesion wipe out?"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lacuna.analysis.base import BaseAnalysis
from lacuna.atlas.config import resolve_targets
from lacuna.atlas.scoring import score_focal
from lacuna.atlas.store import load_atlas
from lacuna.core.data_types import ParcelData, ScalarMetric
from lacuna.core.subject_data import SubjectData

logger = logging.getLogger(__name__)


class LocalNeurotransmitterMapping(BaseAnalysis):
    """Local neurotransmitter mapping: NT scores within the lesion footprint.

    Computes, for each NT target, the mean z-scored NT density within the
    lesion mask (excluding zero-valued voxels).

    Parameters
    ----------
    atlas_cache_dir : Path
        Directory containing the prepared NT atlas (from `lacuna prepare lntm`).
    targets : str or list[str]
        Target selection. Preset name ("all", "dopaminergic", etc.) or
        explicit list of target names. Default "all".
    enriched : bool
        If True, use REACT-enriched atlas instead of static.
    parcel_atlases : list[str] or None
        Atlas names for regional scoring.
    aggregation : str
        "mean" or "sum". Default "mean".
    verbose : bool
        Enable verbose logging.
    keep_intermediate : bool
        Keep intermediate results.
    """

    TARGET_SPACE = None  # Adapt atlas to mask space
    TARGET_RESOLUTION = None
    batch_strategy = "sequential"

    def __init__(
        self,
        atlas_cache_dir: str | Path,
        targets: str | list[str] = "all",
        enriched: bool = False,
        parcel_atlases: list[str] | None = None,
        aggregation: str = "mean",
        verbose: bool = False,
        keep_intermediate: bool = False,
    ):
        super().__init__(verbose=verbose, keep_intermediate=keep_intermediate)
        self.atlas_cache_dir = Path(atlas_cache_dir)
        self._target_spec = targets
        self.enriched = enriched
        self.parcel_atlases = parcel_atlases
        self.aggregation = aggregation

    def _validate_inputs(self, mask_data: SubjectData) -> None:
        """Validate that the atlas cache exists and targets are available."""
        atlas = load_atlas(self.atlas_cache_dir)
        self._atlas = atlas
        self._resolved_targets = resolve_targets(self._target_spec, atlas.targets)

    def _run_analysis(self, mask_data: SubjectData) -> dict[str, Any]:
        """Compute local NT scores."""
        atlas = self._atlas

        # Resample atlas to mask space if needed
        mask_img = mask_data.mask_img
        mask_affine = mask_img.affine
        mask_shape = mask_img.shape[:3]

        atlas_shape = atlas.get_map(atlas.targets[0]).shape[:3]
        if atlas_shape != mask_shape:
            atlas = atlas.resample_to(mask_affine, mask_shape)

        # Subset to requested targets
        atlas = atlas.subset(self._resolved_targets)

        lesion_mask = mask_img.get_fdata().astype(bool)

        # Global scores
        global_scores = score_focal(atlas, lesion_mask, aggregation=self.aggregation)

        results = {}
        for target, score in global_scores.items():
            results[target] = ScalarMetric(
                name=target,
                data=score,
                data_type="scalar",
                metadata={"analysis": "lntm", "aggregation": self.aggregation},
            )

        # Regional scores (if parcellation requested)
        if self.parcel_atlases:
            self._compute_regional_scores(
                atlas, lesion_mask, mask_data, results
            )

        return results

    def _compute_regional_scores(
        self,
        atlas,
        lesion_mask,
        mask_data,
        results,
    ):
        """Compute per-parcel NT scores."""
        from lacuna.analysis.parcel_aggregation import ParcelAggregation

        # Load atlases using the existing parcellation infrastructure
        pa = ParcelAggregation(parcel_names=self.parcel_atlases)
        pa._validate_inputs(mask_data)

        for atlas_info in pa._atlases:
            atlas_name = atlas_info["name"]
            atlas_img = atlas_info["image"]
            labels = atlas_info["labels"]
            atlas_data = atlas_img.get_fdata()

            for target in atlas.targets:
                parcel_scores = {}
                for label_idx, label_name in enumerate(labels, start=1):
                    parcel_mask = (atlas_data == label_idx)
                    combined = lesion_mask & parcel_mask
                    if not combined.any():
                        continue
                    target_scores = score_focal(
                        atlas.subset([target]),
                        lesion_mask,
                        aggregation=self.aggregation,
                        parcel_mask=parcel_mask,
                    )
                    parcel_scores[label_name] = target_scores[target]

                if parcel_scores:
                    key = f"{target}_{atlas_name}"
                    results[key] = ParcelData(
                        name=key,
                        data=parcel_scores,
                        region_labels=list(parcel_scores.keys()),
                        parcel_names=[atlas_name],
                        aggregation_method=self.aggregation,
                    )

    def _get_parameters(self) -> dict:
        return {
            "atlas_cache_dir": str(self.atlas_cache_dir),
            "targets": self._target_spec,
            "enriched": self.enriched,
            "aggregation": self.aggregation,
            "parcel_atlases": self.parcel_atlases,
        }
```

- [ ] **Step 4: Update analysis/__init__.py to include LocalNeurotransmitterMapping**

Add to `src/lacuna/analysis/__init__.py`:
```python
from lacuna.analysis.local_neurotransmitter_mapping import LocalNeurotransmitterMapping
```
And add `"LocalNeurotransmitterMapping"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/analysis/test_local_ntm.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna
git add src/lacuna/analysis/local_neurotransmitter_mapping.py src/lacuna/analysis/__init__.py tests/unit/analysis/test_local_ntm.py
git commit -m "feat(analysis): add LocalNeurotransmitterMapping (lntm) analysis"
```

---

### Task 9: StructuralNeurotransmitterMapping Analysis

**Files:**
- Create: `src/lacuna/analysis/structural_neurotransmitter_mapping.py`
- Test: `tests/unit/analysis/test_structural_ntm.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/analysis/test_structural_ntm.py
"""Tests for StructuralNeurotransmitterMapping analysis."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis.structural_neurotransmitter_mapping import (
    StructuralNeurotransmitterMapping,
)
from lacuna.atlas.store import build_nt_atlas, save_atlas
from lacuna.core.data_types import ScalarMetric


def _create_pet_map(tmp_path, target, tracer, pub, shape=(91, 109, 91)):
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    rng = np.random.default_rng(hash((target, tracer, pub)) % 2**32)
    data = rng.random(shape).astype(np.float32) + 0.1
    fname = f"target-{target}_tracer-{tracer}_n-10_dx-hc_pub-{pub}_space-MNI152NLin6Asym_desc-proc.nii.gz"
    path = tmp_path / fname
    nib.save(nib.Nifti1Image(data, affine), str(path))


@pytest.fixture
def atlas_cache(tmp_path):
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _create_pet_map(pet_dir, "D1", "sch23390", "kaller2017")
    _create_pet_map(pet_dir, "5HT1a", "cumi101", "beliveau2017")
    atlas = build_nt_atlas(pet_dir)
    cache_dir = tmp_path / "cache"
    save_atlas(atlas, cache_dir)
    return cache_dir


class TestStructuralNTMConstruction:
    def test_basic_construction(self, atlas_cache):
        sntm = StructuralNeurotransmitterMapping(
            atlas_cache_dir=atlas_cache,
            connectome_name="dTOR-985",
            check_dependencies=False,
        )
        assert sntm._target_spec == "all"

    def test_targets_parameter(self, atlas_cache):
        sntm = StructuralNeurotransmitterMapping(
            atlas_cache_dir=atlas_cache,
            connectome_name="dTOR-985",
            targets=["D1"],
            check_dependencies=False,
        )
        assert sntm._target_spec == ["D1"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/analysis/test_structural_ntm.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement StructuralNeurotransmitterMapping**

```python
# src/lacuna/analysis/structural_neurotransmitter_mapping.py
"""Structural Neurotransmitter Mapping (sntm).

Scores NT atlas values at endpoints of lesion-disconnected streamlines.
Answers: "what NT-weighted structural connectivity does the lesion disrupt?"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from lacuna.analysis.base import BaseAnalysis
from lacuna.atlas.config import resolve_targets
from lacuna.atlas.scoring import score_structural_endpoints
from lacuna.atlas.store import load_atlas
from lacuna.core.data_types import ParcelData, ScalarMetric, Tractogram
from lacuna.core.subject_data import SubjectData

logger = logging.getLogger(__name__)


class StructuralNeurotransmitterMapping(BaseAnalysis):
    """Structural neurotransmitter mapping: NT scores via structural disconnection.

    For each lesion-intersecting streamline, computes the mean NT value at its
    two endpoints, then sums across all intersecting streamlines.

    Parameters
    ----------
    atlas_cache_dir : Path
        Directory containing the prepared NT atlas.
    connectome_name : str
        Name of the structural connectome (e.g., "dTOR-985").
    targets : str or list[str]
        Target selection. Default "all".
    enriched : bool
        If True, use REACT-enriched atlas.
    parcel_atlases : list[str] or None
        Atlas names for regional scoring.
    precomputed_weights_dir : Path or None
        Directory with precomputed endpoint NT weights (from `lacuna prepare sntm`).
    check_dependencies : bool
        Check for MRtrix3 availability.
    n_jobs : int
        Number of parallel jobs for MRtrix.
    verbose : bool
        Enable verbose logging.
    keep_intermediate : bool
        Keep intermediate results.
    """

    TARGET_SPACE = None  # Set dynamically from connectome
    TARGET_RESOLUTION = None
    batch_strategy = "sequential"

    def __init__(
        self,
        atlas_cache_dir: str | Path,
        connectome_name: str,
        targets: str | list[str] = "all",
        enriched: bool = False,
        parcel_atlases: list[str] | None = None,
        precomputed_weights_dir: str | Path | None = None,
        check_dependencies: bool = True,
        n_jobs: int = 1,
        verbose: bool = False,
        keep_intermediate: bool = False,
    ):
        super().__init__(verbose=verbose, keep_intermediate=keep_intermediate)
        self.atlas_cache_dir = Path(atlas_cache_dir)
        self.connectome_name = connectome_name
        self._target_spec = targets
        self.enriched = enriched
        self.parcel_atlases = parcel_atlases
        self.precomputed_weights_dir = (
            Path(precomputed_weights_dir) if precomputed_weights_dir else None
        )
        self.n_jobs = n_jobs

        if check_dependencies:
            from lacuna.utils.mrtrix import check_mrtrix_available
            check_mrtrix_available()

    def _validate_inputs(self, mask_data: SubjectData) -> None:
        """Validate atlas, connectome, and resolve targets."""
        atlas = load_atlas(self.atlas_cache_dir)
        self._atlas = atlas
        self._resolved_targets = resolve_targets(self._target_spec, atlas.targets)

    def _run_analysis(self, mask_data: SubjectData) -> dict[str, Any]:
        """Compute structural NT scores."""
        from lacuna.utils.mrtrix import filter_tractogram_by_mask

        atlas = self._atlas.subset(self._resolved_targets)

        # Resilience chain: find or compute filtered tractogram
        filtered_tck_path = self._find_or_compute_filtered_tractogram(mask_data)

        if filtered_tck_path is None:
            # Empty lesion or no intersecting streamlines
            results = {}
            for target in self._resolved_targets:
                results[target] = ScalarMetric(
                    name=target, data=0.0, data_type="scalar",
                    metadata={"analysis": "sntm"},
                )
            results["streamline_count"] = ScalarMetric(
                name="streamline_count", data=0, data_type="scalar",
            )
            return results

        # Get endpoint coordinates and intersecting IDs
        endpoints_start, endpoints_end, intersecting_ids = (
            self._get_endpoint_data(mask_data, filtered_tck_path, atlas)
        )

        # Score
        scores, count = score_structural_endpoints(
            atlas, endpoints_start, endpoints_end, intersecting_ids
        )

        results = {}
        for target, score in scores.items():
            results[target] = ScalarMetric(
                name=target, data=score, data_type="scalar",
                metadata={"analysis": "sntm", "streamline_count": count},
            )
        results["streamline_count"] = ScalarMetric(
            name="streamline_count", data=count, data_type="scalar",
        )

        return results

    def _find_or_compute_filtered_tractogram(self, mask_data):
        """Resilience chain: find existing filtered tractogram or compute one.

        Checks:
        1. SubjectData.results for SNM filtered_tractogram
        2. Output directory for previously written SNM outputs
        3. Computes filtering via MRtrix tckedit
        """
        import tempfile
        from lacuna.utils.mrtrix import filter_tractogram_by_mask

        # Check 1: SubjectData results
        if "StructuralNetworkMapping" in mask_data.results:
            snm_results = mask_data.results["StructuralNetworkMapping"]
            if "filtered_tractogram" in snm_results:
                tck = snm_results["filtered_tractogram"]
                if isinstance(tck, Tractogram) and tck.tractogram_path.exists():
                    logger.info("Reusing filtered tractogram from SNM results")
                    return tck.tractogram_path

        # Check 3: Compute via MRtrix
        logger.info("Computing filtered tractogram via MRtrix")
        from lacuna.data.atlases import load_structural_connectome
        connectome = load_structural_connectome(self.connectome_name)

        tmp_dir = tempfile.mkdtemp(prefix="sntm_")
        mask_path = Path(tmp_dir) / "lesion_mask.nii.gz"
        import nibabel as nib
        nib.save(mask_data.mask_img, str(mask_path))

        filtered_path = Path(tmp_dir) / "filtered.tck"
        filter_tractogram_by_mask(
            tractogram_path=connectome.tractogram_path,
            mask=str(mask_path),
            output_path=str(filtered_path),
            n_jobs=self.n_jobs,
            force=True,
            verbose=self.verbose,
        )

        if not filtered_path.exists():
            return None
        return filtered_path

    def _get_endpoint_data(self, mask_data, filtered_tck_path, atlas):
        """Extract endpoint coordinates and compute intersecting streamline IDs.

        Uses MRtrix tckresample -endpoints to get endpoint coordinates,
        then samples NT values at those locations.
        """
        import tempfile
        from lacuna.utils.mrtrix import run_mrtrix_command

        tmp_dir = Path(filtered_tck_path).parent

        # Get endpoints from the full tractogram
        from lacuna.data.atlases import load_structural_connectome
        connectome = load_structural_connectome(self.connectome_name)

        endpoints_tck = tmp_dir / "endpoints.tck"
        run_mrtrix_command(
            f"tckresample {connectome.tractogram_path} {endpoints_tck} -endpoints",
            verbose=self.verbose,
        )

        # Load endpoints and convert to voxel coordinates
        import nibabel as nib
        endpoints_tractogram = nib.streamlines.load(str(endpoints_tck))
        streamlines = endpoints_tractogram.streamlines

        # Convert world coordinates to voxel indices for atlas sampling
        ref_img = atlas.get_map(atlas.targets[0])
        inv_affine = np.linalg.inv(ref_img.affine)

        n_streamlines = len(streamlines)
        endpoints_start = np.zeros((n_streamlines, 3), dtype=np.int32)
        endpoints_end = np.zeros((n_streamlines, 3), dtype=np.int32)

        for i, sl in enumerate(streamlines):
            start_world = sl[0]
            end_world = sl[-1]
            start_vox = (inv_affine[:3, :3] @ start_world + inv_affine[:3, 3]).astype(np.int32)
            end_vox = (inv_affine[:3, :3] @ end_world + inv_affine[:3, 3]).astype(np.int32)
            endpoints_start[i] = np.clip(start_vox, 0, np.array(ref_img.shape[:3]) - 1)
            endpoints_end[i] = np.clip(end_vox, 0, np.array(ref_img.shape[:3]) - 1)

        # Determine intersecting streamline IDs from filtered tractogram
        filtered = nib.streamlines.load(str(filtered_tck_path))
        intersecting_ids = np.arange(len(filtered.streamlines))

        return endpoints_start, endpoints_end, intersecting_ids

    def _get_parameters(self) -> dict:
        return {
            "atlas_cache_dir": str(self.atlas_cache_dir),
            "connectome_name": self.connectome_name,
            "targets": self._target_spec,
            "enriched": self.enriched,
        }
```

- [ ] **Step 4: Update analysis/__init__.py**

Add import and `__all__` entry for `StructuralNeurotransmitterMapping`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/analysis/test_structural_ntm.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna
git add src/lacuna/analysis/structural_neurotransmitter_mapping.py src/lacuna/analysis/__init__.py tests/unit/analysis/test_structural_ntm.py
git commit -m "feat(analysis): add StructuralNeurotransmitterMapping (sntm) analysis"
```

---

### Task 10: FunctionalNeurotransmitterMapping Analysis

**Files:**
- Create: `src/lacuna/analysis/functional_neurotransmitter_mapping.py`
- Test: `tests/unit/analysis/test_functional_ntm.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/analysis/test_functional_ntm.py
"""Tests for FunctionalNeurotransmitterMapping analysis."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis.functional_neurotransmitter_mapping import (
    FunctionalNeurotransmitterMapping,
)
from lacuna.atlas.store import build_nt_atlas, save_atlas
from lacuna.core.data_types import ScalarMetric


def _create_pet_map(tmp_path, target, tracer, pub, shape=(91, 109, 91)):
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    rng = np.random.default_rng(hash((target, tracer, pub)) % 2**32)
    data = rng.random(shape).astype(np.float32) + 0.1
    fname = f"target-{target}_tracer-{tracer}_n-10_dx-hc_pub-{pub}_space-MNI152NLin6Asym_desc-proc.nii.gz"
    path = tmp_path / fname
    nib.save(nib.Nifti1Image(data, affine), str(path))


@pytest.fixture
def atlas_cache(tmp_path):
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _create_pet_map(pet_dir, "D1", "sch23390", "kaller2017")
    _create_pet_map(pet_dir, "5HT1a", "cumi101", "beliveau2017")
    atlas = build_nt_atlas(pet_dir)
    cache_dir = tmp_path / "cache"
    save_atlas(atlas, cache_dir)
    return cache_dir


class TestFunctionalNTMConstruction:
    def test_basic_construction(self, atlas_cache):
        fntm = FunctionalNeurotransmitterMapping(
            atlas_cache_dir=atlas_cache,
            connectome_name="GSP1000",
        )
        assert fntm._target_spec == "all"
        assert fntm.enriched is False

    def test_enriched_parameter(self, atlas_cache):
        fntm = FunctionalNeurotransmitterMapping(
            atlas_cache_dir=atlas_cache,
            connectome_name="GSP1000",
            enriched=True,
        )
        assert fntm.enriched is True

    def test_targets_parameter(self, atlas_cache):
        fntm = FunctionalNeurotransmitterMapping(
            atlas_cache_dir=atlas_cache,
            connectome_name="GSP1000",
            targets=["D1"],
        )
        assert fntm._target_spec == ["D1"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/analysis/test_functional_ntm.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement FunctionalNeurotransmitterMapping**

```python
# src/lacuna/analysis/functional_neurotransmitter_mapping.py
"""Functional Neurotransmitter Mapping (fntm).

Scores NT atlas values weighted by lesion functional connectivity.
Static mode: NT atlas × fLNM z-map.
REACT-enriched mode: global = temporal correlation with NT timeseries,
                     regional = REACT atlas × fLNM z-map.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from lacuna.analysis.base import BaseAnalysis
from lacuna.atlas.config import resolve_targets
from lacuna.atlas.scoring import score_functional_overlap, score_react_temporal
from lacuna.atlas.store import load_atlas
from lacuna.core.data_types import ParcelData, ScalarMetric, VoxelMap
from lacuna.core.subject_data import SubjectData

logger = logging.getLogger(__name__)


class FunctionalNeurotransmitterMapping(BaseAnalysis):
    """Functional neurotransmitter mapping: NT scores via functional connectivity.

    Computes functional connectivity of the lesion (internally, using the same
    method as FunctionalNetworkMapping), then scores the resulting z-map
    against the NT atlas.

    Parameters
    ----------
    atlas_cache_dir : Path
        Directory containing the prepared NT atlas.
    connectome_name : str
        Name of the functional connectome (e.g., "GSP1000").
    targets : str or list[str]
        Target selection. Default "all".
    enriched : bool
        If True, use REACT-enriched scoring.
    react_cache_dir : Path or None
        Directory with REACT outputs (required if enriched=True).
    parcel_atlases : list[str] or None
        Atlas names for regional scoring.
    method : str
        Lesion timeseries extraction method ("boes" or "pini").
    n_jobs : int
        Number of parallel jobs.
    verbose : bool
        Enable verbose logging.
    keep_intermediate : bool
        Keep intermediate results.
    """

    TARGET_SPACE = None  # Set from connectome
    TARGET_RESOLUTION = None
    batch_strategy = "sequential"

    def __init__(
        self,
        atlas_cache_dir: str | Path,
        connectome_name: str,
        targets: str | list[str] = "all",
        enriched: bool = False,
        react_cache_dir: str | Path | None = None,
        parcel_atlases: list[str] | None = None,
        method: str = "boes",
        n_jobs: int = 1,
        verbose: bool = False,
        keep_intermediate: bool = False,
    ):
        super().__init__(verbose=verbose, keep_intermediate=keep_intermediate)
        self.atlas_cache_dir = Path(atlas_cache_dir)
        self.connectome_name = connectome_name
        self._target_spec = targets
        self.enriched = enriched
        self.react_cache_dir = Path(react_cache_dir) if react_cache_dir else None
        self.parcel_atlases = parcel_atlases
        self.method = method
        self.n_jobs = n_jobs

    def _validate_inputs(self, mask_data: SubjectData) -> None:
        """Validate atlas, connectome, and REACT data if enriched."""
        atlas = load_atlas(self.atlas_cache_dir)
        self._atlas = atlas
        self._resolved_targets = resolve_targets(self._target_spec, atlas.targets)

        if self.enriched and self.react_cache_dir is None:
            raise ValueError(
                "REACT cache directory required for enriched mode. "
                "Run 'lacuna prepare react' first."
            )

    def _run_analysis(self, mask_data: SubjectData) -> dict[str, Any]:
        """Compute functional NT scores."""
        atlas = self._atlas.subset(self._resolved_targets)

        # Compute functional connectivity z-map internally
        z_map = self._compute_functional_connectivity(mask_data)

        if self.enriched:
            return self._run_enriched(mask_data, atlas, z_map)
        else:
            return self._run_static(atlas, z_map)

    def _run_static(self, atlas, z_map):
        """Static mode: NT atlas × fLNM z-map."""
        # Resample atlas to z-map space if needed
        z_shape = z_map.shape[:3]
        atlas_shape = atlas.get_map(atlas.targets[0]).shape[:3]
        if atlas_shape != z_shape:
            atlas = atlas.resample_to(z_map.affine, z_shape)

        scores = score_functional_overlap(atlas, z_map)

        results = {}
        for target, score in scores.items():
            results[target] = ScalarMetric(
                name=target, data=score, data_type="scalar",
                metadata={"analysis": "fntm", "mode": "static"},
            )
        return results

    def _run_enriched(self, mask_data, atlas, z_map):
        """REACT-enriched mode: temporal for global, spatial for regional."""
        results = {}

        # Global: temporal correlation with REACT stage 1 timeseries
        react_data = self._load_react_data()
        lesion_ts = self._extract_lesion_timeseries(mask_data)

        # Average REACT stage 1 NT timeseries across subjects
        all_stage1 = react_data["stage1_timeseries"]
        avg_stage1 = np.mean(all_stage1, axis=0)  # (n_timepoints, n_targets)

        # Build target-keyed timeseries dict
        nt_timeseries = {}
        react_targets = react_data["stage2_atlas"].targets
        for i, target in enumerate(react_targets):
            if target in self._resolved_targets:
                nt_timeseries[target] = avg_stage1[:, i]

        temporal_scores = score_react_temporal(nt_timeseries, lesion_ts)

        for target, score in temporal_scores.items():
            results[target] = ScalarMetric(
                name=target, data=score, data_type="scalar",
                metadata={"analysis": "fntm", "mode": "enriched"},
            )

        return results

    def _compute_functional_connectivity(self, mask_data):
        """Compute fLNM z-map using FunctionalNetworkMapping logic."""
        from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

        fnm = FunctionalNetworkMapping(
            connectome_name=self.connectome_name,
            method=self.method,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            compute_p_map=False,
            fdr_alpha=None,
            return_in_input_space=False,
        )
        result = fnm.run(mask_data)
        z_map_result = result.results["FunctionalNetworkMapping"]["zmap"]
        return z_map_result.data  # nib.Nifti1Image

    def _extract_lesion_timeseries(self, mask_data):
        """Extract mean lesion BOLD timeseries from connectome data.

        Reuses the extraction logic from FunctionalNetworkMapping.
        """
        from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

        fnm = FunctionalNetworkMapping(
            connectome_name=self.connectome_name,
            method=self.method,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        # Access the internal timeseries extraction
        # This would need to be refactored as a shared utility
        # For now, return a placeholder
        raise NotImplementedError(
            "Lesion timeseries extraction for REACT enriched mode "
            "requires refactoring FNM internals into shared utility."
        )

    def _load_react_data(self):
        """Load REACT stage 1 timeseries and stage 2 atlas from cache."""
        from lacuna.atlas.store import load_atlas

        stage2_atlas = load_atlas(self.react_cache_dir / "stage2_atlas")

        # Load stage 1 timeseries
        stage1_dir = self.react_cache_dir / "stage1_timeseries"
        stage1_list = []
        for ts_file in sorted(stage1_dir.glob("*.npy")):
            stage1_list.append(np.load(ts_file))

        return {
            "stage2_atlas": stage2_atlas,
            "stage1_timeseries": stage1_list,
        }

    def _get_parameters(self) -> dict:
        return {
            "atlas_cache_dir": str(self.atlas_cache_dir),
            "connectome_name": self.connectome_name,
            "targets": self._target_spec,
            "enriched": self.enriched,
            "method": self.method,
        }
```

- [ ] **Step 4: Update analysis/__init__.py**

Add import and `__all__` entry for `FunctionalNeurotransmitterMapping`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/analysis/test_functional_ntm.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna
git add src/lacuna/analysis/functional_neurotransmitter_mapping.py src/lacuna/analysis/__init__.py tests/unit/analysis/test_functional_ntm.py
git commit -m "feat(analysis): add FunctionalNeurotransmitterMapping (fntm) analysis"
```

---

### Task 11: CLI — `lacuna prepare` Subcommand

**Files:**
- Create: `src/lacuna/cli/prepare.py`
- Modify: `src/lacuna/cli/parser.py`
- Modify: `src/lacuna/cli/main.py`

- [ ] **Step 1: Create prepare.py module**

```python
# src/lacuna/cli/prepare.py
"""CLI implementation for the 'lacuna prepare' subcommand.

Handles precomputation of non-subject-specific data needed by analyses.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_prepare_lntm(args) -> None:
    """Prepare the NT atlas: average per target, z-score, cache."""
    from lacuna.atlas.store import build_nt_atlas, save_atlas
    from lacuna.atlas.config import parse_map_selection
    from lacuna.io.fetch import get_data_dir

    source_dir = Path(args.source_dir) if args.source_dir else get_data_dir() / "atlases" / "neurotransmitter" / "raw"
    cache_dir = Path(args.cache_dir) if args.cache_dir else get_data_dir() / "atlases" / "neurotransmitter" / "prepared"
    map_config = parse_map_selection(Path(args.map_config) if args.map_config else None)

    if not source_dir.exists():
        raise FileNotFoundError(
            f"PET atlas source directory not found: {source_dir}\n"
            f"Run 'lacuna fetch ntatlas' first to download the raw PET maps."
        )

    logger.info("Building NT atlas from %s", source_dir)
    atlas = build_nt_atlas(source_dir, map_config=map_config)
    save_atlas(atlas, cache_dir)
    logger.info("NT atlas saved to %s (%d targets)", cache_dir, len(atlas.targets))
    print(f"NT atlas prepared: {len(atlas.targets)} targets saved to {cache_dir}")


def run_prepare_sntm(args) -> None:
    """Precompute endpoint NT weights for all streamlines."""
    from lacuna.atlas.store import load_atlas
    from lacuna.io.fetch import get_data_dir

    atlas_dir = get_data_dir() / "atlases" / "neurotransmitter" / "prepared"
    if not (atlas_dir / "manifest.json").exists():
        logger.info("NT atlas not found, running prepare lntm first...")
        # Auto-trigger prepare lntm
        import types
        lntm_args = types.SimpleNamespace(source_dir=None, cache_dir=None, map_config=None)
        run_prepare_lntm(lntm_args)

    atlas = load_atlas(atlas_dir)
    logger.info("Loaded NT atlas with %d targets", len(atlas.targets))

    cache_dir = Path(args.cache_dir) if args.cache_dir else get_data_dir() / "sntm" / args.connectome_path.replace("/", "_")

    # Load tractogram and compute endpoint weights
    logger.info("Computing endpoint NT weights for tractogram...")
    _precompute_endpoint_weights(atlas, Path(args.connectome_path), cache_dir)
    print(f"Endpoint weights saved to {cache_dir}")


def _precompute_endpoint_weights(atlas, tractogram_path, cache_dir):
    """Sample NT atlas values at all streamline endpoints, cache as float16 matrix."""
    import nibabel as nib
    import numpy as np
    from lacuna.utils.mrtrix import run_mrtrix_command

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Get endpoints via tckresample
    endpoints_path = cache_dir / "endpoints.tck"
    run_mrtrix_command(
        f"tckresample {tractogram_path} {endpoints_path} -endpoints",
        verbose=True,
    )

    # Load endpoints
    endpoints_tractogram = nib.streamlines.load(str(endpoints_path))
    streamlines = endpoints_tractogram.streamlines
    n_streamlines = len(streamlines)

    # Convert to voxel coordinates
    ref_img = atlas.get_map(atlas.targets[0])
    inv_affine = np.linalg.inv(ref_img.affine)
    shape = ref_img.shape[:3]

    starts = np.zeros((n_streamlines, 3), dtype=np.int32)
    ends = np.zeros((n_streamlines, 3), dtype=np.int32)
    for i, sl in enumerate(streamlines):
        s = (inv_affine[:3, :3] @ sl[0] + inv_affine[:3, 3]).astype(np.int32)
        e = (inv_affine[:3, :3] @ sl[-1] + inv_affine[:3, 3]).astype(np.int32)
        starts[i] = np.clip(s, 0, np.array(shape) - 1)
        ends[i] = np.clip(e, 0, np.array(shape) - 1)

    # Sample NT values at endpoints
    n_targets = len(atlas.targets)
    weights = np.zeros((n_targets, n_streamlines), dtype=np.float16)
    for j, target in enumerate(atlas.targets):
        data = atlas.get_map(target).get_fdata()
        val_start = data[starts[:, 0], starts[:, 1], starts[:, 2]]
        val_end = data[ends[:, 0], ends[:, 1], ends[:, 2]]
        weights[j] = ((val_start + val_end) / 2.0).astype(np.float16)

    # Save
    np.save(cache_dir / "endpoint_weights.npy", weights)
    np.save(cache_dir / "endpoints_start.npy", starts)
    np.save(cache_dir / "endpoints_end.npy", ends)
    with open(cache_dir / "targets.txt", "w") as f:
        f.write("\n".join(atlas.targets))

    logger.info("Saved endpoint weights: shape %s", weights.shape)


def run_prepare_react(args) -> None:
    """Run REACT stage 1+2 on normative fMRI data."""
    from lacuna.atlas.react import compute_react_atlas
    from lacuna.atlas.store import load_atlas, save_atlas
    from lacuna.io.fetch import get_data_dir

    atlas_dir = get_data_dir() / "atlases" / "neurotransmitter" / "prepared"
    atlas = load_atlas(atlas_dir)

    cache_dir = Path(args.cache_dir) if args.cache_dir else get_data_dir() / "react"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load normative fMRI data
    logger.info("Loading normative fMRI connectome: %s", args.connectome_name)
    # This would use the existing connectome loading infrastructure
    # The exact implementation depends on how the connectome HDF5 is structured
    raise NotImplementedError(
        "REACT preparation requires connectome loading integration. "
        "Implement after connectome HDF5 structure is confirmed."
    )
```

- [ ] **Step 2: Add prepare subcommand to CLI parser**

In `src/lacuna/cli/parser.py`, add a `prepare` subparser under the main parser:

```python
# Add inside build_parser(), after the existing subcommands:
prepare_parser = subparsers.add_parser(
    "prepare",
    help="Precompute non-subject-specific data for analyses.",
)
prepare_subparsers = prepare_parser.add_subparsers(dest="prepare_target")

# prepare lntm
prepare_lntm = prepare_subparsers.add_parser("lntm", help="Prepare NT atlas")
prepare_lntm.add_argument("--source-dir", type=str, default=None, help="Directory with raw PET NIfTI maps")
prepare_lntm.add_argument("--cache-dir", type=str, default=None, help="Output cache directory")
prepare_lntm.add_argument("--map-config", type=str, default=None, help="YAML map selection config")

# prepare sntm
prepare_sntm = prepare_subparsers.add_parser("sntm", help="Precompute structural endpoint weights")
prepare_sntm.add_argument("--connectome-path", type=str, required=True, help="Path to tractogram")
prepare_sntm.add_argument("--cache-dir", type=str, default=None, help="Output cache directory")

# prepare react
prepare_react = prepare_subparsers.add_parser("react", help="Run REACT on normative data")
prepare_react.add_argument("--connectome-name", type=str, required=True, help="Normative fMRI connectome name")
prepare_react.add_argument("--cache-dir", type=str, default=None, help="Output cache directory")
```

- [ ] **Step 3: Add prepare dispatch to CLI main**

In `src/lacuna/cli/main.py`, add dispatch for the `prepare` command:

```python
# In the main dispatch logic (where "fetch", "run", etc. are routed):
elif args.command == "prepare":
    from lacuna.cli.prepare import run_prepare_lntm, run_prepare_sntm, run_prepare_react
    if args.prepare_target == "lntm":
        run_prepare_lntm(args)
    elif args.prepare_target == "sntm":
        run_prepare_sntm(args)
    elif args.prepare_target == "react":
        run_prepare_react(args)
    else:
        prepare_parser.print_help()
```

- [ ] **Step 4: Add lntm/sntm/fntm analysis aliases to run command**

In `src/lacuna/cli/main.py`, update the analysis alias mapping:

```python
ANALYSIS_ALIASES = {
    "rd": "LocalDamage",
    "ld": "LocalDamage",
    "fnm": "FunctionalNetworkMapping",
    "snm": "StructuralNetworkMapping",
    "afnm": "AcceleratedFunctionalNetworkMapping",
    "lntm": "LocalNeurotransmitterMapping",
    "sntm": "StructuralNeurotransmitterMapping",
    "fntm": "FunctionalNeurotransmitterMapping",
}
```

- [ ] **Step 5: Add run parsers for lntm/sntm/fntm**

In `src/lacuna/cli/parser.py`, add parser builders for the three NT analyses with shared options:

```python
def _add_ntm_common_args(parser):
    """Add arguments shared by all NTM analyses."""
    parser.add_argument("--targets", type=str, default="all", help="Target preset or comma-separated list")
    parser.add_argument("--enriched", action="store_true", help="Use REACT-enriched atlas")
    parser.add_argument("--parcel-atlases", nargs="+", help="Atlases for regional scoring")
    parser.add_argument("--custom-parcellation", nargs=4, action="append",
                        metavar=("NAME", "NIFTI", "LABELS", "SPACE"),
                        help="Custom parcellation (repeatable)")
```

- [ ] **Step 6: Commit**

```bash
cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna
git add src/lacuna/cli/prepare.py src/lacuna/cli/parser.py src/lacuna/cli/main.py
git commit -m "feat(cli): add lacuna prepare subcommand and lntm/sntm/fntm run support"
```

---

### Task 12: Integration Testing

**Files:**
- Create: `tests/integration/test_lntm_pipeline.py`

- [ ] **Step 1: Write integration test for the local NTM pipeline**

```python
# tests/integration/test_lntm_pipeline.py
"""Integration test: full lntm pipeline from raw PET maps to scores."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.atlas.store import build_nt_atlas, save_atlas
from lacuna.analysis.local_neurotransmitter_mapping import LocalNeurotransmitterMapping
from lacuna.core.data_types import ScalarMetric
from lacuna.core.subject_data import SubjectData


def _create_pet_map(tmp_path, target, tracer, pub, shape=(91, 109, 91)):
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    rng = np.random.default_rng(hash((target, tracer, pub)) % 2**32)
    data = rng.random(shape).astype(np.float32) + 0.1
    fname = f"target-{target}_tracer-{tracer}_n-10_dx-hc_pub-{pub}_space-MNI152NLin6Asym_desc-proc.nii.gz"
    path = tmp_path / fname
    nib.save(nib.Nifti1Image(data, affine), str(path))


class TestLNTMIntegration:
    def test_full_pipeline(self, tmp_path):
        """Test: raw PET maps → build atlas → save → load → score lesion."""
        # 1. Create fake PET maps
        pet_dir = tmp_path / "pet_raw"
        pet_dir.mkdir()
        _create_pet_map(pet_dir, "D1", "sch23390", "kaller2017")
        _create_pet_map(pet_dir, "5HT1a", "cumi101", "beliveau2017")
        _create_pet_map(pet_dir, "5HT1a", "way100635", "savli2012")

        # 2. Build and cache atlas
        atlas = build_nt_atlas(pet_dir)
        cache_dir = tmp_path / "cache"
        save_atlas(atlas, cache_dir)

        # 3. Create lesion subject
        affine = np.eye(4) * 2
        affine[3, 3] = 1
        mask_data = np.zeros((91, 109, 91), dtype=np.int8)
        mask_data[40:50, 50:60, 40:50] = 1
        subject = SubjectData(
            mask_img=nib.Nifti1Image(mask_data, affine),
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata={"subject_id": "sub-001"},
        )

        # 4. Run lntm
        lntm = LocalNeurotransmitterMapping(atlas_cache_dir=cache_dir)
        result = lntm.run(subject)

        # 5. Verify outputs
        lntm_results = result.results["LocalNeurotransmitterMapping"]
        assert "D1" in lntm_results
        assert "5HT1a" in lntm_results
        assert isinstance(lntm_results["D1"], ScalarMetric)
        assert np.isfinite(lntm_results["D1"].get_data())

        # 6. Verify provenance
        assert any(
            "LocalNeurotransmitterMapping" in str(p)
            for p in result.provenance
        )

    def test_target_subsetting(self, tmp_path):
        """Test that --targets filters output correctly."""
        pet_dir = tmp_path / "pet_raw"
        pet_dir.mkdir()
        _create_pet_map(pet_dir, "D1", "sch23390", "kaller2017")
        _create_pet_map(pet_dir, "5HT1a", "cumi101", "beliveau2017")

        atlas = build_nt_atlas(pet_dir)
        cache_dir = tmp_path / "cache"
        save_atlas(atlas, cache_dir)

        affine = np.eye(4) * 2
        affine[3, 3] = 1
        mask_data = np.zeros((91, 109, 91), dtype=np.int8)
        mask_data[45, 55, 45] = 1
        subject = SubjectData(
            mask_img=nib.Nifti1Image(mask_data, affine),
            space="MNI152NLin6Asym",
            resolution=2.0,
        )

        lntm = LocalNeurotransmitterMapping(
            atlas_cache_dir=cache_dir,
            targets=["D1"],
        )
        result = lntm.run(subject)
        lntm_results = result.results["LocalNeurotransmitterMapping"]
        assert "D1" in lntm_results
        assert "5HT1a" not in lntm_results

    def test_pipeline_chaining(self, tmp_path):
        """Test lntm works in a Pipeline with other analyses."""
        from lacuna.core.pipeline import Pipeline

        pet_dir = tmp_path / "pet_raw"
        pet_dir.mkdir()
        _create_pet_map(pet_dir, "D1", "sch23390", "kaller2017")

        atlas = build_nt_atlas(pet_dir)
        cache_dir = tmp_path / "cache"
        save_atlas(atlas, cache_dir)

        affine = np.eye(4) * 2
        affine[3, 3] = 1
        mask_data = np.zeros((91, 109, 91), dtype=np.int8)
        mask_data[45:48, 55:58, 45:48] = 1
        subject = SubjectData(
            mask_img=nib.Nifti1Image(mask_data, affine),
            space="MNI152NLin6Asym",
            resolution=2.0,
        )

        pipe = Pipeline(name="test_ntm")
        pipe.add(LocalNeurotransmitterMapping(atlas_cache_dir=cache_dir))
        result = pipe.run(subject)

        assert "LocalNeurotransmitterMapping" in result.results
```

- [ ] **Step 2: Run integration tests**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/integration/test_lntm_pipeline.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna
git add tests/integration/test_lntm_pipeline.py
git commit -m "test: add integration tests for lntm pipeline"
```

---

### Task 13: Error Handling and Target Conflict Detection

**Files:**
- Modify: `src/lacuna/atlas/config.py`
- Test: `tests/unit/atlas/test_config.py` (add tests)

- [ ] **Step 1: Add error handling tests**

Add to `tests/unit/atlas/test_config.py`:

```python
class TestTargetConflictDetection:
    def test_run_target_excluded_at_prepare_time(self):
        """Requesting a target at run time that was excluded during prepare."""
        available = ["D1", "5HT1a"]  # DAT was excluded
        with pytest.raises(ValueError, match="DAT.*not available"):
            resolve_targets(["D1", "DAT"], available)

    def test_helpful_error_message_lists_available(self):
        available = ["D1", "5HT1a"]
        with pytest.raises(ValueError, match="Available targets"):
            resolve_targets(["GABA"], available)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/unit/atlas/test_config.py -v`
Expected: All PASS (error handling already implemented in Task 2)

- [ ] **Step 3: Commit**

```bash
cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna
git add tests/unit/atlas/test_config.py
git commit -m "test: add target conflict detection tests"
```

---

### Task 14: Run Full Test Suite and Fix Issues

- [ ] **Step 1: Run the complete test suite**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -m pytest tests/ -v --tb=short -q 2>&1 | tail -30`
Expected: All tests pass. If failures, diagnose and fix.

- [ ] **Step 2: Run import check**

Run: `cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna && python -c "from lacuna.atlas import VoxelAtlas, build_nt_atlas, score_focal; from lacuna.analysis import LocalNeurotransmitterMapping, LocalDamage; print('All imports OK')"`
Expected: "All imports OK"

- [ ] **Step 3: Commit any fixes**

```bash
cd /home/marvin/mount/hdd8tb/CSI_MVCI/lacuna
git add -A
git commit -m "fix: resolve test suite issues from NTM integration"
```
