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
        Map selection configuration. Supports two formats:

        1. Flat filter dict (simple):
           - {"exclude": ["DAT", "D1"]} — exclude listed targets
           - {"publications": ["beliveau2017"]} — keep only listed pubs

        2. Per-target dict (advanced, from YAML config):
           - {"DAT": "exclude", "5HT1a": ["beliveau2017"]}
           - Values: "exclude", "all", or list of publication keys.
           - Targets not in config default to "all".

    Returns
    -------
    VoxelAtlas
        Z-scored, averaged atlas with one map per target.
    """
    source_dir = Path(source_dir)

    # Discover and group NIfTI files by target
    file_groups: dict[str, list[Path]] = defaultdict(list)
    nifti_files = sorted(source_dir.glob("*.nii.gz"))

    if not nifti_files:
        raise ValueError(f"No .nii.gz files found in {source_dir}")

    for nifti_path in nifti_files:
        try:
            target = parse_target_from_filename(nifti_path.name)
        except ValueError:
            logger.warning("Skipping file with no target: %s", nifti_path.name)
            continue
        file_groups[target].append(nifti_path)

    if not file_groups:
        raise ValueError(f"No PET NIfTI files with valid target found in {source_dir}")

    # Apply map selection config
    if map_config:
        file_groups = _apply_map_config(file_groups, map_config)

    if not file_groups:
        raise ValueError("No targets remaining after applying map_config filters.")

    # Build averaged, z-scored maps
    maps: dict[str, nib.Nifti1Image] = {}
    reference_affine = None

    for target in sorted(file_groups):
        paths = file_groups[target]
        logger.info("Building %s from %d maps", target, len(paths))

        loaded = [nib.load(str(p)) for p in paths]

        if reference_affine is None:
            reference_affine = loaded[0].affine

        # Average excluding zeros
        averaged = _average_excluding_zeros([img.get_fdata() for img in loaded])

        # Z-score (excluding zeros)
        z_scored = _zscore_excluding_zeros(averaged)

        maps[target] = nib.Nifti1Image(z_scored.astype(np.float32), reference_affine)

    # Detect resolution from affine
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


def _apply_map_config(
    file_groups: dict[str, list[Path]],
    map_config: dict[str, Any],
) -> dict[str, list[Path]]:
    """Apply map_config filtering to file groups.

    Supports both flat-filter format and per-target format.
    """
    # Detect format: flat filter uses "exclude" or "publications" as top-level keys
    if "exclude" in map_config or "publications" in map_config:
        return _apply_flat_config(file_groups, map_config)
    return _apply_per_target_config(file_groups, map_config)


def _apply_flat_config(
    file_groups: dict[str, list[Path]],
    map_config: dict[str, Any],
) -> dict[str, list[Path]]:
    """Apply flat-format config: {exclude: [...], publications: [...]}."""
    result = dict(file_groups)

    # Exclude targets
    exclude_targets = set(map_config.get("exclude", []))
    if exclude_targets:
        result = {t: p for t, p in result.items() if t not in exclude_targets}

    # Filter by publications
    publications = map_config.get("publications")
    if publications:
        pubs_wanted = set(publications)
        filtered = {}
        for target, paths in result.items():
            selected = [
                p for p in paths
                if parse_publication_from_filename(p.name) in pubs_wanted
            ]
            if selected:
                filtered[target] = selected
        result = filtered

    return result


def _apply_per_target_config(
    file_groups: dict[str, list[Path]],
    map_config: dict[str, Any],
) -> dict[str, list[Path]]:
    """Apply per-target config: {target: "exclude"|"all"|[pub_list]}."""
    result: dict[str, list[Path]] = {}
    for target, paths in file_groups.items():
        selection = map_config.get(target, "all")
        if selection == "exclude":
            logger.info("Excluding target: %s", target)
            continue
        if selection == "all":
            result[target] = paths
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
            result[target] = selected
    return result


def _average_excluding_zeros(arrays: list[np.ndarray]) -> np.ndarray:
    """Average multiple arrays, excluding zeros from the mean.

    Zeros indicate outside-coverage voxels (e.g., cortical-only tracers).
    Voxels where ALL arrays are zero remain zero.
    """
    if len(arrays) == 1:
        return arrays[0].copy()

    stacked = np.stack(arrays, axis=0)
    nonzero_mask = stacked != 0
    count = nonzero_mask.sum(axis=0)
    safe_count = np.maximum(count, 1)
    total = np.where(nonzero_mask, stacked, 0).sum(axis=0)
    averaged = total / safe_count
    averaged[count == 0] = 0.0
    return averaged


def _zscore_excluding_zeros(data: np.ndarray) -> np.ndarray:
    """Z-score an array, computing stats only on nonzero voxels.

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

    Stores each map as a NIfTI file and metadata as JSON manifest.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    maps_dir = cache_dir / "maps"
    maps_dir.mkdir(exist_ok=True)
    for target in atlas.targets:
        nib.save(atlas.get_map(target), str(maps_dir / f"{target}.nii.gz"))

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
