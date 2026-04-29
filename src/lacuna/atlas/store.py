"""Atlas lifecycle: build, save, load, and cache voxel atlases.

Builds a neurotransmitter VoxelAtlas from the curated representative
PET maps fetched via `lacuna fetch ntatlas`. One map per target
(no averaging across publications); maps are z-scored and grouped
by neurotransmitter system in metadata.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

from lacuna.atlas.types import VoxelAtlas
from lacuna.data.ntatlas import load_collection, parse_target

logger = logging.getLogger(__name__)


def build_nt_atlas(source_dir: Path) -> VoxelAtlas:
    """Build a neurotransmitter VoxelAtlas from representative PET maps.

    Loads the bundled NiSpace-data collection (one recommended map per
    target) and z-scores each map (excluding zero voxels). System
    grouping (Dopamine: [D1, D23, ...]) and target → map_id traceability
    are preserved in the atlas metadata.

    Parameters
    ----------
    source_dir : Path
        Directory containing the NIfTI files downloaded by
        ``lacuna fetch ntatlas``.

    Returns
    -------
    VoxelAtlas
        Z-scored atlas keyed by target (e.g. "D1", "5HT1a").
    """
    source_dir = Path(source_dir)
    coll = load_collection()

    maps: dict[str, nib.Nifti1Image] = {}
    target_to_map_id: dict[str, str] = {}
    reference_affine: np.ndarray | None = None

    for map_id in (mid for ids in coll["systems"].values() for mid in ids):
        path = source_dir / f"{map_id}_space-MNI152NLin6Asym_desc-proc.nii.gz"
        if not path.exists():
            logger.warning("Missing PET map, skipping: %s", path.name)
            continue

        img = nib.load(str(path))
        if reference_affine is None:
            reference_affine = img.affine

        z_scored = _zscore_excluding_zeros(img.get_fdata())
        target = parse_target(map_id)
        maps[target] = nib.Nifti1Image(z_scored.astype(np.float32), reference_affine)
        target_to_map_id[target] = map_id

    if not maps:
        raise FileNotFoundError(
            f"No PET maps found in {source_dir}.\n"
            f"Run 'lacuna fetch ntatlas --output-dir {source_dir}' first."
        )

    voxel_sizes = np.sqrt(np.sum(reference_affine[:3, :3] ** 2, axis=0))
    resolution = float(np.mean(voxel_sizes))

    loaded = set(maps)
    systems_by_target = {
        system: [t for t in (parse_target(mid) for mid in mids) if t in loaded]
        for system, mids in coll["systems"].items()
    }
    systems_by_target = {s: ts for s, ts in systems_by_target.items() if ts}

    return VoxelAtlas(
        maps=maps,
        space="MNI152NLin6Asym",
        resolution=resolution,
        domain="neurotransmitter",
        metadata={
            "nispace_commit": coll["nispace_commit"],
            "collection": coll["collection_name"],
            "systems": systems_by_target,
            "target_to_map_id": target_to_map_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


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
            f"Run 'lacuna fetch ntatlas --output-dir {cache_dir}' to create one."
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
