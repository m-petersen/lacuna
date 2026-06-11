"""Brain mask loading.

Downloads (with SHA-256 verification) and caches a binary brain mask for a given
coordinate space and resolution via ``pooch``, then validates that the file is a
binary mask on Lacuna's canonical grid for that space/resolution before use.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lacuna.assets.masks.registry import BRAIN_MASK_REGISTRY, mask_name
from lacuna.spatial.transform import _canonicalize_space_variant

logger = logging.getLogger(__name__)


def _validate_mask(path: Path, space: str, resolution: float) -> None:
    """Validate a downloaded mask: canonical grid, 3D, binary.

    Raises
    ------
    ValueError
        If the file is not a binary 3D mask on the expected space/resolution grid.
    """
    import nibabel as nib
    import numpy as np

    from lacuna.core.spaces import REFERENCE_AFFINES, REFERENCE_SHAPES

    img = nib.load(str(path))
    key = (space, float(resolution))
    ref_shape = REFERENCE_SHAPES.get(key)
    ref_affine = REFERENCE_AFFINES.get(key)

    if img.ndim != 3:
        raise ValueError(f"Brain mask {path} is {img.ndim}D; expected a 3D mask.")
    if ref_shape is not None and tuple(img.shape) != tuple(ref_shape):
        raise ValueError(
            f"Brain mask {path} has shape {tuple(img.shape)}, expected {tuple(ref_shape)} "
            f"for {space}@{resolution:g}mm."
        )
    if ref_affine is not None and not np.allclose(img.affine, ref_affine, atol=1e-3):
        raise ValueError(
            f"Brain mask {path} affine does not match the canonical {space}@{resolution:g}mm grid."
        )
    values = np.unique(np.asanyarray(img.dataobj))
    if not set(values.tolist()) <= {0, 1}:
        raise ValueError(f"Brain mask {path} is not binary (values: {values[:8]} ...).")


def load_brain_mask(space: str, resolution: float, *, validate: bool = True) -> Path:
    """Load a binary brain mask for ``space``/``resolution``, caching on first use.

    Anatomically identical spaces (e.g. MNI152NLin2009[abc]Asym) are normalized to
    their canonical form before lookup.

    Parameters
    ----------
    space : str
        Coordinate space identifier (e.g. "MNI152NLin6Asym").
    resolution : float
        Voxel resolution in mm (1.0 or 2.0).
    validate : bool, default True
        Verify the downloaded file is a binary 3D mask on the canonical grid.

    Returns
    -------
    Path
        Path to the locally cached brain mask (.nii.gz).

    Raises
    ------
    KeyError
        If no mask is registered for the space/resolution.
    FileNotFoundError
        If the mask has no URL or the download fails.
    ValueError
        If validation fails.
    """
    canonical_space = _canonicalize_space_variant(space)
    name = mask_name(canonical_space, float(resolution))
    metadata = BRAIN_MASK_REGISTRY.get(name)  # KeyError if unknown

    if not metadata.url:
        raise FileNotFoundError(f"Brain mask '{name}' has no download URL registered.")

    try:
        import pooch
    except ImportError as e:  # pragma: no cover - pooch is a hard dependency
        raise ImportError(
            "pooch is required to download brain masks. Install with: pip install pooch"
        ) from e

    from lacuna.utils.cache import get_cache_dir

    cache_dir = Path(get_cache_dir()) / "masks"
    logger.debug(f"Loading brain mask {name} from {metadata.url}")
    try:
        path = pooch.retrieve(
            url=metadata.url,
            known_hash=f"sha256:{metadata.sha256}" if metadata.sha256 else None,
            fname=f"{name}_desc-brain_mask.nii.gz",
            path=cache_dir,
            progressbar=True,
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to download brain mask '{name}' from {metadata.url}: {e}"
        ) from e

    path = Path(path)
    if validate:
        _validate_mask(path, canonical_space, float(resolution))
    return path


__all__ = ["load_brain_mask"]
