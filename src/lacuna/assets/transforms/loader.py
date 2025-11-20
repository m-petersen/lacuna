"""Transform loading with TemplateFlow integration.

This module provides functions to load spatial transforms between template spaces,
automatically downloading from TemplateFlow as needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lacuna.assets.transforms.registry import TRANSFORM_REGISTRY
from lacuna.spatial.transform import _canonicalize_space_variant

logger = logging.getLogger(__name__)


def load_transform(name: str) -> Path:
    """Load a spatial transform by name.

    Downloads from TemplateFlow on first use and caches locally.
    Tries both forward and reverse directions since TemplateFlow may
    only have the transform stored in one direction.

    Supports space equivalence: anatomically identical spaces like
    MNI152NLin2009[abc]Asym are automatically normalized to their
    canonical form (cAsym).

    Parameters
    ----------
    name : str
        Transform name from registry (e.g., "MNI152NLin6Asym_to_MNI152NLin2009cAsym")

    Returns
    -------
    Path
        Path to transform .h5 file

    Raises
    ------
    KeyError
        If transform not found in registry
    FileNotFoundError
        If transform download fails

    Examples
    --------
    >>> from lacuna.assets.transforms import load_transform
    >>>
    >>> # Load transform
    >>> transform_path = load_transform("MNI152NLin6Asym_to_MNI152NLin2009cAsym")
    >>> print(transform_path.exists())
    True
    """
    # Normalize the requested transform name to handle space aliases
    # e.g., "MNI152NLin2009aAsym_to_X" -> "MNI152NLin2009cAsym_to_X"
    parts = name.split("_to_")
    if len(parts) == 2:
        from_space, to_space = parts
        from_space_normalized = _canonicalize_space_variant(from_space)
        to_space_normalized = _canonicalize_space_variant(to_space)
        normalized_name = f"{from_space_normalized}_to_{to_space_normalized}"
    else:
        normalized_name = name

    # Get metadata from registry using normalized name
    metadata = TRANSFORM_REGISTRY.get(normalized_name)

    # Normalize spaces from metadata to handle equivalence
    from_space_normalized = _canonicalize_space_variant(metadata.from_space)
    to_space_normalized = _canonicalize_space_variant(metadata.to_space)

    # Log if normalization occurred
    if from_space_normalized != metadata.from_space or to_space_normalized != metadata.to_space:
        logger.info(
            f"Using space equivalence: {metadata.from_space} → {from_space_normalized}, "
            f"{metadata.to_space} → {to_space_normalized} (anatomically identical spaces)"
        )

    try:
        import templateflow.api as tflow
    except ImportError as e:
        raise ImportError(
            "TemplateFlow is required for transform loading. "
            "Install with: pip install templateflow"
        ) from e

    import time

    def _wait_for_file(path: Path, timeout: float = 10.0) -> bool:
        """Wait for file to exist with exponential backoff."""
        start_time = time.time()
        delay = 0.1  # Start with 100ms

        while time.time() - start_time < timeout:
            if path.exists() and path.stat().st_size > 0:
                return True
            time.sleep(delay)
            delay = min(delay * 1.5, 1.0)  # Exponential backoff, max 1s

        return False

    # TemplateFlow naming convention:
    # tpl-{target}_from-{source}_mode-image_xfm.h5
    # The transform is stored under the target template

    logger.info(
        f"Loading transform: {from_space_normalized} → {to_space_normalized} "
        f"(original request: {name})"
    )

    # Pre-create cache directories to avoid TemplateFlow's unlink bug
    # TemplateFlow tries to delete files before checking if they exist
    cache_dir = Path.home() / ".cache" / "templateflow"

    for space in [metadata.to_space, metadata.from_space]:
        space_dir = cache_dir / f"tpl-{space}"
        space_dir.mkdir(parents=True, exist_ok=True)

    # Try forward direction (source -> target)
    transform_path = None
    forward_error = None

    logger.debug(
        f"Querying TemplateFlow for forward transform: {from_space_normalized} → {to_space_normalized}"
    )

    try:
        transform_path = tflow.get(
            to_space_normalized,
            **{"from": from_space_normalized},
            mode="image",
            suffix="xfm",
            extension=".h5",
        )

        if transform_path is not None and transform_path:
            # TemplateFlow can return a list or a single path
            if isinstance(transform_path, list):
                if transform_path:  # Non-empty list
                    transform_path = transform_path[0]
                else:
                    transform_path = None

            if transform_path:
                path = Path(transform_path)
                if _wait_for_file(path):
                    # Verify file integrity
                    file_size = path.stat().st_size
                    if file_size < 1024:  # Suspiciously small (< 1KB)
                        logger.warning(
                            f"Transform file seems corrupted (size: {file_size} bytes): {path}. "
                            "Removing and will re-download."
                        )
                        path.unlink()
                        # Retry download
                        transform_path = tflow.get(
                            to_space_normalized,
                            **{"from": from_space_normalized},
                            mode="image",
                            suffix="xfm",
                            extension=".h5",
                        )
                        if isinstance(transform_path, list) and transform_path:
                            transform_path = transform_path[0]
                        path = Path(transform_path) if transform_path else None
                        if not path or not path.exists():
                            raise FileNotFoundError(
                                "Failed to download valid transform file after retry"
                            )
                        file_size = path.stat().st_size

                    logger.info(
                        f"✓ Transform loaded: {path.name} " f"({file_size / (1024**2):.1f} MB)"
                    )
                    return path

    except Exception as e:
        forward_error = e
        logger.debug(f"Forward transform query failed: {e}")

    # Try reverse direction (target -> source)
    reverse_error = None

    logger.debug(
        f"Querying TemplateFlow for reverse transform: {to_space_normalized} → {from_space_normalized}"
    )

    try:
        transform_path = tflow.get(
            from_space_normalized,
            **{"from": to_space_normalized},
            mode="image",
            suffix="xfm",
            extension=".h5",
        )

        if transform_path is not None and transform_path:
            # TemplateFlow can return a list or a single path
            if isinstance(transform_path, list):
                if transform_path:  # Non-empty list
                    transform_path = transform_path[0]
                else:
                    transform_path = None

            if transform_path:
                path = Path(transform_path)
                if _wait_for_file(path):
                    # Verify file integrity
                    file_size = path.stat().st_size
                    if file_size < 1024:  # Suspiciously small (< 1KB)
                        logger.warning(
                            f"Transform file seems corrupted (size: {file_size} bytes): {path}. "
                            "Removing and will re-download."
                        )
                        path.unlink()
                        # Retry download
                        transform_path = tflow.get(
                            from_space_normalized,
                            **{"from": to_space_normalized},
                            mode="image",
                            suffix="xfm",
                            extension=".h5",
                        )
                        if isinstance(transform_path, list) and transform_path:
                            transform_path = transform_path[0]
                        path = Path(transform_path) if transform_path else None
                        if not path or not path.exists():
                            raise FileNotFoundError(
                                "Failed to download valid transform file after retry"
                            )
                        file_size = path.stat().st_size

                    logger.info(
                        f"✓ Transform loaded (reverse): {path.name} "
                        f"({file_size / (1024**2):.1f} MB)"
                    )
                    return path

            if transform_path:
                path = Path(transform_path)
                if _wait_for_file(path):
                    return path

    except Exception as e:
        reverse_error = e

    # If we get here, transform wasn't found in either direction
    error_details = []
    if forward_error:
        error_details.append(f"Forward: {forward_error}")
    if reverse_error:
        error_details.append(f"Reverse: {reverse_error}")

    error_msg = (
        f"Transform {normalized_name} not found in TemplateFlow "
        f"({from_space_normalized} ↔ {to_space_normalized}). "
    )

    if error_details:
        error_msg += f" Errors: {'; '.join(error_details)}"
    else:
        error_msg += "The transform file may not be available or download may have failed."

    raise FileNotFoundError(error_msg)


def is_transform_cached(name: str) -> bool:
    """Check if transform is already cached locally.

    Parameters
    ----------
    name : str
        Transform name from registry

    Returns
    -------
    bool
        True if transform is cached, False otherwise

    Examples
    --------
    >>> from lacuna.assets.transforms import is_transform_cached
    >>> is_transform_cached("MNI152NLin6Asym_to_MNI152NLin2009cAsym")
    True
    """
    try:
        transform_path = load_transform(name)
        return transform_path.exists()
    except (FileNotFoundError, KeyError):
        return False


__all__ = [
    "load_transform",
    "is_transform_cached",
]
