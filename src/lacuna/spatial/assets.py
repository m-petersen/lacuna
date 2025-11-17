"""Asset management for transforms and templates via TemplateFlow.

This module provides the DataAssetManager for retrieving spatial data assets
(templates, transforms) with caching and TemplateFlow integration.

Note: Atlas management is handled separately in the lacuna.atlas module.
"""

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from lacuna.core.exceptions import (
    TransformDownloadError,
    TransformNotAvailableError,
)

if TYPE_CHECKING:
    from templateflow import api as tfapi
else:
    # Lazy import to avoid ResourceWarning during test collection
    tfapi = None

logger = logging.getLogger(__name__)


# Space equivalence groups - spaces that are anatomically identical
# MNI152NLin2009aAsym, bAsym, and cAsym are anatomically identical
# They only differ in the preprocessing pipeline used to create them
EQUIVALENT_SPACES = {
    "MNI152NLin2009aAsym": "MNI152NLin2009cAsym",
    "MNI152NLin2009bAsym": "MNI152NLin2009cAsym",
    "MNI152NLin2009cAsym": "MNI152NLin2009cAsym",  # Identity for completeness
}


def _canonicalize_space_variant(space_id: str) -> str:
    """Canonicalize MNI space variant identifiers to canonical forms.
    
    For anatomically identical spaces (e.g., MNI152NLin2009[abc]Asym),
    returns the canonical representative (MNI152NLin2009cAsym).
    
    Args:
        space_id: Space identifier to canonicalize
        
    Returns:
        Canonical space identifier
    """
    return EQUIVALENT_SPACES.get(space_id, space_id)


def _get_tfapi():
    """Lazy import of templateflow API to avoid ResourceWarning during test collection."""
    global tfapi
    if tfapi is None:
        from templateflow import api as _tfapi
        tfapi = _tfapi
    return tfapi


# Default cache directory for downloaded assets
DEFAULT_ASSET_CACHE_DIR = Path.home() / ".cache" / "lacuna" / "assets"

# Path to bundled templates in package
BUNDLED_TEMPLATES_DIR = Path(__file__).parent.parent / "data" / "templates"

# Supported spaces and their resolutions
SUPPORTED_TEMPLATE_CONFIGS = {
    "MNI152NLin6Asym": [1, 2],
    "MNI152NLin2009cAsym": [1, 2],
    "MNI152NLin2009bAsym": [0.5],
}


class DataAssetManager:
    """Manager for spatial data assets (templates, transforms).
    
    .. deprecated:: 0.2.0
        `DataAssetManager` is deprecated and will be removed in version 0.3.0.
        Use the modular asset system instead:
        
        - For templates: `from lacuna.assets import load_template`
        - For transforms: `from lacuna.assets import load_transform`
        
        Example migration:
        
        Old::
        
            from lacuna.spatial.assets import DataAssetManager
            manager = DataAssetManager()
            template = manager.get_template("MNI152NLin2009cAsym", resolution=1)
            transform = manager.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")
        
        New::
        
            from lacuna.assets import load_template, load_transform
            template = load_template("MNI152NLin2009cAsym_res-1")
            transform = load_transform("MNI152NLin6Asym_to_MNI152NLin2009cAsym")

    Handles retrieval and caching of:
    - MNI templates (from bundled files or TemplateFlow)
    - Spatial transforms (from TemplateFlow)

    Note: Atlas management is handled by lacuna.atlas module.

    Attributes:
        cache_dir: Directory for caching downloaded assets
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialize asset manager.

        Args:
            cache_dir: Directory for asset cache (default: ~/.cache/lacuna/assets)
        """
        import warnings
        warnings.warn(
            "DataAssetManager is deprecated and will be removed in version 0.3.0. "
            "Use lacuna.assets.load_template() and lacuna.assets.load_transform() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        
        self.cache_dir = cache_dir if cache_dir is not None else DEFAULT_ASSET_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache for retrieved assets
        self._transform_cache: dict[tuple[str, str], Any] = {}
        self._template_cache: dict[tuple[str, int | float], Path] = {}

    def get_transform(self, source_space: str, target_space: str) -> Path | None:
        """Retrieve transform between coordinate spaces.

        Args:
            source_space: Source coordinate space identifier
            target_space: Target coordinate space identifier

        Returns:
            Path to transform file, or None if not available

        Raises:
            TransformNotAvailableError: If transform pair is not supported
            TransformDownloadError: If download fails

        Examples:
            >>> manager = DataAssetManager()
            >>> transform = manager.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")
        """
        # Normalize space identifiers to handle equivalent spaces
        source_normalized = _canonicalize_space_variant(source_space)
        target_normalized = _canonicalize_space_variant(target_space)
        
        # Log if space normalization occurred
        if source_space != source_normalized or target_space != target_normalized:
            logger.info(
                f"Using space equivalence: {source_space} → {source_normalized}, "
                f"{target_space} → {target_normalized} (anatomically identical spaces)"
            )
        
        # Check if transform is supported (using normalized spaces)
        if not self._is_transform_supported(source_normalized, target_normalized):
            # Get list of supported transform pairs
            supported_spaces = list(SUPPORTED_TEMPLATE_CONFIGS.keys())
            supported_pairs = [
                f"{s1} <-> {s2}"
                for i, s1 in enumerate(supported_spaces)
                for s2 in supported_spaces[i + 1 :]
            ]
            raise TransformNotAvailableError(source_space, target_space, supported_pairs)

        # Check cache (bidirectional, using normalized spaces)
        cache_key = tuple(sorted([source_normalized, target_normalized]))
        if cache_key in self._transform_cache:
            logger.debug(f"Using cached transform: {source_space} <-> {target_space}")
            return self._transform_cache[cache_key]

        # Try to download from TemplateFlow (using normalized spaces)
        try:
            transform_path = self._download_transform_from_templateflow(source_normalized, target_normalized)

            if transform_path is not None:
                # Cache the result
                self._transform_cache[cache_key] = transform_path
                logger.info(f"Downloaded and cached transform: {source_space} <-> {target_space} (normalized to {source_normalized} <-> {target_normalized})")
                return transform_path

            return None

        except ImportError as e:
            logger.error(f"TemplateFlow not available: {e}")
            raise TransformDownloadError(
                source_space,
                target_space,
                "TemplateFlow package not installed. Install with: pip install templateflow",
            ) from e

        except Exception as e:
            logger.error(f"Failed to download transform: {e}")
            raise TransformDownloadError(source_space, target_space, f"Download failed: {e}") from e

    def _download_transform_from_templateflow(
        self, source_space: str, target_space: str
    ) -> Path | None:
        """Download transform from TemplateFlow.

        Args:
            source_space: Source coordinate space
            target_space: Target coordinate space

        Returns:
            Path to downloaded transform file

        Raises:
            Exception: If download fails
        """
        import time
        tfapi = _get_tfapi()
        
        def _wait_for_file(path: Path, timeout: float = 10.0) -> bool:
            """Wait for file to exist with exponential backoff.
            
            Args:
                path: Path to check
                timeout: Maximum time to wait in seconds
                
            Returns:
                True if file exists, False if timeout reached
            """
            start_time = time.time()
            delay = 0.1  # Start with 100ms
            
            while time.time() - start_time < timeout:
                if path.exists() and path.stat().st_size > 0:
                    # File exists and has content
                    return True
                time.sleep(delay)
                delay = min(delay * 1.5, 1.0)  # Exponential backoff, max 1s
            
            return False
        
        # TemplateFlow naming convention:
        # tpl-{target}_from-{source}_mode-image_xfm.h5
        # The transform is stored under the target template
        try:
            # Try forward direction (source -> target)
            logger.debug(f"Attempting to download transform: {source_space} -> {target_space}")
            transform_path = tfapi.get(
                template=target_space,
                **{"from": source_space},
                mode="image",
                suffix="xfm",
                extension=".h5",
            )

            if transform_path is not None:
                path = Path(transform_path)
                # Wait for file to be fully written
                if _wait_for_file(path):
                    logger.info(f"Transform downloaded: {path}")
                    return path
                else:
                    logger.warning(f"Transform download timeout: {path}")

        except Exception as e:
            logger.debug(f"Forward direction failed: {e}")

            # Try reverse direction (target -> source)
            try:
                logger.debug(f"Attempting reverse: {target_space} -> {source_space}")
                transform_path = tfapi.get(
                    template=source_space,
                    **{"from": target_space},
                    mode="image",
                    suffix="xfm",
                    extension=".h5",
                )

                if transform_path is not None:
                    path = Path(transform_path)
                    # Wait for file to be fully written
                    if _wait_for_file(path):
                        logger.info(f"Transform downloaded (reverse): {path}")
                        return path
                    else:
                        logger.warning(f"Transform download timeout (reverse): {path}")

            except Exception as e2:
                logger.debug(f"Reverse direction also failed: {e2}")
                raise Exception(
                    f"Could not download transform in either direction: {e}, {e2}"
                ) from e2

        return None

    def _is_transform_supported(self, source_space: str, target_space: str) -> bool:
        """Check if transform between spaces is supported.

        Args:
            source_space: Source space
            target_space: Target space

        Returns:
            True if transform is supported
        """
        # Native space cannot transform
        if source_space == "native" or target_space == "native":
            return False

        # Both must be MNI templates
        supported_spaces = set(SUPPORTED_TEMPLATE_CONFIGS.keys())
        return source_space in supported_spaces and target_space in supported_spaces

    def get_template(self, space: str, resolution: int | float) -> Path | None:
        """Retrieve template image for coordinate space.

        Prefers bundled templates over downloads.

        Args:
            space: Coordinate space identifier
            resolution: Template resolution in mm (1, 2, etc.)

        Returns:
            Path to template file

        Raises:
            ValueError: If space or resolution is invalid

        Examples:
            >>> manager = DataAssetManager()
            >>> template = manager.get_template("MNI152NLin6Asym", resolution=2)
        """
        # Normalize space identifier to handle equivalent spaces
        space_normalized = _canonicalize_space_variant(space)
        
        # Log if space normalization occurred
        if space != space_normalized:
            logger.info(
                f"Using space equivalence: {space} → {space_normalized} "
                f"(anatomically identical spaces)"
            )
        
        # Validate space (using normalized identifier)
        if space_normalized not in SUPPORTED_TEMPLATE_CONFIGS:
            raise ValueError(
                f"Unsupported space: {space}. "
                f"Supported spaces: {list(SUPPORTED_TEMPLATE_CONFIGS.keys())}"
            )

        # Validate resolution (using normalized space)
        if resolution not in SUPPORTED_TEMPLATE_CONFIGS[space_normalized]:
            raise ValueError(
                f"Invalid resolution {resolution}mm for {space}. "
                f"Supported resolutions: {SUPPORTED_TEMPLATE_CONFIGS[space_normalized]}"
            )

        # Check cache first (using normalized space)
        cache_key = (space_normalized, resolution)
        if cache_key in self._template_cache:
            logger.debug(f"Using cached template: {space} @ {resolution}mm (normalized to {space_normalized})")
            return self._template_cache[cache_key]

        # Check bundled templates (using normalized space)
        bundled_path = self._find_bundled_template(space_normalized, resolution)
        if bundled_path is not None and bundled_path.exists():
            logger.debug(f"Using bundled template: {bundled_path}")
            self._template_cache[cache_key] = bundled_path
            return bundled_path

        # Try TemplateFlow (using normalized space)
        logger.debug(f"Trying to download template from TemplateFlow: {space} @ {resolution}mm (normalized to {space_normalized})")
        try:
            template_path = self._download_template_from_templateflow(space_normalized, resolution)
            if template_path is not None:
                self._template_cache[cache_key] = template_path
                logger.info(f"Downloaded template from TemplateFlow: {space} -> {space_normalized} @ {resolution}mm")
                return template_path
        except Exception as e:
            logger.warning(f"Failed to download template from TemplateFlow: {e}")

        logger.warning(
            f"Template not found for {space} at {resolution}mm. "
            f"Checked: bundled files, TemplateFlow"
        )
        return None

    def _download_template_from_templateflow(
        self, space: str, resolution: int | float
    ) -> Path | None:
        """Download template from TemplateFlow.

        Args:
            space: Coordinate space identifier
            resolution: Template resolution in mm

        Returns:
            Path to downloaded template file, or None if not found

        Raises:
            Nothing - returns None on any error
        """
        tfapi = _get_tfapi()
        
        try:
            # Try with desc='brain' first (most common for lesion analysis)
            result = tfapi.get(
                template=space,
                resolution=resolution,
                suffix="T1w",
                desc="brain",
            )

            # tfapi.get() returns a Path object directly, not a list
            if result is not None:
                return Path(result)

            # Fallback: try without desc field
            result = tfapi.get(
                template=space,
                resolution=resolution,
                suffix="T1w",
            )

            if result is not None:
                return Path(result)

            return None

        except Exception as e:
            logger.debug(f"TemplateFlow download failed: {e}")
            return None

    def _find_bundled_template(self, space: str, resolution: int | float) -> Path | None:
        """Find bundled template file.

        Args:
            space: Coordinate space
            resolution: Resolution in mm

        Returns:
            Path to template if found, None otherwise
        """
        # Format resolution for filename
        if isinstance(resolution, float):
            res_str = f"{int(resolution * 100):02d}"  # 0.5 -> "05", 2.0 -> "20"
        else:
            res_str = f"{resolution:02d}"  # 1 -> "01", 2 -> "02"

        # Try different naming patterns
        patterns = [
            f"tpl-{space}_res-{res_str}_desc-brain_T1w.nii.gz",
            f"tpl-{space}_res-{res_str}_T1w.nii.gz",
        ]

        for pattern in patterns:
            template_path = BUNDLED_TEMPLATES_DIR / pattern
            if template_path.exists():
                return template_path

        return None


def get_transform_path(source_space: str, target_space: str) -> Path | None:
    """Get path to transform file between coordinate spaces.

    This is a convenience function that creates a DataAssetManager and
    retrieves the transform path.

    Args:
        source_space: Source coordinate space identifier
        target_space: Target coordinate space identifier

    Returns:
        Path to transform file, or None if not available

    Raises:
        TransformNotAvailableError: If transform pair is not supported

    Examples:
        >>> path = get_transform_path("MNI152NLin6Asym", "MNI152NLin2009cAsym")
        >>> if path:
        ...     print(f"Transform found at: {path}")
    """
    manager = DataAssetManager()
    return manager.get_transform(source_space, target_space)


__all__ = [
    "DataAssetManager",
    "get_transform_path",
    "DEFAULT_ASSET_CACHE_DIR",
    "BUNDLED_TEMPLATES_DIR",
    "SUPPORTED_TEMPLATE_CONFIGS",
]
