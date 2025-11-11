"""Asset management for transforms, templates, and atlases via TemplateFlow.

This module provides the DataAssetManager for retrieving spatial data assets
(templates, atlases, transforms) with caching and TemplateFlow integration.
"""

import logging
from pathlib import Path
from typing import Any

from lacuna.core.exceptions import (
    TransformDownloadError,
    TransformNotAvailableError,
)

logger = logging.getLogger(__name__)

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

# Known atlas names (for validation)
KNOWN_ATLASES = [
    "HarvardOxford",
    "AAL",
    "Schaefer",
    "Desikan",
    "Destrieux",
]


class DataAssetManager:
    """Manager for spatial data assets (templates, atlases, transforms).

    Handles retrieval and caching of:
    - MNI templates (from bundled files or TemplateFlow)
    - Anatomical atlases (from TemplateFlow)
    - Spatial transforms (from TemplateFlow)

    Attributes:
        cache_dir: Directory for caching downloaded assets
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialize asset manager.

        Args:
            cache_dir: Directory for asset cache (default: ~/.cache/lacuna/assets)
        """
        self.cache_dir = cache_dir if cache_dir is not None else DEFAULT_ASSET_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache for retrieved assets
        self._transform_cache: dict[tuple[str, str], Any] = {}

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
        # Check if transform is supported
        if not self._is_transform_supported(source_space, target_space):
            # Get list of supported transform pairs
            supported_spaces = list(SUPPORTED_TEMPLATE_CONFIGS.keys())
            supported_pairs = [
                f"{s1} <-> {s2}"
                for i, s1 in enumerate(supported_spaces)
                for s2 in supported_spaces[i + 1 :]
            ]
            raise TransformNotAvailableError(source_space, target_space, supported_pairs)

        # Check cache (bidirectional)
        cache_key = tuple(sorted([source_space, target_space]))
        if cache_key in self._transform_cache:
            logger.debug(f"Using cached transform: {source_space} <-> {target_space}")
            return self._transform_cache[cache_key]

        # Try to download from TemplateFlow
        try:
            transform_path = self._download_transform_from_templateflow(source_space, target_space)

            if transform_path is not None:
                # Cache the result
                self._transform_cache[cache_key] = transform_path
                logger.info(f"Downloaded and cached transform: {source_space} <-> {target_space}")
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
            ImportError: If templateflow package is not available
            Exception: If download fails
        """
        try:
            from templateflow import api as tfapi
        except ImportError as e:
            raise ImportError(
                "templateflow package is required for downloading transforms. "
                "Install with: pip install templateflow"
            ) from e

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
                return Path(transform_path)

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
                    return Path(transform_path)

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
        # Validate space
        if space not in SUPPORTED_TEMPLATE_CONFIGS:
            raise ValueError(
                f"Unsupported space: {space}. "
                f"Supported spaces: {list(SUPPORTED_TEMPLATE_CONFIGS.keys())}"
            )

        # Validate resolution
        if resolution not in SUPPORTED_TEMPLATE_CONFIGS[space]:
            raise ValueError(
                f"Invalid resolution {resolution}mm for {space}. "
                f"Supported resolutions: {SUPPORTED_TEMPLATE_CONFIGS[space]}"
            )

        # Check bundled templates first
        bundled_path = self._find_bundled_template(space, resolution)
        if bundled_path is not None and bundled_path.exists():
            logger.debug(f"Using bundled template: {bundled_path}")
            return bundled_path

        # Try TemplateFlow (not implemented yet)
        logger.warning(
            f"Bundled template not found for {space} at {resolution}mm. "
            f"TemplateFlow integration pending."
        )
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

    def get_atlas(self, name: str, space: str) -> Path | None:
        """Retrieve atlas in specified coordinate space.

        Args:
            name: Atlas name (e.g., "HarvardOxford", "AAL")
            space: Target coordinate space

        Returns:
            Path to atlas file, or None if not available

        Raises:
            ValueError: If atlas name is not recognized

        Examples:
            >>> manager = DataAssetManager()
            >>> atlas = manager.get_atlas("HarvardOxford", space="MNI152NLin6Asym")
        """
        # Validate atlas name
        if name not in KNOWN_ATLASES:
            raise ValueError(f"Unknown atlas: {name}. Known atlases: {KNOWN_ATLASES}")

        # Try to get from TemplateFlow or bundled atlases
        # Not implemented yet
        logger.info(f"Atlas {name} in {space} not yet available (atlas integration pending)")
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
    "KNOWN_ATLASES",
]
