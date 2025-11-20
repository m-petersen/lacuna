"""Atlas loading functions for bundled and user-registered atlases.

This module provides functions to load atlases from the registry.
Supports both bundled atlases (shipped with Lacuna) and user-registered
custom atlases.

Examples
--------
>>> from lacuna.atlas.loader import load_atlas
>>>
>>> # Load a bundled atlas
>>> atlas = load_atlas("Schaefer2018_100Parcels7Networks")
>>> print(atlas.image.shape)
>>> print(list(atlas.labels.keys())[:5])
>>>
>>> # Access metadata
>>> print(atlas.metadata.space)
>>> print(atlas.metadata.citation)
"""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib

from lacuna.atlas.registry import ATLAS_REGISTRY, AtlasMetadata

# Path to bundled atlases in package
BUNDLED_ATLASES_DIR = Path(__file__).parent.parent / "data" / "atlases"


@dataclass
class Atlas:
    """Loaded atlas with image data, labels, and metadata.

    Attributes
    ----------
    image : nib.Nifti1Image
        The atlas image (3D or 4D for probabilistic atlases)
    labels : dict[int, str]
        Mapping from region ID to region name
    metadata : AtlasMetadata
        Atlas metadata from registry
    """

    image: nib.Nifti1Image
    labels: dict[int, str]
    metadata: AtlasMetadata


def load_atlas(
    atlas_name: str,
) -> Atlas:
    """Load an atlas by name from the registry.

    Loads bundled atlases or user-registered custom atlases.

    Parameters
    ----------
    atlas_name : str
        Name of the atlas from ATLAS_REGISTRY

    Returns
    -------
    Atlas
        Loaded atlas with image, labels, and metadata

    Raises
    ------
    KeyError
        If atlas_name is not in the registry
    FileNotFoundError
        If atlas files are not found

    Examples
    --------
    >>> atlas = load_atlas("Schaefer2018_100Parcels7Networks")
    >>> print(f"Atlas has {len(atlas.labels)} regions")
    >>> print(f"Space: {atlas.metadata.space}")
    """
    # Get metadata from registry
    if atlas_name not in ATLAS_REGISTRY:
        available = list(ATLAS_REGISTRY.keys())
        raise KeyError(
            f"Atlas '{atlas_name}' not found in registry. "
            f"Available atlases: {', '.join(available)}"
        )

    metadata = ATLAS_REGISTRY[atlas_name]

    # Determine atlas directory (bundled or custom)
    # If atlas_filename is absolute path, use it directly
    # Otherwise, look in bundled atlases directory
    atlas_filename_path = Path(metadata.atlas_filename)
    if atlas_filename_path.is_absolute():
        atlas_path = atlas_filename_path
    else:
        atlas_path = BUNDLED_ATLASES_DIR / metadata.atlas_filename
    if not atlas_path.exists():
        raise FileNotFoundError(
            f"Atlas file not found: {atlas_path}\n" f"Expected: {metadata.atlas_filename}"
        )

    image = nib.load(atlas_path)

    # Load labels (same logic: absolute or relative to bundled dir)
    labels_filename_path = Path(metadata.labels_filename)
    if labels_filename_path.is_absolute():
        labels_path = labels_filename_path
    else:
        labels_path = BUNDLED_ATLASES_DIR / metadata.labels_filename
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}\n" f"Expected: {metadata.labels_filename}"
        )

    labels = _load_labels_file(labels_path)

    return Atlas(image=image, labels=labels, metadata=metadata)


def _load_labels_file(labels_path: Path) -> dict[int, str]:
    """Load atlas labels from text file.

    Two formats are supported:
    1. "region_id region_name" format (e.g., "1 Left-Hemisphere")
    2. One region name per line (region_id is line number, starting from 1)

    Lines starting with # are treated as comments.

    Parameters
    ----------
    labels_path : Path
        Path to labels text file

    Returns
    -------
    dict[int, str]
        Mapping from region ID to region name
    """
    labels = {}

    with open(labels_path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Try to parse "region_id region_name" format first
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                try:
                    region_id = int(parts[0])
                    region_name = parts[1]
                    labels[region_id] = region_name
                    continue
                except ValueError:
                    # Not "ID name" format, fall through to line-number format
                    pass

            # Use line number as region ID (for format like Schaefer atlas)
            # This handles the case where each line is just a region name
            if line:  # Non-empty line
                region_id = line_num
                # Adjust for skipped comment/empty lines
                # Find the actual region number by counting non-comment lines seen so far
                actual_region_id = len(labels) + 1
                labels[actual_region_id] = line

    return labels
