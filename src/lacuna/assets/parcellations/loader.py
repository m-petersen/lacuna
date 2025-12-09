"""Parcellation loading functions for bundled and user-registered parcellations.

This module provides functions to load parcellations from the registry.
Supports both bundled parcellations (shipped with Lacuna) and user-registered
custom parcellations.

Examples
--------
>>> from lacuna.assets.parcellations import load_parcellation
>>>
>>> # Load a bundled parcellation
>>> parcellation = load_parcellation("Schaefer2018_100Parcels7Networks")
>>> print(parcellation.image.shape)
>>> print(list(parcellation.labels.keys())[:5])
>>>
>>> # Access metadata
>>> print(parcellation.metadata.space)
>>> print(parcellation.metadata.citation)
"""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib

from lacuna.assets.parcellations.registry import PARCELLATION_REGISTRY, ParcellationMetadata

# Path to bundled parcellations in package (stored in data/atlases for now)
BUNDLED_PARCELLATIONS_DIR = Path(__file__).parent.parent.parent / "data" / "atlases"


@dataclass
class Parcellation:
    """Loaded parcellation with image data, labels, and metadata.

    Attributes
    ----------
    image : nib.Nifti1Image
        The parcellation image (3D or 4D for probabilistic parcellations)
    labels : dict[int, str]
        Mapping from region ID to region name
    metadata : ParcellationMetadata
        Parcellation metadata from registry
    """

    image: nib.Nifti1Image
    labels: dict[int, str]
    metadata: ParcellationMetadata


def load_parcellation(
    parcellation_name: str,
) -> Parcellation:
    """Load an parcellation by name from the registry.

    Loads bundled parcellations or user-registered custom parcellations.

    Parameters
    ----------
    parcellation_name : str
        Name of the parcellation from PARCELLATION_REGISTRY

    Returns
    -------
    Parcellation
        Loaded parcellation with image, labels, and metadata

    Raises
    ------
    KeyError
        If parcellation_name is not in the registry
    FileNotFoundError
        If parcellation files are not found

    Examples
    --------
    >>> parcellation = load_parcellation("Schaefer2018_100Parcels7Networks")
    >>> print(f"Parcellation has {len(parcellation.labels)} regions")
    >>> print(f"Space: {parcellation.metadata.space}")
    """
    # Get metadata from registry
    if parcellation_name not in PARCELLATION_REGISTRY:
        available = list(PARCELLATION_REGISTRY.keys())
        raise KeyError(
            f"Parcellation '{parcellation_name}' not found in registry. "
            f"Available parcellations: {', '.join(available)}"
        )

    metadata = PARCELLATION_REGISTRY[parcellation_name]

    # Determine parcellation directory (bundled or custom)
    # If parcellation_filename is absolute path, use it directly
    # Otherwise, look in bundled parcellations directory
    parcellation_filename_path = Path(metadata.parcellation_filename)
    if parcellation_filename_path.is_absolute():
        parcellation_path = parcellation_filename_path
    else:
        parcellation_path = BUNDLED_PARCELLATIONS_DIR / metadata.parcellation_filename
    if not parcellation_path.exists():
        raise FileNotFoundError(
            f"Parcellation file not found: {parcellation_path}\n"
            f"Expected: {metadata.parcellation_filename}"
        )

    image = nib.load(parcellation_path)

    # Load labels (same logic: absolute or relative to bundled dir)
    labels_filename_path = Path(metadata.labels_filename)
    if labels_filename_path.is_absolute():
        labels_path = labels_filename_path
    else:
        labels_path = BUNDLED_PARCELLATIONS_DIR / metadata.labels_filename
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}\n" f"Expected: {metadata.labels_filename}"
        )

    labels = _load_labels_file(labels_path)

    return Parcellation(image=image, labels=labels, metadata=metadata)


def _load_labels_file(labels_path: Path) -> dict[int, str]:
    """Load parcellation labels from text file.

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

            # Use line number as region ID (for format like Schaefer parcellation)
            # This handles the case where each line is just a region name
            if line:  # Non-empty line
                region_id = line_num
                # Adjust for skipped comment/empty lines
                # Find the actual region number by counting non-comment lines seen so far
                actual_region_id = len(labels) + 1
                labels[actual_region_id] = line

    return labels
