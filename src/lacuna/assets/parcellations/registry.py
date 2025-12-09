"""Parcellation registry with metadata for all supported parcellations.

This module provides a centralized registry of bundled parcellations with their
metadata (space, resolution, description, etc.). The registry enables:

- Discovery of available parcellations
- Filtering parcellations by space/resolution
- Access to parcellation metadata without loading image data
- Consistent naming and citation tracking

Examples
--------
>>> from lacuna.assets.parcellations import list_parcellations, PARCELLATION_REGISTRY
>>>
>>> # List all available parcellations
>>> parcellations = list_parcellations()
>>> print([a.name for a in parcellations])
>>>
>>> # Filter by coordinate space
>>> mni6_parcellations = list_parcellations(space="MNI152NLin6Asym")
>>>
>>> # Get specific parcellation metadata
>>> schaefer = PARCELLATION_REGISTRY["Schaefer2018_100Parcels7Networks"]
>>> print(schaefer.space, schaefer.resolution)
"""

from dataclasses import dataclass, field
from pathlib import Path

from lacuna.assets.base import AssetRegistry, SpatialAssetMetadata


@dataclass(frozen=True)
class ParcellationMetadata(SpatialAssetMetadata):
    """Metadata for a neuroimaging parcellation.

    Inherits from SpatialAssetMetadata to include space and resolution validation.

    Attributes
    ----------
    name : str
        Unique identifier for the parcellation
    space : str
        Coordinate space (e.g., "MNI152NLin6Asym")
    resolution : float
        Resolution in mm (e.g., 1.0, 2.0)
    description : str
        Human-readable description
    parcellation_filename : str
        Filename of the NIfTI parcellation file
    labels_filename : str
        Filename of the labels text file
    citation : str, optional
        Citation information for the parcellation
    networks : list[str], optional
        List of network names if parcellation has network organization
    n_regions : int, optional
        Number of regions/parcels in the parcellation
    is_4d : bool, optional
        Whether the parcellation is 4D (multiple volumes) or 3D (single volume).
        Default is False. 4D parcellations are transformed volume-by-volume
        and aggregated independently.
    region_labels : list[str] | None, optional
        Human-readable labels for each region (1-indexed, matching ROI values).
        If None, labels will be auto-generated as "region_001", "region_002", etc.
        Loaded automatically from labels_filename during parcellation registration.
    """

    parcellation_filename: str = ""
    labels_filename: str = ""
    citation: str | None = None
    networks: list[str] = field(default_factory=list)
    n_regions: int | None = None
    is_4d: bool = False
    region_labels: list[str] | None = None


# Global registry instance
_parcellation_registry = AssetRegistry[ParcellationMetadata]("parcellation")

# Registry of bundled parcellations
# Maps parcellation name -> ParcellationMetadata
PARCELLATION_REGISTRY: dict[str, ParcellationMetadata] = {
    # Schaefer 2018 Cortical Parcellation - 100 Parcels
    "Schaefer2018_100Parcels7Networks": ParcellationMetadata(
        name="Schaefer2018_100Parcels7Networks",
        space="MNI152NLin6Asym",
        resolution=1,
        description="Schaefer 2018 cortical parcellation with 100 parcels organized into 7 networks",
        parcellation_filename="tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-100Parcels7Networks_dseg.nii.gz",
        labels_filename="tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-100Parcels7Networks_dseg_labels.txt",
        citation="Schaefer et al. (2018), Cerebral Cortex, 28(9), 3095-3114",
        networks=[
            "Visual",
            "Somatomotor",
            "DorsalAttention",
            "VentralAttention",
            "Limbic",
            "Frontoparietal",
            "Default",
        ],
        n_regions=100,
    ),
    # Schaefer 2018 - 200 Parcels
    "Schaefer2018_200Parcels7Networks": ParcellationMetadata(
        name="Schaefer2018_200Parcels7Networks",
        space="MNI152NLin6Asym",
        resolution=1,
        description="Schaefer 2018 cortical parcellation with 200 parcels organized into 7 networks",
        parcellation_filename="tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.nii.gz",
        labels_filename="tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_labels.txt",
        citation="Schaefer et al. (2018), Cerebral Cortex, 28(9), 3095-3114",
        networks=[
            "Visual",
            "Somatomotor",
            "DorsalAttention",
            "VentralAttention",
            "Limbic",
            "Frontoparietal",
            "Default",
        ],
        n_regions=200,
    ),
    # Schaefer 2018 - 400 Parcels
    "Schaefer2018_400Parcels7Networks": ParcellationMetadata(
        name="Schaefer2018_400Parcels7Networks",
        space="MNI152NLin6Asym",
        resolution=1,
        description="Schaefer 2018 cortical parcellation with 400 parcels organized into 7 networks",
        parcellation_filename="tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-400Parcels7Networks_dseg.nii.gz",
        labels_filename="tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-400Parcels7Networks_dseg_labels.txt",
        citation="Schaefer et al. (2018), Cerebral Cortex, 28(9), 3095-3114",
        networks=[
            "Visual",
            "Somatomotor",
            "DorsalAttention",
            "VentralAttention",
            "Limbic",
            "Frontoparietal",
            "Default",
        ],
        n_regions=400,
    ),
    # Schaefer 2018 - 1000 Parcels
    "Schaefer2018_1000Parcels7Networks": ParcellationMetadata(
        name="Schaefer2018_1000Parcels7Networks",
        space="MNI152NLin6Asym",
        resolution=1,
        description="Schaefer 2018 cortical parcellation with 1000 parcels organized into 7 networks",
        parcellation_filename="tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-1000Parcels7Networks_dseg.nii.gz",
        labels_filename="tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-1000Parcels7Networks_dseg_labels.txt",
        citation="Schaefer et al. (2018), Cerebral Cortex, 28(9), 3095-3114",
        networks=[
            "Visual",
            "Somatomotor",
            "DorsalAttention",
            "VentralAttention",
            "Limbic",
            "Frontoparietal",
            "Default",
        ],
        n_regions=1000,
    ),
    # Tian Subcortical Parcellation - Scale 1
    "TianSubcortex_3TS1": ParcellationMetadata(
        name="TianSubcortex_3TS1",
        space="MNI152NLin6Asym",
        resolution=1,
        description="Tian subcortical parcellation at Scale 1 (3 Tesla)",
        parcellation_filename="tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS1_dseg.nii.gz",
        labels_filename="tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS1_dseg_labels.txt",
        citation="Tian et al. (2020), Nature Neuroscience, 23(11), 1421-1432",
        n_regions=16,
    ),
    # Tian Subcortical Parcellation - Scale 2
    "TianSubcortex_3TS2": ParcellationMetadata(
        name="TianSubcortex_3TS2",
        space="MNI152NLin6Asym",
        resolution=1,
        description="Tian subcortical parcellation at Scale 2 (3 Tesla)",
        parcellation_filename="tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS2_dseg.nii.gz",
        labels_filename="tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS2_dseg_labels.txt",
        citation="Tian et al. (2020), Nature Neuroscience, 23(11), 1421-1432",
        n_regions=32,
    ),
    # Tian Subcortical Parcellation - Scale 3
    "TianSubcortex_3TS3": ParcellationMetadata(
        name="TianSubcortex_3TS3",
        space="MNI152NLin6Asym",
        resolution=1,
        description="Tian subcortical parcellation at Scale 3 (3 Tesla)",
        parcellation_filename="tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS3_dseg.nii.gz",
        labels_filename="tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS3_dseg_labels.txt",
        citation="Tian et al. (2020), Nature Neuroscience, 23(11), 1421-1432",
        n_regions=54,
    ),
    # HCP White Matter Tracts
    "HCP1065_thr0p1": ParcellationMetadata(
        name="HCP1065_thr0p1",
        space="MNI152NLin2009aAsym",
        resolution=1,
        description="HCP-1065 probabilistic white matter tract parcellation with 0.1 threshold (4D: 64 tracts)",
        parcellation_filename="tpl-MNI152Nlin2009aAsym_res-01_atlas-HCP1065_desc-thr0p1_probseg.nii.gz",
        labels_filename="tpl-MNI152Nlin2009aAsym_res-01_atlas-HCP1065_desc-thr0p1_probseg_labels.txt",
        citation="Yeh et al. (2022), Nature Communications, 13(1), 4933",
        n_regions=64,  # 64 white matter tracts (one per volume)
        is_4d=True,  # 4D parcellation with one tract per volume
    ),
}


def list_parcellations(
    space: str | None = None,
    resolution: int | None = None,
) -> list[ParcellationMetadata]:
    """List available parcellations with optional filtering.

    Parameters
    ----------
    space : str, optional
        Filter by coordinate space (e.g., "MNI152NLin6Asym")
    resolution : int, optional
        Filter by resolution in mm (e.g., 1, 2)

    Returns
    -------
    list[ParcellationMetadata]
        List of parcellation metadata matching the filters

    Examples
    --------
    >>> # List all parcellations
    >>> parcellations = list_parcellations()
    >>>
    >>> # Filter by space
    >>> mni6_parcellations = list_parcellations(space="MNI152NLin6Asym")
    >>>
    >>> # Filter by resolution
    >>> res1_parcellations = list_parcellations(resolution=1)
    >>>
    >>> # Combined filters
    >>> filtered = list_parcellations(space="MNI152NLin6Asym", resolution=1)
    """
    parcellations = list(PARCELLATION_REGISTRY.values())

    if space is not None:
        parcellations = [a for a in parcellations if a.space == space]

    if resolution is not None:
        parcellations = [a for a in parcellations if a.resolution == resolution]

    # Sort by name for consistent ordering
    parcellations = sorted(parcellations, key=lambda a: a.name)

    return parcellations


def register_parcellation(metadata: ParcellationMetadata) -> None:
    """Register a custom parcellation with the registry.

    Allows users to add their own parcellations to the registry for use with
    Lacuna's analysis modules.

    Parameters
    ----------
    metadata : ParcellationMetadata
        Complete metadata for the custom parcellation. The parcellation_filename and
        labels_filename should be absolute paths to the parcellation files.

    Raises
    ------
    ValueError
        If an parcellation with the same name already exists in the registry

    Examples
    --------
    >>> from pathlib import Path
    >>> from lacuna.parcellation.registry import register_parcellation, ParcellationMetadata
    >>>
    >>> # Register a custom parcellation
    >>> custom_metadata = ParcellationMetadata(
    ...     name="MyCustomParcellation",
    ...     space="MNI152NLin6Asym",
    ...     resolution=1,
    ...     description="My custom parcellation",
    ...     parcellation_filename="/path/to/my_parcellation.nii.gz",
    ...     labels_filename="/path/to/my_parcellation_labels.txt",
    ... )
    >>> register_parcellation(custom_metadata)
    """
    if metadata.name in PARCELLATION_REGISTRY:
        raise ValueError(
            f"Parcellation '{metadata.name}' already registered. "
            f"Use a different name or unregister the existing parcellation first."
        )

    PARCELLATION_REGISTRY[metadata.name] = metadata


def register_parcellation_from_files(
    name: str,
    parcellation_path: str | Path,
    labels_path: str | Path,
    space: str,
    resolution: int,
    description: str,
    citation: str | None = None,
    networks: list[str] | None = None,
    n_regions: int | None = None,
) -> None:
    """Register a custom parcellation from file paths.

    Convenience function that creates ParcellationMetadata from file paths and
    registers the parcellation.

    Parameters
    ----------
    name : str
        Unique identifier for the parcellation
    parcellation_path : str or Path
        Path to the NIfTI parcellation file
    labels_path : str or Path
        Path to the labels text file
    space : str
        Coordinate space (e.g., "MNI152NLin6Asym")
    resolution : int
        Resolution in mm
    description : str
        Human-readable description
    citation : str, optional
        Citation information
    networks : list[str], optional
        List of network names if parcellation has network organization
    n_regions : int, optional
        Number of regions in the parcellation

    Examples
    --------
    >>> from lacuna.parcellation.registry import register_parcellation_from_files
    >>>
    >>> register_parcellation_from_files(
    ...     name="MyParcellation",
    ...     parcellation_path="/data/parcellations/my_parcellation.nii.gz",
    ...     labels_path="/data/parcellations/my_parcellation_labels.txt",
    ...     space="MNI152NLin6Asym",
    ...     resolution=2,
    ...     description="Custom 2mm parcellation",
    ... )
    """
    # Convert to absolute paths
    parcellation_path = Path(parcellation_path).resolve()
    labels_path = Path(labels_path).resolve()

    # Verify files exist
    if not parcellation_path.exists():
        raise FileNotFoundError(f"Parcellation file not found: {parcellation_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # Load region labels from file
    from lacuna.assets.parcellations.loader import _load_labels_file

    labels_dict = _load_labels_file(labels_path)
    # Convert to list ordered by region ID (1-indexed)
    max_region = max(labels_dict.keys()) if labels_dict else 0
    region_labels = [labels_dict.get(i, f"region_{i:03d}") for i in range(1, max_region + 1)]

    metadata = ParcellationMetadata(
        name=name,
        space=space,
        resolution=resolution,
        description=description,
        parcellation_filename=str(parcellation_path),
        labels_filename=str(labels_path),
        citation=citation,
        networks=networks or [],
        n_regions=n_regions,
        region_labels=region_labels,
    )

    register_parcellation(metadata)


def unregister_parcellation(name: str) -> None:
    """Remove an parcellation from the registry.

    Parameters
    ----------
    name : str
        Name of the parcellation to unregister

    Raises
    ------
    KeyError
        If parcellation is not in the registry

    Examples
    --------
    >>> from lacuna.parcellation.registry import unregister_parcellation
    >>> unregister_parcellation("MyCustomParcellation")
    """
    if name not in PARCELLATION_REGISTRY:
        raise KeyError(f"Parcellation '{name}' not found in registry")

    del PARCELLATION_REGISTRY[name]


def register_parcellations_from_directory(
    directory: str | Path,
    space: str | None = None,
    resolution: int | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Register all parcellations found in a directory.

    Discovers parcellation files in the directory and registers them. Each parcellation should have:
    - NIfTI file (.nii or .nii.gz)
    - Labels file with same base name + "_labels.txt" or ".txt"

    If space/resolution are not provided, attempts to parse from BIDS-style filenames
    (tpl-{SPACE}_res-{RES}_...) or detect from image headers.

    Parameters
    ----------
    directory : str or Path
        Directory containing parcellation files
    space : str, optional
        Default coordinate space for parcellations without BIDS naming
    resolution : int, optional
        Default resolution for parcellations without BIDS naming
    overwrite : bool, default=False
        If True, overwrite existing parcellations with same names

    Returns
    -------
    list[str]
        Names of successfully registered parcellations

    Raises
    ------
    FileNotFoundError
        If directory doesn't exist

    Examples
    --------
    >>> from lacuna.parcellation.registry import register_parcellations_from_directory
    >>>
    >>> # Register all parcellations from a directory
    >>> registered = register_parcellations_from_directory("/data/my_parcellations")
    >>> print(f"Registered {len(registered)} parcellations: {registered}")
    >>>
    >>> # Register with explicit space/resolution for non-BIDS parcellations
    >>> registered = register_parcellations_from_directory(
    ...     "/data/custom_parcellations",
    ...     space="MNI152NLin6Asym",
    ...     resolution=2
    ... )
    """
    import nibabel as nib

    from lacuna.core.spaces import detect_space_from_header

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    registered_names = []

    # Find all NIfTI files
    nifti_patterns = ["*.nii.gz", "*.nii"]
    nifti_files = []
    for pattern in nifti_patterns:
        nifti_files.extend(directory.glob(pattern))

    for nifti_path in nifti_files:
        # Get base name
        if nifti_path.name.endswith(".nii.gz"):
            base_name = nifti_path.name[:-7]
        else:
            base_name = nifti_path.name[:-4]

        # Skip if already registered and not overwriting
        if base_name in PARCELLATION_REGISTRY and not overwrite:
            continue

        # Look for corresponding labels file
        labels_path = directory / f"{base_name}_labels.txt"
        if not labels_path.exists():
            labels_path = directory / f"{base_name}.txt"

        if not labels_path.exists():
            # Skip parcellation without labels
            continue

        # Parse space and resolution from filename or use defaults
        parcellation_space = space
        parcellation_resolution = resolution

        # Try BIDS-style filename parsing
        parts = base_name.split("_")
        for part in parts:
            if part.startswith("tpl-"):
                parcellation_space = part[4:]
            elif part.startswith("res-"):
                try:
                    res_str = part[4:]
                    parcellation_resolution = (
                        int(res_str) if res_str.isdigit() else parcellation_resolution
                    )
                except (ValueError, AttributeError):
                    pass

        # Fall back to header detection if needed
        if parcellation_space is None or parcellation_resolution is None:
            try:
                parcellation_img = nib.load(nifti_path)
                detected = detect_space_from_header(parcellation_img)
                if detected is not None:
                    detected_space, detected_res = detected
                    if parcellation_space is None:
                        parcellation_space = detected_space
                    if parcellation_resolution is None:
                        parcellation_resolution = detected_res
            except Exception:
                pass

        # Skip if we couldn't determine space/resolution
        if parcellation_space is None or parcellation_resolution is None:
            continue

        # Register the parcellation
        try:
            register_parcellation_from_files(
                name=base_name,
                parcellation_path=nifti_path,
                labels_path=labels_path,
                space=parcellation_space,
                resolution=parcellation_resolution,
                description=f"Parcellation from {directory.name}",
            )
            registered_names.append(base_name)
        except Exception:
            # Skip parcellations that fail to register
            continue

    return registered_names


def _load_bundled_parcellation_labels(labels_filename: str) -> list[str] | None:
    """Load labels for a bundled parcellation from the data directory.

    Parameters
    ----------
    labels_filename : str
        Relative path to labels file in bundled parcellations directory

    Returns
    -------
    list[str] | None
        Ordered list of region labels (1-indexed), or None if file doesn't exist
    """
    from pathlib import Path

    bundled_dir = Path(__file__).parent.parent.parent / "data" / "parcellations"
    labels_path = bundled_dir / labels_filename

    if not labels_path.exists():
        return None

    # Inline label loading to avoid circular import
    labels_dict = {}
    with open(labels_path) as f:
        for _line_num, line in enumerate(f, start=1):
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
                    labels_dict[region_id] = region_name
                    continue
                except ValueError:
                    pass

            # Use line number as region ID
            if line:
                actual_region_id = len(labels_dict) + 1
                labels_dict[actual_region_id] = line

    if not labels_dict:
        return None

    # Convert to list ordered by region ID (1-indexed)
    max_region = max(labels_dict.keys())
    return [labels_dict.get(i, f"region_{i:03d}") for i in range(1, max_region + 1)]


# Auto-load labels for all bundled parcellations after registry initialization
def _populate_bundled_parcellation_labels() -> None:
    """Populate region_labels for all bundled parcellations in PARCELLATION_REGISTRY."""
    for parcellation_name, metadata in PARCELLATION_REGISTRY.items():
        if metadata.region_labels is None and metadata.labels_filename:
            labels = _load_bundled_parcellation_labels(metadata.labels_filename)
            if labels is not None:
                # Create new metadata with labels (dataclass is frozen)
                updated_metadata = ParcellationMetadata(
                    name=metadata.name,
                    space=metadata.space,
                    resolution=metadata.resolution,
                    description=metadata.description,
                    parcellation_filename=metadata.parcellation_filename,
                    labels_filename=metadata.labels_filename,
                    citation=metadata.citation,
                    networks=metadata.networks,
                    n_regions=metadata.n_regions,
                    is_4d=metadata.is_4d,
                    region_labels=labels,
                )
                PARCELLATION_REGISTRY[parcellation_name] = updated_metadata


# Populate labels when module is imported
_populate_bundled_parcellation_labels()
