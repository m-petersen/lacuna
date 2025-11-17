"""Atlas registry with metadata for all supported atlases.

This module provides a centralized registry of bundled atlases with their
metadata (space, resolution, description, etc.). The registry enables:

- Discovery of available atlases
- Filtering atlases by space/resolution
- Access to atlas metadata without loading image data
- Consistent naming and citation tracking

Examples
--------
>>> from lacuna.assets.atlases import list_atlases, ATLAS_REGISTRY
>>> 
>>> # List all available atlases
>>> atlases = list_atlases()
>>> print([a.name for a in atlases])
>>> 
>>> # Filter by coordinate space
>>> mni6_atlases = list_atlases(space="MNI152NLin6Asym")
>>> 
>>> # Get specific atlas metadata
>>> schaefer = ATLAS_REGISTRY["Schaefer2018_100Parcels7Networks"]
>>> print(schaefer.space, schaefer.resolution)
"""

from dataclasses import dataclass, field
from pathlib import Path

from lacuna.assets.base import SpatialAssetMetadata, AssetRegistry


@dataclass(frozen=True)
class AtlasMetadata(SpatialAssetMetadata):
    """Metadata for a neuroimaging atlas.
    
    Inherits from SpatialAssetMetadata to include space and resolution validation.
    
    Attributes
    ----------
    name : str
        Unique identifier for the atlas
    space : str
        Coordinate space (e.g., "MNI152NLin6Asym")
    resolution : float
        Resolution in mm (e.g., 1.0, 2.0)
    description : str
        Human-readable description
    atlas_filename : str
        Filename of the NIfTI atlas file
    labels_filename : str
        Filename of the labels text file
    citation : str, optional
        Citation information for the atlas
    networks : list[str], optional
        List of network names if atlas has network organization
    n_regions : int, optional
        Number of regions/parcels in the atlas
    is_4d : bool, optional
        Whether the atlas is 4D (multiple volumes) or 3D (single volume).
        Default is False. 4D atlases are transformed volume-by-volume 
        and aggregated independently.
    """
    
    atlas_filename: str = ""
    labels_filename: str = ""
    citation: str | None = None
    networks: list[str] = field(default_factory=list)
    n_regions: int | None = None
    is_4d: bool = False


# Global registry instance
_atlas_registry = AssetRegistry[AtlasMetadata]("atlas")

# Registry of bundled atlases
# Maps atlas name -> AtlasMetadata
ATLAS_REGISTRY: dict[str, AtlasMetadata] = {
    # Schaefer 2018 Cortical Parcellation - 100 Parcels
    "Schaefer2018_100Parcels7Networks": AtlasMetadata(
        name="Schaefer2018_100Parcels7Networks",
        space="MNI152NLin6Asym",
        resolution=1,
        description="Schaefer 2018 cortical parcellation with 100 parcels organized into 7 networks",
        atlas_filename="tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-100Parcels7Networks_dseg.nii.gz",
        labels_filename="tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-100Parcels7Networks_dseg_labels.txt",
        citation="Schaefer et al. (2018), Cerebral Cortex, 28(9), 3095-3114",
        networks=["Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Limbic", "Frontoparietal", "Default"],
        n_regions=100,
    ),
    
    # Schaefer 2018 - 200 Parcels
    "Schaefer2018_200Parcels7Networks": AtlasMetadata(
        name="Schaefer2018_200Parcels7Networks",
        space="MNI152NLin6Asym",
        resolution=1,
        description="Schaefer 2018 cortical parcellation with 200 parcels organized into 7 networks",
        atlas_filename="tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-200Parcels7Networks_dseg.nii.gz",
        labels_filename="tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-200Parcels7Networks_dseg_labels.txt",
        citation="Schaefer et al. (2018), Cerebral Cortex, 28(9), 3095-3114",
        networks=["Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Limbic", "Frontoparietal", "Default"],
        n_regions=200,
    ),
    
    # Schaefer 2018 - 400 Parcels
    "Schaefer2018_400Parcels7Networks": AtlasMetadata(
        name="Schaefer2018_400Parcels7Networks",
        space="MNI152NLin6Asym",
        resolution=1,
        description="Schaefer 2018 cortical parcellation with 400 parcels organized into 7 networks",
        atlas_filename="tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-400Parcels7Networks_dseg.nii.gz",
        labels_filename="tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-400Parcels7Networks_dseg_labels.txt",
        citation="Schaefer et al. (2018), Cerebral Cortex, 28(9), 3095-3114",
        networks=["Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Limbic", "Frontoparietal", "Default"],
        n_regions=400,
    ),
    
    # Schaefer 2018 - 1000 Parcels
    "Schaefer2018_1000Parcels7Networks": AtlasMetadata(
        name="Schaefer2018_1000Parcels7Networks",
        space="MNI152NLin6Asym",
        resolution=1,
        description="Schaefer 2018 cortical parcellation with 1000 parcels organized into 7 networks",
        atlas_filename="tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-1000Parcels7Networks_dseg.nii.gz",
        labels_filename="tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-1000Parcels7Networks_dseg_labels.txt",
        citation="Schaefer et al. (2018), Cerebral Cortex, 28(9), 3095-3114",
        networks=["Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Limbic", "Frontoparietal", "Default"],
        n_regions=1000,
    ),
    
    # Tian Subcortical Atlas - Scale 1
    "TianSubcortex_3TS1": AtlasMetadata(
        name="TianSubcortex_3TS1",
        space="MNI152NLin6Asym",
        resolution=1,
        description="Tian subcortical atlas at Scale 1 (3 Tesla)",
        atlas_filename="tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS1_dseg.nii.gz",
        labels_filename="tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS1_dseg_labels.txt",
        citation="Tian et al. (2020), Nature Neuroscience, 23(11), 1421-1432",
        n_regions=16,
    ),
    
    # Tian Subcortical Atlas - Scale 2
    "TianSubcortex_3TS2": AtlasMetadata(
        name="TianSubcortex_3TS2",
        space="MNI152NLin6Asym",
        resolution=1,
        description="Tian subcortical atlas at Scale 2 (3 Tesla)",
        atlas_filename="tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS2_dseg.nii.gz",
        labels_filename="tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS2_dseg_labels.txt",
        citation="Tian et al. (2020), Nature Neuroscience, 23(11), 1421-1432",
        n_regions=32,
    ),
    
    # Tian Subcortical Atlas - Scale 3
    "TianSubcortex_3TS3": AtlasMetadata(
        name="TianSubcortex_3TS3",
        space="MNI152NLin6Asym",
        resolution=1,
        description="Tian subcortical atlas at Scale 3 (3 Tesla)",
        atlas_filename="tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS3_dseg.nii.gz",
        labels_filename="tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS3_dseg_labels.txt",
        citation="Tian et al. (2020), Nature Neuroscience, 23(11), 1421-1432",
        n_regions=54,
    ),
    
    # HCP White Matter Tracts
    "HCP1065_thr0p1": AtlasMetadata(
        name="HCP1065_thr0p1",
        space="MNI152NLin2009aAsym",
        resolution=1,
        description="HCP-1065 probabilistic white matter tract atlas with 0.1 threshold (4D: 64 tracts)",
        atlas_filename="tpl-MNI152Nlin2009aAsym_res-01_atlas-HCP1065_desc-thr0p1_probseg.nii.gz",
        labels_filename="tpl-MNI152Nlin2009aAsym_res-01_atlas-HCP1065_desc-thr0p1_probseg_labels.txt",
        citation="Yeh et al. (2022), Nature Communications, 13(1), 4933",
        n_regions=64,  # 64 white matter tracts (one per volume)
        is_4d=True,  # 4D atlas with one tract per volume
    ),
}


def list_atlases(
    space: str | None = None,
    resolution: int | None = None,
) -> list[AtlasMetadata]:
    """List available atlases with optional filtering.
    
    Parameters
    ----------
    space : str, optional
        Filter by coordinate space (e.g., "MNI152NLin6Asym")
    resolution : int, optional
        Filter by resolution in mm (e.g., 1, 2)
    
    Returns
    -------
    list[AtlasMetadata]
        List of atlas metadata matching the filters
    
    Examples
    --------
    >>> # List all atlases
    >>> atlases = list_atlases()
    >>> 
    >>> # Filter by space
    >>> mni6_atlases = list_atlases(space="MNI152NLin6Asym")
    >>> 
    >>> # Filter by resolution
    >>> res1_atlases = list_atlases(resolution=1)
    >>> 
    >>> # Combined filters
    >>> filtered = list_atlases(space="MNI152NLin6Asym", resolution=1)
    """
    atlases = list(ATLAS_REGISTRY.values())
    
    if space is not None:
        atlases = [a for a in atlases if a.space == space]
    
    if resolution is not None:
        atlases = [a for a in atlases if a.resolution == resolution]
    
    return atlases


def register_atlas(metadata: AtlasMetadata) -> None:
    """Register a custom atlas with the registry.
    
    Allows users to add their own atlases to the registry for use with
    Lacuna's analysis modules.
    
    Parameters
    ----------
    metadata : AtlasMetadata
        Complete metadata for the custom atlas. The atlas_filename and
        labels_filename should be absolute paths to the atlas files.
    
    Raises
    ------
    ValueError
        If an atlas with the same name already exists in the registry
    
    Examples
    --------
    >>> from pathlib import Path
    >>> from lacuna.atlas.registry import register_atlas, AtlasMetadata
    >>> 
    >>> # Register a custom atlas
    >>> custom_metadata = AtlasMetadata(
    ...     name="MyCustomAtlas",
    ...     space="MNI152NLin6Asym",
    ...     resolution=1,
    ...     description="My custom parcellation",
    ...     atlas_filename="/path/to/my_atlas.nii.gz",
    ...     labels_filename="/path/to/my_atlas_labels.txt",
    ... )
    >>> register_atlas(custom_metadata)
    """
    if metadata.name in ATLAS_REGISTRY:
        raise ValueError(
            f"Atlas '{metadata.name}' already registered. "
            f"Use a different name or unregister the existing atlas first."
        )
    
    ATLAS_REGISTRY[metadata.name] = metadata


def register_atlas_from_files(
    name: str,
    atlas_path: str | Path,
    labels_path: str | Path,
    space: str,
    resolution: int,
    description: str,
    citation: str | None = None,
    networks: list[str] | None = None,
    n_regions: int | None = None,
) -> None:
    """Register a custom atlas from file paths.
    
    Convenience function that creates AtlasMetadata from file paths and
    registers the atlas.
    
    Parameters
    ----------
    name : str
        Unique identifier for the atlas
    atlas_path : str or Path
        Path to the NIfTI atlas file
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
        List of network names if atlas has network organization
    n_regions : int, optional
        Number of regions in the atlas
    
    Examples
    --------
    >>> from lacuna.atlas.registry import register_atlas_from_files
    >>> 
    >>> register_atlas_from_files(
    ...     name="MyAtlas",
    ...     atlas_path="/data/atlases/my_atlas.nii.gz",
    ...     labels_path="/data/atlases/my_atlas_labels.txt",
    ...     space="MNI152NLin6Asym",
    ...     resolution=2,
    ...     description="Custom 2mm atlas",
    ... )
    """
    # Convert to absolute paths
    atlas_path = Path(atlas_path).resolve()
    labels_path = Path(labels_path).resolve()
    
    # Verify files exist
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    metadata = AtlasMetadata(
        name=name,
        space=space,
        resolution=resolution,
        description=description,
        atlas_filename=str(atlas_path),
        labels_filename=str(labels_path),
        citation=citation,
        networks=networks or [],
        n_regions=n_regions,
    )
    
    register_atlas(metadata)


def unregister_atlas(name: str) -> None:
    """Remove an atlas from the registry.
    
    Parameters
    ----------
    name : str
        Name of the atlas to unregister
    
    Raises
    ------
    KeyError
        If atlas is not in the registry
    
    Examples
    --------
    >>> from lacuna.atlas.registry import unregister_atlas
    >>> unregister_atlas("MyCustomAtlas")
    """
    if name not in ATLAS_REGISTRY:
        raise KeyError(f"Atlas '{name}' not found in registry")
    
    del ATLAS_REGISTRY[name]


def register_atlases_from_directory(
    directory: str | Path,
    space: str | None = None,
    resolution: int | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Register all atlases found in a directory.
    
    Discovers atlas files in the directory and registers them. Each atlas should have:
    - NIfTI file (.nii or .nii.gz)
    - Labels file with same base name + "_labels.txt" or ".txt"
    
    If space/resolution are not provided, attempts to parse from BIDS-style filenames
    (tpl-{SPACE}_res-{RES}_...) or detect from image headers.
    
    Parameters
    ----------
    directory : str or Path
        Directory containing atlas files
    space : str, optional
        Default coordinate space for atlases without BIDS naming
    resolution : int, optional
        Default resolution for atlases without BIDS naming
    overwrite : bool, default=False
        If True, overwrite existing atlases with same names
    
    Returns
    -------
    list[str]
        Names of successfully registered atlases
    
    Raises
    ------
    FileNotFoundError
        If directory doesn't exist
    
    Examples
    --------
    >>> from lacuna.atlas.registry import register_atlases_from_directory
    >>> 
    >>> # Register all atlases from a directory
    >>> registered = register_atlases_from_directory("/data/my_atlases")
    >>> print(f"Registered {len(registered)} atlases: {registered}")
    >>> 
    >>> # Register with explicit space/resolution for non-BIDS atlases
    >>> registered = register_atlases_from_directory(
    ...     "/data/custom_atlases",
    ...     space="MNI152NLin6Asym",
    ...     resolution=2
    ... )
    """
    from lacuna.core.spaces import detect_space_from_header
    import nibabel as nib
    
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
        if nifti_path.name.endswith('.nii.gz'):
            base_name = nifti_path.name[:-7]
        else:
            base_name = nifti_path.name[:-4]
        
        # Skip if already registered and not overwriting
        if base_name in ATLAS_REGISTRY and not overwrite:
            continue
        
        # Look for corresponding labels file
        labels_path = directory / f"{base_name}_labels.txt"
        if not labels_path.exists():
            labels_path = directory / f"{base_name}.txt"
        
        if not labels_path.exists():
            # Skip atlas without labels
            continue
        
        # Parse space and resolution from filename or use defaults
        atlas_space = space
        atlas_resolution = resolution
        
        # Try BIDS-style filename parsing
        parts = base_name.split('_')
        for part in parts:
            if part.startswith('tpl-'):
                atlas_space = part[4:]
            elif part.startswith('res-'):
                try:
                    res_str = part[4:]
                    atlas_resolution = int(res_str) if res_str.isdigit() else atlas_resolution
                except (ValueError, AttributeError):
                    pass
        
        # Fall back to header detection if needed
        if atlas_space is None or atlas_resolution is None:
            try:
                atlas_img = nib.load(nifti_path)
                detected = detect_space_from_header(atlas_img)
                if detected is not None:
                    detected_space, detected_res = detected
                    if atlas_space is None:
                        atlas_space = detected_space
                    if atlas_resolution is None:
                        atlas_resolution = detected_res
            except Exception:
                pass
        
        # Skip if we couldn't determine space/resolution
        if atlas_space is None or atlas_resolution is None:
            continue
        
        # Register the atlas
        try:
            register_atlas_from_files(
                name=base_name,
                atlas_path=nifti_path,
                labels_path=labels_path,
                space=atlas_space,
                resolution=atlas_resolution,
                description=f"Atlas from {directory.name}",
            )
            registered_names.append(base_name)
        except Exception:
            # Skip atlases that fail to register
            continue
    
    return registered_names
