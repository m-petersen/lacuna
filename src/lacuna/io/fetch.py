"""
Data fetching and caching utilities for reference datasets.

This module provides automatic downloading and caching of atlases and templates
using Pooch. Connectomes are user-managed due to size and licensing.
"""

import os
from pathlib import Path

import pooch

from ..core.exceptions import AtlasNotFoundError


def get_data_dir() -> Path:
    """
    Get the data cache directory following XDG Base Directory specification.

    Priority:
    1. LACUNA_DATA_DIR environment variable (explicit user choice)
    2. XDG_CACHE_HOME/ldk (XDG standard)
    3. ~/.cache/lacuna (fallback)

    Returns
    -------
    Path
        Absolute path to data cache directory

    Examples
    --------
    >>> data_dir = get_data_dir()
    >>> print(data_dir)
    PosixPath('/home/user/.cache/lacuna')

    >>> import os
    >>> os.environ['LACUNA_DATA_DIR'] = '/mnt/nvme/ldk_data'
    >>> data_dir = get_data_dir()
    >>> print(data_dir)
    PosixPath('/mnt/nvme/ldk_data')
    """
    if env_dir := os.getenv("LACUNA_DATA_DIR"):
        return Path(env_dir).expanduser().resolve()

    if xdg_cache := os.getenv("XDG_CACHE_HOME"):
        return Path(xdg_cache) / "lacuna"

    return Path.home() / ".cache" / "lacuna"


# Pooch registry for pre-registered atlases
# TODO: Replace with actual hosting URLs when available
ATLAS_REGISTRY = pooch.create(
    path=get_data_dir() / "atlases",
    base_url="https://github.com/lacuna/ldk-data/raw/main/atlases/",
    registry={
        # Harvard-Oxford Cortical Atlas (48 regions)
        "harvard-oxford-cortical.nii.gz": "sha256:placeholder_hash_ho_cortical",
        "harvard-oxford-cortical_labels.txt": "sha256:placeholder_hash_ho_labels",
        # AAL3 Atlas (170 regions)
        "aal3.nii.gz": "sha256:placeholder_hash_aal3",
        "aal3_labels.txt": "sha256:placeholder_hash_aal3_labels",
        # Schaefer 2018 100 parcels
        "schaefer2018-100parcels-7networks.nii.gz": "sha256:placeholder_hash_schaefer100",
        "schaefer2018-100parcels-7networks_labels.txt": "sha256:placeholder_hash_schaefer100_labels",
        # Schaefer 2018 400 parcels
        "schaefer2018-400parcels-7networks.nii.gz": "sha256:placeholder_hash_schaefer400",
        "schaefer2018-400parcels-7networks_labels.txt": "sha256:placeholder_hash_schaefer400_labels",
    },
)


# Pooch registry for pre-registered tractograms
# TODO: Replace with actual hosting URLs when available
TRACTOGRAM_REGISTRY = pooch.create(
    path=get_data_dir() / "tractograms",
    base_url="https://github.com/lacuna/ldk-data/raw/main/tractograms/",
    registry={
        # dTOR985 - Default structural connectome (985 subjects)
        # Distributed in TrackVis .trk format, needs conversion to .tck
        "dTOR985.trk": "sha256:placeholder_hash_dtor985",
    },
)


def list_available_atlases() -> list[str]:
    """
    List all pre-registered atlases that can be automatically downloaded.

    Returns
    -------
    List[str]
        Names of available atlases

    Examples
    --------
    >>> atlases = list_available_atlases()
    >>> print(atlases)
    ['harvard-oxford-cortical', 'aal3', 'schaefer2018-100parcels-7networks', ...]
    """
    # Extract base names (without .nii.gz extension)
    atlas_files = [f for f in ATLAS_REGISTRY.registry.keys() if f.endswith(".nii.gz")]
    return [Path(f).stem for f in atlas_files]


def discover_atlas_files(atlas_path: Path) -> tuple[Path, Path]:
    """
    Discover atlas image and label files from a path.

    Handles both:
    - Direct path to .nii.gz file (finds paired _labels.txt)
    - Directory containing atlas files

    Parameters
    ----------
    atlas_path : Path
        Path to atlas .nii.gz file or directory

    Returns
    -------
    Tuple[Path, Path]
        (image_path, labels_path) pair

    Raises
    ------
    AtlasNotFoundError
        If atlas files cannot be found or paired

    Examples
    --------
    >>> img, labels = discover_atlas_files(Path("/path/to/custom_atlas.nii.gz"))
    >>> print(img, labels)
    /path/to/custom_atlas.nii.gz /path/to/custom_atlas_labels.txt
    """
    atlas_path = Path(atlas_path)

    if atlas_path.is_file() and atlas_path.suffix == ".gz":
        # Direct path to .nii.gz file
        img_path = atlas_path

        # Try to find paired label file
        label_candidates = [
            atlas_path.parent / f"{atlas_path.stem.replace('.nii', '')}_labels.txt",
            atlas_path.parent / f"{atlas_path.stem.replace('.nii', '')}.txt",
            atlas_path.with_suffix(".txt"),
        ]

        for label_path in label_candidates:
            if label_path.exists():
                return img_path, label_path

        raise AtlasNotFoundError(
            f"Could not find label file for atlas: {img_path}\n"
            f"Tried: {[str(p) for p in label_candidates]}"
        )

    elif atlas_path.is_dir():
        # Directory - find .nii.gz and matching .txt
        nifti_files = list(atlas_path.glob("*.nii.gz"))

        if not nifti_files:
            raise AtlasNotFoundError(f"No .nii.gz files found in directory: {atlas_path}")

        if len(nifti_files) > 1:
            raise AtlasNotFoundError(
                f"Multiple .nii.gz files found in {atlas_path}. "
                "Please specify the exact atlas file."
            )

        return discover_atlas_files(nifti_files[0])

    else:
        raise AtlasNotFoundError(
            f"Atlas path does not exist or is not a file/directory: {atlas_path}"
        )


def _fetch_registered_atlas(name: str) -> tuple[Path, Path]:
    """
    Fetch a pre-registered atlas from the Pooch registry.

    Downloads if not cached, verifies checksums.

    Parameters
    ----------
    name : str
        Atlas name (without .nii.gz extension)

    Returns
    -------
    Tuple[Path, Path]
        (image_path, labels_path) pair

    Raises
    ------
    AtlasNotFoundError
        If atlas is not in registry
    """
    img_filename = f"{name}.nii.gz"
    labels_filename = f"{name}_labels.txt"

    if img_filename not in ATLAS_REGISTRY.registry:
        raise AtlasNotFoundError(
            f"Atlas '{name}' is not in the pre-registered catalog.\n"
            f"Available atlases: {list_available_atlases()}"
        )

    # Fetch both files (downloads if missing, uses cache otherwise)
    img_path = Path(ATLAS_REGISTRY.fetch(img_filename, progressbar=True))
    labels_path = Path(ATLAS_REGISTRY.fetch(labels_filename, progressbar=True))

    return img_path, labels_path


def _scan_atlas_directory(atlas_dir: Path, pattern: str) -> tuple[Path, Path] | None:
    """
    Scan a directory for atlas files matching a pattern.

    Parameters
    ----------
    atlas_dir : Path
        Directory to scan
    pattern : str
        Atlas name pattern to match

    Returns
    -------
    Optional[Tuple[Path, Path]]
        (image_path, labels_path) if found, None otherwise
    """
    # Try exact match
    img_candidates = [
        atlas_dir / f"{pattern}.nii.gz",
        atlas_dir / pattern / f"{pattern}.nii.gz",
    ]

    for img_path in img_candidates:
        if img_path.exists():
            try:
                return discover_atlas_files(img_path)
            except AtlasNotFoundError:
                continue

    # Try glob pattern
    matches = list(atlas_dir.glob(f"*{pattern}*.nii.gz"))
    if matches:
        try:
            return discover_atlas_files(matches[0])
        except AtlasNotFoundError:
            pass

    return None


def get_atlas(name_or_path: str) -> tuple[Path, Path]:
    """
    Get atlas files with automatic download or discovery.

    Resolution order:
    1. If path exists → use directly (custom atlas)
    2. If in pre-registered catalog → fetch via Pooch (download if missing)
    3. If LACUNA_ATLAS_DIR set → scan directory for matching files
    4. Else → raise AtlasNotFoundError

    Parameters
    ----------
    name_or_path : str
        Atlas name (e.g., 'aal3') or path to .nii.gz file

    Returns
    -------
    Tuple[Path, Path]
        (image_path, labels_path) pair

    Raises
    ------
    AtlasNotFoundError
        If atlas cannot be found through any resolution method

    Examples
    --------
    >>> # Pre-registered atlas (downloads if needed)
    >>> img, labels = get_atlas("aal3")

    >>> # Custom atlas file
    >>> img, labels = get_atlas("/path/to/custom_atlas.nii.gz")

    >>> # Atlas in custom directory
    >>> import os
    >>> os.environ['LACUNA_ATLAS_DIR'] = '/lab/atlases'
    >>> img, labels = get_atlas("my_custom_atlas")
    """
    # 1. Check if it's a path that exists
    if Path(name_or_path).exists():
        return discover_atlas_files(Path(name_or_path))

    # 2. Try pre-registered catalog
    if name_or_path in list_available_atlases():
        return _fetch_registered_atlas(name_or_path)

    # 3. Check custom atlas directory
    if atlas_dir := os.getenv("LACUNA_ATLAS_DIR"):
        result = _scan_atlas_directory(Path(atlas_dir), name_or_path)
        if result:
            return result

    # 4. Not found anywhere
    raise AtlasNotFoundError(
        f"Atlas '{name_or_path}' not found.\n"
        f"Available pre-registered atlases: {list_available_atlases()}\n"
        f"You can:\n"
        f"  - Provide a direct path to a .nii.gz file\n"
        f"  - Set LACUNA_ATLAS_DIR to search a custom directory\n"
        f"  - Check spelling of atlas name"
    )


def get_connectome_path(name_or_path: str) -> Path:
    """
    Resolve connectome file path.

    Resolution order:
    1. If path exists → use directly
    2. If LACUNA_CONNECTOME_DIR set → look for named file
    3. Else → raise ConnectomeNotFoundError with setup instructions

    Parameters
    ----------
    name_or_path : str
        Connectome name or path to HDF5 file

    Returns
    -------
    Path
        Path to connectome HDF5 file

    Raises
    ------
    FileNotFoundError
        If connectome cannot be found

    Examples
    --------
    >>> # Direct path
    >>> path = get_connectome_path("/data/gsp1000_chunk_000.h5")

    >>> # Named connectome in custom directory
    >>> import os
    >>> os.environ['LACUNA_CONNECTOME_DIR'] = '/data/connectomes'
    >>> path = get_connectome_path("gsp1000_chunk_000")
    """
    from ..core.exceptions import ConnectomeNotFoundError

    # 1. Direct path
    if Path(name_or_path).exists():
        return Path(name_or_path)

    # 2. Check connectome directory
    if connectome_dir := os.getenv("LACUNA_CONNECTOME_DIR"):
        candidates = [
            Path(connectome_dir) / name_or_path,
            Path(connectome_dir) / f"{name_or_path}.h5",
            Path(connectome_dir) / f"{name_or_path}.hdf5",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

    # 3. Not found - provide helpful error
    raise ConnectomeNotFoundError(
        f"Connectome '{name_or_path}' not found.\n\n"
        "Connectomes are not automatically downloaded due to size and licensing.\n"
        "To prepare your connectome:\n\n"
        "1. Download GSP1000 from Harvard Dataverse:\n"
        "   https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ILXIKS\n\n"
        "2. Convert to LDK format:\n"
        "   ldk convert gsp1000 /path/to/raw /output/dir\n\n"
        "3. Set environment variable:\n"
        "   export LACUNA_CONNECTOME_DIR=/output/dir\n\n"
        "Or provide direct path: FunctionalNetworkMapping(connectome_path='/path/to/file.h5')"
    )


def get_tractogram(name: str = "dTOR985", *, convert_to_tck: bool = True) -> Path:
    """
    Get structural tractogram with automatic download and conversion.

    The default tractogram is dTOR985 (985 subjects), which is automatically
    downloaded in TrackVis .trk format and optionally converted to MRtrix3 .tck
    format for use with StructuralNetworkMapping.

    Parameters
    ----------
    name : str, default="dTOR985"
        Tractogram name (currently only "dTOR985" supported)
    convert_to_tck : bool, default=True
        Whether to convert .trk to .tck format using MRtrix3 tckconvert
        Required for StructuralNetworkMapping

    Returns
    -------
    Path
        Path to tractogram file (.tck if converted, .trk otherwise)

    Raises
    ------
    ValueError
        If tractogram name not recognized
    RuntimeError
        If MRtrix3 not available and convert_to_tck=True

    Examples
    --------
    >>> # Get dTOR985 as .tck (default, ready for analysis)
    >>> tck_path = get_tractogram("dTOR985")
    >>> analysis = StructuralNetworkMapping(tractogram_path=tck_path)

    >>> # Get raw .trk file without conversion
    >>> trk_path = get_tractogram("dTOR985", convert_to_tck=False)

    Notes
    -----
    - First call downloads ~2GB .trk file from repository
    - Conversion to .tck creates ~5-10GB file (MRtrix3 format)
    - Files cached in ~/.cache/lacuna/tractograms/
    - Conversion requires MRtrix3 installation
    """
    if name not in TRACTOGRAM_REGISTRY.registry:
        raise ValueError(
            f"Tractogram '{name}' not recognized.\n"
            f"Available tractograms: {list(TRACTOGRAM_REGISTRY.registry.keys())}"
        )

    # Fetch .trk file (downloads if needed, uses cache otherwise)
    trk_filename = f"{name}.trk"
    trk_path = Path(TRACTOGRAM_REGISTRY.fetch(trk_filename, progressbar=True))

    if not convert_to_tck:
        return trk_path

    # Convert to .tck format
    tck_path = trk_path.parent / f"{name}.tck"

    if tck_path.exists():
        print(f"Using cached .tck file: {tck_path}")
        return tck_path

    print(f"Converting {name} to MRtrix3 .tck format...")

    # Import here to avoid circular import
    from .convert import trk_to_tck

    try:
        return trk_to_tck(trk_path, tck_path)
    except RuntimeError as e:
        print(
            f"\nWARNING: Could not convert to .tck format: {e}\n"
            f"Returning .trk file. You'll need to convert manually:\n"
            f"  ldk.io.trk_to_tck('{trk_path}', '{tck_path}')"
        )
        return trk_path
