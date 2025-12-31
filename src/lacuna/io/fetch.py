"""
Data fetching and caching utilities for reference datasets.

This module provides automatic downloading and caching of atlases and templates
using Pooch. Connectomes are user-managed due to size and licensing.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pooch

from ..core.exceptions import AtlasNotFoundError
from .downloaders import ConnectomeSource, FetchProgress, FetchResult


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
# NOTE: This registry is NOT FUNCTIONAL - URLs and hashes are placeholders.
# The actual parcellation files are bundled in lacuna/data/atlases/ and accessed
# via lacuna.assets.parcellations module. This registry is reserved for future
# remote hosting capability.
# TODO: Replace with actual hosting URLs when lacuna-data repository is created
ATLAS_REGISTRY = pooch.create(
    path=get_data_dir() / "atlases",
    base_url="https://github.com/lacuna/data/raw/main/atlases/",  # placeholder
    registry={
        # Placeholder entries - NOT FUNCTIONAL
        "schaefer2018-100parcels-7networks.nii.gz": "sha256:placeholder",
        "schaefer2018-100parcels-7networks_labels.txt": "sha256:placeholder",
        "schaefer2018-400parcels-7networks.nii.gz": "sha256:placeholder",
        "schaefer2018-400parcels-7networks_labels.txt": "sha256:placeholder",
    },
)


# Pooch registry for pre-registered tractograms
# NOTE: This registry is NOT FUNCTIONAL - URLs and hashes are placeholders.
# Tractograms should be registered via lacuna.assets.connectomes module.
# TODO: Replace with actual hosting URLs when lacuna-data repository is created
TRACTOGRAM_REGISTRY = pooch.create(
    path=get_data_dir() / "tractograms",
    base_url="https://github.com/lacuna/ldk-data/raw/main/tractograms/",  # placeholder
    registry={
        # Placeholder entry - NOT FUNCTIONAL
        "dTOR985.trk": "sha256:placeholder",
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
    ['harvard-oxford-cortical', 'schaefer2018-100parcels-7networks', ...]
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
        Atlas name (e.g., 'schaefer2018-100parcels-7networks') or path to .nii.gz file

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
    >>> img, labels = get_atlas("schaefer2018-100parcels-7networks")

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


# ============================================================================
# Connectome Fetching Functions
# ============================================================================


def fetch_gsp1000(
    output_dir: str | Path,
    *,
    api_key: str | None = None,
    batches: int = 10,
    register: bool = True,
    register_name: str = "GSP1000",
    force: bool = False,
    progress_callback: Callable[[FetchProgress], None] | None = None,
) -> FetchResult:
    """
    Download, process, and register the GSP1000 functional connectome.

    Downloads the Brain Genomics Superstruct Project 1000-subject resting-state
    fMRI dataset from Harvard Dataverse, converts to HDF5 batch format, and
    optionally registers for use with FunctionalNetworkMapping.

    Parameters
    ----------
    output_dir : str or Path
        Directory for output HDF5 batch files.
    api_key : str, optional
        Harvard Dataverse API key. If not provided, looks for DATAVERSE_API_KEY
        environment variable.
    batches : int, default=10
        Number of HDF5 batch files to create. More batches = lower RAM usage.
        Recommendations: 4GB RAM → 100, 8GB → 50, 16GB → 25, 32GB+ → 10.
    register : bool, default=True
        Automatically register connectome after processing.
    register_name : str, default="GSP1000"
        Name for connectome registration.
    force : bool, default=False
        Overwrite existing files and registrations.
    progress_callback : callable, optional
        Function called with FetchProgress updates during operation.

    Returns
    -------
    FetchResult
        Result containing output paths, registration status, and timing.

    Raises
    ------
    AuthenticationError
        If API key is missing or invalid.
    DownloadError
        If download fails after retries.
    ProcessingError
        If NIfTI to HDF5 conversion fails.

    Examples
    --------
    >>> from lacuna.io import fetch_gsp1000
    >>>
    >>> # Basic usage with API key
    >>> result = fetch_gsp1000(
    ...     output_dir="/data/connectomes/gsp1000",
    ...     api_key="your-dataverse-api-key",
    ...     batches=50  # for 8GB RAM systems
    ... )
    >>> print(result.summary())

    >>> # Using environment variable for API key
    >>> import os
    >>> os.environ["DATAVERSE_API_KEY"] = "your-key"
    >>> result = fetch_gsp1000("/data/connectomes/gsp1000")

    >>> # With progress tracking
    >>> def on_progress(p: FetchProgress):
    ...     print(f"{p.phase}: {p.percent_complete:.1f}%")
    >>> result = fetch_gsp1000("/data", progress_callback=on_progress)
    """
    import time

    from ..core.exceptions import AuthenticationError, DownloadError, ProcessingError
    from .convert import gsp1000_to_ldk
    from .downloaders import CONNECTOME_SOURCES, FetchProgress, FetchResult
    from .downloaders.dataverse import DataverseDownloader

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    download_time = 0.0
    processing_time = 0.0
    warnings: list[str] = []

    source = CONNECTOME_SOURCES["gsp1000"]

    # Create directories
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Phase 1: Download
        download_start = time.time()

        if progress_callback:
            progress_callback(
                FetchProgress(
                    phase="download",
                    current_file="",
                    files_completed=0,
                    files_total=1,
                    message="Initializing download...",
                )
            )

        downloader = DataverseDownloader(source, api_key=api_key)
        downloader.download(
            output_path=raw_dir,
            progress_callback=progress_callback,
        )

        download_time = time.time() - download_start

        # Phase 2: Processing (convert to HDF5 batches)
        processing_start = time.time()

        if progress_callback:
            progress_callback(
                FetchProgress(
                    phase="processing",
                    current_file="",
                    files_completed=0,
                    files_total=1,
                    message="Converting to HDF5 format...",
                )
            )

        # Calculate subjects per batch from batches count
        # GSP1000 has ~1000 subjects
        subjects_per_chunk = max(1, 1000 // batches)

        # Find brain mask (should be included in download or use template)
        mask_candidates = list(raw_dir.glob("*mask*.nii.gz")) + list(
            raw_dir.glob("*MNI152*.nii.gz")
        )
        if mask_candidates:
            mask_path = mask_candidates[0]
        else:
            # Use templateflow mask as fallback
            try:
                import templateflow.api as tflow

                mask_path = Path(
                    tflow.get(
                        "MNI152NLin6Asym",
                        resolution=2,
                        desc="brain",
                        suffix="mask",
                    )
                )
            except Exception as e:
                raise ProcessingError(
                    operation="locate brain mask",
                    reason=f"No brain mask found in download and templateflow failed: {e}",
                ) from e

        # Run conversion
        output_files = gsp1000_to_ldk(
            gsp_dir=raw_dir,
            mask_path=mask_path,
            output_dir=processed_dir,
            subjects_per_chunk=subjects_per_chunk,
            overwrite=force,
        )

        processing_time = time.time() - processing_start

        # Phase 3: Registration (if requested)
        registered = False
        if register:
            if progress_callback:
                progress_callback(
                    FetchProgress(
                        phase="registration",
                        current_file="",
                        files_completed=0,
                        files_total=1,
                        message=f"Registering as '{register_name}'...",
                    )
                )
            # TODO: Implement actual registration with registry
            # For now, just mark as "would register"
            registered = True
            warnings.append(
                "Registration placeholder: actual registry integration not yet implemented"
            )

        duration = time.time() - start_time

        return FetchResult(
            success=True,
            connectome_name="gsp1000",
            output_dir=processed_dir,
            output_files=output_files,
            registered=registered,
            register_name=register_name if registered else None,
            duration_seconds=duration,
            download_time_seconds=download_time,
            processing_time_seconds=processing_time,
            warnings=warnings,
        )

    except (AuthenticationError, DownloadError, ProcessingError):
        raise
    except Exception as e:
        raise ProcessingError(
            operation="fetch_gsp1000",
            reason=str(e),
        ) from e


def fetch_dtor985(
    output_dir: str | Path,
    *,
    keep_original: bool = True,
    register: bool = True,
    register_name: str = "dTOR985",
    force: bool = False,
    progress_callback: Callable[[FetchProgress], None] | None = None,
) -> FetchResult:
    """
    Download, convert, and register the dTOR985 structural tractogram.

    Downloads the Diffusion Tensor Imaging Open Resource 985-subject tractogram
    from Figshare in TrackVis (.trk) format, converts to MRtrix3 (.tck) format,
    and optionally registers for use with StructuralNetworkMapping.

    Parameters
    ----------
    output_dir : str or Path
        Directory for output .tck file.
    keep_original : bool, default=True
        Keep original .trk file after conversion. Set False to save disk space.
    register : bool, default=True
        Automatically register tractogram after processing.
    register_name : str, default="dTOR985"
        Name for tractogram registration.
    force : bool, default=False
        Overwrite existing files and registrations.
    progress_callback : callable, optional
        Function called with FetchProgress updates during operation.

    Returns
    -------
    FetchResult
        Result containing output path, registration status, and timing.

    Raises
    ------
    DownloadError
        If download fails (network or Cloudflare issues).
    ProcessingError
        If .trk to .tck conversion fails.

    Examples
    --------
    >>> from lacuna.io import fetch_dtor985
    >>>
    >>> # Basic usage
    >>> result = fetch_dtor985("/data/connectomes/dtor985")
    >>> print(result.output_files[0])  # Path to .tck file

    >>> # Save disk space by removing .trk after conversion
    >>> result = fetch_dtor985("/data", keep_original=False)
    """
    import time

    from ..core.exceptions import DownloadError, ProcessingError
    from .convert import trk_to_tck
    from .downloaders import CONNECTOME_SOURCES, FetchProgress, FetchResult
    from .downloaders.figshare import FigshareDownloader

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    download_time = 0.0
    processing_time = 0.0
    warnings: list[str] = []

    source = CONNECTOME_SOURCES["dtor985"]

    try:
        # Phase 1: Download
        download_start = time.time()

        if progress_callback:
            progress_callback(
                FetchProgress(
                    phase="download",
                    current_file="",
                    files_completed=0,
                    files_total=1,
                    message="Downloading dTOR985 tractogram...",
                )
            )

        downloader = FigshareDownloader(source)
        downloaded_files = downloader.download(
            output_path=output_dir,
            progress_callback=progress_callback,
        )

        if not downloaded_files:
            raise DownloadError(
                url=source.download_url or "",
                reason="No files downloaded",
            )

        trk_path = downloaded_files[0]
        download_time = time.time() - download_start

        # Phase 2: Convert to .tck format
        processing_start = time.time()

        if progress_callback:
            progress_callback(
                FetchProgress(
                    phase="processing",
                    current_file=trk_path.name,
                    files_completed=0,
                    files_total=1,
                    message="Converting to .tck format...",
                )
            )

        tck_path = trk_path.with_suffix(".tck")

        if tck_path.exists() and not force:
            warnings.append(f"Using existing .tck file: {tck_path}")
        else:
            tck_path = trk_to_tck(trk_path, tck_path)

        # Remove original if requested
        if not keep_original and trk_path.exists():
            trk_path.unlink()

        processing_time = time.time() - processing_start

        # Phase 3: Registration (if requested)
        registered = False
        if register:
            if progress_callback:
                progress_callback(
                    FetchProgress(
                        phase="registration",
                        current_file="",
                        files_completed=0,
                        files_total=1,
                        message=f"Registering as '{register_name}'...",
                    )
                )
            # TODO: Implement actual registration with registry
            registered = True
            warnings.append(
                "Registration placeholder: actual registry integration not yet implemented"
            )

        duration = time.time() - start_time

        output_files = [tck_path]
        if keep_original and trk_path.exists():
            output_files.insert(0, trk_path)

        return FetchResult(
            success=True,
            connectome_name="dtor985",
            output_dir=output_dir,
            output_files=output_files,
            registered=registered,
            register_name=register_name if registered else None,
            duration_seconds=duration,
            download_time_seconds=download_time,
            processing_time_seconds=processing_time,
            warnings=warnings,
        )

    except (DownloadError, ProcessingError):
        raise
    except Exception as e:
        raise ProcessingError(
            operation="fetch_dtor985",
            reason=str(e),
        ) from e


def fetch_connectome(
    name: str,
    output_dir: str | Path,
    **kwargs,
) -> FetchResult:
    """
    Generic fetch function that dispatches to specific connectome fetchers.

    Parameters
    ----------
    name : str
        Connectome name ('gsp1000', 'dtor985').
    output_dir : str or Path
        Directory for output files.
    **kwargs
        Additional arguments passed to specific fetch function.

    Returns
    -------
    FetchResult
        Result from the specific fetch operation.

    Raises
    ------
    ValueError
        If connectome name is not recognized.

    Examples
    --------
    >>> from lacuna.io import fetch_connectome
    >>> result = fetch_connectome("gsp1000", "/data", api_key="key", batches=50)
    """
    from .downloaders import CONNECTOME_SOURCES

    name = name.lower()

    if name not in CONNECTOME_SOURCES:
        available = ", ".join(CONNECTOME_SOURCES.keys())
        raise ValueError(f"Unknown connectome '{name}'. Available: {available}")

    if name == "gsp1000":
        return fetch_gsp1000(output_dir, **kwargs)
    elif name == "dtor985":
        return fetch_dtor985(output_dir, **kwargs)
    else:
        raise ValueError(f"No fetch implementation for '{name}'")


def list_fetchable_connectomes() -> list[ConnectomeSource]:
    """
    List all connectomes available for fetching.

    Returns
    -------
    list of ConnectomeSource
        Available connectome sources with metadata.

    Examples
    --------
    >>> from lacuna.io import list_fetchable_connectomes
    >>> for source in list_fetchable_connectomes():
    ...     print(f"{source.name}: {source.display_name}")
    ...     print(f"  Type: {source.type}, Size: ~{source.estimated_size_gb}GB")
    """
    from .downloaders import CONNECTOME_SOURCES

    return list(CONNECTOME_SOURCES.values())


def get_fetch_status(name: str) -> dict:
    """
    Get the current status of a connectome (downloaded, processed, registered).

    Parameters
    ----------
    name : str
        Connectome name ('gsp1000', 'dtor985').

    Returns
    -------
    dict
        Status information including:
        - downloaded: bool
        - processed: bool
        - registered: bool
        - location: Path | None
        - size_bytes: int | None
    """
    from .downloaders import CONNECTOME_SOURCES

    name = name.lower()
    if name not in CONNECTOME_SOURCES:
        raise ValueError(f"Unknown connectome '{name}'")

    # Check cache directory
    cache_dir = get_data_dir() / "connectomes" / name
    processed_dir = cache_dir / "processed"

    downloaded = cache_dir.exists() and any(cache_dir.iterdir())
    processed = processed_dir.exists() and any(processed_dir.iterdir())

    # Calculate size if exists
    size_bytes = None
    location = None
    if processed:
        location = processed_dir
        size_bytes = sum(f.stat().st_size for f in processed_dir.rglob("*") if f.is_file())
    elif downloaded:
        location = cache_dir
        size_bytes = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())

    return {
        "downloaded": downloaded,
        "processed": processed,
        "registered": False,  # TODO: Check actual registry
        "location": location,
        "size_bytes": size_bytes,
    }
