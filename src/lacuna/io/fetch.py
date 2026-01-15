"""
Data fetching and caching utilities for connectomes and tractograms.

This module provides automatic downloading, conversion, and caching of
connectome datasets (GSP1000, dTOR985) for use with Lacuna analyses.

Atlases are bundled in the package and accessed via `lacuna.assets.parcellations`.
"""

from __future__ import annotations

import os
import time
import warnings
from collections.abc import Callable
from pathlib import Path

from ..core.exceptions import AtlasNotFoundError
from .downloaders import ConnectomeSource, FetchProgress, FetchResult


def get_data_dir() -> Path:
    """
    Get the data cache directory following XDG Base Directory specification.

    Priority:
    1. LACUNA_DATA_DIR environment variable (explicit user choice)
    2. XDG_CACHE_HOME/lacuna (XDG standard)
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
    >>> os.environ['LACUNA_DATA_DIR'] = '/mnt/nvme/lacuna_data'
    >>> data_dir = get_data_dir()
    >>> print(data_dir)
    PosixPath('/mnt/nvme/lacuna_data')
    """
    if env_dir := os.getenv("LACUNA_DATA_DIR"):
        return Path(env_dir).expanduser().resolve()

    if xdg_cache := os.getenv("XDG_CACHE_HOME"):
        return Path(xdg_cache) / "lacuna"

    return Path.home() / ".cache" / "lacuna"


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


def get_connectome_path(name_or_path: str) -> Path:
    """
    Resolve a connectome name or path to its file location.

    For registered connectomes, looks up path in registry.
    For paths, validates existence.

    Parameters
    ----------
    name_or_path : str
        Either a registered connectome name (e.g., "GSP1000") or
        a direct path to .h5 file or directory.

    Returns
    -------
    Path
        Resolved path to connectome data.

    Raises
    ------
    FileNotFoundError
        If connectome cannot be resolved.

    Examples
    --------
    >>> path = get_connectome_path("GSP1000")  # Registered name
    >>> path = get_connectome_path("/data/my_connectome.h5")  # Direct path
    """
    # Check if it's a path
    path = Path(name_or_path)
    if path.exists():
        return path

    # Try looking up in registry
    try:
        from ..assets.connectomes import get_functional_connectome

        return get_functional_connectome(name_or_path).data_path
    except (ImportError, KeyError, AttributeError):
        pass

    # Check cache directory
    cache_dir = get_data_dir() / "connectomes"
    candidates = [
        cache_dir / name_or_path,
        cache_dir / name_or_path.lower(),
        cache_dir / f"{name_or_path}.h5",
        cache_dir / f"{name_or_path.lower()}.h5",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Connectome '{name_or_path}' not found.\n"
        "Options:\n"
        "  - Provide a direct path to an existing .h5 file or directory\n"
        "  - Register a connectome using lacuna.assets.connectomes\n"
        "  - Download using: lacuna.io.fetch_gsp1000() or fetch_dtor985()\n\n"
        "Quick start:\n"
        "1. Get API key from https://dataverse.harvard.edu/\n"
        "2. Run:\n"
        "   lacuna fetch gsp1000 /path/to/output --api-key YOUR_KEY\n\n"
        "Or in Python:\n"
        "   from lacuna.io import fetch_gsp1000\n"
        "   fetch_gsp1000('/path/to/output', api_key='YOUR_KEY')"
    )


# ============================================================================
# Connectome Fetching Functions
# ============================================================================


def fetch_gsp1000(
    output_dir: str | Path,
    *,
    api_key: str | None = None,
    batches: int = 10,
    test_mode: bool = False,
    skip_checksum: bool = False,
    register: bool = True,
    register_name: str = "GSP1000",
    force: bool = False,
    progress_callback: Callable[[FetchProgress], None] | None = None,
    verbose: bool = False,
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
    test_mode : bool, default=False
        If True, downloads only 1 tarball (~2GB) to test the full pipeline.
    skip_checksum : bool, default=False
        Skip checksum verification. Use when Dataverse metadata is outdated.
    register : bool, default=True
        Automatically register connectome after processing.
    register_name : str, default="GSP1000"
        Name for connectome registration.
    force : bool, default=False
        Overwrite existing files and registrations.
    progress_callback : callable, optional
        Function called with FetchProgress updates during operation.
    verbose : bool, default=False
        Print informational messages.

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
    >>> result = fetch_gsp1000(
    ...     output_dir="/data/connectomes/gsp1000",
    ...     api_key="your-dataverse-api-key",
    ...     batches=50
    ... )
    >>> print(result.summary())
    """
    from ..core.exceptions import AuthenticationError, DownloadError, ProcessingError
    from .convert import gsp1000_to_hdf5
    from .downloaders import CONNECTOME_SOURCES
    from .downloaders.dataverse import DataverseDownloader

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    download_time = 0.0
    processing_time = 0.0
    warn_list: list[str] = []

    source = CONNECTOME_SOURCES["gsp1000"]

    # Create directories
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Check if processed files already exist
    existing_hdf5 = list(processed_dir.glob("*.h5")) + list(processed_dir.glob("*.hdf5"))
    if existing_hdf5 and not force:
        if verbose:
            print(f"Using existing HDF5 files: {processed_dir} ({len(existing_hdf5)} files)")
        warn_list.append(f"Using existing HDF5 files: {processed_dir}")

        # Skip to registration phase
        registered = _register_gsp1000(
            register, register_name, source, processed_dir, progress_callback, warn_list
        )

        return FetchResult(
            success=True,
            connectome_name="gsp1000",
            output_dir=processed_dir,
            output_files=existing_hdf5,
            registered=registered,
            register_name=register_name if registered else None,
            duration_seconds=time.time() - start_time,
            download_time_seconds=0.0,
            processing_time_seconds=0.0,
            warnings=warn_list,
        )

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
            test_mode=test_mode,
            skip_checksum=skip_checksum,
        )

        download_time = time.time() - download_start

        # Phase 2: Extract tarballs
        if progress_callback:
            progress_callback(
                FetchProgress(
                    phase="processing",
                    current_file="",
                    files_completed=0,
                    files_total=1,
                    message="Extracting tarballs...",
                )
            )

        import tarfile

        tar_files = list(raw_dir.glob("*.tar"))
        for tar_path in tar_files:
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=raw_dir)

        # Phase 3: Convert to HDF5
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

        subjects_per_chunk = 100 if test_mode else max(1, 1000 // batches)
        if test_mode:
            warn_list.append("Test mode: using 1 tarball with minimal batching")

        # Find brain mask
        mask_path = _find_brain_mask(raw_dir)

        # Run conversion
        output_files = gsp1000_to_hdf5(
            gsp_dir=raw_dir,
            mask_path=mask_path,
            output_dir=processed_dir,
            subjects_per_chunk=subjects_per_chunk,
            overwrite=force,
        )

        processing_time = time.time() - processing_start

        # Phase 4: Registration
        registered = _register_gsp1000(
            register, register_name, source, processed_dir, progress_callback, warn_list
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
            warnings=warn_list,
        )

    except (AuthenticationError, DownloadError, ProcessingError):
        raise
    except Exception as e:
        raise ProcessingError(operation="fetch_gsp1000", reason=str(e)) from e


def _find_brain_mask(raw_dir: Path) -> Path:
    """Find brain mask from download or templateflow."""
    from ..core.exceptions import ProcessingError

    mask_candidates = list(raw_dir.glob("*mask*.nii.gz")) + list(raw_dir.glob("*MNI152*.nii.gz"))
    if mask_candidates:
        return mask_candidates[0]

    # Use templateflow mask as fallback
    try:
        import templateflow.api as tflow

        return Path(tflow.get("MNI152NLin6Asym", resolution=2, desc="brain", suffix="mask"))
    except Exception as e:
        raise ProcessingError(
            operation="locate brain mask",
            reason=f"No brain mask found in download and templateflow failed: {e}",
        ) from e


def _register_gsp1000(
    register: bool,
    register_name: str,
    source,
    processed_dir: Path,
    progress_callback: Callable | None,
    warn_list: list[str],
) -> bool:
    """Register GSP1000 connectome."""
    if not register:
        return False

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
    try:
        from ..assets.connectomes import register_functional_connectome

        register_functional_connectome(
            name=register_name,
            space=source.space,
            resolution=2.0,
            data_path=processed_dir,
            n_subjects=source.n_subjects,
            description=source.description or "Downloaded via fetch_gsp1000",
        )
        return True
    except Exception as e:
        warn_list.append(f"Registration failed: {e}")
        return False


def fetch_dtor985(
    output_dir: str | Path,
    *,
    api_key: str | None = None,
    keep_original: bool = True,
    register: bool = True,
    register_name: str = "dTOR985",
    force: bool = False,
    progress_callback: Callable[[FetchProgress], None] | None = None,
    verbose: bool = False,
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
    api_key : str, optional
        Figshare API key for authenticated downloads. If not provided,
        uses FIGSHARE_API_KEY environment variable. Get one from
        https://figshare.com/account/applications.
    keep_original : bool, default=True
        Keep original .trk file after conversion.
    register : bool, default=True
        Automatically register tractogram after processing.
    register_name : str, default="dTOR985"
        Name for tractogram registration.
    force : bool, default=False
        Overwrite existing files and registrations.
    progress_callback : callable, optional
        Function called with FetchProgress updates during operation.
    verbose : bool, default=False
        Print informational messages.

    Returns
    -------
    FetchResult
        Result containing output path, registration status, and timing.

    Raises
    ------
    DownloadError
        If download fails or API key is missing.
    ProcessingError
        If .trk to .tck conversion fails.

    Examples
    --------
    >>> from lacuna.io import fetch_dtor985
    >>> result = fetch_dtor985("/data/connectomes/dtor985", api_key="YOUR_TOKEN")
    >>> print(result.output_files[0])  # Path to .tck file
    """
    from ..core.exceptions import DownloadError, ProcessingError
    from .convert import trk_to_tck
    from .downloaders import CONNECTOME_SOURCES
    from .downloaders.figshare import FigshareDownloader

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    download_time = 0.0
    processing_time = 0.0
    warn_list: list[str] = []

    source = CONNECTOME_SOURCES["dtor985"]

    # Check if .tck already exists
    tck_path = output_dir / f"{source.name}.tck"
    trk_path = output_dir / f"{source.name}.trk"

    if tck_path.exists() and not force:
        if verbose:
            print(f"Using existing .tck file: {tck_path}")
        warn_list.append(f"Using existing .tck file: {tck_path}")

        registered = _register_dtor985(
            register, register_name, source, tck_path, progress_callback, warn_list
        )

        return FetchResult(
            success=True,
            connectome_name="dtor985",
            output_dir=output_dir,
            output_files=[tck_path],
            registered=registered,
            register_name=register_name if registered else None,
            duration_seconds=time.time() - start_time,
            download_time_seconds=0.0,
            processing_time_seconds=0.0,
            warnings=warn_list,
        )

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

        downloader = FigshareDownloader(source, api_key=api_key)
        downloaded_files = downloader.download(
            output_path=output_dir,
            progress_callback=progress_callback,
        )

        if not downloaded_files:
            raise DownloadError(url=source.download_url or "", reason="No files downloaded")

        trk_path = downloaded_files[0]
        download_time = time.time() - download_start

        # Phase 2: Convert to .tck
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
            if verbose:
                print(f"Using existing .tck file: {tck_path}")
            warn_list.append(f"Using existing .tck file: {tck_path}")
        else:
            tck_path = trk_to_tck(trk_path, tck_path)

        if not keep_original and trk_path.exists():
            trk_path.unlink()

        processing_time = time.time() - processing_start

        # Phase 3: Registration
        registered = _register_dtor985(
            register, register_name, source, tck_path, progress_callback, warn_list
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
            warnings=warn_list,
        )

    except (DownloadError, ProcessingError):
        raise
    except Exception as e:
        raise ProcessingError(operation="fetch_dtor985", reason=str(e)) from e


def _register_dtor985(
    register: bool,
    register_name: str,
    source,
    tck_path: Path,
    progress_callback: Callable | None,
    warn_list: list[str],
) -> bool:
    """Register dTOR985 tractogram."""
    if not register:
        return False

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
    try:
        from ..assets.connectomes import register_structural_connectome

        register_structural_connectome(
            name=register_name,
            space=source.space,
            tractogram_path=tck_path,
            n_subjects=source.n_subjects,
            description=source.description or "Downloaded via fetch_dtor985",
        )
        return True
    except Exception as e:
        warn_list.append(f"Registration failed: {e}")
        return False


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


# ============================================================================
# Deprecated Functions
# ============================================================================


def list_available_atlases() -> list[str]:
    """
    List available atlases.

    .. deprecated::
        Use ``lacuna.assets.parcellations.list_parcellations()`` instead.
        Atlases are bundled in the package, not fetched remotely.

    Returns
    -------
    list[str]
        List of bundled atlas names.
    """
    warnings.warn(
        "list_available_atlases() is deprecated. "
        "Use lacuna.assets.parcellations.list_parcellations() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from ..assets.parcellations import list_parcellations

        return [p.name for p in list_parcellations()]
    except ImportError:
        return []


def get_atlas(name_or_path: str) -> tuple[Path, Path]:
    """
    Get atlas files.

    .. deprecated::
        Use ``lacuna.assets.parcellations.load_parcellation()`` instead.
        Atlases are bundled in the package, not fetched remotely.

    Parameters
    ----------
    name_or_path : str
        Atlas name or path.

    Returns
    -------
    tuple[Path, Path]
        (image_path, labels_path) pair.
    """
    warnings.warn(
        "get_atlas() is deprecated. "
        "Use lacuna.assets.parcellations.load_parcellation() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Try as path first
    path = Path(name_or_path)
    if path.exists():
        return discover_atlas_files(path)

    # Try bundled atlases
    try:
        from ..assets.parcellations import load_parcellation

        parc = load_parcellation(name_or_path)
        return parc.image_path, parc.labels_path
    except (ImportError, KeyError) as e:
        raise AtlasNotFoundError(
            f"Atlas '{name_or_path}' not found. "
            "Use lacuna.assets.parcellations.list_parcellations() to see available atlases."
        ) from e


def get_tractogram(name: str = "dTOR985", *, convert_to_tck: bool = True) -> Path:
    """
    Get structural tractogram.

    .. deprecated::
        Use ``lacuna.io.fetch_dtor985()`` instead.

    Parameters
    ----------
    name : str
        Tractogram name.
    convert_to_tck : bool
        Convert to .tck format.

    Returns
    -------
    Path
        Path to tractogram file.
    """
    warnings.warn(
        "get_tractogram() is deprecated. Use lacuna.io.fetch_dtor985() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    cache_dir = get_data_dir() / "tractograms"
    tck_path = cache_dir / f"{name}.tck"
    trk_path = cache_dir / f"{name}.trk"

    if tck_path.exists() and convert_to_tck:
        return tck_path
    elif trk_path.exists():
        if convert_to_tck:
            from .convert import trk_to_tck

            return trk_to_tck(trk_path, tck_path)
        return trk_path

    raise FileNotFoundError(
        f"Tractogram '{name}' not found. Use lacuna.io.fetch_dtor985() to download it."
    )
