"""
Data fetching and caching utilities for connectomes and tractograms.

This module provides automatic downloading, conversion, and caching of
connectome datasets (GSP1000, dTOR985) for use with Lacuna analyses.

Atlases are bundled in the package and accessed via `lacuna.assets.parcellations`.
"""

from __future__ import annotations

import json
import os
import time
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
    stale_test_data = False
    existing_hdf5 = list(processed_dir.glob("*.h5")) + list(processed_dir.glob("*.hdf5"))
    if existing_hdf5 and not force:
        # Detect stale test-mode data: single chunk with ≤10 subjects
        stale_test_data = False
        if not test_mode and len(existing_hdf5) == 1:
            try:
                import h5py

                with h5py.File(existing_hdf5[0], "r") as hf:
                    if hf.attrs.get("n_subjects", 0) <= 10:
                        stale_test_data = True
            except Exception:
                pass

        if stale_test_data:
            if verbose:
                print(
                    "Existing HDF5 appears to be from test mode " "— overwriting with full dataset"
                )
            warn_list.append("Overwriting stale test-mode HDF5 data")
        else:
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

        if test_mode:
            subjects_per_chunk = 10
            max_subjects = 10
            warn_list.append("Test mode: using first 10 subjects in single chunk")
        else:
            subjects_per_chunk = max(1, 1000 // batches)
            max_subjects = None

        # Find brain mask
        mask_path = _find_brain_mask(raw_dir)

        # Run conversion (overwrite if force or stale test-mode data detected)
        output_files = gsp1000_to_hdf5(
            gsp_dir=raw_dir,
            mask_path=mask_path,
            output_dir=processed_dir,
            subjects_per_chunk=subjects_per_chunk,
            max_subjects=max_subjects,
            overwrite=force or stale_test_data,
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
            description=source.description
            or f"Downloaded via fetch_dtor985 ({source.n_subjects} subjects)",
        )
        return True
    except Exception as e:
        warn_list.append(f"Registration failed: {e}")
        return False


def fetch_hcp1065(
    output_dir: str | Path,
    *,
    keep_original: bool = True,
    register: bool = True,
    register_name: str = "HCP1065",
    force: bool = False,
    progress_callback: Callable[[FetchProgress], None] | None = None,
    verbose: bool = False,
) -> FetchResult:
    """
    Download, merge, and register the HCP1065 structural tractogram.

    Downloads the Human Connectome Project 1065-subject averaged tractography
    atlas from GitHub Releases as a zip of TrackVis (.trk) files, merges all
    tract files (excluding cranial nerves) into a single MRtrix3 (.tck) file,
    and optionally registers for use with StructuralNetworkMapping.

    Parameters
    ----------
    output_dir : str or Path
        Directory for output .tck file.
    keep_original : bool, default=True
        Keep original .zip file and extracted tracts after merging.
    register : bool, default=True
        Automatically register tractogram after processing.
    register_name : str, default="HCP1065"
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
        If download fails.
    ProcessingError
        If extraction or merging fails.

    Examples
    --------
    >>> from lacuna.io import fetch_hcp1065
    >>> result = fetch_hcp1065("/data/connectomes/hcp1065")
    >>> print(result.output_files[0])  # Path to .tck file
    """
    from ..core.exceptions import DownloadError, ProcessingError
    from .convert import merge_trk_to_tck
    from .downloaders import CONNECTOME_SOURCES
    from .downloaders.github import GithubReleaseDownloader

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    download_time = 0.0
    processing_time = 0.0
    warn_list: list[str] = []

    source = CONNECTOME_SOURCES["hcp1065"]

    # Check if .tck already exists
    tck_path = output_dir / f"{source.name}.tck"

    if tck_path.exists() and not force:
        if verbose:
            print(f"Using existing .tck file: {tck_path}")
        warn_list.append(f"Using existing .tck file: {tck_path}")

        registered = _register_hcp1065(
            register, register_name, source, tck_path, progress_callback, warn_list
        )

        return FetchResult(
            success=True,
            connectome_name="hcp1065",
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
        # Phase 1: Download zip
        download_start = time.time()

        if progress_callback:
            progress_callback(
                FetchProgress(
                    phase="download",
                    current_file="",
                    files_completed=0,
                    files_total=1,
                    message="Downloading HCP1065 tractography atlas...",
                )
            )

        downloader = GithubReleaseDownloader(source)
        downloaded_files = downloader.download(
            output_path=output_dir,
            progress_callback=progress_callback,
        )

        if not downloaded_files:
            raise DownloadError(url=source.download_url or "", reason="No files downloaded")

        zip_path = downloaded_files[0]
        download_time = time.time() - download_start

        # Phase 2: Extract zip
        processing_start = time.time()

        if progress_callback:
            progress_callback(
                FetchProgress(
                    phase="processing",
                    current_file=zip_path.name,
                    files_completed=0,
                    files_total=1,
                    message="Extracting tract files...",
                )
            )

        import zipfile

        extract_dir = output_dir / "hcp1065_tracts"
        if not extract_dir.exists() or not any(extract_dir.iterdir()) or force:
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

        # Phase 3: Merge .trk files to single .tck
        if progress_callback:
            progress_callback(
                FetchProgress(
                    phase="processing",
                    current_file="",
                    files_completed=0,
                    files_total=1,
                    message="Merging tract files to .tck format...",
                )
            )

        if tck_path.exists() and not force:
            if verbose:
                print(f"Using existing .tck file: {tck_path}")
            warn_list.append(f"Using existing .tck file: {tck_path}")
        else:
            tck_path = merge_trk_to_tck(
                source_dir=extract_dir,
                output_path=tck_path,
                overwrite=force,
            )

        # Cleanup originals if requested
        if not keep_original:
            import shutil

            if zip_path.exists():
                zip_path.unlink()
            if extract_dir.exists():
                shutil.rmtree(extract_dir)

        processing_time = time.time() - processing_start

        # Phase 4: Registration
        registered = _register_hcp1065(
            register, register_name, source, tck_path, progress_callback, warn_list
        )

        duration = time.time() - start_time

        output_files = [tck_path]
        if keep_original and zip_path.exists():
            output_files.insert(0, zip_path)

        return FetchResult(
            success=True,
            connectome_name="hcp1065",
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
        raise ProcessingError(operation="fetch_hcp1065", reason=str(e)) from e


def _register_hcp1065(
    register: bool,
    register_name: str,
    source,
    tck_path: Path,
    progress_callback: Callable | None,
    warn_list: list[str],
) -> bool:
    """Register HCP1065 tractogram."""
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
            description=source.description
            or f"Downloaded via fetch_hcp1065 ({source.n_subjects} subjects)",
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
    elif name == "hcp1065":
        return fetch_hcp1065(output_dir, **kwargs)
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
# Neurotransmitter Atlas Fetching
# ============================================================================


def fetch_ntatlas(
    output_dir: str | Path,
    *,
    force: bool = False,
    progress_callback: Callable[[FetchProgress], None] | None = None,
) -> FetchResult:
    """Download and prepare the NT PET atlas from NiSpace-data.

    Downloads the curated representative maps (one recommended map per
    target) in MNI152NLin6Asym space at the pinned NiSpace-data commit,
    verifies each against ``file_hashes.json``, then z-scores and saves
    the resulting :class:`~lacuna.atlas.types.VoxelAtlas` directly to
    ``output_dir``. The output is a ready-to-use atlas cache:
    ``output_dir/manifest.json`` + ``output_dir/maps/<target>.nii.gz``.

    Parameters
    ----------
    output_dir : str or Path
        Directory where the prepared atlas is written.
    force : bool, default=False
        Re-download and rebuild even if a manifest already exists.
    progress_callback : callable, optional
        Called with ``FetchProgress`` updates per file.

    Raises
    ------
    DownloadError
        On download failure or SHA-256 mismatch.
    """
    import tempfile
    import urllib.error
    import urllib.request

    from lacuna.atlas.store import build_nt_atlas, save_atlas
    from lacuna.core.exceptions import DownloadError
    from lacuna.data.ntatlas import all_map_ids, hashes_url, map_rel_path, map_url

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    if not force and (output_dir / "manifest.json").exists():
        return FetchResult(
            success=True,
            connectome_name="ntatlas",
            output_dir=output_dir,
            output_files=sorted((output_dir / "maps").glob("*.nii.gz")),
            duration_seconds=time.time() - start_time,
            warnings=["Existing atlas reused. Use force=True to re-download."],
        )

    try:
        with urllib.request.urlopen(hashes_url(), timeout=60) as resp:
            file_hashes = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise DownloadError(hashes_url(), f"Failed to fetch file_hashes.json: {e}") from e

    map_ids = all_map_ids()
    with tempfile.TemporaryDirectory(prefix="lacuna_ntatlas_") as raw_dir_str:
        raw_dir = Path(raw_dir_str)
        for idx, map_id in enumerate(map_ids):
            url = map_url(map_id)
            filename = f"{map_id}_space-MNI152NLin6Asym_desc-proc.nii.gz"
            out_path = raw_dir / filename
            expected_hash = file_hashes[map_rel_path(map_id)]

            if progress_callback is not None:
                progress_callback(
                    FetchProgress(
                        phase="download",
                        current_file=filename,
                        files_completed=idx,
                        files_total=len(map_ids),
                    )
                )

            try:
                urllib.request.urlretrieve(url, out_path)
            except urllib.error.URLError as e:
                raise DownloadError(url, f"Failed to download: {e}") from e
            if _sha256(out_path) != expected_hash:
                raise DownloadError(url, f"SHA-256 mismatch for {filename}")

        atlas = build_nt_atlas(raw_dir)

    save_atlas(atlas, output_dir)

    duration = time.time() - start_time
    return FetchResult(
        success=True,
        connectome_name="ntatlas",
        output_dir=output_dir,
        output_files=sorted((output_dir / "maps").glob("*.nii.gz")),
        duration_seconds=duration,
        download_time_seconds=duration,
    )


def _sha256(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
