"""
Unified cache and temp directory management for Lacuna.

Provides consistent cache and temp locations that can be configured via
environment variables. This is particularly important for HPC environments
where /tmp may not be writable or has limited space.

Environment Variables
---------------------
LACUNA_CACHE_DIR : str
    Base directory for all Lacuna cache files (downloads, transforms, etc.)
    Default: ~/.cache/lacuna (Linux/macOS), %LOCALAPPDATA%/lacuna/cache (Windows)

LACUNA_TEMP_DIR : str
    Base directory for temporary files created during analysis.
    Default: Falls back to LACUNA_CACHE_DIR/tmp if not set.
    This is useful for HPC systems where /tmp is not writable.
"""

import os
import tempfile
from pathlib import Path


def get_cache_dir() -> Path:
    """Get the Lacuna cache directory.

    The cache directory can be configured via the LACUNA_CACHE_DIR environment
    variable. If not set, defaults to:
    - $XDG_CACHE_HOME/lacuna on Linux/macOS (typically ~/.cache/lacuna)
    - %LOCALAPPDATA%/lacuna/cache on Windows
    - /tmp/lacuna_cache as fallback

    Returns
    -------
    Path
        Path to cache directory (created if doesn't exist)

    Examples
    --------
    Configure custom cache location:

    >>> import os
    >>> os.environ['LACUNA_CACHE_DIR'] = '/path/to/my/cache'
    >>> from lacuna.utils.cache import get_cache_dir
    >>> cache_dir = get_cache_dir()
    """
    # Check environment variable first
    if cache_dir_env := os.environ.get("LACUNA_CACHE_DIR"):
        cache_dir = Path(cache_dir_env)
    else:
        # Platform-specific default
        if os.name == "nt":  # Windows
            # Use %LOCALAPPDATA%/lacuna/cache
            local_app_data = os.environ.get("LOCALAPPDATA")
            if local_app_data:
                cache_dir = Path(local_app_data) / "lacuna" / "cache"
            else:
                cache_dir = Path.home() / "AppData" / "Local" / "lacuna" / "cache"
        else:  # Linux/macOS
            # Use $XDG_CACHE_HOME/lacuna or ~/.cache/lacuna
            xdg_cache = os.environ.get("XDG_CACHE_HOME")
            if xdg_cache:
                cache_dir = Path(xdg_cache) / "lacuna"
            else:
                cache_dir = Path.home() / ".cache" / "lacuna"

    # Create directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def get_tdi_cache_dir() -> Path:
    """Get the TDI cache subdirectory.

    Returns
    -------
    Path
        Path to TDI cache directory (created if doesn't exist)
    """
    tdi_cache = get_cache_dir() / "tdi"
    tdi_cache.mkdir(parents=True, exist_ok=True)
    return tdi_cache


def get_transform_cache_dir() -> Path:
    """Get the transform cache subdirectory.

    Returns
    -------
    Path
        Path to transform cache directory (created if doesn't exist)
    """
    transform_cache = get_cache_dir() / "transforms"
    transform_cache.mkdir(parents=True, exist_ok=True)
    return transform_cache


def get_temp_base_dir() -> Path:
    """Get the base temporary directory for Lacuna.

    The temp directory can be configured via the LACUNA_TEMP_DIR environment
    variable. If not set, falls back to LACUNA_CACHE_DIR/tmp. This is important
    for HPC environments where /tmp may not be writable or has limited space.

    Returns
    -------
    Path
        Path to temp base directory (created if doesn't exist)

    Examples
    --------
    Configure custom temp location for HPC:

    >>> import os
    >>> os.environ['LACUNA_TEMP_DIR'] = '/scratch/user/tmp'
    >>> from lacuna.utils.cache import get_temp_base_dir
    >>> temp_dir = get_temp_base_dir()
    """
    # Check environment variable first
    if temp_dir_env := os.environ.get("LACUNA_TEMP_DIR"):
        temp_base = Path(temp_dir_env)
    else:
        # Fall back to cache dir / tmp
        temp_base = get_cache_dir() / "tmp"

    temp_base.mkdir(parents=True, exist_ok=True)
    return temp_base


def make_temp_file(suffix: str = "", prefix: str = "", delete: bool = False, mode: str = "w+b"):
    """Create a temporary file in Lacuna's temp directory.

    This is a replacement for tempfile.NamedTemporaryFile that uses
    the configurable LACUNA_TEMP_DIR instead of the system default.

    Parameters
    ----------
    suffix : str, optional
        File suffix (e.g., '.nii.gz')
    prefix : str, optional
        File prefix
    delete : bool, optional
        Whether to delete the file when closed (default: False for compatibility
        with external tools that need to read the file)
    mode : str, optional
        File mode (default: 'w+b' for binary write)

    Returns
    -------
    tempfile.NamedTemporaryFile
        A NamedTemporaryFile object with the file in LACUNA_TEMP_DIR

    Examples
    --------
    >>> with make_temp_file(suffix='.nii.gz') as f:
    ...     nib.save(img, f.name)
    ...     # Use f.name with external tool
    """
    temp_dir = get_temp_base_dir()
    return tempfile.NamedTemporaryFile(
        suffix=suffix,
        prefix=prefix,
        dir=temp_dir,
        delete=delete,
        mode=mode,
    )


def get_temp_dir(prefix: str = "") -> Path:
    """Get a temporary directory within the Lacuna temp location.

    Creates a unique temporary directory that can be configured via
    LACUNA_TEMP_DIR (or LACUNA_CACHE_DIR) environment variable, providing
    consistent temp location across the package. This is important for HPC
    environments where /tmp may not be writable.

    Parameters
    ----------
    prefix : str, optional
        Prefix for the temp directory name

    Returns
    -------
    Path
        Path to created temporary directory

    Examples
    --------
    >>> temp_dir = get_temp_dir(prefix="snm_sub01_")
    >>> # temp_dir is e.g. ~/.cache/lacuna/tmp/snm_sub01_abc123/
    >>> # Or if LACUNA_TEMP_DIR=/scratch: /scratch/snm_sub01_abc123/
    """
    import uuid

    temp_base = get_temp_base_dir()

    # Create unique subdirectory
    unique_suffix = uuid.uuid4().hex[:8]
    temp_dir = temp_base / f"{prefix}{unique_suffix}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    return temp_dir
