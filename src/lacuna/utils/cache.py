"""
Unified cache directory management for Lacuna.

Provides a consistent cache location that can be configured via environment variable.
"""

import os
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
