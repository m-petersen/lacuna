"""
Automatic batch strategy selection based on analysis characteristics.

This module implements the logic for selecting the optimal batch processing
strategy based on:
- Analysis type (via batch_strategy class attribute)
- Number of subjects
- Available system resources
- User preferences
"""

from __future__ import annotations

import os
from collections.abc import Callable

from lacuna.analysis.base import BaseAnalysis
from lacuna.batch.strategies import BatchStrategy, ParallelStrategy, VectorizedStrategy


def select_strategy(
    analysis: BaseAnalysis,
    n_subjects: int,
    n_jobs: int = -1,
    force_strategy: str | None = None,
    backend: str = "loky",
    lesion_batch_size: int | None = None,
    batch_result_callback: Callable | None = None,
) -> BatchStrategy:
    """
    Select the optimal batch processing strategy.

    Selection Logic:
    1. If force_strategy specified, use it (with validation)
    2. Check analysis.batch_strategy class attribute
    3. Consider system resources and memory requirements
    4. Return appropriate BatchStrategy instance

    Parameters
    ----------
    analysis : BaseAnalysis
        Analysis instance to be executed
    n_subjects : int
        Number of subjects to process
    n_jobs : int, default=-1
        Number of parallel jobs requested
    force_strategy : str or None, default=None
        Override automatic selection. Options:
        - "parallel": Force parallel processing
        - "vectorized": Force vectorized processing
        - "streaming": Force streaming processing (future)
    backend : str, default='loky'
        Joblib backend for parallel processing:
        - 'loky': Robust multiprocessing (best for standalone scripts)
        - 'threading': Thread-based (use in Jupyter notebooks to avoid pickling issues)
        - 'multiprocessing': Standard multiprocessing
    lesion_batch_size : int or None, default=None
        For vectorized strategy: number of lesions to process together in memory.
        - None: Process all lesions at once (fastest, high memory)
        - N: Process N lesions at a time (balanced speed/memory)
        Only applies to vectorized strategy.

    Returns
    -------
    BatchStrategy
        Instantiated strategy ready for execution

    Raises
    ------
    ValueError
        If force_strategy is invalid or not yet implemented

    Examples
    --------
    >>> from lacuna.batch.selection import select_strategy
    >>> from lacuna.analysis import RegionalDamage
    >>>
    >>> # Default (loky backend for scripts)
    >>> analysis = RegionalDamage()
    >>> strategy = select_strategy(analysis, n_subjects=50)
    >>> print(strategy.name)
    'parallel'
    >>>
    >>> # Threading backend for Jupyter
    >>> strategy = select_strategy(analysis, n_subjects=50, backend='threading')
    >>>
    >>> # Vectorized with memory control
    >>> from lacuna.analysis import FunctionalNetworkMapping
    >>> analysis = FunctionalNetworkMapping(...)
    >>> strategy = select_strategy(analysis, n_subjects=100, lesion_batch_size=20)
    """
    # Handle force override
    if force_strategy is not None:
        force_strategy = force_strategy.lower()

        if force_strategy == "parallel":
            return ParallelStrategy(n_jobs=n_jobs, backend=backend)
        elif force_strategy == "vectorized":
            return VectorizedStrategy(
                n_jobs=n_jobs,
                lesion_batch_size=lesion_batch_size,
                batch_result_callback=batch_result_callback,
            )
        elif force_strategy == "streaming":
            raise ValueError(
                "Strategy 'streaming' is not yet implemented. "
                "Available strategies: 'parallel', 'vectorized'"
            )
        else:
            raise ValueError(
                f"Unknown strategy '{force_strategy}'. "
                f"Available strategies: 'parallel', 'vectorized', 'streaming' (future)"
            )

    # Get preferred strategy from analysis class
    preferred_strategy = getattr(analysis, "batch_strategy", "parallel")

    # Dispatch to appropriate strategy
    if preferred_strategy == "parallel":
        return ParallelStrategy(n_jobs=n_jobs, backend=backend)
    elif preferred_strategy == "vectorized":
        return VectorizedStrategy(
            n_jobs=n_jobs,
            lesion_batch_size=lesion_batch_size,
            batch_result_callback=batch_result_callback,
        )
    elif preferred_strategy == "streaming":
        # Future: StreamingStrategy
        raise NotImplementedError(
            f"Analysis {analysis.__class__.__name__} prefers 'streaming' strategy, "
            "but this is not yet implemented. Use 'parallel' or 'vectorized' instead."
        )
    else:
        # Unknown strategy - fall back to parallel
        import warnings

        warnings.warn(
            f"Unknown batch_strategy '{preferred_strategy}' in {analysis.__class__.__name__}. "
            f"Falling back to parallel processing.",
            RuntimeWarning,
            stacklevel=2,
        )
        return ParallelStrategy(n_jobs=n_jobs, backend=backend)


def get_available_cores() -> int:
    """
    Get the number of CPU cores available to this process.

    Returns
    -------
    int
        Number of available cores

    Notes
    -----
    Uses os.sched_getaffinity() if available (Linux), otherwise falls back
    to os.cpu_count(). The affinity-based count respects cgroup limits and
    CPU pinning in containerized environments.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def estimate_memory_requirement(analysis: BaseAnalysis, n_subjects: int) -> int:
    """
    Estimate memory requirements for batch processing.

    Parameters
    ----------
    analysis : BaseAnalysis
        Analysis to be executed
    n_subjects : int
        Number of subjects

    Returns
    -------
    int
        Estimated memory requirement in bytes

    Notes
    -----
    This is a rough estimation based on typical neuroimaging data sizes:
    - Base lesion NIfTI: ~30 MB in memory
    - Network analyses with connectomes: +500 MB per subject
    - Atlas aggregation: minimal additional memory

    Future implementations may query actual data sizes or use profiling.
    """
    # Base estimation: typical lesion NIfTI in memory
    base_per_subject = 30 * 1024 * 1024  # 30 MB

    # Check if analysis needs connectome data
    if hasattr(analysis, "connectome_path"):
        base_per_subject += 500 * 1024 * 1024  # +500 MB for connectome

    return base_per_subject * n_subjects


def check_memory_available() -> int:
    """
    Get available system memory in bytes.

    Returns
    -------
    int
        Available memory in bytes, or -1 if psutil not available

    Notes
    -----
    Requires psutil package (optional dependency). If not available,
    returns -1 to indicate unknown memory status.
    """
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        return -1  # Unknown - psutil not installed
