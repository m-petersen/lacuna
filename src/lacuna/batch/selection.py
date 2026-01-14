"""
Batch strategy selection based on analysis class attribute.

This module provides simple dispatch to the appropriate batch processing
strategy based on the analysis.batch_strategy class attribute.
"""

from __future__ import annotations

import os
import warnings
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
    Select batch processing strategy based on analysis.batch_strategy attribute.

    Parameters
    ----------
    analysis : BaseAnalysis
        Analysis instance to be executed
    n_subjects : int
        Number of subjects to process (currently unused, reserved for future)
    n_jobs : int, default=-1
        Number of parallel jobs requested
    force_strategy : str or None, default=None
        Override automatic selection. Options:
        - "parallel": Force parallel processing
        - "vectorized": Force vectorized processing
    backend : str, default='loky'
        Joblib backend for parallel processing:
        - 'loky': Robust multiprocessing (best for standalone scripts)
        - 'threading': Thread-based (use in Jupyter notebooks)
        - 'multiprocessing': Standard multiprocessing
    lesion_batch_size : int or None, default=None
        For vectorized strategy: number of lesions to process together.
    batch_result_callback : callable or None, default=None
        Callback function called after each batch is processed.

    Returns
    -------
    BatchStrategy
        Instantiated strategy ready for execution

    Raises
    ------
    ValueError
        If force_strategy is invalid
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
        else:
            raise ValueError(
                f"Unknown strategy '{force_strategy}'. "
                f"Available strategies: 'parallel', 'vectorized'"
            )

    # Get strategy from analysis class attribute
    preferred_strategy = getattr(analysis, "batch_strategy", "parallel")

    # Dispatch to appropriate strategy
    if preferred_strategy == "vectorized":
        return VectorizedStrategy(
            n_jobs=n_jobs,
            lesion_batch_size=lesion_batch_size,
            batch_result_callback=batch_result_callback,
        )
    elif preferred_strategy == "parallel":
        return ParallelStrategy(n_jobs=n_jobs, backend=backend)
    else:
        # Unknown strategy - fall back to parallel with warning
        warnings.warn(
            f"Unknown batch_strategy '{preferred_strategy}' in "
            f"{analysis.__class__.__name__}. Falling back to parallel.",
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
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1
