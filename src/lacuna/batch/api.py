"""
Main API for batch processing multiple subjects through analysis pipelines.

This module provides the primary entry point for batch processing with
automatic strategy selection and progress monitoring.
"""

from __future__ import annotations

from collections.abc import Callable

from tqdm import tqdm

from lacuna.analysis.base import BaseAnalysis
from lacuna.batch.selection import select_strategy
from lacuna.core.data_types import ParcelData, VoxelMap
from lacuna.core.mask_data import MaskData


def _detect_input_type(inputs: list) -> str:
    """
    Detect the type of inputs in the batch list.

    Parameters
    ----------
    inputs : list
        List of inputs to check

    Returns
    -------
    str
        One of 'mask_data', 'voxel_map', or 'mixed'

    Raises
    ------
    ValueError
        If inputs list is empty
    """
    if not inputs:
        raise ValueError("inputs cannot be empty")

    has_mask_data = False
    has_voxel_map = False

    for item in inputs:
        if isinstance(item, MaskData):
            has_mask_data = True
        elif isinstance(item, VoxelMap):
            has_voxel_map = True
        else:
            # Unknown type - could be other compatible types
            pass

    if has_mask_data and has_voxel_map:
        return "mixed"
    elif has_voxel_map:
        return "voxel_map"
    else:
        return "mask_data"


def batch_process(
    inputs: list[MaskData | VoxelMap] | None = None,
    analysis: BaseAnalysis | None = None,
    n_jobs: int = -1,
    show_progress: bool = True,
    strategy: str | None = None,
    backend: str = "loky",
    lesion_batch_size: int | None = None,
    batch_result_callback: Callable | None = None,
    *,
    mask_data_list: list[MaskData] | None = None,  # Deprecated, use inputs
) -> list[MaskData | ParcelData]:
    """
    Process multiple subjects through an analysis pipeline with automatic optimization.

    This function automatically selects the optimal processing strategy based on
    the analysis type and available system resources. It provides progress monitoring
    and graceful error handling for individual subject failures.

    Parameters
    ----------
    inputs : list[MaskData] or list[VoxelMap]
        List of MaskData or VoxelMap objects to process.
        All items must be of the same type (no mixing).
    analysis : BaseAnalysis
        Analysis instance to apply to each input
    n_jobs : int, default=-1
        Number of parallel jobs:
        - -1: Use all available CPU cores
        - 1: Sequential processing (useful for debugging)
        - N: Use N parallel workers
    show_progress : bool, default=True
        Display progress bar during processing
    strategy : str or None, default=None
        Force specific strategy:
        - None: Automatic selection based on analysis.batch_strategy
        - "parallel": Force parallel processing
        - "vectorized": Force vectorized processing
        - "streaming": Force streaming processing (future)
    backend : str, default='loky'
        Joblib backend for parallel processing:
    lesion_batch_size : int or None, default=None
        For vectorized strategy: number of lesions to process together in memory.
        - None: Process all lesions at once (fastest, high memory)
        - N: Process N lesions at a time (balanced speed/memory)
        Only applies when using vectorized strategy. Ignored for parallel strategy.
    batch_result_callback : callable or None, default=None
        Callback function called after each lesion batch is processed.
        Signature: callback(batch_results: list[MaskData]) -> None
        Use this to save results immediately and free memory.
        Example: batch_result_callback=lambda batch: [save(r) for r in batch]
        - 'loky': Robust multiprocessing (best for standalone scripts)
        - 'threading': Thread-based parallelism (use in Jupyter notebooks to avoid pickling issues)
        - 'multiprocessing': Standard multiprocessing
        Note: Only applies when using parallel processing (n_jobs > 1)

    Returns
    -------
    list[MaskData]
        List of processed MaskData objects with results added.
        Subjects that failed processing are excluded (warnings are emitted).

    Raises
    ------
    ValueError
        If mask_data_list is empty or analysis is invalid
    RuntimeError
        If strategy selection or execution fails

    Examples
    --------
    Basic usage with automatic optimization:

    >>> from lacuna import batch_process
    >>> from lacuna.analysis import RegionalDamage
    >>> from lacuna.io import load_bids_dataset
    >>>
    >>> # Load subjects
    >>> dataset = load_bids_dataset("path/to/bids")
    >>> lesions = list(dataset.values())
    >>>
    >>> # Process with automatic strategy selection
    >>> analysis = RegionalDamage()
    >>> results = batch_process(lesions, analysis)
    >>> print(f"Processed {len(results)} subjects")

    Use in Jupyter notebooks (threading backend to avoid pickling issues):

    >>> results = batch_process(
    ...     lesions,
    ...     analysis,
    ...     n_jobs=-1,
    ...     backend='threading'  # Works in Jupyter!
    ... )

    Control parallelization:

    >>> # Use all cores (default)
    >>> results = batch_process(lesions, analysis, n_jobs=-1)
    >>>
    >>> # Use 4 cores
    >>> results = batch_process(lesions, analysis, n_jobs=4)
    >>>
    >>> # Sequential (debugging)
    >>> results = batch_process(lesions, analysis, n_jobs=1)

    Chain multiple analyses:

    >>> from lacuna.analysis import RegionalDamage, ParcelAggregation
    >>>
    >>> # First analysis
    >>> regional = RegionalDamage()
    >>> after_regional = batch_process(lesions, regional)
    >>>
    >>> # Second analysis on results
    >>> aggregation = ParcelAggregation(source="mask_img")
    >>> final = batch_process(after_regional, aggregation)

    Notes
    -----
    - Progress bar requires tqdm package
    - Parallel processing requires joblib package
    - Individual subject failures emit warnings but don't stop the batch
    - Strategy selection is automatic based on analysis.batch_strategy attribute
    """
    # Handle deprecated mask_data_list parameter
    if mask_data_list is not None:
        import warnings

        warnings.warn(
            "mask_data_list parameter is deprecated, use 'inputs' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if inputs is not None:
            raise ValueError("Cannot specify both 'inputs' and 'mask_data_list'")
        inputs = mask_data_list

    # Validate inputs
    if not inputs:
        raise ValueError("inputs cannot be empty")

    # Validate analysis parameter
    if analysis is None:
        raise ValueError("analysis parameter is required")

    # Check for mixed input types
    input_type = _detect_input_type(inputs)
    if input_type == "mixed":
        raise TypeError(
            "batch_process does not support mixed input types. "
            "All items must be either MaskData or VoxelMap, not both."
        )

    if not isinstance(analysis, BaseAnalysis):
        raise ValueError(f"analysis must be a BaseAnalysis instance, got {type(analysis)}")

    # Select processing strategy
    strategy_instance = select_strategy(
        analysis=analysis,
        n_subjects=len(inputs),
        n_jobs=n_jobs,
        force_strategy=strategy,
        backend=backend,
        lesion_batch_size=lesion_batch_size,
        batch_result_callback=batch_result_callback,
    )

    # Setup progress tracking
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(
            total=len(inputs),
            desc=f"Processing with {strategy_instance.name} strategy",
            unit="subject",
        )

        def progress_callback(idx: int) -> None:
            """Update progress bar."""
            if progress_bar:
                progress_bar.update(1)

    else:
        progress_callback = None

    # Execute batch processing
    try:
        results = strategy_instance.execute(
            mask_data_list=inputs,
            analysis=analysis,
            progress_callback=progress_callback,
        )
    finally:
        # Close progress bar
        if progress_bar:
            progress_bar.close()

    return results
