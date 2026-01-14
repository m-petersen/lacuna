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
from lacuna.core.subject_data import SubjectData


def _detect_input_type(inputs: list) -> str:
    """
    Detect and validate the type of inputs in the batch list.

    Parameters
    ----------
    inputs : list
        List of inputs to check

    Returns
    -------
    str
        One of 'subject_data' or 'voxel_map'

    Raises
    ------
    ValueError
        If inputs list is empty
    TypeError
        If inputs contain invalid types or mixed SubjectData/VoxelMap
    """
    if not inputs:
        raise ValueError("inputs cannot be empty")

    has_subject_data = False
    has_voxel_map = False
    invalid_items = []

    for i, item in enumerate(inputs):
        if isinstance(item, SubjectData):
            has_subject_data = True
        elif isinstance(item, VoxelMap):
            has_voxel_map = True
        else:
            invalid_items.append((i, type(item).__name__))

    # Check for invalid types first
    if invalid_items:
        # Build informative error message
        if len(invalid_items) <= 3:
            examples = ", ".join(f"inputs[{i}] is {t}" for i, t in invalid_items)
        else:
            examples = ", ".join(f"inputs[{i}] is {t}" for i, t in invalid_items[:3])
            examples += f", ... and {len(invalid_items) - 3} more"
        raise TypeError(
            f"batch_process requires all inputs to be SubjectData or VoxelMap instances. "
            f"Found invalid types: {examples}"
        )

    # Check for mixed types
    if has_subject_data and has_voxel_map:
        raise TypeError(
            "batch_process does not support mixed input types. "
            "All items must be either SubjectData or VoxelMap, not both."
        )

    if has_voxel_map:
        return "voxel_map"
    else:
        return "subject_data"


def batch_process(
    inputs: list[SubjectData | VoxelMap] | None = None,
    analysis: BaseAnalysis | None = None,
    n_jobs: int = -1,
    show_progress: bool = True,
    strategy: str | None = None,
    backend: str = "loky",
    lesion_batch_size: int | None = None,
    batch_result_callback: Callable | None = None,
) -> list[SubjectData | ParcelData]:
    """
    Process multiple subjects through an analysis pipeline with automatic optimization.

    This function automatically selects the optimal processing strategy based on
    the analysis type and available system resources. It provides progress monitoring
    and graceful error handling for individual subject failures.

    Parameters
    ----------
    inputs : list[SubjectData] or list[VoxelMap]
        List of SubjectData or VoxelMap objects to process.
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
    backend : str, default='loky'
        Joblib backend for parallel processing:
    lesion_batch_size : int or None, default=None
        For vectorized strategy: number of lesions to process together in memory.
        - None: Process all lesions at once (fastest, high memory)
        - N: Process N lesions at a time (balanced speed/memory)
        Only applies when using vectorized strategy. Ignored for parallel strategy.
    batch_result_callback : callable or None, default=None
        Callback function called after each lesion batch is processed.
        Signature: callback(batch_results: list[SubjectData]) -> None
        Use this to save results immediately and free memory.
        Example: batch_result_callback=lambda batch: [save(r) for r in batch]
        - 'loky': Robust multiprocessing (best for standalone scripts)
        - 'threading': Thread-based parallelism (use in Jupyter notebooks to avoid pickling issues)
        - 'multiprocessing': Standard multiprocessing
        Note: Only applies when using parallel processing (n_jobs > 1)

    Returns
    -------
    list[SubjectData]
        List of processed SubjectData objects with results added.
        Subjects that failed processing are excluded (warnings are emitted).

    Raises
    ------
    ValueError
        If inputs is empty or analysis is invalid
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
    >>> aggregation = ParcelAggregation(source="maskimg")
    >>> final = batch_process(after_regional, aggregation)

    Notes
    -----
    - Progress bar requires tqdm package
    - Parallel processing requires joblib package
    - Individual subject failures emit warnings but don't stop the batch
    - Strategy selection is automatic based on analysis.batch_strategy attribute
    """
    # Validate inputs
    if not inputs:
        raise ValueError("inputs cannot be empty")

    # Validate analysis parameter
    if analysis is None:
        raise ValueError("analysis parameter is required")

    # Validate input types (raises TypeError for invalid/mixed types)
    _detect_input_type(inputs)

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
            inputs=inputs,
            analysis=analysis,
            progress_callback=progress_callback,
        )
    finally:
        # Close progress bar
        if progress_bar:
            progress_bar.close()

    return results
