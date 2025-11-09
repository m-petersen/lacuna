"""
Main API for batch processing multiple subjects through analysis pipelines.

This module provides the primary entry point for batch processing with
automatic strategy selection and progress monitoring.
"""

from __future__ import annotations

from collections.abc import Callable

from tqdm import tqdm

from ldk.analysis.base import BaseAnalysis
from ldk.batch.selection import select_strategy
from ldk.core.lesion_data import LesionData


def batch_process(
    lesion_data_list: list[LesionData],
    analysis: BaseAnalysis,
    n_jobs: int = -1,
    show_progress: bool = True,
    strategy: str | None = None,
    backend: str = "loky",
    lesion_batch_size: int | None = None,
    batch_result_callback: Callable | None = None,
) -> list[LesionData]:
    """
    Process multiple lesions through an analysis pipeline with automatic optimization.

    This function automatically selects the optimal processing strategy based on
    the analysis type and available system resources. It provides progress monitoring
    and graceful error handling for individual subject failures.

    Parameters
    ----------
    lesion_data_list : list[LesionData]
        List of LesionData objects to process
    analysis : BaseAnalysis
        Analysis instance to apply to each lesion
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
        Signature: callback(batch_results: list[LesionData]) -> None
        Use this to save results immediately and free memory.
        Example: batch_result_callback=lambda batch: [save(r) for r in batch]
        - 'loky': Robust multiprocessing (best for standalone scripts)
        - 'threading': Thread-based parallelism (use in Jupyter notebooks to avoid pickling issues)
        - 'multiprocessing': Standard multiprocessing
        Note: Only applies when using parallel processing (n_jobs > 1)

    Returns
    -------
    list[LesionData]
        List of processed LesionData objects with results added.
        Subjects that failed processing are excluded (warnings are emitted).

    Raises
    ------
    ValueError
        If lesion_data_list is empty or analysis is invalid
    RuntimeError
        If strategy selection or execution fails

    Examples
    --------
    Basic usage with automatic optimization:

    >>> from ldk import batch_process
    >>> from ldk.analysis import RegionalDamage
    >>> from ldk.io import load_bids_dataset
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

    >>> from ldk.analysis import RegionalDamage, AtlasAggregation
    >>>
    >>> # First analysis
    >>> regional = RegionalDamage()
    >>> after_regional = batch_process(lesions, regional)
    >>>
    >>> # Second analysis on results
    >>> aggregation = AtlasAggregation(source="lesion_img")
    >>> final = batch_process(after_regional, aggregation)

    Notes
    -----
    - Progress bar requires tqdm package
    - Parallel processing requires joblib package
    - Individual subject failures emit warnings but don't stop the batch
    - Strategy selection is automatic based on analysis.batch_strategy attribute
    """
    # Validate inputs
    if not lesion_data_list:
        raise ValueError("lesion_data_list cannot be empty")

    if not isinstance(analysis, BaseAnalysis):
        raise ValueError(f"analysis must be a BaseAnalysis instance, got {type(analysis)}")

    # Select processing strategy
    strategy_instance = select_strategy(
        analysis=analysis,
        n_subjects=len(lesion_data_list),
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
            total=len(lesion_data_list),
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
            lesion_data_list=lesion_data_list,
            analysis=analysis,
            progress_callback=progress_callback,
        )
    finally:
        # Close progress bar
        if progress_bar:
            progress_bar.close()

    return results
