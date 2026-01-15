"""
Batch processing strategies for different analysis types.

This module implements the Strategy pattern for batch processing, enabling
automatic optimization based on analysis characteristics:

- **ParallelStrategy**: Independent per-subject processing with multiprocessing
- **VectorizedStrategy**: Batch matrix operations for connectome analyses
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable

from joblib import Parallel, delayed

from lacuna.analysis.base import BaseAnalysis
from lacuna.core.subject_data import SubjectData


def _format_subject_id(subject_data: SubjectData) -> str:
    """
    Format a human-readable identifier for a subject.

    Combines subject_id, session_id, and label into a compact string.

    Parameters
    ----------
    subject_data : SubjectData
        Subject data with metadata.

    Returns
    -------
    str
        Formatted identifier like 'sub-001/ses-01/lesion' or 'sub-001/lesion'
    """
    parts = []
    metadata = subject_data.metadata

    subject_id = metadata.get("subject_id", "unknown")
    parts.append(subject_id)

    session_id = metadata.get("session_id")
    if session_id:
        parts.append(session_id)

    label = metadata.get("label")
    if label:
        parts.append(label)

    return "/".join(parts)


def _process_one_subject(
    mask_data: SubjectData, idx: int, analysis: BaseAnalysis
) -> tuple[int, SubjectData | None]:
    """
    Process a single subject with error handling.

    This function is defined at module level to ensure it can be pickled
    by all multiprocessing backends (including standard 'multiprocessing').

    Parameters
    ----------
    mask_data : SubjectData
        The subject data to process
    idx : int
        The subject index in the batch
    analysis : BaseAnalysis
        The analysis to run on the subject

    Returns
    -------
    tuple[int, SubjectData | None]
        Tuple of (index, result) where result is None if processing failed
    """
    try:
        result = analysis.run(mask_data)
        return idx, result
    except Exception as e:
        # Get formatted subject identifier
        subject_id = _format_subject_id(mask_data)
        warnings.warn(
            f"Analysis failed for {subject_id}: {e}",
            RuntimeWarning,
            stacklevel=2,
        )
        return idx, None


class BatchStrategy(ABC):
    """
    Abstract base class for batch processing strategies.

    Each strategy implements a different approach to processing multiple subjects:
    - Parallel: Uses multiprocessing for independent analyses
    - Vectorized: Stacks data for batch matrix operations

    Parameters
    ----------
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available cores.
        Only relevant for ParallelStrategy.
    """

    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs

    @abstractmethod
    def execute(
        self,
        inputs: list[SubjectData],
        analysis: BaseAnalysis,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[SubjectData]:
        """
        Execute analysis on all lesions using this strategy.

        Parameters
        ----------
        inputs : list[SubjectData]
            List of lesions to process
        analysis : BaseAnalysis
            Analysis instance to apply to each lesion
        progress_callback : callable or None
            Optional callback function to report progress.
            Called with current index after each subject completes.

        Returns
        -------
        list[SubjectData]
            List of processed SubjectData objects with results added

        Raises
        ------
        RuntimeError
            If execution fails
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging and debugging."""
        pass


class ParallelStrategy(BatchStrategy):
    """
    Parallel batch processing using joblib multiprocessing.

    Best for independent per-subject analyses (RegionalDamage, ParcelAggregation)
    Speedup on multi-core systems (proportional to available cores)
    Low memory overhead

    This strategy processes each subject independently using joblib.Parallel.
    The backend can be configured to handle different environments:
    - 'loky' (default): Robust multiprocessing for standalone scripts
    - 'threading': Thread-based parallelism for Jupyter notebooks
    - 'multiprocessing': Standard multiprocessing (less robust than loky)

    Parameters
    ----------
    n_jobs : int, default=-1
        Number of parallel jobs:
        - -1: Use all available CPU cores
        - 1: Sequential processing (useful for debugging)
        - N: Use N parallel workers
    backend : str, default='loky'
        Joblib backend to use:
        - 'loky': Robust multiprocessing (best for scripts)
        - 'threading': Thread-based (use in Jupyter notebooks)
        - 'multiprocessing': Standard multiprocessing

    Examples
    --------
    >>> from lacuna.batch.strategies import ParallelStrategy
    >>> from lacuna.analysis import RegionalDamage
    >>>
    >>> # For standalone scripts (default)
    >>> strategy = ParallelStrategy(n_jobs=4)
    >>> results = strategy.execute(lesions, RegionalDamage())
    >>>
    >>> # For Jupyter notebooks
    >>> strategy = ParallelStrategy(n_jobs=4, backend='threading')
    >>> results = strategy.execute(lesions, RegionalDamage())
    """

    def __init__(self, n_jobs: int = -1, backend: str = "loky"):
        super().__init__(n_jobs)
        self.backend = backend

        # Resolve -1 to actual core count
        if self.n_jobs == -1:
            import os

            self.n_jobs = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else os.cpu_count() or 1
            )

    def execute(
        self,
        inputs: list[SubjectData],
        analysis: BaseAnalysis,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[SubjectData]:
        """
        Execute parallel batch processing.

        Processes subjects in parallel using joblib. Each subject is processed
        independently, and failures are caught and reported as warnings without
        stopping the entire batch.

        Parameters
        ----------
        inputs : list[SubjectData]
            Subjects to process
        analysis : BaseAnalysis
            Analysis to apply
        progress_callback : callable or None
            Progress reporting function

        Returns
        -------
        list[SubjectData]
            Successfully processed subjects (failures are filtered out)
        """
        # Execute in parallel
        if self.n_jobs == 1:
            # Sequential processing (useful for debugging)
            results = []
            for i, lesion in enumerate(inputs):
                result = _process_one_subject(lesion, i, analysis)
                results.append(result)
                if progress_callback:
                    progress_callback(i)
        else:
            # Parallel processing with joblib using user-specified backend
            # Uses module-level _process_one_subject for pickle compatibility
            # with all backends including standard 'multiprocessing'
            results = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                delayed(_process_one_subject)(lesion, i, analysis)
                for i, lesion in enumerate(inputs)
            )
            # Update progress bar once for the entire batch (not per-item to avoid duplicates)
            if progress_callback:
                for _ in range(len(results)):
                    progress_callback(0)  # Index doesn't matter, just triggers update

        # Sort by original index and filter out failures
        results = sorted(results, key=lambda x: x[0])
        successful_results = [r[1] for r in results if r[1] is not None]

        # Warn if any subjects failed
        n_failed = len(inputs) - len(successful_results)
        if n_failed > 0:
            warnings.warn(
                f"{n_failed} out of {len(inputs)} subjects failed processing. "
                "Check warnings above for details.",
                RuntimeWarning,
                stacklevel=2,
            )

        return successful_results

    @property
    def name(self) -> str:
        return "parallel"


class VectorizedStrategy(BatchStrategy):
    """
    Vectorized batch processing using batched NumPy operations.

    Best for matrix-based analyses (FunctionalNetworkMapping)
    Speedup via optimized BLAS operations and reduced overhead
    Moderate memory overhead (processes lesions in configurable batches)

    This strategy leverages vectorized operations to process multiple lesions
    simultaneously through each connectome batch. Instead of:
        for lesion in lesions:
            for connectome_batch in batches:
                process(lesion, connectome_batch)

    It does:
        for connectome_batch in batches:
            process_all_lesions_together(lesions_batch, connectome_batch)

    This dramatically reduces overhead and enables efficient BLAS operations.

    The analysis class must implement:
        run_batch(inputs: list[SubjectData]) -> list[SubjectData]

    Parameters
    ----------
    n_jobs : int, default=-1
        Not used for vectorized processing (uses BLAS parallelization instead).
        Kept for interface compatibility.
    lesion_batch_size : int or None, default=None
        Number of lesions to process together in memory.
        - None: Process all lesions together (maximum speed, high memory)
        - N: Process N lesions at a time (balanced speed/memory)
        Useful when processing hundreds of lesions.

    Examples
    --------
    >>> from lacuna.batch.strategies import VectorizedStrategy
    >>> from lacuna.analysis import FunctionalNetworkMapping
    >>>
    >>> # Process all lesions together (fastest)
    >>> strategy = VectorizedStrategy()
    >>> results = strategy.execute(lesions, FunctionalNetworkMapping(...))
    >>>
    >>> # Process 50 lesions at a time (memory-constrained)
    >>> strategy = VectorizedStrategy(lesion_batch_size=50)
    >>> results = strategy.execute(lesions, FunctionalNetworkMapping(...))
    """

    def __init__(
        self,
        n_jobs: int = -1,
        lesion_batch_size: int | None = None,
        batch_result_callback: Callable[[list[SubjectData]], None] | None = None,
    ):
        super().__init__(n_jobs)
        self.lesion_batch_size = lesion_batch_size
        self.batch_result_callback = batch_result_callback

    def execute(
        self,
        inputs: list[SubjectData],
        analysis: BaseAnalysis,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[SubjectData]:
        """
        Execute vectorized batch processing.

        Calls analysis.run_batch() which processes multiple lesions together
        using vectorized operations. Falls back to sequential processing if
        run_batch() is not implemented.

        Parameters
        ----------
        inputs : list[SubjectData]
            Subjects to process
        analysis : BaseAnalysis
            Analysis to apply (must implement run_batch method)
        progress_callback : callable or None
            Progress reporting function (called after each lesion batch)

        Returns
        -------
        list[SubjectData]
            Processed subjects

        Raises
        ------
        NotImplementedError
            If analysis doesn't implement run_batch()
        """
        # Check if analysis supports batch processing
        if not hasattr(analysis, "run_batch") or not callable(analysis.run_batch):
            raise NotImplementedError(
                f"Analysis {analysis.__class__.__name__} does not implement "
                f"run_batch() method required for vectorized strategy. "
                f"Please use parallel strategy instead or implement run_batch()."
            )

        # Process all lesions together if no batch size specified
        if self.lesion_batch_size is None:
            results = analysis.run_batch(inputs)

            # Call batch result callback if provided
            if self.batch_result_callback:
                self.batch_result_callback(results)

            # Update progress
            if progress_callback:
                for i in range(len(results)):
                    progress_callback(i)

            return results

        # Process lesions in batches
        all_results = []
        n_lesions = len(inputs)

        for batch_start in range(0, n_lesions, self.lesion_batch_size):
            batch_end = min(batch_start + self.lesion_batch_size, n_lesions)
            lesion_batch = inputs[batch_start:batch_end]

            # Process this batch
            batch_results = analysis.run_batch(lesion_batch)

            # Call batch result callback if provided (for immediate saving)
            if self.batch_result_callback:
                self.batch_result_callback(batch_results)

            all_results.extend(batch_results)

            # Update progress
            if progress_callback:
                for i in range(batch_start, batch_end):
                    progress_callback(i)

        return all_results

    @property
    def name(self) -> str:
        return "vectorized"
