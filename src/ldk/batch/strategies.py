"""
Batch processing strategies for different analysis types.

This module implements the Strategy pattern for batch processing, enabling
automatic optimization based on analysis characteristics:

- **ParallelStrategy**: Independent per-subject processing with multiprocessing
- **VectorizedStrategy**: Batch matrix operations (future - Phase 5)
- **StreamingStrategy**: Memory-constrained sequential processing (future)
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable

from joblib import Parallel, delayed

from ldk.analysis.base import BaseAnalysis
from ldk.core.lesion_data import LesionData


class BatchStrategy(ABC):
    """
    Abstract base class for batch processing strategies.

    Each strategy implements a different approach to processing multiple subjects:
    - Parallel: Uses multiprocessing for independent analyses
    - Vectorized: Stacks data for batch matrix operations
    - Streaming: Processes sequentially with immediate disk writes

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
        lesion_data_list: list[LesionData],
        analysis: BaseAnalysis,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[LesionData]:
        """
        Execute analysis on all lesions using this strategy.

        Parameters
        ----------
        lesion_data_list : list[LesionData]
            List of lesions to process
        analysis : BaseAnalysis
            Analysis instance to apply to each lesion
        progress_callback : callable or None
            Optional callback function to report progress.
            Called with current index after each subject completes.

        Returns
        -------
        list[LesionData]
            List of processed LesionData objects with results added

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

    Best for: Independent per-subject analyses (RegionalDamage, AtlasAggregation)
    Speedup: 4-8x on multi-core systems (proportional to available cores)
    Memory: Low overhead (~1.2x per-subject memory usage)

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
    >>> from ldk.batch.strategies import ParallelStrategy
    >>> from ldk.analysis import RegionalDamage
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
        lesion_data_list: list[LesionData],
        analysis: BaseAnalysis,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[LesionData]:
        """
        Execute parallel batch processing.

        Processes subjects in parallel using joblib. Each subject is processed
        independently, and failures are caught and reported as warnings without
        stopping the entire batch.

        Parameters
        ----------
        lesion_data_list : list[LesionData]
            Subjects to process
        analysis : BaseAnalysis
            Analysis to apply
        progress_callback : callable or None
            Progress reporting function

        Returns
        -------
        list[LesionData]
            Successfully processed subjects (failures are filtered out)
        """

        def _process_one(lesion_data: LesionData, idx: int) -> tuple[int, LesionData | None]:
            """Process single subject with error handling."""

            try:
                result = analysis.run(lesion_data)
                # Note: progress_callback not used here - progress tracking handled differently
                return idx, result
            except Exception as e:
                # Get subject ID from metadata if available
                subject_id = lesion_data.metadata.get("subject_id", f"index_{idx}")
                warnings.warn(
                    f"Analysis failed for subject {subject_id} (index {idx}): {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return idx, None

        # Execute in parallel
        if self.n_jobs == 1:
            # Sequential processing (useful for debugging)
            results = []
            for i, lesion in enumerate(lesion_data_list):
                result = _process_one(lesion, i)
                results.append(result)
                if progress_callback:
                    progress_callback(i)
        else:
            # Parallel processing with joblib using user-specified backend
            # Note: We can't pass progress_callback to workers due to pickling issues
            # Progress bar will update in chunks instead
            results = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                delayed(_process_one)(lesion, i) for i, lesion in enumerate(lesion_data_list)
            )
            # Update progress bar all at once after parallel processing
            if progress_callback:
                for i in range(len(results)):
                    progress_callback(i)

        # Sort by original index and filter out failures
        results = sorted(results, key=lambda x: x[0])
        successful_results = [r[1] for r in results if r[1] is not None]

        # Warn if any subjects failed
        n_failed = len(lesion_data_list) - len(successful_results)
        if n_failed > 0:
            warnings.warn(
                f"{n_failed} out of {len(lesion_data_list)} subjects failed processing. "
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

    Best for: Matrix-based analyses (FunctionalNetworkMapping)
    Speedup: 10-50x via optimized BLAS operations and reduced overhead
    Memory: Moderate (processes lesions in configurable batches)

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
        run_batch(lesion_data_list: list[LesionData]) -> list[LesionData]

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
    >>> from ldk.batch.strategies import VectorizedStrategy
    >>> from ldk.analysis import FunctionalNetworkMapping
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
        batch_result_callback: Callable[[list[LesionData]], None] | None = None,
    ):
        super().__init__(n_jobs)
        self.lesion_batch_size = lesion_batch_size
        self.batch_result_callback = batch_result_callback

    def execute(
        self,
        lesion_data_list: list[LesionData],
        analysis: BaseAnalysis,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[LesionData]:
        """
        Execute vectorized batch processing.

        Calls analysis.run_batch() which processes multiple lesions together
        using vectorized operations. Falls back to sequential processing if
        run_batch() is not implemented.

        Parameters
        ----------
        lesion_data_list : list[LesionData]
            Subjects to process
        analysis : BaseAnalysis
            Analysis to apply (must implement run_batch method)
        progress_callback : callable or None
            Progress reporting function (called after each lesion batch)

        Returns
        -------
        list[LesionData]
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
            results = analysis.run_batch(lesion_data_list)

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
        n_lesions = len(lesion_data_list)

        for batch_start in range(0, n_lesions, self.lesion_batch_size):
            batch_end = min(batch_start + self.lesion_batch_size, n_lesions)
            lesion_batch = lesion_data_list[batch_start:batch_end]

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


# Future strategies (Phase 5+)

# class StreamingStrategy(BatchStrategy):
#     """
#     Streaming batch processing for memory-constrained environments.
#
#     Best for: Large connectome analyses with limited RAM
#     Speedup: None (sequential), but enables processing of datasets
#              that don't fit in memory
#     Memory: Low (one subject at a time + immediate disk writes)
#     """
#     pass
