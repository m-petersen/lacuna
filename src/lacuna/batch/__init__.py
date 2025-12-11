"""
Batch processing infrastructure for multi-subject analysis pipelines.

This module provides efficient batch processing capabilities with automatic
strategy selection based on analysis characteristics and computational resources.

Key Components
--------------
- **batch_process()**: Main entry point for batch processing
- **BatchStrategy**: Abstract base class for processing strategies
- **ParallelStrategy**: Multi-core parallelization using joblib
- **VectorizedStrategy**: Batch matrix operations for connectome analyses
- **select_strategy()**: Automatic strategy selection logic

Examples
--------
>>> from lacuna.batch import batch_process
>>> from lacuna.analysis import RegionalDamage
>>> from lacuna.io import load_bids_dataset
>>>
>>> # Load subjects
>>> dataset = load_bids_dataset("path/to/bids")
>>> lesions = list(dataset.values())
>>>
>>> # Batch process with automatic optimization
>>> analysis = RegionalDamage()
>>> results = batch_process(lesions, analysis)
>>>
>>> # Process with specific number of cores
>>> results = batch_process(lesions, analysis, n_jobs=4)
"""

from lacuna.batch.api import batch_process
from lacuna.batch.extract import (
    extract,
    extract_parcel_table,
    extract_scalars,
    extract_voxelmaps,
)
from lacuna.batch.selection import select_strategy
from lacuna.batch.strategies import BatchStrategy, ParallelStrategy, VectorizedStrategy

__all__ = [
    "batch_process",
    "extract",
    "extract_parcel_table",
    "extract_scalars",
    "extract_voxelmaps",
    "select_strategy",
    "BatchStrategy",
    "ParallelStrategy",
    "VectorizedStrategy",
]
