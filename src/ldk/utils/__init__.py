"""Utility functions for data preprocessing and connectome preparation.

This module provides tools for working with neuroimaging datasets and
preparing data for lesion network mapping analyses.

Key Components
--------------
GSP1000 Utilities:
    - create_connectome_batches: Convert GSP1000 data to optimized HDF5 batches
    - validate_connectome_batches: Verify integrity of batch files
"""

from ldk.utils.gsp1000 import create_connectome_batches, validate_connectome_batches

__all__ = [
    "create_connectome_batches",
    "validate_connectome_batches",
]
