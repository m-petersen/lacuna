"""Utility functions for data preprocessing and connectome preparation.

This module provides tools for working with neuroimaging datasets and
preparing data for lesion network mapping analyses.

Key Components
--------------
GSP1000 Utilities:
    - create_connectome_batches: Convert GSP1000 data to optimized HDF5 batches
    - validate_connectome_batches: Verify integrity of batch files

Logging Utilities:
    - ConsoleLogger: Consistent console logger for user-facing messages
    - log_section, log_info, log_success, log_warning, log_error, log_progress: Convenience functions
"""

from ldk.utils.gsp1000 import create_connectome_batches, validate_connectome_batches
from ldk.utils.logging import (
    ConsoleLogger,
    log_error,
    log_info,
    log_progress,
    log_section,
    log_success,
    log_warning,
)

__all__ = [
    "create_connectome_batches",
    "validate_connectome_batches",
    "ConsoleLogger",
    "log_section",
    "log_info",
    "log_success",
    "log_warning",
    "log_error",
    "log_progress",
]
