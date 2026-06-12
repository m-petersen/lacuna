"""Utility functions for data preprocessing and connectome preparation.

This module provides tools for working with neuroimaging datasets and
preparing data for lesion network mapping analyses.

Key Components
--------------
Logging Utilities:
    - ConsoleLogger: Consistent console logger for user-facing messages
    - log_section, log_info, log_success, log_warning, log_error, log_progress: Convenience functions

Suggestion Utilities:
    - suggest_similar: Find similar strings for error message suggestions
    - format_suggestions: Format suggestions for error messages
"""

from lacuna.utils.logging import (
    ConsoleLogger,
    log_error,
    log_info,
    log_progress,
    log_section,
    log_success,
    log_warning,
)
from lacuna.utils.suggestions import format_suggestions, suggest_similar

__all__ = [
    "ConsoleLogger",
    "log_section",
    "log_info",
    "log_success",
    "log_warning",
    "log_error",
    "log_progress",
    "suggest_similar",
    "format_suggestions",
]
