"""
Consistent logging and user message formatting for Lacuna.

Provides a unified system for displaying progress, success, warnings, and errors
to users with consistent formatting across all modules.
"""

import logging
from enum import Enum


class MessageType(Enum):
    """Types of messages that can be displayed."""

    INFO = ""  # General information
    SUCCESS = ""  # Operation completed successfully
    WARNING = ""  # Warning message
    ERROR = ""  # Error message
    PROGRESS = ""  # Progress update
    SECTION = "="  # Section header
    SUBSECTION = "-"  # Subsection header


class ConsoleLogger:
    """
    Consistent console logger for user-facing messages.

    Uses the standard Python logging module for output, ensuring consistent
    formatting with timestamps and module names across all Lacuna modules.

    Parameters
    ----------
    verbose : bool, default=True
        If True, print messages. If False, silent mode (no output).
    width : int, default=70
        Width for section headers
    indent : str, default="  "
        Indentation string for nested messages
    name : str, default="lacuna"
        Logger name for the Python logging module

    Examples
    --------
    >>> logger = ConsoleLogger(verbose=True)
    >>> logger.section("PROCESSING DATA")
    2026-01-15 10:00:00 - lacuna - INFO - ============================================
    2026-01-15 10:00:00 - lacuna - INFO - PROCESSING DATA
    2026-01-15 10:00:00 - lacuna - INFO - ============================================

    >>> logger.info("Loading connectome...")
    2026-01-15 10:00:00 - lacuna - INFO - Loading connectome...

    >>> logger.success("Analysis complete", details={"subjects": 10, "time": 42.3})
    2026-01-15 10:00:00 - lacuna - INFO - Analysis complete
    2026-01-15 10:00:00 - lacuna - INFO -   subjects: 10
    2026-01-15 10:00:00 - lacuna - INFO -   time: 42.3
    """

    def __init__(
        self,
        verbose: bool = False,
        width: int = 70,
        indent: str = "  ",
        name: str = "lacuna.analysis",
    ):
        """Initialize console logger."""
        self.verbose = verbose
        self.width = width
        self.indent = indent
        self._logger = logging.getLogger(name)

    def _log(self, message: str, level: int = logging.INFO) -> None:
        """
        Log message if verbose mode is enabled.

        Parameters
        ----------
        message : str
            Message to log
        level : int
            Logging level (default: INFO)
        """
        if self.verbose:
            self._logger.log(level, message)

    def section(self, title: str) -> None:
        """
        Print a major section header.

        Parameters
        ----------
        title : str
            Section title

        Examples
        --------
        >>> logger.section("ANALYSIS PIPELINE")
        """
        if self.verbose:
            separator = "=" * self.width
            self._log("")
            self._log(separator)
            self._log(title)
            self._log(separator)

    def subsection(self, title: str) -> None:
        """
        Print a minor subsection header.

        Parameters
        ----------
        title : str
            Subsection title

        Examples
        --------
        >>> logger.subsection("Loading data")
        """
        if self.verbose:
            separator = "-" * self.width
            self._log("")
            self._log(separator)
            self._log(title)
            self._log(separator)

    def debug(self, message: str, indent_level: int = 0) -> None:
        """
        Print a debug message (only shown with -vv verbosity).

        Parameters
        ----------
        message : str
            Debug message
        indent_level : int, default=0
            Indentation level (0, 1, 2, ...)

        Examples
        --------
        >>> logger.debug("Computing correlation maps")
        """
        indent = self.indent * indent_level
        self._log(f"{indent}{message}", level=logging.DEBUG)

    def info(self, message: str, indent_level: int = 0) -> None:
        """
        Print an informational message.

        Parameters
        ----------
        message : str
            Information message
        indent_level : int, default=0
            Indentation level (0, 1, 2, ...)

        Examples
        --------
        >>> logger.info("Loading mask information...")
        """
        indent = self.indent * indent_level
        self._log(f"{indent}{message}")

    def success(
        self,
        message: str,
        details: dict | None = None,
        indent_level: int = 0,
    ) -> None:
        """
        Print a success message with optional details.

        Parameters
        ----------
        message : str
            Success message
        details : dict, optional
            Dictionary of key-value pairs to display
        indent_level : int, default=0
            Indentation level

        Examples
        --------
        >>> logger.success("Analysis complete", details={"time": 42.3, "subjects": 10})
        """
        indent = self.indent * indent_level
        self._log(f"{indent}{message}")

        if details and self.verbose:
            detail_indent = self.indent * (indent_level + 1)
            for key, value in details.items():
                # Format numbers nicely
                if isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                elif isinstance(value, int) and value >= 1000:
                    formatted_value = f"{value:,}"
                else:
                    formatted_value = str(value)

                self._log(f"{detail_indent}{key}: {formatted_value}")

    def warning(self, message: str, indent_level: int = 0) -> None:
        """
        Print a warning message.

        Parameters
        ----------
        message : str
            Warning message
        indent_level : int, default=0
            Indentation level

        Examples
        --------
        >>> logger.warning("Mask size smaller than expected")
        """
        indent = self.indent * indent_level
        self._log(f"{indent}{message}", level=logging.WARNING)

    def error(self, message: str, indent_level: int = 0) -> None:
        """
        Print an error message.

        Parameters
        ----------
        message : str
            Error message
        indent_level : int, default=0
            Indentation level

        Examples
        --------
        >>> logger.error("Failed to load connectome")
        """
        indent = self.indent * indent_level
        self._log(f"{indent}{message}", level=logging.ERROR)

    def progress(
        self,
        message: str,
        current: int | None = None,
        total: int | None = None,
        percent: float | None = None,
        indent_level: int = 0,
    ) -> None:
        """
        Print a progress update.

        Parameters
        ----------
        message : str
            Progress message
        current : int, optional
            Current item number
        total : int, optional
            Total items
        percent : float, optional
            Completion percentage (0-100)
        indent_level : int, default=0
            Indentation level

        Examples
        --------
        >>> logger.progress("Processing batch", current=3, total=10)
        >>> logger.progress("Loading data", percent=65.5)
        """
        indent = self.indent * indent_level
        progress_str = f"{indent}{message}"

        if current is not None and total is not None:
            progress_str += f" [{current}/{total}]"
        elif percent is not None:
            progress_str += f" [{percent:.1f}%]"

        self._log(progress_str)

    def result_summary(self, title: str, metrics: dict, indent_level: int = 0) -> None:
        """
        Print a formatted summary of results.

        Parameters
        ----------
        title : str
            Summary title
        metrics : dict
            Dictionary of metric name: value pairs
        indent_level : int, default=0
            Indentation level

        Examples
        --------
        >>> logger.result_summary("Analysis Results", {
        ...     "Mean correlation": 0.4523,
        ...     "Std correlation": 0.1234,
        ...     "Range": "[-0.45, 0.89]"
        ... })
        """
        indent = self.indent * indent_level
        self._log(f"{indent}{title}:")

        detail_indent = self.indent * (indent_level + 1)
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, int) and value >= 1000:
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)

            self._log(f"{detail_indent}{key}: {formatted_value}")

    def blank_line(self) -> None:
        """Print a blank line for spacing."""
        if self.verbose:
            self._log("")


# Convenience functions for quick logging without creating logger instance
def log_section(title: str, width: int = 70, verbose: bool = False) -> None:
    """Print a section header."""
    logger = ConsoleLogger(verbose=verbose, width=width)
    logger.section(title)


def log_info(message: str, verbose: bool = False) -> None:
    """Print an info message."""
    logger = ConsoleLogger(verbose=verbose)
    logger.info(message)


def log_success(message: str, details: dict | None = None, verbose: bool = False) -> None:
    """Print a success message."""
    logger = ConsoleLogger(verbose=verbose)
    logger.success(message, details=details)


def log_warning(message: str, verbose: bool = False) -> None:
    """Print a warning message."""
    logger = ConsoleLogger(verbose=verbose)
    logger.warning(message)


def log_error(message: str, verbose: bool = False) -> None:
    """Print an error message."""
    logger = ConsoleLogger(verbose=verbose)
    logger.error(message)


def log_progress(
    message: str,
    current: int | None = None,
    total: int | None = None,
    verbose: bool = False,
) -> None:
    """Print a progress message."""
    logger = ConsoleLogger(verbose=verbose)
    logger.progress(message, current=current, total=total)
