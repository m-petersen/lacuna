"""
Consistent logging and user message formatting for LDK.

Provides a unified system for displaying progress, success, warnings, and errors
to users with consistent formatting across all modules.
"""

from enum import Enum


class MessageType(Enum):
    """Types of messages that can be displayed."""

    INFO = "·"  # General information
    SUCCESS = "✓"  # Operation completed successfully
    WARNING = "⚡"  # Warning message
    ERROR = "✗"  # Error message
    PROGRESS = "→"  # Progress update
    SECTION = "="  # Section header
    SUBSECTION = "-"  # Subsection header


class ConsoleLogger:
    """
    Consistent console logger for user-facing messages.

    Provides formatted output with consistent symbols, indentation, and styling
    across all LDK modules.

    Parameters
    ----------
    log_level : int, default=1
        Logging verbosity level:
        - 0: Silent (no output)
        - 1: Standard (high-level progress, results, summaries)
        - 2: Verbose (detailed operations, resampling, timing)
    verbose : bool, default=True
        Deprecated. Use log_level instead. If provided, overrides log_level.
    width : int, default=70
        Width for section headers
    indent : str, default="  "
        Indentation string for nested messages

    Examples
    --------
    >>> logger = ConsoleLogger(log_level=1)
    >>> logger.section("PROCESSING DATA")
    ===========================================
    PROCESSING DATA
    ===========================================

    >>> logger.info("Loading connectome...")
    ·  Loading connectome...

    >>> logger.success("Analysis complete", details={"subjects": 10, "time": 42.3})
    ✓ Analysis complete
      - subjects: 10
      - time: 42.3s

    >>> logger.progress("Batch 3/10", percent=30)
    → Batch 3/10 [30%]
    """

    def __init__(
        self,
        log_level: int = 1,
        verbose: bool | None = None,
        width: int = 70,
        indent: str = "  ",
    ):
        """Initialize console logger."""
        # Backward compatibility: if verbose is explicitly provided, map to log_level
        if verbose is not None:
            self.log_level = 1 if verbose else 0
        else:
            self.log_level = log_level

        self.width = width
        self.indent = indent

    @property
    def verbose(self) -> bool:
        """Backward compatibility property for verbose attribute."""
        return self.log_level > 0

    def _print(self, message: str, min_level: int = 1) -> None:
        """
        Print message if log level is sufficient.

        Parameters
        ----------
        message : str
            Message to print
        min_level : int, default=1
            Minimum log level required to display this message
            - Use 1 for standard messages (default)
            - Use 2 for verbose/debug messages
        """
        if self.log_level >= min_level:
            print(message, flush=True)

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
        if self.log_level >= 1:
            separator = "=" * self.width
            self._print(f"\n{separator}", min_level=1)
            self._print(title, min_level=1)
            self._print(separator, min_level=1)

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
        if self.log_level >= 1:
            separator = "-" * self.width
            self._print(f"\n{separator}", min_level=1)
            self._print(title, min_level=1)
            self._print(separator, min_level=1)

    def info(self, message: str, indent_level: int = 0, verbose: bool = False) -> None:
        """
        Print an informational message.

        Parameters
        ----------
        message : str
            Information message
        indent_level : int, default=0
            Indentation level (0, 1, 2, ...)
        verbose : bool, default=False
            If True, only show at log_level=2 (verbose mode).
            If False, show at log_level=1 (standard mode).

        Examples
        --------
        >>> logger.info("Loading mask information...")
        ·  Loading mask information..."
        """
        min_level = 2 if verbose else 1
        indent = self.indent * indent_level
        self._print(f"{indent}{MessageType.INFO.value}  {message}", min_level=min_level)

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
        ✓ Analysis complete
          - time: 42.3s
          - subjects: 10
        """
        indent = self.indent * indent_level
        self._print(f"{indent}{MessageType.SUCCESS.value} {message}", min_level=1)

        if details and self.log_level >= 1:
            detail_indent = self.indent * (indent_level + 1)
            for key, value in details.items():
                # Format numbers nicely
                if isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                elif isinstance(value, int) and value >= 1000:
                    formatted_value = f"{value:,}"
                else:
                    formatted_value = str(value)

                self._print(f"{detail_indent}- {key}: {formatted_value}", min_level=1)

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
        >>> logger.warning("Lesion size smaller than expected")
        ⚡  Lesion size smaller than expected"
        """
        indent = self.indent * indent_level
        self._print(f"{indent}{MessageType.WARNING.value}  {message}", min_level=1)

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
        ✗ Failed to load connectome"
        """
        indent = self.indent * indent_level
        self._print(f"{indent}{MessageType.ERROR.value} {message}", min_level=1)

    def progress(
        self,
        message: str,
        current: int | None = None,
        total: int | None = None,
        percent: float | None = None,
        indent_level: int = 0,
        verbose: bool = False,
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
        verbose : bool, default=False
            If True, only show at log_level=2 (verbose mode).
            If False, show at log_level=1 (standard mode).

        Examples
        --------
        >>> logger.progress("Processing batch", current=3, total=10)
        →  Processing batch [3/10]

        >>> logger.progress("Loading data", percent=65.5)
        →  Loading data [65.5%]"
        """
        min_level = 2 if verbose else 1
        indent = self.indent * indent_level
        progress_str = f"{indent}{MessageType.PROGRESS.value}  {message}"

        if current is not None and total is not None:
            progress_str += f" [{current}/{total}]"
        elif percent is not None:
            progress_str += f" [{percent:.1f}%]"

        self._print(progress_str, min_level=min_level)

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
        Analysis Results:
          - Mean correlation: 0.4523
          - Std correlation: 0.1234
          - Range: [-0.45, 0.89]
        """
        indent = self.indent * indent_level
        self._print(f"{indent}{title}:", min_level=1)

        detail_indent = self.indent * (indent_level + 1)
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, int) and value >= 1000:
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)

            self._print(f"{detail_indent}- {key}: {formatted_value}", min_level=1)

    def blank_line(self) -> None:
        """Print a blank line for spacing."""
        if self.log_level >= 1:
            print()


# Convenience functions for quick logging without creating logger instance
def log_section(title: str, width: int = 70, log_level: int = 1) -> None:
    """Print a section header."""
    logger = ConsoleLogger(log_level=log_level, width=width)
    logger.section(title)


def log_info(message: str, log_level: int = 1, verbose: bool = False) -> None:
    """Print an info message."""
    logger = ConsoleLogger(log_level=log_level)
    logger.info(message, verbose=verbose)


def log_success(message: str, details: dict | None = None, log_level: int = 1) -> None:
    """Print a success message."""
    logger = ConsoleLogger(log_level=log_level)
    logger.success(message, details=details)


def log_warning(message: str, log_level: int = 1) -> None:
    """Print a warning message."""
    logger = ConsoleLogger(log_level=log_level)
    logger.warning(message)


def log_error(message: str, log_level: int = 1) -> None:
    """Print an error message."""
    logger = ConsoleLogger(log_level=log_level)
    logger.error(message)


def log_progress(
    message: str,
    current: int | None = None,
    total: int | None = None,
    log_level: int = 1,
    verbose: bool = False,
) -> None:
    """Print a progress message."""
    logger = ConsoleLogger(log_level=log_level)
    logger.progress(message, current=current, total=total, verbose=verbose)
