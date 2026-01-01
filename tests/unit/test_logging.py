"""Unit tests for logging utilities."""

import io
from contextlib import redirect_stdout

from lacuna.utils.logging import (
    ConsoleLogger,
    MessageType,
    log_error,
    log_info,
    log_progress,
    log_section,
    log_success,
    log_warning,
)


class TestConsoleLogger:
    """Tests for ConsoleLogger class."""

    def test_logger_initialization(self):
        """Test logger can be initialized with different parameters."""
        logger = ConsoleLogger(verbose=True, width=80, indent="    ")
        assert logger.verbose is True
        assert logger.width == 80
        assert logger.indent == "    "

    def test_logger_verbose_false(self):
        """Test that verbose=False suppresses output."""
        logger = ConsoleLogger(verbose=False)

        output = io.StringIO()
        with redirect_stdout(output):
            logger.info("This should not print")
            logger.success("Neither should this")

        assert output.getvalue() == ""

    def test_section_formatting(self):
        """Test section header formatting."""
        logger = ConsoleLogger(verbose=True, width=40)

        output = io.StringIO()
        with redirect_stdout(output):
            logger.section("TEST SECTION")

        result = output.getvalue()
        assert "=" * 40 in result
        assert "TEST SECTION" in result
        assert result.count("=" * 40) == 2  # Top and bottom separator

    def test_subsection_formatting(self):
        """Test subsection header formatting."""
        logger = ConsoleLogger(verbose=True, width=40)

        output = io.StringIO()
        with redirect_stdout(output):
            logger.subsection("Test Subsection")

        result = output.getvalue()
        assert "-" * 40 in result
        assert "Test Subsection" in result
        assert result.count("-" * 40) == 2

    def test_info_message(self):
        """Test info message formatting."""
        logger = ConsoleLogger(verbose=True)

        output = io.StringIO()
        with redirect_stdout(output):
            logger.info("Loading data...")

        result = output.getvalue()
        assert MessageType.INFO.value in result
        assert "Loading data..." in result

    def test_success_message_without_details(self):
        """Test success message without details."""
        logger = ConsoleLogger(verbose=True)

        output = io.StringIO()
        with redirect_stdout(output):
            logger.success("Analysis complete")

        result = output.getvalue()
        assert MessageType.SUCCESS.value in result
        assert "Analysis complete" in result

    def test_success_message_with_details(self):
        """Test success message with details dictionary."""
        logger = ConsoleLogger(verbose=True)

        output = io.StringIO()
        with redirect_stdout(output):
            logger.success("Analysis complete", details={"time": 42.3, "subjects": 10})

        result = output.getvalue()
        assert MessageType.SUCCESS.value in result
        assert "Analysis complete" in result
        assert "time: 42.30" in result
        assert "subjects: 10" in result

    def test_warning_message(self):
        """Test warning message formatting."""
        logger = ConsoleLogger(verbose=True)

        output = io.StringIO()
        with redirect_stdout(output):
            logger.warning("Lesion size smaller than expected")

        result = output.getvalue()
        assert MessageType.WARNING.value in result
        assert "Lesion size smaller than expected" in result

    def test_error_message(self):
        """Test error message formatting."""
        logger = ConsoleLogger(verbose=True)

        output = io.StringIO()
        with redirect_stdout(output):
            logger.error("Failed to load connectome")

        result = output.getvalue()
        assert MessageType.ERROR.value in result
        assert "Failed to load connectome" in result

    def test_progress_with_current_total(self):
        """Test progress message with current/total."""
        logger = ConsoleLogger(verbose=True)

        output = io.StringIO()
        with redirect_stdout(output):
            logger.progress("Processing batch", current=3, total=10)

        result = output.getvalue()
        assert MessageType.PROGRESS.value in result
        assert "Processing batch" in result
        assert "[3/10]" in result

    def test_progress_with_percent(self):
        """Test progress message with percentage."""
        logger = ConsoleLogger(verbose=True)

        output = io.StringIO()
        with redirect_stdout(output):
            logger.progress("Loading data", percent=65.5)

        result = output.getvalue()
        assert MessageType.PROGRESS.value in result
        assert "Loading data" in result
        assert "[65.5%]" in result

    def test_result_summary(self):
        """Test result summary formatting."""
        logger = ConsoleLogger(verbose=True)

        output = io.StringIO()
        with redirect_stdout(output):
            logger.result_summary(
                "Analysis Results",
                {"Mean correlation": 0.4523, "Std correlation": 0.1234, "Range": "[-0.45, 0.89]"},
            )

        result = output.getvalue()
        assert "Analysis Results:" in result
        assert "Mean correlation: 0.4523" in result
        assert "Std correlation: 0.1234" in result
        assert "Range: [-0.45, 0.89]" in result

    def test_indentation_levels(self):
        """Test that indentation works correctly."""
        logger = ConsoleLogger(verbose=True, indent="  ")

        output = io.StringIO()
        with redirect_stdout(output):
            logger.info("Level 0", indent_level=0)
            logger.info("Level 1", indent_level=1)
            logger.info("Level 2", indent_level=2)

        lines = output.getvalue().split("\n")
        # Check that indentation increases
        assert lines[0].startswith(MessageType.INFO.value)
        assert lines[1].startswith("  " + MessageType.INFO.value)
        assert lines[2].startswith("    " + MessageType.INFO.value)

    def test_blank_line(self):
        """Test blank line output."""
        logger = ConsoleLogger(verbose=True)

        output = io.StringIO()
        with redirect_stdout(output):
            logger.info("First")
            logger.blank_line()
            logger.info("Second")

        result = output.getvalue()
        assert "\n\n" in result  # Blank line creates extra newline


class TestConvenienceFunctions:
    """Tests for convenience logging functions."""

    def test_log_section(self):
        """Test log_section convenience function."""
        output = io.StringIO()
        with redirect_stdout(output):
            log_section("TEST", verbose=True)

        result = output.getvalue()
        assert "TEST" in result
        assert "=" in result

    def test_log_info(self):
        """Test log_info convenience function."""
        output = io.StringIO()
        with redirect_stdout(output):
            log_info("Test message", verbose=True)

        result = output.getvalue()
        assert MessageType.INFO.value in result
        assert "Test message" in result

    def test_log_success(self):
        """Test log_success convenience function."""
        output = io.StringIO()
        with redirect_stdout(output):
            log_success("Success message", details={"count": 5}, verbose=True)

        result = output.getvalue()
        assert MessageType.SUCCESS.value in result
        assert "Success message" in result
        assert "count: 5" in result

    def test_log_warning(self):
        """Test log_warning convenience function."""
        output = io.StringIO()
        with redirect_stdout(output):
            log_warning("Warning message", verbose=True)

        result = output.getvalue()
        assert MessageType.WARNING.value in result
        assert "Warning message" in result

    def test_log_error(self):
        """Test log_error convenience function."""
        output = io.StringIO()
        with redirect_stdout(output):
            log_error("Error message", verbose=True)

        result = output.getvalue()
        assert MessageType.ERROR.value in result
        assert "Error message" in result

    def test_log_progress(self):
        """Test log_progress convenience function."""
        output = io.StringIO()
        with redirect_stdout(output):
            log_progress("Processing", current=5, total=10, verbose=True)

        result = output.getvalue()
        assert MessageType.PROGRESS.value in result
        assert "Processing" in result
        assert "[5/10]" in result


class TestMessageType:
    """Tests for MessageType enum."""

    def test_message_type_values(self):
        """Test that all message types have values."""
        assert MessageType.INFO.value == "·"
        assert MessageType.SUCCESS.value == "✓"
        assert MessageType.WARNING.value == "⚡"
        assert MessageType.ERROR.value == "✗"
        assert MessageType.PROGRESS.value == "→"
        assert MessageType.SECTION.value == "="
        assert MessageType.SUBSECTION.value == "-"


class TestNumberFormatting:
    """Tests for number formatting in logger output."""

    def test_float_formatting(self):
        """Test that floats are formatted correctly."""
        logger = ConsoleLogger(verbose=True)

        output = io.StringIO()
        with redirect_stdout(output):
            logger.success("Test", details={"value": 3.14159})

        result = output.getvalue()
        assert "value: 3.14" in result

    def test_large_int_formatting(self):
        """Test that large integers get comma separators."""
        logger = ConsoleLogger(verbose=True)

        output = io.StringIO()
        with redirect_stdout(output):
            logger.success("Test", details={"count": 1234567})

        result = output.getvalue()
        assert "count: 1,234,567" in result

    def test_small_int_no_separator(self):
        """Test that small integers don't get separators."""
        logger = ConsoleLogger(verbose=True)

        output = io.StringIO()
        with redirect_stdout(output):
            logger.success("Test", details={"count": 999})

        result = output.getvalue()
        assert "count: 999" in result
        assert "," not in result
