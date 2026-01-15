"""Unit tests for logging utilities."""

import logging

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

    def test_logger_verbose_false(self, caplog):
        """Test that verbose=False suppresses output."""
        logger = ConsoleLogger(verbose=False)

        with caplog.at_level(logging.INFO):
            logger.info("This should not print")
            logger.success("Neither should this")

        assert len(caplog.records) == 0

    def test_section_formatting(self, caplog):
        """Test section header formatting."""
        logger = ConsoleLogger(verbose=True, width=40)

        with caplog.at_level(logging.INFO):
            logger.section("TEST SECTION")

        # Should have 4 records: empty line, separator, title, separator
        messages = [r.message for r in caplog.records]
        assert "=" * 40 in messages
        assert "TEST SECTION" in messages

    def test_subsection_formatting(self, caplog):
        """Test subsection header formatting."""
        logger = ConsoleLogger(verbose=True, width=40)

        with caplog.at_level(logging.INFO):
            logger.subsection("Test Subsection")

        messages = [r.message for r in caplog.records]
        assert "-" * 40 in messages
        assert "Test Subsection" in messages

    def test_info_message(self, caplog):
        """Test info message formatting."""
        logger = ConsoleLogger(verbose=True)

        with caplog.at_level(logging.INFO):
            logger.info("Loading data...")

        assert "Loading data..." in caplog.text

    def test_success_message_without_details(self, caplog):
        """Test success message without details."""
        logger = ConsoleLogger(verbose=True)

        with caplog.at_level(logging.INFO):
            logger.success("Analysis complete")

        assert "Analysis complete" in caplog.text

    def test_success_message_with_details(self, caplog):
        """Test success message with details dictionary."""
        logger = ConsoleLogger(verbose=True)

        with caplog.at_level(logging.INFO):
            logger.success("Analysis complete", details={"time": 42.3, "subjects": 10})

        assert "Analysis complete" in caplog.text
        assert "time: 42.30" in caplog.text
        assert "subjects: 10" in caplog.text

    def test_warning_message(self, caplog):
        """Test warning message formatting."""
        logger = ConsoleLogger(verbose=True)

        with caplog.at_level(logging.WARNING):
            logger.warning("Mask size smaller than expected")

        assert "Mask size smaller than expected" in caplog.text
        assert caplog.records[0].levelno == logging.WARNING

    def test_error_message(self, caplog):
        """Test error message formatting."""
        logger = ConsoleLogger(verbose=True)

        with caplog.at_level(logging.ERROR):
            logger.error("Failed to load connectome")

        assert "Failed to load connectome" in caplog.text
        assert caplog.records[0].levelno == logging.ERROR

    def test_progress_with_current_total(self, caplog):
        """Test progress message with current/total."""
        logger = ConsoleLogger(verbose=True)

        with caplog.at_level(logging.INFO):
            logger.progress("Processing batch", current=3, total=10)

        assert "Processing batch" in caplog.text
        assert "[3/10]" in caplog.text

    def test_progress_with_percent(self, caplog):
        """Test progress message with percentage."""
        logger = ConsoleLogger(verbose=True)

        with caplog.at_level(logging.INFO):
            logger.progress("Loading data", percent=65.5)

        assert "Loading data" in caplog.text
        assert "[65.5%]" in caplog.text

    def test_result_summary(self, caplog):
        """Test result summary formatting."""
        logger = ConsoleLogger(verbose=True)

        with caplog.at_level(logging.INFO):
            logger.result_summary(
                "Analysis Results",
                {"Mean correlation": 0.4523, "Std correlation": 0.1234, "Range": "[-0.45, 0.89]"},
            )

        assert "Analysis Results:" in caplog.text
        assert "Mean correlation: 0.4523" in caplog.text
        assert "Std correlation: 0.1234" in caplog.text
        assert "Range: [-0.45, 0.89]" in caplog.text

    def test_indentation_levels(self, caplog):
        """Test that indentation works correctly."""
        logger = ConsoleLogger(verbose=True, indent="  ")

        with caplog.at_level(logging.INFO):
            logger.info("Level 0", indent_level=0)
            logger.info("Level 1", indent_level=1)
            logger.info("Level 2", indent_level=2)

        messages = [r.message for r in caplog.records]
        assert messages[0] == "Level 0"
        assert messages[1] == "  Level 1"
        assert messages[2] == "    Level 2"

    def test_blank_line(self, caplog):
        """Test blank line output."""
        logger = ConsoleLogger(verbose=True)

        with caplog.at_level(logging.INFO):
            logger.info("First")
            logger.blank_line()
            logger.info("Second")

        # Should have 3 records
        assert len(caplog.records) == 3


class TestConvenienceFunctions:
    """Tests for convenience logging functions."""

    def test_log_section(self, caplog):
        """Test log_section convenience function."""
        with caplog.at_level(logging.INFO):
            log_section("TEST", verbose=True)

        assert "TEST" in caplog.text

    def test_log_info(self, caplog):
        """Test log_info convenience function."""
        with caplog.at_level(logging.INFO):
            log_info("Test message", verbose=True)

        assert "Test message" in caplog.text

    def test_log_success(self, caplog):
        """Test log_success convenience function."""
        with caplog.at_level(logging.INFO):
            log_success("Success message", details={"count": 5}, verbose=True)

        assert "Success message" in caplog.text
        assert "count: 5" in caplog.text

    def test_log_warning(self, caplog):
        """Test log_warning convenience function."""
        with caplog.at_level(logging.WARNING):
            log_warning("Warning message", verbose=True)

        assert "Warning message" in caplog.text

    def test_log_error(self, caplog):
        """Test log_error convenience function."""
        with caplog.at_level(logging.ERROR):
            log_error("Error message", verbose=True)

        assert "Error message" in caplog.text

    def test_log_progress(self, caplog):
        """Test log_progress convenience function."""
        with caplog.at_level(logging.INFO):
            log_progress("Processing", current=5, total=10, verbose=True)

        assert "Processing" in caplog.text
        assert "[5/10]" in caplog.text


class TestMessageType:
    """Tests for MessageType enum."""

    def test_message_type_values(self):
        """Test that all message types have values."""
        # Symbols removed - now empty strings
        assert MessageType.INFO.value == ""
        assert MessageType.SUCCESS.value == ""
        assert MessageType.WARNING.value == ""
        assert MessageType.ERROR.value == ""
        assert MessageType.PROGRESS.value == ""
        assert MessageType.SECTION.value == "="
        assert MessageType.SUBSECTION.value == "-"


class TestNumberFormatting:
    """Tests for number formatting in logger output."""

    def test_float_formatting(self, caplog):
        """Test that floats are formatted correctly."""
        logger = ConsoleLogger(verbose=True)

        with caplog.at_level(logging.INFO):
            logger.success("Test", details={"value": 3.14159})

        assert "value: 3.14" in caplog.text

    def test_large_int_formatting(self, caplog):
        """Test that large integers get comma separators."""
        logger = ConsoleLogger(verbose=True)

        with caplog.at_level(logging.INFO):
            logger.success("Test", details={"count": 1234567})

        assert "count: 1,234,567" in caplog.text

    def test_small_int_no_separator(self, caplog):
        """Test that small integers don't get separators."""
        logger = ConsoleLogger(verbose=True)

        with caplog.at_level(logging.INFO):
            logger.success("Test", details={"count": 999})

        assert "count: 999" in caplog.text
        assert "1," not in caplog.text  # No comma for 999
