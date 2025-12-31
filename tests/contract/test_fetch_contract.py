"""
Contract tests for the connectome fetching API.

These tests define the expected API signatures and behaviors for the
fetch functions. Implementation must satisfy these contracts.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import get_type_hints


class TestFetchGSP1000Contract:
    """Contract tests for fetch_gsp1000 function."""

    def test_fetch_gsp1000_exists_in_module(self):
        """fetch_gsp1000 should be importable from lacuna.io."""
        from lacuna.io import fetch_gsp1000

        assert callable(fetch_gsp1000)

    def test_fetch_gsp1000_signature(self):
        """fetch_gsp1000 should have the expected parameter signature."""
        from lacuna.io import fetch_gsp1000

        sig = inspect.signature(fetch_gsp1000)
        params = sig.parameters

        # Required parameters
        assert "output_dir" in params
        assert params["output_dir"].kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        )

        # Optional keyword-only parameters
        assert "api_key" in params
        assert params["api_key"].default is None

        assert "batches" in params
        assert params["batches"].default == 10

        assert "register" in params
        assert params["register"].default is True

        assert "force" in params
        assert params["force"].default is False

        assert "progress_callback" in params
        assert params["progress_callback"].default is None

    def test_fetch_gsp1000_returns_fetch_result(self):
        """fetch_gsp1000 should return FetchResult."""
        from lacuna.io import fetch_gsp1000
        from lacuna.io.downloaders import FetchResult

        hints = get_type_hints(fetch_gsp1000)
        assert "return" in hints
        assert hints["return"] is FetchResult

    def test_fetch_gsp1000_accepts_path_like(self):
        """fetch_gsp1000 should accept str or Path for output_dir."""
        from lacuna.io import fetch_gsp1000

        hints = get_type_hints(fetch_gsp1000)
        # Should accept str | Path
        assert "output_dir" in hints


class TestFetchdTOR985Contract:
    """Contract tests for fetch_dtor985 function."""

    def test_fetch_dtor985_exists_in_module(self):
        """fetch_dtor985 should be importable from lacuna.io."""
        from lacuna.io import fetch_dtor985

        assert callable(fetch_dtor985)

    def test_fetch_dtor985_signature(self):
        """fetch_dtor985 should have the expected parameter signature."""
        from lacuna.io import fetch_dtor985

        sig = inspect.signature(fetch_dtor985)
        params = sig.parameters

        # Required parameters
        assert "output_dir" in params

        # Optional keyword-only parameters
        assert "keep_original" in params
        assert params["keep_original"].default is True

        assert "register" in params
        assert params["register"].default is True

        assert "force" in params
        assert params["force"].default is False

        assert "progress_callback" in params
        assert params["progress_callback"].default is None

    def test_fetch_dtor985_returns_fetch_result(self):
        """fetch_dtor985 should return FetchResult."""
        from lacuna.io import fetch_dtor985
        from lacuna.io.downloaders import FetchResult

        hints = get_type_hints(fetch_dtor985)
        assert "return" in hints
        assert hints["return"] is FetchResult


class TestFetchConnectomeContract:
    """Contract tests for fetch_connectome dispatcher function."""

    def test_fetch_connectome_exists_in_module(self):
        """fetch_connectome should be importable from lacuna.io."""
        from lacuna.io import fetch_connectome

        assert callable(fetch_connectome)

    def test_fetch_connectome_signature(self):
        """fetch_connectome should have name as first positional arg."""
        from lacuna.io import fetch_connectome

        sig = inspect.signature(fetch_connectome)
        params = list(sig.parameters.keys())

        # First param should be name
        assert params[0] == "name"
        # Second param should be output_dir
        assert params[1] == "output_dir"


class TestListFetchableConnectomesContract:
    """Contract tests for list_fetchable_connectomes function."""

    def test_list_fetchable_connectomes_exists_in_module(self):
        """list_fetchable_connectomes should be importable from lacuna.io."""
        from lacuna.io import list_fetchable_connectomes

        assert callable(list_fetchable_connectomes)

    def test_list_fetchable_connectomes_returns_list(self):
        """list_fetchable_connectomes should return list[ConnectomeSource]."""
        from lacuna.io import list_fetchable_connectomes
        from lacuna.io.downloaders import ConnectomeSource

        result = list_fetchable_connectomes()
        assert isinstance(result, list)
        assert len(result) >= 2  # At least gsp1000 and dtor985
        assert all(isinstance(s, ConnectomeSource) for s in result)

    def test_list_includes_gsp1000_and_dtor985(self):
        """list_fetchable_connectomes should include gsp1000 and dtor985."""
        from lacuna.io import list_fetchable_connectomes

        result = list_fetchable_connectomes()
        names = [s.name for s in result]
        assert "gsp1000" in names
        assert "dtor985" in names


class TestConnectomeSourcesRegistry:
    """Contract tests for CONNECTOME_SOURCES registry."""

    def test_connectome_sources_is_dict(self):
        """CONNECTOME_SOURCES should be a dict."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES

        assert isinstance(CONNECTOME_SOURCES, dict)

    def test_connectome_sources_has_gsp1000(self):
        """CONNECTOME_SOURCES should have gsp1000 entry."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES, ConnectomeSource

        assert "gsp1000" in CONNECTOME_SOURCES
        gsp = CONNECTOME_SOURCES["gsp1000"]
        assert isinstance(gsp, ConnectomeSource)
        assert gsp.type == "functional"
        assert gsp.source_type == "dataverse"

    def test_connectome_sources_has_dtor985(self):
        """CONNECTOME_SOURCES should have dtor985 entry."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES, ConnectomeSource

        assert "dtor985" in CONNECTOME_SOURCES
        dtor = CONNECTOME_SOURCES["dtor985"]
        assert isinstance(dtor, ConnectomeSource)
        assert dtor.type == "structural"
        assert dtor.source_type == "figshare"


class TestFetchExceptionsContract:
    """Contract tests for fetch-related exceptions."""

    def test_fetch_exceptions_exist(self):
        """All fetch exceptions should be importable from lacuna.core.exceptions."""
        from lacuna.core.exceptions import (
            AuthenticationError,
            ChecksumError,
            DownloadError,
            FetchError,
            ProcessingError,
        )

        # All should inherit from FetchError
        assert issubclass(AuthenticationError, FetchError)
        assert issubclass(DownloadError, FetchError)
        assert issubclass(ProcessingError, FetchError)
        assert issubclass(ChecksumError, FetchError)

    def test_fetch_error_inherits_from_lacuna_error(self):
        """FetchError should inherit from LacunaError."""
        from lacuna.core.exceptions import FetchError, LacunaError

        assert issubclass(FetchError, LacunaError)


class TestFetchResultContract:
    """Contract tests for FetchResult dataclass."""

    def test_fetch_result_fields(self):
        """FetchResult should have expected fields."""
        from lacuna.io.downloaders import FetchResult

        # Create a minimal instance
        result = FetchResult(
            success=True,
            connectome_name="test",
            output_dir=Path("/tmp"),
        )

        # Check expected attributes
        assert hasattr(result, "success")
        assert hasattr(result, "connectome_name")
        assert hasattr(result, "output_dir")
        assert hasattr(result, "output_files")
        assert hasattr(result, "registered")
        assert hasattr(result, "register_name")
        assert hasattr(result, "duration_seconds")
        assert hasattr(result, "warnings")
        assert hasattr(result, "error")

    def test_fetch_result_summary_method(self):
        """FetchResult should have a summary() method."""
        from lacuna.io.downloaders import FetchResult

        result = FetchResult(
            success=True,
            connectome_name="test",
            output_dir=Path("/tmp"),
        )

        assert hasattr(result, "summary")
        assert callable(result.summary)
        summary = result.summary()
        assert isinstance(summary, str)


class TestFetchProgressContract:
    """Contract tests for FetchProgress dataclass."""

    def test_fetch_progress_fields(self):
        """FetchProgress should have expected fields."""
        from lacuna.io.downloaders import FetchProgress

        progress = FetchProgress(
            phase="download",
            current_file="test.nii.gz",
            files_completed=0,
            files_total=10,
        )

        assert hasattr(progress, "phase")
        assert hasattr(progress, "current_file")
        assert hasattr(progress, "files_completed")
        assert hasattr(progress, "files_total")
        assert hasattr(progress, "bytes_transferred")
        assert hasattr(progress, "bytes_total")
        assert hasattr(progress, "message")

    def test_fetch_progress_percent_properties(self):
        """FetchProgress should have percent calculation properties."""
        from lacuna.io.downloaders import FetchProgress

        progress = FetchProgress(
            phase="download",
            current_file="test.nii.gz",
            files_completed=5,
            files_total=10,
            bytes_transferred=500,
            bytes_total=1000,
        )

        assert progress.percent_complete == 50.0
        assert progress.download_percent == 50.0
