"""
Integration tests for connectome fetching.

These tests require network access and are marked as @pytest.mark.slow.
They test the full fetch workflow including download, processing, and registration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.slow
@pytest.mark.integration
class TestFetchGSP1000Integration:
    """Integration tests for fetch_gsp1000 function."""

    def test_fetch_gsp1000_missing_api_key(self, tmp_path):
        """fetch_gsp1000 should raise AuthenticationError without API key."""
        from lacuna.core.exceptions import AuthenticationError
        from lacuna.io import fetch_gsp1000

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError):
                fetch_gsp1000(output_dir=tmp_path)

    def test_fetch_gsp1000_creates_output_dir(self, tmp_path):
        """fetch_gsp1000 should create output directory if it doesn't exist."""
        from lacuna.core.exceptions import AuthenticationError
        from lacuna.io import fetch_gsp1000

        output_dir = tmp_path / "nonexistent" / "path"
        assert not output_dir.exists()

        # Will fail due to missing API key, but directory should be created
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError):
                fetch_gsp1000(output_dir=output_dir)

        assert output_dir.exists()

    def test_fetch_gsp1000_with_mocked_download(self, tmp_path):
        """fetch_gsp1000 should work with mocked downloader."""
        from lacuna.io import fetch_gsp1000
        from lacuna.io.downloaders.dataverse import DataverseDownloader

        # Create mock NIfTI file structure for testing
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)
        sub_dir = raw_dir / "sub-001" / "func"
        sub_dir.mkdir(parents=True)

        # Create minimal NIfTI-like file (just for testing, not a real NIfTI)
        test_file = sub_dir / "sub-001_bld001_rest_finalmask.nii.gz"
        test_file.write_bytes(b"fake nifti data")

        # Mock the DataverseDownloader
        with patch.object(
            DataverseDownloader,
            "__init__",
            lambda self, source, api_key=None: setattr(self, "source", source)
            or setattr(self, "api_key", "fake")
            or setattr(self, "session", MagicMock()),
        ):
            with patch.object(
                DataverseDownloader,
                "download",
                return_value=[test_file],
            ):
                from lacuna.core.exceptions import ProcessingError

                # This will fail at processing due to fake file, but tests the flow
                with pytest.raises((ProcessingError, OSError, RuntimeError, ValueError)):
                    fetch_gsp1000(
                        output_dir=tmp_path,
                        api_key="fake-key",
                        batches=1,
                        register=False,
                    )


@pytest.mark.slow
@pytest.mark.integration
class TestFetchDtor985Integration:
    """Integration tests for fetch_dtor985 function."""

    def test_fetch_dtor985_creates_output_dir(self, tmp_path):
        """fetch_dtor985 should create output directory if it doesn't exist."""
        from lacuna.io import fetch_dtor985
        from lacuna.io.downloaders.figshare import FigshareDownloader

        output_dir = tmp_path / "nonexistent" / "path"
        assert not output_dir.exists()

        # Mock the download to avoid network call
        with patch.object(
            FigshareDownloader,
            "download",
            return_value=[output_dir / "dtor985.trk"],
        ):
            with patch(
                "lacuna.io.convert.trk_to_tck",
                side_effect=RuntimeError("MRtrix3 not available"),
            ):
                from lacuna.core.exceptions import ProcessingError

                with pytest.raises(ProcessingError):
                    fetch_dtor985(output_dir=output_dir)

        # Directory should be created even if the operation fails later
        assert output_dir.exists()

    def test_fetch_dtor985_with_mocked_download(self, tmp_path):
        """fetch_dtor985 should work with mocked downloader and converter."""
        from lacuna.io import fetch_dtor985
        from lacuna.io.downloaders.figshare import FigshareDownloader

        # Create mock TRK file
        trk_file = tmp_path / "dtor985.trk"
        trk_file.write_bytes(b"fake tractogram data")

        # Mock the FigshareDownloader
        with patch.object(FigshareDownloader, "download", return_value=[trk_file]):
            # Mock trk_to_tck converter
            def mock_convert(src, dst):
                dst.write_bytes(b"converted tractogram")
                return dst

            with patch("lacuna.io.convert.trk_to_tck", side_effect=mock_convert):
                result = fetch_dtor985(
                    output_dir=tmp_path,
                    keep_original=True,
                    register=False,
                )

                assert result.success
                assert result.connectome_name == "dtor985"
                assert len(result.output_files) >= 1


@pytest.mark.slow
@pytest.mark.integration
class TestFetchConnectomeIntegration:
    """Integration tests for fetch_connectome dispatcher."""

    def test_fetch_connectome_gsp1000_dispatch(self, tmp_path):
        """fetch_connectome should dispatch to fetch_gsp1000."""
        from lacuna.core.exceptions import AuthenticationError
        from lacuna.io import fetch_connectome

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError):
                fetch_connectome("gsp1000", output_dir=tmp_path)

    def test_fetch_connectome_dtor985_dispatch(self, tmp_path):
        """fetch_connectome should dispatch to fetch_dtor985."""
        from lacuna.io import fetch_connectome
        from lacuna.io.downloaders.figshare import FigshareDownloader

        # Mock to avoid actual download
        with patch.object(
            FigshareDownloader,
            "download",
            return_value=[tmp_path / "dtor985.trk"],
        ):
            with patch(
                "lacuna.io.convert.trk_to_tck",
                side_effect=RuntimeError("MRtrix3 not available"),
            ):
                from lacuna.core.exceptions import ProcessingError

                with pytest.raises(ProcessingError):
                    fetch_connectome("dtor985", output_dir=tmp_path)

    def test_fetch_connectome_unknown_raises(self, tmp_path):
        """fetch_connectome should raise ValueError for unknown connectome."""
        from lacuna.io import fetch_connectome

        with pytest.raises(ValueError, match="Unknown connectome"):
            fetch_connectome("unknown_connectome", output_dir=tmp_path)


@pytest.mark.integration
class TestListFetchableConnectomes:
    """Integration tests for list_fetchable_connectomes."""

    def test_list_fetchable_connectomes_returns_sources(self):
        """list_fetchable_connectomes should return ConnectomeSource objects."""
        from lacuna.io import list_fetchable_connectomes
        from lacuna.io.downloaders import ConnectomeSource

        sources = list_fetchable_connectomes()

        assert len(sources) >= 2
        assert all(isinstance(s, ConnectomeSource) for s in sources)

        names = [s.name for s in sources]
        assert "gsp1000" in names
        assert "dtor985" in names


@pytest.mark.integration
class TestGetFetchStatus:
    """Integration tests for get_fetch_status."""

    def test_get_fetch_status_returns_dict(self, tmp_path):
        """get_fetch_status should return status dict."""
        from lacuna.io import get_fetch_status

        # Mock the data directory to use tmp_path
        with patch("lacuna.io.fetch.get_data_dir", return_value=tmp_path):
            status = get_fetch_status("gsp1000")

            assert isinstance(status, dict)
            assert "downloaded" in status
            assert "processed" in status
            assert "registered" in status
            assert "location" in status

    def test_get_fetch_status_unknown_raises(self):
        """get_fetch_status should raise ValueError for unknown connectome."""
        from lacuna.io import get_fetch_status

        with pytest.raises(ValueError, match="Unknown connectome"):
            get_fetch_status("unknown_connectome")
