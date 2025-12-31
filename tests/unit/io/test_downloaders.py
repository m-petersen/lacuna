"""
Unit tests for connectome downloaders.

These tests use mocked HTTP responses to test downloader functionality
without making actual network requests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestDataverseDownloader:
    """Unit tests for DataverseDownloader."""

    def test_dataverse_downloader_requires_api_key(self):
        """DataverseDownloader should raise AuthenticationError without API key."""
        from lacuna.core.exceptions import AuthenticationError
        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.dataverse import DataverseDownloader

        source = CONNECTOME_SOURCES["gsp1000"]

        # Clear any env var that might be set
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError) as exc_info:
                DataverseDownloader(source, api_key=None)

            assert "API key required" in str(exc_info.value)

    def test_dataverse_downloader_accepts_api_key(self):
        """DataverseDownloader should accept API key via constructor."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.dataverse import DataverseDownloader

        source = CONNECTOME_SOURCES["gsp1000"]
        downloader = DataverseDownloader(source, api_key="test-key")

        assert downloader.api_key == "test-key"
        assert downloader.session.headers["X-Dataverse-key"] == "test-key"

    def test_dataverse_downloader_uses_env_var(self):
        """DataverseDownloader should use DATAVERSE_API_KEY env var."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.dataverse import DataverseDownloader

        source = CONNECTOME_SOURCES["gsp1000"]

        with patch.dict("os.environ", {"DATAVERSE_API_KEY": "env-key"}):
            downloader = DataverseDownloader(source)
            assert downloader.api_key == "env-key"

    def test_get_dataset_files_parses_response(self):
        """_get_dataset_files should parse Dataverse API response correctly."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.dataverse import DataverseDownloader

        source = CONNECTOME_SOURCES["gsp1000"]

        mock_response = {
            "status": "OK",
            "data": [
                {
                    "dataFile": {
                        "id": 12345,
                        "filename": "test_file.nii.gz",
                        "filesize": 1024,
                        "checksum": {"type": "MD5", "value": "abc123"},
                    }
                }
            ],
        }

        with patch.dict("os.environ", {"DATAVERSE_API_KEY": "test-key"}):
            downloader = DataverseDownloader(source)

            with patch.object(downloader.session, "get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.json.return_value = mock_response
                mock_resp.status_code = 200
                mock_get.return_value = mock_resp

                files = downloader._get_dataset_files()

                assert len(files) == 1
                assert files[0]["id"] == 12345
                assert files[0]["filename"] == "test_file.nii.gz"
                assert files[0]["checksum"] == "abc123"

    def test_verify_checksum_correct(self, tmp_path):
        """_verify_checksum should return True for matching checksum."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.dataverse import DataverseDownloader

        source = CONNECTOME_SOURCES["gsp1000"]

        # Create test file with known content
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        # MD5 of "hello world"
        expected_md5 = "5eb63bbbe01eeed093cb22bb8f5acdc3"

        with patch.dict("os.environ", {"DATAVERSE_API_KEY": "test-key"}):
            downloader = DataverseDownloader(source)
            assert downloader._verify_checksum(test_file, expected_md5) is True
            assert downloader._verify_checksum(test_file, "wrong-hash") is False


class TestFigshareDownloader:
    """Unit tests for FigshareDownloader."""

    def test_figshare_downloader_creates_scraper(self):
        """FigshareDownloader should create cloudscraper instance."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.figshare import FigshareDownloader

        source = CONNECTOME_SOURCES["dtor985"]
        downloader = FigshareDownloader(source)

        assert downloader.scraper is not None

    def test_figshare_downloader_extracts_filename(self):
        """FigshareDownloader should extract filename from URL."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.figshare import FigshareDownloader

        source = CONNECTOME_SOURCES["dtor985"]
        downloader = FigshareDownloader(source)

        # Test URL with filename
        url = "https://figshare.com/ndownloader/files/12345/tractogram.trk"
        filename = downloader._get_filename_from_url(url)
        assert filename == "tractogram.trk"

        # Test URL without clear filename
        url = "https://figshare.com/ndownloader/files/12345"
        filename = downloader._get_filename_from_url(url)
        assert filename is None

    def test_figshare_download_requires_url(self, tmp_path):
        """FigshareDownloader should raise DownloadError if no URL configured."""
        from lacuna.core.exceptions import DownloadError
        from lacuna.io.downloaders import ConnectomeSource
        from lacuna.io.downloaders.figshare import FigshareDownloader

        source = ConnectomeSource(
            name="test",
            display_name="Test",
            type="structural",
            description="Test",
            source_type="figshare",
            download_url=None,  # No URL
        )

        downloader = FigshareDownloader(source)

        with pytest.raises(DownloadError) as exc_info:
            downloader.download(tmp_path)

        assert "No download_url configured" in str(exc_info.value)

    def test_figshare_download_with_mock(self, tmp_path):
        """FigshareDownloader should download file using cloudscraper."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.figshare import FigshareDownloader

        source = CONNECTOME_SOURCES["dtor985"]
        downloader = FigshareDownloader(source)

        # Mock the scraper.get method
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content.return_value = [b"test data"]

        with patch.object(downloader.scraper, "get", return_value=mock_response):
            files = downloader.download(tmp_path)

            assert len(files) == 1
            assert files[0].exists()


class TestGetApiKey:
    """Unit tests for get_api_key helper function."""

    def test_get_api_key_prefers_cli_arg(self):
        """get_api_key should prefer CLI argument over env var."""
        from lacuna.io.downloaders import get_api_key

        with patch.dict("os.environ", {"DATAVERSE_API_KEY": "env-key"}):
            result = get_api_key(cli_key="cli-key")
            assert result == "cli-key"

    def test_get_api_key_falls_back_to_env(self):
        """get_api_key should use env var if no CLI arg."""
        from lacuna.io.downloaders import get_api_key

        with patch.dict("os.environ", {"DATAVERSE_API_KEY": "env-key"}):
            result = get_api_key(cli_key=None)
            assert result == "env-key"

    def test_get_api_key_returns_none_if_not_found(self):
        """get_api_key should return None if no key found anywhere."""
        from lacuna.io.downloaders import get_api_key

        with patch.dict("os.environ", {}, clear=True):
            result = get_api_key(cli_key=None)
            assert result is None


class TestConnectomeSources:
    """Unit tests for CONNECTOME_SOURCES registry."""

    def test_gsp1000_source_configuration(self):
        """GSP1000 source should have correct configuration."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES

        gsp = CONNECTOME_SOURCES["gsp1000"]

        assert gsp.name == "gsp1000"
        assert gsp.type == "functional"
        assert gsp.source_type == "dataverse"
        assert gsp.persistent_id is not None
        assert "doi:" in gsp.persistent_id
        assert gsp.n_subjects == 1000
        assert gsp.estimated_size_gb > 0

    def test_dtor985_source_configuration(self):
        """dTOR985 source should have correct configuration."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES

        dtor = CONNECTOME_SOURCES["dtor985"]

        assert dtor.name == "dtor985"
        assert dtor.type == "structural"
        assert dtor.source_type == "figshare"
        assert dtor.download_url is not None
        assert "figshare" in dtor.download_url
        assert dtor.n_subjects == 985
        assert dtor.estimated_size_gb > 0
