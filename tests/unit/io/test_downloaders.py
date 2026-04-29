"""
Unit tests for connectome downloaders.

These tests use mocked HTTP responses to test downloader functionality
without making actual network requests.
"""

from __future__ import annotations

import json
from pathlib import Path
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

    def test_figshare_downloader_requires_api_key(self, tmp_path):
        """FigshareDownloader should raise DownloadError without API key."""
        from lacuna.core.exceptions import DownloadError
        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.figshare import FigshareDownloader

        source = CONNECTOME_SOURCES["dtor985"]

        # Clear any env var that might be set
        with patch.dict("os.environ", {}, clear=True):
            downloader = FigshareDownloader(source, api_key=None)

            with pytest.raises(DownloadError) as exc_info:
                downloader.download(tmp_path)

            assert "Figshare API key required" in str(exc_info.value)

    def test_figshare_downloader_accepts_api_key(self):
        """FigshareDownloader should accept API key via constructor."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.figshare import FigshareDownloader

        source = CONNECTOME_SOURCES["dtor985"]
        downloader = FigshareDownloader(source, api_key="test-key")

        assert downloader.api_key == "test-key"

    def test_figshare_downloader_uses_env_var(self):
        """FigshareDownloader should use FIGSHARE_API_KEY env var."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.figshare import FigshareDownloader

        source = CONNECTOME_SOURCES["dtor985"]

        with patch.dict("os.environ", {"FIGSHARE_API_KEY": "env-key"}):
            downloader = FigshareDownloader(source)
            assert downloader.api_key == "env-key"

    def test_figshare_downloader_extracts_filename(self):
        """FigshareDownloader should extract filename from URL."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.figshare import FigshareDownloader

        source = CONNECTOME_SOURCES["dtor985"]
        downloader = FigshareDownloader(source, api_key="test-key")

        # Test URL with filename
        url = "https://figshare.com/ndownloader/files/12345/tractogram.trk"
        filename = downloader._get_filename_from_url(url)
        assert filename == "tractogram.trk"

        # Test URL without clear filename
        url = "https://figshare.com/ndownloader/files/12345"
        filename = downloader._get_filename_from_url(url)
        assert filename is None

    def test_figshare_download_requires_article_id(self, tmp_path):
        """FigshareDownloader should raise DownloadError if no article_id configured."""
        from lacuna.core.exceptions import DownloadError
        from lacuna.io.downloaders import ConnectomeSource
        from lacuna.io.downloaders.figshare import FigshareDownloader

        source = ConnectomeSource(
            name="test",
            display_name="Test",
            type="structural",
            description="Test",
            source_type="figshare",
            article_id=None,  # No article ID
        )

        downloader = FigshareDownloader(source, api_key="test-key")

        with pytest.raises(DownloadError) as exc_info:
            downloader.download(tmp_path)

        assert "No article_id configured" in str(exc_info.value)

    def test_figshare_download_with_mock(self, tmp_path):
        """FigshareDownloader should download file using API."""
        import requests

        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.figshare import FigshareDownloader

        source = CONNECTOME_SOURCES["dtor985"]
        downloader = FigshareDownloader(source, api_key="test-key")

        # Mock API response for file info
        mock_api_response = MagicMock()
        mock_api_response.status_code = 200
        mock_api_response.json.return_value = [
            {
                "name": "tractogram.trk",
                "download_url": "https://figshare.com/download/12345",
                "size": 11000000000,  # 11GB
            }
        ]

        # Mock download response
        mock_download_response = MagicMock()
        mock_download_response.headers = {
            "content-length": "11000000000",
            "content-type": "application/octet-stream",
        }
        # Create fake content that's larger than 10KB to pass validation
        fake_content = (
            b"TRACK" + b"\x00" * 995 + b"\xe8\x03\x00\x00"
        )  # Valid TRK header (1000 bytes at end)
        mock_download_response.iter_content.return_value = [fake_content]

        def mock_get(url, **kwargs):
            if "api.figshare.com" in url:
                return mock_api_response
            else:
                return mock_download_response

        with patch.object(requests, "get", side_effect=mock_get):
            # Patch validation to skip .trk header check since mock file is tiny
            with patch.object(downloader, "_validate_downloaded_file"):
                files = downloader.download(tmp_path)

                assert len(files) == 1
                assert files[0].exists()
                assert files[0].name == "tractogram.trk"


class TestGithubReleaseDownloader:
    """Unit tests for GithubReleaseDownloader."""

    def test_github_downloader_no_api_key_needed(self):
        """GithubReleaseDownloader should create instance without any API key."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.github import GithubReleaseDownloader

        source = CONNECTOME_SOURCES["hcp1065"]
        downloader = GithubReleaseDownloader(source)

        assert downloader.source == source

    def test_github_downloader_requires_download_url(self, tmp_path):
        """GithubReleaseDownloader should raise DownloadError if no download_url."""
        from lacuna.core.exceptions import DownloadError
        from lacuna.io.downloaders import ConnectomeSource
        from lacuna.io.downloaders.github import GithubReleaseDownloader

        source = ConnectomeSource(
            name="test",
            display_name="Test",
            type="structural",
            description="Test",
            source_type="github",
            download_url=None,
        )

        downloader = GithubReleaseDownloader(source)

        with pytest.raises(DownloadError) as exc_info:
            downloader.download(tmp_path)

        assert "No download_url configured" in str(exc_info.value)

    def test_github_downloader_extracts_filename(self):
        """GithubReleaseDownloader should extract filename from URL."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.github import GithubReleaseDownloader

        source = CONNECTOME_SOURCES["hcp1065"]
        downloader = GithubReleaseDownloader(source)

        filename = downloader._get_filename_from_url(source.download_url)
        assert filename == "hcp1065_avg_tracts_trk.zip"

    def test_github_download_with_mock(self, tmp_path):
        """GithubReleaseDownloader should download file via HTTP GET."""
        import requests

        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.github import GithubReleaseDownloader

        source = CONNECTOME_SOURCES["hcp1065"]
        downloader = GithubReleaseDownloader(source)

        # Mock download response
        mock_response = MagicMock()
        mock_response.headers = {
            "content-length": "1000",
            "content-type": "application/zip",
        }
        mock_response.iter_content.return_value = [b"fake zip data"]

        with patch.object(requests, "get", return_value=mock_response):
            files = downloader.download(tmp_path)

            assert len(files) == 1
            assert files[0].exists()
            assert files[0].name == "hcp1065_avg_tracts_trk.zip"

    def test_github_download_skips_existing(self, tmp_path):
        """GithubReleaseDownloader should skip already downloaded files."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES
        from lacuna.io.downloaders.github import GithubReleaseDownloader

        source = CONNECTOME_SOURCES["hcp1065"]
        downloader = GithubReleaseDownloader(source)

        # Create existing file
        existing = tmp_path / "hcp1065_avg_tracts_trk.zip"
        existing.write_bytes(b"existing data")

        files = downloader.download(tmp_path)

        assert len(files) == 1
        assert files[0] == existing


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
        assert dtor.article_id == 25209947
        assert dtor.n_subjects == 985
        assert dtor.n_subjects == 985
        assert dtor.estimated_size_gb > 0

    def test_hcp1065_source_configuration(self):
        """HCP1065 source should have correct configuration."""
        from lacuna.io.downloaders import CONNECTOME_SOURCES

        hcp = CONNECTOME_SOURCES["hcp1065"]

        assert hcp.name == "hcp1065"
        assert hcp.type == "structural"
        assert hcp.source_type == "github"
        assert hcp.download_url is not None
        assert "github.com" in hcp.download_url
        assert hcp.n_subjects == 1065
        assert hcp.space == "MNI152NLin2009cAsym"
        assert hcp.estimated_size_gb > 0


class TestOsfDownloader:
    """Unit tests for OsfDownloader."""

    def test_osf_downloader_init(self):
        """OsfDownloader should store node and folder IDs."""
        from lacuna.io.downloaders.osf import OsfDownloader

        dl = OsfDownloader(node_id="yz9mb", folder_id="abc123")
        assert dl.node_id == "yz9mb"
        assert dl.folder_id == "abc123"

    def test_osf_list_files_parses_response(self):
        """list_files should parse OSF API response into file dicts."""
        from lacuna.io.downloaders.osf import OsfDownloader

        dl = OsfDownloader(node_id="yz9mb", folder_id="abc123")

        mock_json = {
            "data": [
                {
                    "attributes": {
                        "kind": "file",
                        "name": "target-5HT1a.nii.gz",
                        "size": 473848,
                    },
                    "links": {"download": "https://osf.io/download/file1/"},
                },
                {
                    "attributes": {
                        "kind": "folder",
                        "name": "subfolder",
                        "size": 0,
                    },
                    "links": {},
                },
            ],
            "links": {"next": None},
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_json
        mock_resp.status_code = 200

        with patch("lacuna.io.downloaders.osf.requests.get", return_value=mock_resp):
            files = dl.list_files()

        assert len(files) == 1
        assert files[0]["name"] == "target-5HT1a.nii.gz"
        assert files[0]["download_url"] == "https://osf.io/download/file1/"
        assert files[0]["size"] == 473848

    def test_osf_list_files_paginates(self):
        """list_files should follow pagination links."""
        from lacuna.io.downloaders.osf import OsfDownloader

        dl = OsfDownloader(node_id="yz9mb", folder_id="abc123")

        page1_json = {
            "data": [
                {
                    "attributes": {"kind": "file", "name": "file1.nii.gz", "size": 100},
                    "links": {"download": "https://osf.io/download/f1/"},
                },
            ],
            "links": {"next": "https://api.osf.io/v2/next_page"},
        }
        page2_json = {
            "data": [
                {
                    "attributes": {"kind": "file", "name": "file2.nii.gz", "size": 200},
                    "links": {"download": "https://osf.io/download/f2/"},
                },
            ],
            "links": {"next": None},
        }

        resp1 = MagicMock()
        resp1.json.return_value = page1_json
        resp2 = MagicMock()
        resp2.json.return_value = page2_json

        with patch(
            "lacuna.io.downloaders.osf.requests.get", side_effect=[resp1, resp2]
        ):
            files = dl.list_files()

        assert len(files) == 2
        assert files[0]["name"] == "file1.nii.gz"
        assert files[1]["name"] == "file2.nii.gz"

    def test_osf_download_skips_existing(self, tmp_path):
        """download should skip files that already exist."""
        from lacuna.io.downloaders.osf import OsfDownloader

        dl = OsfDownloader(node_id="yz9mb", folder_id="abc123")

        # Create existing file
        existing = tmp_path / "existing.nii.gz"
        existing.write_bytes(b"data")

        with patch.object(
            dl,
            "list_files",
            return_value=[
                {"name": "existing.nii.gz", "download_url": "https://osf.io/download/x/", "size": 4},
            ],
        ):
            result = dl.download(tmp_path)

        assert result == [existing]

    def test_osf_download_creates_dir(self, tmp_path):
        """download should create output directory if missing."""
        from lacuna.io.downloaders.osf import OsfDownloader

        dl = OsfDownloader(node_id="yz9mb", folder_id="abc123")
        out = tmp_path / "new" / "dir"

        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "5"}
        mock_resp.iter_content.return_value = [b"hello"]

        with (
            patch.object(dl, "list_files", return_value=[
                {"name": "f.nii.gz", "download_url": "https://osf.io/download/f/", "size": 5},
            ]),
            patch("lacuna.io.downloaders.osf.requests.get", return_value=mock_resp),
        ):
            result = dl.download(out)

        assert out.exists()
        assert len(result) == 1
        assert result[0].name == "f.nii.gz"

    def test_osf_download_empty_folder_raises(self, tmp_path):
        """download should raise DownloadError for empty folder."""
        from lacuna.core.exceptions import DownloadError
        from lacuna.io.downloaders.osf import OsfDownloader

        dl = OsfDownloader(node_id="yz9mb", folder_id="abc123")

        with patch.object(dl, "list_files", return_value=[]):
            with pytest.raises(DownloadError, match="No files found"):
                dl.download(tmp_path)

    def test_osf_list_files_api_error_raises(self):
        """list_files should raise DownloadError on API failure."""
        import requests

        from lacuna.core.exceptions import DownloadError
        from lacuna.io.downloaders.osf import OsfDownloader

        dl = OsfDownloader(node_id="yz9mb", folder_id="abc123")

        with patch(
            "lacuna.io.downloaders.osf.requests.get",
            side_effect=requests.ConnectionError("offline"),
        ):
            with pytest.raises(DownloadError, match="OSF API request failed"):
                dl.list_files()


class TestFetchNtatlas:
    """Unit tests for fetch_ntatlas."""

    def test_fetch_ntatlas_importable(self):
        """fetch_ntatlas should be importable from lacuna.io."""
        from lacuna.io import fetch_ntatlas

        assert callable(fetch_ntatlas)

    def test_fetch_ntatlas_skips_existing_with_matching_hash(self, tmp_path):
        """Existing files with correct SHA-256 should not be re-downloaded."""
        import hashlib
        import io

        from lacuna.data.ntatlas import all_map_ids, load_collection
        from lacuna.io.fetch import fetch_ntatlas

        coll = load_collection()
        map_ids = all_map_ids()

        # Build a fake hash manifest where every map's hash matches the
        # placeholder content we will write to disk.
        content = b"fake nifti payload"
        good_hash = hashlib.sha256(content).hexdigest()
        hashes = {
            coll["map_path_template"].format(map_id=mid): good_hash
            for mid in map_ids
        }
        hashes_payload = json.dumps(hashes).encode("utf-8")

        # Pre-populate output dir with content matching the hash
        for mid in map_ids:
            (tmp_path / f"{mid}_space-MNI152NLin6Asym_desc-proc.nii.gz").write_bytes(content)
        (tmp_path / "metadata.csv").write_bytes(b"fake")

        download_count = {"n": 0}

        def fake_urlretrieve(url, out_path):
            download_count["n"] += 1
            Path(out_path).write_bytes(content)

        class FakeResp:
            def __init__(self, data):
                self._data = data

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def read(self):
                return self._data

        with patch("urllib.request.urlopen", return_value=FakeResp(hashes_payload)), \
             patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            result = fetch_ntatlas(output_dir=tmp_path)

        assert result.success
        assert len(result.output_files) == len(map_ids)
        # Nothing should have been downloaded — all hashes matched.
        assert download_count["n"] == 0

    def test_fetch_ntatlas_force_redownloads(self, tmp_path):
        """fetch_ntatlas with force=True should download even if files exist."""
        import hashlib
        from lacuna.data.ntatlas import all_map_ids, load_collection
        from lacuna.io.fetch import fetch_ntatlas

        coll = load_collection()
        map_ids = all_map_ids()
        content = b"new payload"
        good_hash = hashlib.sha256(content).hexdigest()
        hashes = {
            coll["map_path_template"].format(map_id=mid): good_hash
            for mid in map_ids
        }
        hashes_payload = json.dumps(hashes).encode("utf-8")

        for mid in map_ids:
            (tmp_path / f"{mid}_space-MNI152NLin6Asym_desc-proc.nii.gz").write_bytes(b"old")

        download_count = {"n": 0}

        def fake_urlretrieve(url, out_path):
            download_count["n"] += 1
            Path(out_path).write_bytes(content)

        class FakeResp:
            def __init__(self, data):
                self._data = data

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def read(self):
                return self._data

        with patch("urllib.request.urlopen", return_value=FakeResp(hashes_payload)), \
             patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            result = fetch_ntatlas(output_dir=tmp_path, force=True)

        assert result.success
        # Force redownload: every map + metadata.csv re-downloaded
        assert download_count["n"] == len(map_ids) + 1

    def test_fetch_ntatlas_hash_mismatch_raises(self, tmp_path):
        """Downloaded file with wrong hash should raise DownloadError."""
        from lacuna.core.exceptions import DownloadError
        from lacuna.data.ntatlas import all_map_ids, load_collection
        from lacuna.io.fetch import fetch_ntatlas

        coll = load_collection()
        map_ids = all_map_ids()
        # All hashes set to something that won't match downloaded content
        hashes = {
            coll["map_path_template"].format(map_id=mid): "0" * 64
            for mid in map_ids
        }
        hashes_payload = json.dumps(hashes).encode("utf-8")

        def fake_urlretrieve(url, out_path):
            Path(out_path).write_bytes(b"wrong content")

        class FakeResp:
            def __init__(self, data):
                self._data = data

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def read(self):
                return self._data

        with patch("urllib.request.urlopen", return_value=FakeResp(hashes_payload)), \
             patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve), \
             pytest.raises(DownloadError, match="Hash mismatch"):
            fetch_ntatlas(output_dir=tmp_path)

    def test_ntatlas_in_parser_choices(self):
        """ntatlas should be a valid choice in the fetch parser."""
        from lacuna.cli.parser import build_parser

        parser = build_parser()
        # Should not raise
        args = parser.parse_args(["fetch", "ntatlas"])
        assert args.connectome == "ntatlas"
