"""
Figshare downloader implementation.

Handles downloads from Figshare using the authenticated API.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from tqdm import tqdm

from ...core.exceptions import DownloadError
from .base import BaseDownloader, ConnectomeSource, FetchProgress

if TYPE_CHECKING:
    pass


# Environment variable for Figshare API token
FIGSHARE_API_KEY_ENV = "FIGSHARE_API_KEY"


class FigshareDownloader(BaseDownloader):
    """
    Downloader for Figshare files using authenticated API.

    Uses Figshare API with authentication token to get download URLs
    that bypass AWS WAF protection.

    Parameters
    ----------
    source : ConnectomeSource
        Configuration for the connectome source.
    api_key : str, optional
        Figshare API key. If not provided, uses FIGSHARE_API_KEY env var.
    """

    # Figshare API base URL
    API_BASE = "https://api.figshare.com/v2"

    def __init__(
        self,
        source: ConnectomeSource,
        api_key: str | None = None,
    ):
        super().__init__(source)

        # Get API key from param or environment
        self.api_key = api_key or os.environ.get(FIGSHARE_API_KEY_ENV)

    def download(
        self,
        output_path: Path,
        progress_callback: Callable[[FetchProgress], None] | None = None,
    ) -> list[Path]:
        """
        Download file from Figshare using authenticated API.

        Parameters
        ----------
        output_path : Path
            Directory to download files to.
        progress_callback : callable, optional
            Function called with FetchProgress updates.

        Returns
        -------
        list[Path]
            List of downloaded file paths (single file for Figshare).

        Raises
        ------
        DownloadError
            If download fails or API key is missing.
        """
        # Check for API key
        if not self.api_key:
            raise DownloadError(
                url="",
                reason=(
                    f"Figshare API key required. Set via:\n"
                    f"  - Environment variable: {FIGSHARE_API_KEY_ENV}\n"
                    f"  - Command line: --api-key YOUR_KEY\n\n"
                    f"Get a free API key from:\n"
                    f"  https://figshare.com/account/applications\n"
                    f"  (Create 'Personal token' under 'Applications')"
                ),
            )

        # Check for article_id
        if not self.source.article_id:
            raise DownloadError(
                url="",
                reason="No article_id configured for Figshare source",
            )

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get file info from API
        file_info = self._get_file_info()
        filename = file_info["name"]
        download_url = file_info["download_url"]
        total_size = file_info.get("size", 0)

        output_file = output_path / filename

        # Skip if already exists
        if output_file.exists():
            existing_size = output_file.stat().st_size
            if existing_size == total_size and total_size > 0:
                if progress_callback:
                    progress_callback(
                        FetchProgress(
                            phase="download",
                            current_file=filename,
                            files_completed=1,
                            files_total=1,
                            message=f"Already downloaded: {filename}",
                        )
                    )
                return [output_file]

        # Report progress
        if progress_callback:
            progress_callback(
                FetchProgress(
                    phase="download",
                    current_file=filename,
                    files_completed=0,
                    files_total=1,
                    message=f"Downloading {filename}",
                )
            )

        # Download file using authenticated URL
        self._download_file(
            url=download_url,
            output_file=output_file,
            total_size=total_size,
            progress_callback=progress_callback,
        )

        return [output_file]

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with API authentication."""
        return {"Authorization": f"token {self.api_key}"}

    def _get_file_info(self) -> dict:
        """
        Get file information from Figshare API.

        Returns
        -------
        dict
            File info including name, download_url, and size.

        Raises
        ------
        DownloadError
            If API request fails.
        """
        api_url = f"{self.API_BASE}/articles/{self.source.article_id}/files"

        try:
            response = requests.get(
                api_url,
                headers=self._get_headers(),
                timeout=30,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise DownloadError(
                    url=api_url,
                    reason=(
                        "Figshare API authentication failed. "
                        "Check that your API key is valid.\n\n"
                        "Get a new API key from:\n"
                        "  https://figshare.com/account/applications"
                    ),
                ) from e
            raise DownloadError(
                url=api_url,
                reason=f"Figshare API error: {e}",
            ) from e
        except Exception as e:
            raise DownloadError(
                url=api_url,
                reason=f"Failed to connect to Figshare API: {e}",
            ) from e

        files = response.json()
        if not files:
            raise DownloadError(
                url=api_url,
                reason="No files found in Figshare article",
            )

        # Get the first (and usually only) file
        return files[0]

    def _get_filename_from_url(self, url: str) -> str | None:
        """Extract filename from URL path (kept for compatibility)."""
        from urllib.parse import unquote, urlparse

        parsed = urlparse(url)
        path = unquote(parsed.path)
        if "/" in path:
            filename = path.split("/")[-1]
            if "." in filename:
                return filename
        return None

    def _download_file(
        self,
        url: str,
        output_file: Path,
        total_size: int = 0,
        progress_callback: Callable[[FetchProgress], None] | None = None,
    ) -> None:
        """
        Download a single file from Figshare.

        Parameters
        ----------
        url : str
            Authenticated download URL.
        output_file : Path
            Output file path.
        total_size : int
            Expected file size in bytes.
        progress_callback : callable, optional
            Progress callback function.

        Raises
        ------
        DownloadError
            If download fails.
        """
        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                stream=True,
                timeout=60,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise DownloadError(
                url=url,
                reason=f"Download failed: HTTP {e.response.status_code}",
            ) from e
        except Exception as e:
            raise DownloadError(url=url, reason=str(e)) from e

        # Get size from headers if not provided
        if total_size == 0:
            total_size = int(response.headers.get("content-length", 0))

        # Check for HTML response (should not happen with API auth, but be safe)
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type.lower():
            raise DownloadError(
                url=url,
                reason=(
                    "Received HTML instead of file data. "
                    "The API token may have insufficient permissions."
                ),
            )

        # Use temp file for atomic write
        temp_file = output_file.with_suffix(output_file.suffix + ".tmp")

        try:
            with open(temp_file, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=output_file.name,
                    disable=progress_callback is not None,
                ) as pbar:
                    bytes_downloaded = 0
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            bytes_downloaded += len(chunk)
                            pbar.update(len(chunk))

                            if progress_callback:
                                progress_callback(
                                    FetchProgress(
                                        phase="download",
                                        current_file=output_file.name,
                                        files_completed=0,
                                        files_total=1,
                                        bytes_transferred=bytes_downloaded,
                                        bytes_total=total_size,
                                        message=f"Downloading {output_file.name}",
                                    )
                                )

            # Move to final location
            temp_file.rename(output_file)

            # Validate downloaded file
            self._validate_downloaded_file(output_file, url, total_size)

        except Exception:
            if temp_file.exists():
                temp_file.unlink()
            raise

    def _validate_downloaded_file(
        self,
        output_file: Path,
        url: str,
        expected_size: int = 0,
    ) -> None:
        """
        Validate that the downloaded file is valid.

        Parameters
        ----------
        output_file : Path
            Downloaded file to validate.
        url : str
            Original URL (for error messages).
        expected_size : int
            Expected file size in bytes.

        Raises
        ------
        DownloadError
            If file appears to be invalid.
        """
        file_size = output_file.stat().st_size

        # Check for size mismatch
        if expected_size > 0 and file_size != expected_size:
            output_file.unlink()
            raise DownloadError(
                url=url,
                reason=(
                    f"Downloaded file size ({file_size}) does not match "
                    f"expected size ({expected_size}). Download may be incomplete."
                ),
            )

        # Check for suspiciously small files
        if file_size < 10_000:
            with open(output_file, "rb") as f:
                header = f.read(1000)

            if b"<!DOCTYPE" in header or b"<html" in header:
                output_file.unlink()
                raise DownloadError(
                    url=url,
                    reason="Downloaded file is an HTML page, not the expected data file.",
                )

        # Validate .trk files
        if output_file.suffix == ".trk":
            with open(output_file, "rb") as f:
                f.seek(996)
                hdr_size_bytes = f.read(4)
                if len(hdr_size_bytes) == 4:
                    import struct

                    hdr_size = struct.unpack("<i", hdr_size_bytes)[0]
                    if hdr_size != 1000:
                        output_file.unlink()
                        raise DownloadError(
                            url=url,
                            reason=(
                                f"Invalid .trk file: header size is {hdr_size} "
                                "instead of 1000. File may be corrupted."
                            ),
                        )
