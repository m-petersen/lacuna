"""
GitHub Release downloader implementation.

Handles downloads from GitHub Releases via direct HTTP GET (no authentication required).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from tqdm import tqdm

from ...core.exceptions import DownloadError
from .base import BaseDownloader, ConnectomeSource, FetchProgress


class GithubReleaseDownloader(BaseDownloader):
    """
    Downloader for files hosted on GitHub Releases.

    No authentication is required — files are downloaded via plain HTTP GET.

    Parameters
    ----------
    source : ConnectomeSource
        Configuration for the connectome source. Must have ``download_url`` set.
    """

    def __init__(self, source: ConnectomeSource):
        super().__init__(source)

    def download(
        self,
        output_path: Path,
        progress_callback: Callable[[FetchProgress], None] | None = None,
    ) -> list[Path]:
        """
        Download file from GitHub Releases.

        Parameters
        ----------
        output_path : Path
            Directory to download files to.
        progress_callback : callable, optional
            Function called with FetchProgress updates.

        Returns
        -------
        list[Path]
            List of downloaded file paths (single file).

        Raises
        ------
        DownloadError
            If download fails or download_url is not configured.
        """
        if not self.source.download_url:
            raise DownloadError(
                url="",
                reason="No download_url configured for GitHub source",
            )

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract filename from URL
        filename = self._get_filename_from_url(self.source.download_url)
        output_file = output_path / filename

        # Skip if already exists
        if output_file.exists():
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

        # Download file
        self._download_file(
            url=self.source.download_url,
            output_file=output_file,
            progress_callback=progress_callback,
        )

        return [output_file]

    def _get_filename_from_url(self, url: str) -> str:
        """Extract filename from URL path."""
        parsed = urlparse(url)
        path = unquote(parsed.path)
        filename = path.split("/")[-1]
        if not filename:
            raise DownloadError(url=url, reason="Could not extract filename from URL")
        return filename

    def _download_file(
        self,
        url: str,
        output_file: Path,
        progress_callback: Callable[[FetchProgress], None] | None = None,
    ) -> None:
        """
        Download a single file via HTTP GET.

        Parameters
        ----------
        url : str
            Download URL.
        output_file : Path
            Output file path.
        progress_callback : callable, optional
            Progress callback function.

        Raises
        ------
        DownloadError
            If download fails.
        """
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise DownloadError(
                url=url,
                reason=f"Download failed: HTTP {e.response.status_code}",
            ) from e
        except Exception as e:
            raise DownloadError(url=url, reason=str(e)) from e

        total_size = int(response.headers.get("content-length", 0))

        # Check for HTML response
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type.lower():
            raise DownloadError(
                url=url,
                reason="Received HTML instead of file data. The URL may be invalid.",
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

        except Exception:
            if temp_file.exists():
                temp_file.unlink()
            raise
