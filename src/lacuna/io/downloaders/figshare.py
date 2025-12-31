"""
Figshare downloader implementation.

Handles downloads from Figshare using cloudscraper to bypass
Cloudflare protection.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

import cloudscraper
from tqdm import tqdm

from ...core.exceptions import DownloadError
from .base import BaseDownloader, ConnectomeSource, FetchProgress

if TYPE_CHECKING:
    pass


class FigshareDownloader(BaseDownloader):
    """
    Downloader for Figshare files.

    Uses cloudscraper to handle Cloudflare protection that blocks
    standard requests library.

    Parameters
    ----------
    source : ConnectomeSource
        Configuration for the connectome source.
    """

    def __init__(self, source: ConnectomeSource):
        super().__init__(source)
        self.scraper = cloudscraper.create_scraper(
            browser={
                "browser": "chrome",
                "platform": "linux",
                "mobile": False,
            }
        )

    def download(
        self,
        output_path: Path,
        progress_callback: Callable[[FetchProgress], None] | None = None,
    ) -> list[Path]:
        """
        Download file from Figshare.

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
            If download fails.
        """
        if not self.source.download_url:
            raise DownloadError(
                url="",
                reason="No download_url configured for source",
            )

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract filename from URL or use default
        filename = self._get_filename_from_url(self.source.download_url)
        if not filename:
            filename = f"{self.source.name}.trk"

        output_file = output_path / filename

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

    def _get_filename_from_url(self, url: str) -> str | None:
        """Extract filename from URL path."""
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
        progress_callback: Callable[[FetchProgress], None] | None = None,
    ) -> None:
        """
        Download a single file from Figshare.

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
        # Skip if already exists (basic caching)
        if output_file.exists():
            # Could add checksum verification here
            return

        try:
            response = self.scraper.get(url, stream=True, timeout=60)
            response.raise_for_status()
        except cloudscraper.exceptions.CloudflareChallengeError as e:
            raise DownloadError(
                url=url,
                reason=f"Cloudflare challenge failed: {e}",
            ) from e
        except Exception as e:
            raise DownloadError(url=url, reason=str(e)) from e

        total_size = int(response.headers.get("content-length", 0))

        # Use temp file for atomic write
        temp_file = output_file.with_suffix(output_file.suffix + ".tmp")

        try:
            with open(temp_file, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=output_file.name,
                    disable=progress_callback is not None,  # Disable if using callback
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
