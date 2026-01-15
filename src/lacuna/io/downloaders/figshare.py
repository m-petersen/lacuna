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

        # Check for WAF challenge responses (AWS WAF, Cloudflare, etc.)
        content_type = response.headers.get("content-type", "")
        total_size = int(response.headers.get("content-length", 0))

        # Detect HTML challenge pages masquerading as downloads
        if "text/html" in content_type.lower():
            raise DownloadError(
                url=url,
                reason=(
                    "Figshare returned an HTML page instead of the file. "
                    "This is likely due to AWS WAF anti-bot protection.\n\n"
                    "Please download the file manually:\n"
                    "  1. Open in browser: https://springernature.figshare.com/articles/dataset/"
                    "dTOR_Diffusion_Tensor_Open_Resource_985_Subject_Tracktogram/25058299\n"
                    "  2. Click 'Download' to get the .trk file\n"
                    f"  3. Save to: {output_file}\n"
                    "  4. Run 'lacuna fetch dtor985' again to convert and register"
                ),
            )

        # Validate file size - tractograms should be at least 1GB
        expected_min_size = 1_000_000_000  # 1GB minimum for a full tractogram
        if total_size > 0 and total_size < expected_min_size:
            # Small file might be an error page or partial download
            if total_size < 10_000:  # Less than 10KB is definitely wrong
                raise DownloadError(
                    url=url,
                    reason=(
                        f"Downloaded file is too small ({total_size} bytes). "
                        "Expected a multi-GB tractogram file.\n\n"
                        "Please download the file manually from:\n"
                        "  https://springernature.figshare.com/articles/dataset/"
                        "dTOR_Diffusion_Tensor_Open_Resource_985_Subject_Tracktogram/25058299"
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

            # Validate downloaded file is not an HTML error page
            self._validate_downloaded_file(output_file, url)

        except Exception:
            if temp_file.exists():
                temp_file.unlink()
            raise

    def _validate_downloaded_file(self, output_file: Path, url: str) -> None:
        """
        Validate that the downloaded file is actually a tractogram, not an error page.

        Parameters
        ----------
        output_file : Path
            Downloaded file to validate.
        url : str
            Original URL (for error messages).

        Raises
        ------
        DownloadError
            If file appears to be invalid.
        """
        # Check file size
        file_size = output_file.stat().st_size
        if file_size < 10_000:  # Less than 10KB
            # Read first bytes to check if it's HTML
            with open(output_file, "rb") as f:
                header = f.read(1000)

            # Check for HTML markers
            if b"<!DOCTYPE" in header or b"<html" in header or b"AwsWaf" in header:
                output_file.unlink()  # Remove invalid file
                raise DownloadError(
                    url=url,
                    reason=(
                        "Downloaded file is an HTML page, not the tractogram. "
                        "Figshare's anti-bot protection blocked the download.\n\n"
                        "Please download manually:\n"
                        "  1. Open: https://springernature.figshare.com/articles/dataset/"
                        "dTOR_Diffusion_Tensor_Open_Resource_985_Subject_Tracktogram/25058299\n"
                        "  2. Click 'Download' to get the .trk file\n"
                        f"  3. Save to: {output_file}\n"
                        "  4. Run 'lacuna fetch dtor985' again to convert and register"
                    ),
                )

        # For .trk files, validate the header
        if output_file.suffix == ".trk":
            with open(output_file, "rb") as f:
                # TrackVis header should start with "TRACK" magic bytes at offset 0
                # or have header size of 1000 at bytes 996-1000
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
                                f"Invalid .trk file: header size is {hdr_size} instead of 1000. "
                                "The file may be corrupted or not a valid TrackVis tractogram.\n\n"
                                "Please download manually from:\n"
                                "  https://springernature.figshare.com/articles/dataset/"
                                "dTOR_Diffusion_Tensor_Open_Resource_985_Subject_Tracktogram/25058299"
                            ),
                        )
