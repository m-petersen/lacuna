"""
Harvard Dataverse downloader implementation.

Handles authenticated downloads from Harvard Dataverse using the
X-Dataverse-key header for API authentication.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from tqdm import tqdm

from ...core.exceptions import AuthenticationError, ChecksumError, DownloadError
from .base import BaseDownloader, ConnectomeSource, FetchProgress, get_api_key

if TYPE_CHECKING:
    pass


class DataverseDownloader(BaseDownloader):
    """
    Downloader for Harvard Dataverse datasets.

    Handles authentication via API key and supports resumable downloads
    with checksum verification.

    Parameters
    ----------
    source : ConnectomeSource
        Configuration for the connectome source.
    api_key : str, optional
        Dataverse API key. If not provided, will attempt to get from
        environment variable or config file.

    Raises
    ------
    AuthenticationError
        If no API key is available.
    """

    def __init__(
        self,
        source: ConnectomeSource,
        api_key: str | None = None,
    ):
        super().__init__(source)
        self.api_key = get_api_key(api_key)
        if self.api_key is None:
            raise AuthenticationError(
                source=source.name,
                reason=(
                    "Dataverse API key required. Set DATAVERSE_API_KEY environment "
                    "variable or use --api-key argument."
                ),
            )
        self.session = requests.Session()
        self.session.headers.update({"X-Dataverse-key": self.api_key})

    def download(
        self,
        output_path: Path,
        progress_callback: Callable[[FetchProgress], None] | None = None,
        test_mode: bool = False,
    ) -> list[Path]:
        """
        Download dataset files from Dataverse.

        Parameters
        ----------
        output_path : Path
            Directory to download files to.
        progress_callback : callable, optional
            Function called with FetchProgress updates.
        test_mode : bool, default=False
            If True, downloads only 1 tarball for testing the full pipeline.
            Metadata files (JSON, TXT, masks) are always downloaded.

        Returns
        -------
        list[Path]
            List of downloaded file paths.

        Raises
        ------
        DownloadError
            If download fails.
        AuthenticationError
            If authentication fails.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get dataset metadata
        files_info = self._get_dataset_files()

        # In test mode: download all metadata files + only 1 tarball
        if test_mode:
            metadata_files = []
            tar_files = []
            for f in files_info:
                filename = f.get("filename", "")
                # Metadata files: JSON, TXT, NIfTI masks - always download
                if filename.endswith((".json", ".txt", ".nii.gz", ".nii")):
                    metadata_files.append(f)
                # Tarballs: limit to 1 in test mode
                elif filename.endswith(".tar"):
                    tar_files.append(f)

            # Take only first tarball in test mode
            files_info = metadata_files + tar_files[:1]

        downloaded_files: list[Path] = []

        for i, file_info in enumerate(files_info):
            file_id = file_info["id"]
            filename = file_info["filename"]
            checksum = file_info.get("checksum")
            file_size = file_info.get("size", 0)

            output_file = output_path / filename

            # Report progress
            if progress_callback:
                progress_callback(
                    FetchProgress(
                        phase="download",
                        current_file=filename,
                        files_completed=i,
                        files_total=len(files_info),
                        bytes_total=file_size,
                        message=f"Downloading {filename}",
                    )
                )

            # Skip if already downloaded and checksum matches
            if output_file.exists() and checksum:
                if self._verify_checksum(output_file, checksum):
                    downloaded_files.append(output_file)
                    continue

            # Download file
            self._download_file(
                file_id=file_id,
                output_file=output_file,
                expected_checksum=checksum,
                progress_callback=progress_callback,
                file_index=i,
                total_files=len(files_info),
            )
            downloaded_files.append(output_file)

        return downloaded_files

    def _get_dataset_files(self) -> list[dict]:
        """
        Get list of files in the dataset.

        Returns
        -------
        list[dict]
            List of file metadata dicts with id, filename, checksum, size.

        Raises
        ------
        DownloadError
            If API request fails.
        """
        if not self.source.persistent_id:
            raise DownloadError(
                url=self.source.dataverse_server,
                reason="No persistent_id configured for source",
            )

        # Use the dataset files API
        url = (
            f"{self.source.dataverse_server}/api/datasets/"
            f":persistentId/versions/:latest/files"
            f"?persistentId={self.source.persistent_id}"
        )

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise AuthenticationError(
                    source=self.source.name,
                    reason="Invalid API key",
                ) from e
            raise DownloadError(
                url=url,
                reason=f"HTTP {response.status_code}: {response.text}",
            ) from e
        except requests.exceptions.RequestException as e:
            raise DownloadError(url=url, reason=str(e)) from e

        data = response.json()
        if data.get("status") != "OK":
            raise DownloadError(
                url=url,
                reason=f"API error: {data.get('message', 'Unknown error')}",
            )

        files_info = []
        for file_data in data.get("data", []):
            df = file_data.get("dataFile", {})
            checksum_info = df.get("checksum", {})
            files_info.append(
                {
                    "id": df.get("id"),
                    "filename": df.get("filename", f"file_{df.get('id')}"),
                    "size": df.get("filesize", 0),
                    "checksum": checksum_info.get("value"),
                    "checksum_type": checksum_info.get("type", "MD5").lower(),
                }
            )

        return files_info

    def _download_file(
        self,
        file_id: int,
        output_file: Path,
        expected_checksum: str | None = None,
        progress_callback: Callable[[FetchProgress], None] | None = None,
        file_index: int = 0,
        total_files: int = 1,
    ) -> None:
        """
        Download a single file by ID.

        Parameters
        ----------
        file_id : int
            Dataverse file ID.
        output_file : Path
            Output file path.
        expected_checksum : str, optional
            Expected MD5 checksum for verification.
        progress_callback : callable, optional
            Progress callback function.
        file_index : int
            Current file index for progress.
        total_files : int
            Total number of files for progress.

        Raises
        ------
        DownloadError
            If download fails.
        ChecksumError
            If checksum verification fails.
        """
        url = f"{self.source.dataverse_server}/api/access/datafile/{file_id}"

        try:
            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise AuthenticationError(
                    source=self.source.name,
                    reason="Invalid API key",
                ) from e
            raise DownloadError(
                url=url,
                reason=f"HTTP {response.status_code}",
            ) from e
        except requests.exceptions.RequestException as e:
            raise DownloadError(url=url, reason=str(e)) from e

        total_size = int(response.headers.get("content-length", 0))
        hasher = hashlib.md5()

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
                            hasher.update(chunk)
                            bytes_downloaded += len(chunk)
                            pbar.update(len(chunk))

                            if progress_callback:
                                progress_callback(
                                    FetchProgress(
                                        phase="download",
                                        current_file=output_file.name,
                                        files_completed=file_index,
                                        files_total=total_files,
                                        bytes_transferred=bytes_downloaded,
                                        bytes_total=total_size,
                                        message=f"Downloading {output_file.name}",
                                    )
                                )

            # Verify checksum
            if expected_checksum:
                actual_checksum = hasher.hexdigest()
                if actual_checksum.lower() != expected_checksum.lower():
                    temp_file.unlink()
                    raise ChecksumError(
                        filepath=str(output_file),
                        expected=expected_checksum,
                        actual=actual_checksum,
                    )

            # Move to final location
            temp_file.rename(output_file)

        except Exception:
            if temp_file.exists():
                temp_file.unlink()
            raise

    def _verify_checksum(self, filepath: Path, expected: str) -> bool:
        """Verify file MD5 checksum."""
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest().lower() == expected.lower()
