"""
OSF (Open Science Framework) downloader implementation.

Handles downloads from OSF projects via the OSF API v2.
Used for neurotransmitter PET atlas maps.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import requests
from tqdm import tqdm

from ...core.exceptions import DownloadError
from .base import FetchProgress


class OsfDownloader:
    """
    Downloader for files hosted on OSF (Open Science Framework).

    Lists files in an OSF storage folder via the API and downloads them
    using public download links. No authentication required for public projects.

    Parameters
    ----------
    node_id : str
        OSF project node ID (e.g., ``"yz9mb"``).
    folder_id : str
        OSF storage folder ID containing the files to download.
    """

    API_BASE = "https://api.osf.io/v2"

    def __init__(self, node_id: str, folder_id: str):
        self.node_id = node_id
        self.folder_id = folder_id

    def list_files(self) -> list[dict[str, str]]:
        """
        List files in the OSF folder.

        Returns
        -------
        list[dict[str, str]]
            List of dicts with ``"name"`` and ``"download_url"`` keys.

        Raises
        ------
        DownloadError
            If the API request fails.
        """
        url = (
            f"{self.API_BASE}/nodes/{self.node_id}"
            f"/files/osfstorage/{self.folder_id}/"
        )
        files: list[dict[str, str]] = []
        page_url: str | None = url + "?page[size]=100"

        while page_url:
            try:
                resp = requests.get(page_url, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as exc:
                raise DownloadError(
                    url=page_url,
                    reason=f"OSF API request failed: {exc}",
                ) from exc

            data = resp.json()
            for item in data.get("data", []):
                if item["attributes"]["kind"] != "file":
                    continue
                files.append(
                    {
                        "name": item["attributes"]["name"],
                        "download_url": item["links"]["download"],
                        "size": item["attributes"].get("size", 0),
                    }
                )
            page_url = data.get("links", {}).get("next")

        return files

    def download(
        self,
        output_path: Path,
        progress_callback: Callable[[FetchProgress], None] | None = None,
    ) -> list[Path]:
        """
        Download all files from the OSF folder to *output_path*.

        Existing files are skipped.

        Parameters
        ----------
        output_path : Path
            Directory to download files into.
        progress_callback : callable, optional
            Called with ``FetchProgress`` updates.

        Returns
        -------
        list[Path]
            Paths of downloaded (or already-present) files.

        Raises
        ------
        DownloadError
            If listing or downloading fails.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        files = self.list_files()
        if not files:
            raise DownloadError(
                url=f"osf:{self.node_id}/{self.folder_id}",
                reason="No files found in OSF folder",
            )

        downloaded: list[Path] = []

        for idx, file_info in enumerate(files):
            name = file_info["name"]
            dest = output_path / name

            if dest.exists():
                if progress_callback:
                    progress_callback(
                        FetchProgress(
                            phase="download",
                            current_file=name,
                            files_completed=idx + 1,
                            files_total=len(files),
                            message=f"Already downloaded: {name}",
                        )
                    )
                downloaded.append(dest)
                continue

            if progress_callback:
                progress_callback(
                    FetchProgress(
                        phase="download",
                        current_file=name,
                        files_completed=idx,
                        files_total=len(files),
                        message=f"Downloading {name}",
                    )
                )

            self._download_file(
                url=file_info["download_url"],
                output_file=dest,
                total_files=len(files),
                file_index=idx,
                progress_callback=progress_callback,
            )
            downloaded.append(dest)

        if progress_callback:
            progress_callback(
                FetchProgress(
                    phase="download",
                    current_file="",
                    files_completed=len(files),
                    files_total=len(files),
                    message="All files downloaded",
                )
            )

        return downloaded

    # ------------------------------------------------------------------

    def _download_file(
        self,
        url: str,
        output_file: Path,
        total_files: int,
        file_index: int,
        progress_callback: Callable[[FetchProgress], None] | None = None,
    ) -> None:
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise DownloadError(
                url=url, reason=f"Download failed: {exc}"
            ) from exc

        total_size = int(resp.headers.get("content-length", 0))
        temp_file = output_file.with_suffix(output_file.suffix + ".tmp")

        try:
            with open(temp_file, "wb") as fh:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"[{file_index + 1}/{total_files}] {output_file.name}",
                    disable=progress_callback is not None,
                ) as pbar:
                    written = 0
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                            written += len(chunk)
                            pbar.update(len(chunk))

                            if progress_callback:
                                progress_callback(
                                    FetchProgress(
                                        phase="download",
                                        current_file=output_file.name,
                                        files_completed=file_index,
                                        files_total=total_files,
                                        bytes_transferred=written,
                                        bytes_total=total_size,
                                        message=f"Downloading {output_file.name}",
                                    )
                                )

            temp_file.rename(output_file)
        except Exception:
            if temp_file.exists():
                temp_file.unlink()
            raise
