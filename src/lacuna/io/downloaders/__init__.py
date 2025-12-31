"""
Downloaders subpackage for source-specific download logic.

This module provides downloader implementations for various data sources:
- Harvard Dataverse (for GSP1000)
- Figshare (for dTOR985)

Each downloader handles authentication, rate limiting, and error handling
specific to its data source.
"""

from .base import (
    CONNECTOME_SOURCES,
    ConnectomeSource,
    FetchConfig,
    FetchProgress,
    FetchResult,
    get_api_key,
)
from .dataverse import DataverseDownloader
from .figshare import FigshareDownloader

__all__ = [
    "CONNECTOME_SOURCES",
    "ConnectomeSource",
    "FetchConfig",
    "FetchProgress",
    "FetchResult",
    "DataverseDownloader",
    "FigshareDownloader",
    "get_api_key",
]
