"""
Lacuna CLI module.

This module provides the command-line interface for Lacuna, following
the BIDS-Apps specification for neuroimaging pipelines.

Usage:
    lacuna <bids_dir> <output_dir> participant [options]

Example:
    lacuna /data/bids /output participant --functional-connectome /connectomes/gsp1000.h5
"""

from lacuna.cli.main import main
from lacuna.cli.parser import build_parser

__all__ = ["main", "build_parser"]
