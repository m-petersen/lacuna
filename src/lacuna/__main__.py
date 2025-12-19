"""
Lacuna CLI entry point.

This module enables running Lacuna as a module:
    python -m lacuna <bids_dir> <output_dir> participant [options]
"""

from lacuna.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
