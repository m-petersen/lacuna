"""
Fetch command implementation for Lacuna CLI.

This module handles the 'lacuna fetch' subcommand for downloading,
processing, and registering connectomes.

Commands:
    lacuna fetch gsp1000 - Download GSP1000 functional connectome
    lacuna fetch dtor985 - Download dTOR985 structural tractogram
    lacuna fetch --list  - List available connectomes
    lacuna fetch --interactive - Interactive guided setup
"""

from __future__ import annotations

import argparse


def handle_fetch_command(args: argparse.Namespace) -> int:
    """
    Handle the fetch subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error).
    """
    # Import here to avoid circular imports and speed up CLI startup

    # Handle --list flag
    if getattr(args, "list", False):
        return _handle_list()

    # Handle --interactive flag
    if getattr(args, "interactive", False):
        return _handle_interactive(args)

    # Handle --clean flag
    if getattr(args, "clean", False):
        return _handle_clean(args)

    # Handle --clean-all flag
    if getattr(args, "clean_all", False):
        return _handle_clean_all(args)

    # Handle specific connectome fetch
    connectome = getattr(args, "connectome", None)
    if connectome is None:
        print("Error: No connectome specified. Use 'lacuna fetch --list' to see options.")
        print("       Use 'lacuna fetch --interactive' for guided setup.")
        return 1

    if connectome == "gsp1000":
        return _handle_gsp1000(args)
    elif connectome == "dtor985":
        return _handle_dtor985(args)
    else:
        print(f"Error: Unknown connectome '{connectome}'")
        return 1


def _handle_list() -> int:
    """Display available connectomes."""
    from lacuna.io.downloaders import CONNECTOME_SOURCES

    print("\nAvailable connectomes:\n")
    for name, source in CONNECTOME_SOURCES.items():
        print(f"  {name}")
        print(f"    {source.display_name}")
        print(f"    Type: {source.type}")
        print(f"    Size: ~{source.estimated_size_gb:.1f} GB")
        print()

    return 0


def _handle_gsp1000(args: argparse.Namespace) -> int:
    """Handle GSP1000 fetch."""
    from lacuna.core.exceptions import AuthenticationError, DownloadError, ProcessingError
    from lacuna.io import fetch_gsp1000

    # Get output directory
    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        from lacuna.io.fetch import get_data_dir

        output_dir = get_data_dir() / "connectomes" / "gsp1000"

    # Get configuration
    api_key = getattr(args, "api_key", None)
    batches = getattr(args, "batches", 10)
    test_mode = getattr(args, "test_mode", False)
    skip_checksum = getattr(args, "skip_checksum", False)
    force = getattr(args, "force", False)

    print("Fetching GSP1000 functional connectome...")
    print(f"  Output: {output_dir}")
    if test_mode:
        print("  Mode: TEST (1 tarball only)")
    else:
        print(f"  Batches: {batches}")
    if skip_checksum:
        print("  Warning: Checksum verification disabled")
    print()

    try:
        result = fetch_gsp1000(
            output_dir=output_dir,
            api_key=api_key,
            batches=batches,
            test_mode=test_mode,
            skip_checksum=skip_checksum,
            register=False,  # Registration handled by analysis steps
            force=force,
        )
        print("\n✓ GSP1000 fetch complete!")
        print(f"  Files: {len(result.output_files)}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        print(f"  Output: {result.output_dir}")
        return 0

    except AuthenticationError as e:
        print(f"\n✗ Authentication error: {e}")
        print("  Set DATAVERSE_API_KEY environment variable or use --api-key")
        return 1
    except DownloadError as e:
        print(f"\n✗ Download error: {e}")
        return 1
    except ProcessingError as e:
        print(f"\n✗ Processing error: {e}")
        return 1


def _handle_dtor985(args: argparse.Namespace) -> int:
    """Handle dTOR985 fetch."""
    from lacuna.core.exceptions import DownloadError, ProcessingError
    from lacuna.io import fetch_dtor985

    # Get output directory
    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        from lacuna.io.fetch import get_data_dir

        output_dir = get_data_dir() / "connectomes" / "dtor985"

    # Get configuration
    api_key = getattr(args, "api_key", None)
    keep_original = not getattr(args, "no_keep_original_trk", False)
    force = getattr(args, "force", False)

    print("Fetching dTOR985 structural tractogram...")
    print(f"  Output: {output_dir}")
    print(f"  Keep original: {keep_original}")
    print()

    try:
        result = fetch_dtor985(
            output_dir=output_dir,
            api_key=api_key,
            keep_original=keep_original,
            register=False,  # Registration handled by analysis steps
            force=force,
        )
        print("\n✓ dTOR985 fetch complete!")
        print(f"  Files: {len(result.output_files)}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        print(f"  Output: {result.output_dir}")
        return 0

    except DownloadError as e:
        print(f"\n✗ Download error: {e}")
        print("  Set FIGSHARE_API_KEY environment variable or use --api-key")
        return 1
    except ProcessingError as e:
        print(f"\n✗ Processing error: {e}")
        return 1


def _handle_interactive(args: argparse.Namespace) -> int:
    """Handle interactive guided setup."""
    from lacuna.io.downloaders import CONNECTOME_SOURCES

    print("\n🔬 Lacuna Connectome Setup Wizard\n")
    print("This wizard will help you download and configure connectomes")
    print("for lesion network mapping analysis.\n")

    # List available connectomes
    print("Available connectomes:\n")
    sources = list(CONNECTOME_SOURCES.values())
    for i, source in enumerate(sources, 1):
        print(f"  [{i}] {source.display_name}")
        print(f"      Type: {source.type}, Size: ~{source.estimated_size_gb:.1f} GB")
        print()

    # Get user selection
    try:
        selection = input("Select a connectome (number): ").strip()
        idx = int(selection) - 1
        if idx < 0 or idx >= len(sources):
            print("Invalid selection.")
            return 1
    except (ValueError, EOFError):
        print("Invalid input.")
        return 1

    selected = sources[idx]
    print(f"\nYou selected: {selected.display_name}\n")

    # RAM-based batch recommendation for GSP1000
    if selected.name == "gsp1000":
        batches = _get_recommended_batches()
        print(f"Recommended batches for your system: {batches}")
        try:
            custom = input(f"Number of batches [{batches}]: ").strip()
            if custom:
                batches = int(custom)
        except (ValueError, EOFError):
            pass

        # Get API key
        import os

        api_key = os.environ.get("DATAVERSE_API_KEY")
        if not api_key:
            print("\nDataverse API key required for GSP1000.")
            print("Get one at: https://dataverse.harvard.edu/")
            try:
                api_key = input("API key: ").strip()
            except EOFError:
                print("No API key provided.")
                return 1

        # Create namespace and call handler
        args.connectome = "gsp1000"
        args.api_key = api_key
        args.batches = batches
        return _handle_gsp1000(args)

    elif selected.name == "dtor985":
        args.connectome = "dtor985"
        return _handle_dtor985(args)

    return 0


def _get_recommended_batches() -> int:
    """Get recommended batch count based on available RAM."""
    try:
        import os

        # Try to get available memory
        if hasattr(os, "sysconf"):
            # Unix-like systems
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            ram_gb = (pages * page_size) / (1024**3)
        else:
            # Fallback
            ram_gb = 16  # Assume 16GB

        # Recommendations based on RAM
        if ram_gb >= 64:
            return 50
        elif ram_gb >= 32:
            return 100
        elif ram_gb >= 16:
            return 150
        else:
            return 150

    except Exception:
        return 100  # Default


def _handle_clean(args: argparse.Namespace) -> int:
    """Handle --clean flag to remove specific connectome data."""
    from lacuna.io.fetch import get_data_dir

    connectome = getattr(args, "connectome", None)
    if not connectome:
        print("Error: Specify which connectome to clean (e.g., 'lacuna fetch gsp1000 --clean')")
        return 1

    cache_dir = get_data_dir() / "connectomes" / connectome

    if not cache_dir.exists():
        print(f"No cached data found for '{connectome}'")
        return 0

    import shutil

    try:
        confirm = (
            input(f"Remove all data for '{connectome}' at {cache_dir}? [y/N]: ").strip().lower()
        )
        if confirm != "y":
            print("Cancelled.")
            return 0

        shutil.rmtree(cache_dir)
        print(f"✓ Removed cached data for '{connectome}'")
        return 0

    except Exception as e:
        print(f"Error removing data: {e}")
        return 1


def _handle_clean_all(args: argparse.Namespace) -> int:
    """Handle --clean-all flag to remove all connectome data."""
    from lacuna.io.fetch import get_data_dir

    cache_dir = get_data_dir() / "connectomes"

    if not cache_dir.exists():
        print("No cached connectome data found.")
        return 0

    import shutil

    # Calculate total size
    total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)

    try:
        confirm = (
            input(f"Remove ALL connectome data ({size_mb:.1f} MB) at {cache_dir}? [y/N]: ")
            .strip()
            .lower()
        )
        if confirm != "y":
            print("Cancelled.")
            return 0

        shutil.rmtree(cache_dir)
        print(f"✓ Removed all connectome cache data ({size_mb:.1f} MB)")
        return 0

    except Exception as e:
        print(f"Error removing data: {e}")
        return 1
