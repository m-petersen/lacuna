"""Prepare command implementation for the Lacuna CLI.

Handles ``lacuna prepare <analysis>`` — precomputation of the non-subject-specific
data product a given analysis consumes. Targets are named after the analysis they
prepare for (mirroring ``lacuna run``). Currently the only target is ``afnm``,
which reduces a whole-brain functional connectome to the parcel-level connectivity
matrix used by accelerated functional network mapping.

Commands:
    lacuna prepare afnm - Build the parcel-level connectivity matrix that
                          'lacuna run afnm' consumes via --matrix-path.
"""

from __future__ import annotations

import argparse
import sys


def handle_prepare_command(args: argparse.Namespace) -> int:
    """Dispatch the ``lacuna prepare`` subcommand by target.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. ``args.prepare_target`` selects the
        precomputation target.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    target = getattr(args, "prepare_target", None)

    if target == "afnm":
        from lacuna.prepare.parcellate import run_parcellate_functional_cli

        return run_parcellate_functional_cli(args)

    print(
        "Error: 'lacuna prepare' requires a target. Try 'lacuna prepare afnm --help'.",
        file=sys.stderr,
    )
    return 1
