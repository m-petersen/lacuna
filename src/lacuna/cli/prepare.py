"""Prepare command implementation for the Lacuna CLI.

Handles ``lacuna prepare <target>`` — precomputation of non-subject-specific
data products that analyses consume. Currently the only target reduces a
whole-brain functional connectome to a parcel-level connectivity matrix used
by accelerated functional network mapping (AFNM); structural and further
targets are reserved for the future.

Commands:
    lacuna prepare functional - Reduce a functional connectome to a parcel-level
                                connectivity matrix (input to ``lacuna run afnm``).
    lacuna prepare structural - Reserved (not yet implemented).
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

    if target == "functional":
        from lacuna.prepare.parcellate import run_parcellate_functional_cli

        return run_parcellate_functional_cli(args)

    if target == "structural":
        print(
            "Error: 'lacuna prepare structural' is not yet implemented.",
            file=sys.stderr,
        )
        return 1

    print(
        "Error: 'lacuna prepare' requires a target. Try 'lacuna prepare functional --help'.",
        file=sys.stderr,
    )
    return 1
