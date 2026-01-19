"""
BIDSify module - Convert NIfTI files to BIDS format.

This module provides functionality to convert a directory of NIfTI mask files
to a BIDS-compliant directory structure.

Examples
--------
>>> from lacuna.io.bidsify import bidsify
>>>
>>> # Convert directory of masks to BIDS
>>> bidsify(
...     input_dir="/data/masks",
...     output_dir="/data/bids_masks",
...     space="MNI152NLin6Asym",
...     label="lesion",
... )
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

__all__ = ["bidsify", "sanitize_subject_id"]

# Valid spaces for lacuna
VALID_SPACES = ["MNI152NLin6Asym", "MNI152NLin2009cAsym"]


def sanitize_subject_id(filename: str) -> str:
    """
    Sanitize a filename to be a valid BIDS subject ID.

    Removes all non-alphanumeric characters from the filename.

    Parameters
    ----------
    filename : str
        Original filename (without extension)

    Returns
    -------
    str
        Sanitized subject ID containing only alphanumeric characters

    Examples
    --------
    >>> sanitize_subject_id("patient_001")
    'patient001'
    >>> sanitize_subject_id("sub-test")
    'subtest'
    """
    # Remove all non-alphanumeric characters
    return re.sub(r"[^a-zA-Z0-9]", "", filename)


def bidsify(
    input_dir: Path | str,
    output_dir: Path | str,
    space: str,
    session: str | None = None,
    label: str | None = None,
) -> Path:
    """
    Convert a directory of NIfTI files to BIDS format.

    Takes NIfTI mask files from input_dir and creates a BIDS-compliant
    directory structure in output_dir. Each input filename becomes a
    subject ID (with special characters removed).

    Parameters
    ----------
    input_dir : Path or str
        Directory containing NIfTI mask files (.nii or .nii.gz)
    output_dir : Path or str
        Output directory for BIDS dataset
    space : str
        Coordinate space of the masks. Must be one of:
        - "MNI152NLin6Asym"
        - "MNI152NLin2009cAsym"
    session : str, optional
        Session label (e.g., "01" or "baseline"). If provided, creates
        session subdirectory structure.
    label : str, optional
        Label for the mask entity (e.g., "lesion", "tumor"). If not
        provided, no label entity is included in the filename.

    Returns
    -------
    Path
        Path to the created BIDS dataset directory

    Raises
    ------
    FileNotFoundError
        If input_dir does not exist
    ValueError
        If space is not valid or input_dir contains no NIfTI files

    Examples
    --------
    >>> bidsify(
    ...     input_dir="/data/masks",
    ...     output_dir="/data/bids",
    ...     space="MNI152NLin6Asym",
    ...     session="01",
    ...     label="lesion",
    ... )
    PosixPath('/data/bids')

    Notes
    -----
    The output structure follows BIDS conventions:

    Without session::

        output_dir/
        ├── dataset_description.json
        ├── participants.tsv
        └── sub-<id>/
            └── anat/
                └── sub-<id>_space-<space>_[label-<label>_]mask.nii.gz

    With session::

        output_dir/
        ├── dataset_description.json
        ├── participants.tsv
        └── sub-<id>/
            └── ses-<session>/
                └── anat/
                    └── sub-<id>_ses-<session>_space-<space>_[label-<label>_]mask.nii.gz
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Validate input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Validate space
    if space not in VALID_SPACES:
        raise ValueError(
            f"Invalid space '{space}'. Must be one of: {', '.join(VALID_SPACES)}"
        )

    # Find NIfTI files
    nifti_files = list(input_dir.glob("*.nii.gz")) + list(input_dir.glob("*.nii"))
    if not nifti_files:
        raise ValueError(f"No NIfTI files found in {input_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track subjects for participants.tsv
    subjects = []

    # Process each NIfTI file
    for nifti_file in nifti_files:
        # Get subject ID from filename
        stem = nifti_file.name
        # Remove .nii.gz or .nii extension
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        elif stem.endswith(".nii"):
            stem = stem[:-4]

        subject_id = sanitize_subject_id(stem)
        subjects.append(subject_id)

        # Build BIDS path
        if session:
            anat_dir = output_dir / f"sub-{subject_id}" / f"ses-{session}" / "anat"
        else:
            anat_dir = output_dir / f"sub-{subject_id}" / "anat"

        anat_dir.mkdir(parents=True, exist_ok=True)

        # Build BIDS filename
        filename_parts = [f"sub-{subject_id}"]
        if session:
            filename_parts.append(f"ses-{session}")
        filename_parts.append(f"space-{space}")
        if label:
            filename_parts.append(f"label-{label}")
        filename_parts.append("mask.nii.gz")

        bids_filename = "_".join(filename_parts)
        output_path = anat_dir / bids_filename

        # Copy file
        shutil.copy2(nifti_file, output_path)

    # Create dataset_description.json
    _create_dataset_description(output_dir)

    # Create participants.tsv
    _create_participants_tsv(output_dir, subjects)

    return output_dir


def _create_dataset_description(output_dir: Path) -> None:
    """Create BIDS dataset_description.json file."""
    description = {
        "Name": "Lacuna BIDSified Dataset",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "lacuna bidsify",
                "Description": "Converted from raw NIfTI files to BIDS format",
            }
        ],
    }

    with open(output_dir / "dataset_description.json", "w") as f:
        json.dump(description, f, indent=2)


def _create_participants_tsv(output_dir: Path, subjects: list[str]) -> None:
    """Create BIDS participants.tsv file."""
    lines = ["participant_id"]
    for subject_id in sorted(set(subjects)):
        lines.append(f"sub-{subject_id}")

    with open(output_dir / "participants.tsv", "w") as f:
        f.write("\n".join(lines) + "\n")
