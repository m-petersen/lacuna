"""Tutorial data access for Lacuna.

This module provides functions to access the bundled synthetic BIDS dataset
used in Lacuna tutorials and documentation.

Examples
--------
Get paths to tutorial data:

>>> from lacuna.data.tutorials import get_tutorial_bids_dir
>>> bids_dir = get_tutorial_bids_dir()
>>> print(bids_dir)
/path/to/lacuna/data/tutorials

Copy tutorial data to a working directory:

>>> from lacuna.data.tutorials import setup_tutorial_data
>>> tutorial_dir = setup_tutorial_data("~/lacuna_tutorial")
>>> print(tutorial_dir)
/home/user/lacuna_tutorial
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

__all__ = [
    "get_tutorial_bids_dir",
    "get_tutorial_subjects",
    "get_subject_mask_path",
    "setup_tutorial_data",
]


def _get_data_dir() -> Path:
    """Get the path to the bundled tutorial data directory."""
    return Path(__file__).parent


def get_tutorial_bids_dir() -> Path:
    """Get the path to the bundled synthetic BIDS dataset.

    Returns the path to the tutorial BIDS directory bundled with Lacuna.
    This directory contains synthetic lesion masks that can be used for
    learning and testing.

    Returns
    -------
    Path
        Path to the synthetic BIDS dataset directory.

    Examples
    --------
    >>> from lacuna.data.tutorials import get_tutorial_bids_dir
    >>> bids_dir = get_tutorial_bids_dir()
    >>> list(bids_dir.glob("sub-*"))
    [PosixPath('.../sub-01'), PosixPath('.../sub-02'), PosixPath('.../sub-03')]

    Notes
    -----
    The bundled dataset includes:
    - 3 synthetic subjects (sub-01, sub-02, sub-03)
    - Binary lesion masks in MNI152NLin6 space (1mm resolution)
    - BIDS-compliant structure with dataset_description.json and participants.tsv
    """
    data_dir = _get_data_dir()
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Tutorial data directory not found: {data_dir}. "
            "This may indicate a corrupted installation."
        )
    return data_dir


def get_tutorial_subjects() -> list[str]:
    """Get a list of subject IDs in the tutorial dataset.

    Returns
    -------
    list[str]
        Subject IDs (e.g., ["sub-01", "sub-02", "sub-03"]).

    Examples
    --------
    >>> from lacuna.data.tutorials import get_tutorial_subjects
    >>> subjects = get_tutorial_subjects()
    >>> print(subjects)
    ['sub-01', 'sub-02', 'sub-03']
    """
    bids_dir = get_tutorial_bids_dir()
    subjects = sorted(
        d.name for d in bids_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")
    )
    return subjects


def get_subject_mask_path(
    subject_id: str,
    session_id: str = "ses-01",
) -> Path:
    """Get the path to a subject's lesion mask.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., "sub-01").
    session_id : str, optional
        Session identifier. Default: "ses-01".

    Returns
    -------
    Path
        Path to the lesion mask NIfTI file.

    Raises
    ------
    FileNotFoundError
        If the mask file does not exist.

    Examples
    --------
    >>> from lacuna.data.tutorials import get_subject_mask_path
    >>> mask_path = get_subject_mask_path("sub-01")
    >>> print(mask_path.name)
    sub-01_ses-01_space-MNI152NLin6Asym_label-acuteinfarct_mask.nii.gz
    """
    bids_dir = get_tutorial_bids_dir()

    # Build expected path
    anat_dir = bids_dir / subject_id / session_id / "anat"

    if not anat_dir.exists():
        raise FileNotFoundError(
            f"Anatomical directory not found for {subject_id}/{session_id}: {anat_dir}"
        )

    # Find mask file (flexible on exact naming)
    masks = list(anat_dir.glob("*_mask.nii.gz"))
    if not masks:
        raise FileNotFoundError(
            f"No mask file found in {anat_dir}. "
            f"Expected pattern: *_mask.nii.gz"
        )

    # Return first match (typically only one per session)
    return masks[0]


def setup_tutorial_data(
    target_dir: str | PathLike[str],
    *,
    overwrite: bool = False,
) -> Path:
    """Copy tutorial data to a working directory.

    Creates a copy of the synthetic BIDS dataset in the specified location.
    Useful for tutorials where users want to modify or experiment with the data.

    Parameters
    ----------
    target_dir : str or PathLike
        Directory where tutorial data will be copied.
    overwrite : bool, optional
        If True, overwrite existing directory. Default: False.

    Returns
    -------
    Path
        Path to the copied tutorial data directory.

    Raises
    ------
    FileExistsError
        If target_dir exists and overwrite is False.

    Examples
    --------
    >>> from lacuna.data.tutorials import setup_tutorial_data
    >>> tutorial_dir = setup_tutorial_data("~/my_tutorial")
    >>> print(tutorial_dir)
    /home/user/my_tutorial

    >>> # The directory now contains the BIDS dataset
    >>> list(tutorial_dir.glob("sub-*"))
    [PosixPath('/home/user/my_tutorial/sub-01'), ...]

    Notes
    -----
    The copied dataset is identical to the bundled data and includes:
    - dataset_description.json
    - participants.tsv
    - sub-01/ses-01/anat/sub-01_ses-01_space-MNI152NLin6Asym_label-acuteinfarct_mask.nii.gz
    - sub-02/ses-01/anat/sub-02_ses-01_space-MNI152NLin6Asym_label-acuteinfarct_mask.nii.gz
    - sub-03/ses-01/anat/sub-03_ses-01_space-MNI152NLin6Asym_label-acuteinfarct_mask.nii.gz
    """
    target = Path(target_dir).expanduser().resolve()

    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target directory already exists: {target}. "
                "Use overwrite=True to replace it."
            )
        shutil.rmtree(target)

    source = get_tutorial_bids_dir()
    shutil.copytree(source, target)

    return target
