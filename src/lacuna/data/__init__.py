"""
Bundled reference data for lesion decoding toolkit.

This module provides access to lightweight reference atlases bundled with the
package, enabling zero-configuration usage for common analyses.

All bundled atlases use BIDS-compliant naming:
- Template: ``tpl-{template}_res-{resolution}``
- Atlas: ``atlas-{name}_desc-{description}``
- Suffix: ``_dseg`` (discrete segmentation) or ``_probseg`` (probabilistic)

Available atlases:
    - Schaefer 2018 cortical parcellation (100, 200, 400, 1000 parcels)
    - Tian subcortical atlas (3 scales)

Examples
--------
>>> from lacuna.data import get_bundled_atlas_dir, list_bundled_atlases
>>>
>>> # List all bundled atlases
>>> atlases = list_bundled_atlases()
>>> print(atlases[:2])
['tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-1000Parcels7Networks_dseg', ...]
>>>
>>> # Get the bundled atlas directory
>>> atlas_dir = get_bundled_atlas_dir()
>>>
>>> # Use bundled atlases in analysis (default behavior)
>>> from lacuna.analysis import FocalDamage
>>> analysis = FocalDamage()  # Automatically uses bundled atlases!
>>>
>>> # Get specific atlas files
>>> schaefer = 'tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-100Parcels7Networks_dseg'
>>> img_path, labels_path = get_bundled_atlas(schaefer)
"""

from pathlib import Path

__all__ = [
    "get_bundled_atlas_dir",
    "list_bundled_atlases",
    "get_bundled_atlas",
    "get_atlas_citation",
    # Tutorial data
    "get_tutorial_bids_dir",
    "get_tutorial_subjects",
    "get_subject_mask_path",
    "setup_tutorial_data",
]

# Re-export tutorial data functions
from lacuna.data.tutorials import (
    get_subject_mask_path,
    get_tutorial_bids_dir,
    get_tutorial_subjects,
    setup_tutorial_data,
)


def get_bundled_atlas_dir() -> Path:
    """
    Get the directory containing bundled reference atlases.

    Returns
    -------
    Path
        Absolute path to bundled atlases directory

    Examples
    --------
    >>> from lacuna.data import get_bundled_atlas_dir
    >>> atlas_dir = get_bundled_atlas_dir()
    >>> print(atlas_dir)
    PosixPath('/home/user/env/lib/python3.10/site-packages/lacuna/data/atlases')
    """
    return Path(__file__).parent / "atlases"


def list_bundled_atlases() -> list[str]:
    """
    List all bundled atlas names (base names without extensions).

    Returns
    -------
    list of str
        Sorted list of atlas base names

    Examples
    --------
    >>> from lacuna.data import list_bundled_atlases
    >>> atlases = list_bundled_atlases()
    >>> print(atlases[0])  # First Schaefer atlas
    'tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-1000Parcels7Networks_dseg'
    >>> print(len(atlases))  # Schaefer (4) + Tian (3)
    7
    """
    atlas_dir = get_bundled_atlas_dir()

    # Find all .nii.gz files
    nifti_files = list(atlas_dir.glob("*.nii.gz"))

    # Extract base names (remove .nii.gz)
    parcel_names = []
    for f in nifti_files:
        base_name = f.name.replace(".nii.gz", "")
        parcel_names.append(base_name)

    return sorted(parcel_names)


def get_bundled_atlas(name: str) -> tuple[Path, Path]:
    """
    Get paths to a specific bundled atlas image and labels file.

    Parameters
    ----------
    name : str
        Atlas base name (without extension)

    Returns
    -------
    tuple of Path
        (image_path, labels_path) for the requested atlas

    Raises
    ------
    ValueError
        If the requested atlas is not found in bundled data

    Examples
    --------
    >>> from lacuna.data import get_bundled_atlas
    >>> schaefer = 'tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-100Parcels7Networks_dseg'
    >>> img, labels = get_bundled_atlas(schaefer)
    >>> print(img.name)
    'tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-100Parcels7Networks_dseg.nii.gz'
    >>>
    >>> # Check files exist
    >>> print(img.exists(), labels.exists())
    True True
    """
    atlas_dir = get_bundled_atlas_dir()

    # Try to find image file
    img_path = atlas_dir / f"{name}.nii.gz"
    if not img_path.exists():
        available = list_bundled_atlases()
        raise ValueError(
            f"Bundled atlas '{name}' not found. Available atlases: {', '.join(available)}"
        )

    # Try to find labels file (try _labels.txt first, then .txt)
    labels_candidates = [
        atlas_dir / f"{name}_labels.txt",
        atlas_dir / f"{name}.txt",
    ]

    labels_path = None
    for candidate in labels_candidates:
        if candidate.exists():
            labels_path = candidate
            break

    if labels_path is None:
        raise ValueError(
            f"Labels file not found for atlas '{name}'. "
            f"Expected {labels_candidates[0]} or {labels_candidates[1]}"
        )

    return img_path, labels_path


def get_atlas_citation(name: str) -> str:
    """
    Get the citation information for a bundled atlas.

    Parameters
    ----------
    name : str
        Atlas base name (without extension)

    Returns
    -------
    str
        Citation text for the atlas

    Examples
    --------
    >>> from lacuna.data import get_atlas_citation
    >>> schaefer = 'tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-100Parcels7Networks_dseg'
    >>> citation = get_atlas_citation(schaefer)
    >>> print(citation[:20])
    'Schaefer 2018 Atlas'
    """
    # Citation database - keys match actual bundled atlas names
    citations = {
        "tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-100Parcels7Networks_dseg": "Schaefer et al, 2018. https://doi.org/10.1093/cercor/bhx179",
        "tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-200Parcels7Networks_dseg": "Schaefer et al, 2018. https://doi.org/10.1093/cercor/bhx179",
        "tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-400Parcels7Networks_dseg": "Schaefer et al, 2018. https://doi.org/10.1093/cercor/bhx179",
        "tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-1000Parcels7Networks_dseg": "Schaefer et al, 2018. https://doi.org/10.1093/cercor/bhx179",
        "tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS1_dseg": "Tian et al, 2020. https://doi.org/10.1038/s41593-020-00711-6",
        "tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS2_dseg": "Tian et al, 2020. https://doi.org/10.1038/s41593-020-00711-6",
        "tpl-MNI152NLin6Asym_res-01_atlas-TianSubcortex_desc-3TS3_dseg": "Tian et al, 2020. https://doi.org/10.1038/s41593-020-00711-6",
    }

    if name not in citations:
        available = list(citations.keys())
        return f"No citation available for '{name}'. Available: {', '.join(available)}"

    return citations[name]
