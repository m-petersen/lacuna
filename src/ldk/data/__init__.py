"""
Bundled reference data for lesion decoding toolkit.

This module provides access to lightweight reference atlases bundled with the
package, enabling zero-configuration usage for common analyses.

Examples
--------
>>> from ldk.data import get_bundled_atlas_dir, list_bundled_atlases
>>>
>>> # List all bundled atlases
>>> atlases = list_bundled_atlases()
>>> print(atlases)
['schaefer2018-100parcels-7networks', ...]
>>>
>>> # Get the bundled atlas directory
>>> atlas_dir = get_bundled_atlas_dir()
>>> print(atlas_dir)
PosixPath('/path/to/ldk/data/atlases')
>>>
>>> # Use bundled atlases in analysis (default behavior)
>>> from ldk.analysis import RegionalDamage
>>> analysis = RegionalDamage()  # Automatically uses bundled atlases!
>>>
>>> # Get specific atlas files
>>> img_path, labels_path = get_bundled_atlas('aal3')
"""

from pathlib import Path

__all__ = [
    "get_bundled_atlas_dir",
    "list_bundled_atlases",
    "get_bundled_atlas",
    "get_atlas_citation",
]


def get_bundled_atlas_dir() -> Path:
    """
    Get the directory containing bundled reference atlases.

    Returns
    -------
    Path
        Absolute path to bundled atlases directory

    Examples
    --------
    >>> from ldk.data import get_bundled_atlas_dir
    >>> atlas_dir = get_bundled_atlas_dir()
    >>> print(atlas_dir)
    PosixPath('/home/user/env/lib/python3.10/site-packages/ldk/data/atlases')
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
    >>> from ldk.data import list_bundled_atlases
    >>> atlases = list_bundled_atlases()
    >>> print(atlases)
    ['aal3', 'harvard-oxford-cortical', 'schaefer2018-100parcels-7networks']
    >>> print(len(atlases))
    3
    """
    atlas_dir = get_bundled_atlas_dir()

    # Find all .nii.gz files
    nifti_files = list(atlas_dir.glob("*.nii.gz"))

    # Extract base names (remove .nii.gz)
    atlas_names = []
    for f in nifti_files:
        base_name = f.name.replace(".nii.gz", "")
        atlas_names.append(base_name)

    return sorted(atlas_names)


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
    >>> from ldk.data import get_bundled_atlas
    >>> img, labels = get_bundled_atlas('aal3')
    >>> print(img.name, labels.name)
    aal3.nii.gz aal3_labels.txt
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
    >>> from ldk.data import get_atlas_citation
    >>> citation = get_atlas_citation('aal3')
    >>> print(citation)
    AAL3 Atlas:
    Rolls, E.T., Joliot, M. & Tzourio-Mazoyer, N. Implementation of a new...
    """
    # Citation database
    citations = {
        "aal3": """AAL3 Atlas (170 regions):
Rolls, E.T., Joliot, M. & Tzourio-Mazoyer, N. (2015).
Implementation of a new parcellation of the orbitofrontal cortex in the 
automated anatomical labeling atlas.
NeuroImage, 122, 1-5.
https://doi.org/10.1016/j.neuroimage.2015.07.075""",
        "harvard-oxford-cortical": """Harvard-Oxford Cortical Atlas (48 regions):
Desikan, R.S., Ségonne, F., Fischl, B., et al. (2006).
An automated labeling system for subdividing the human cerebral cortex on MRI 
scans into gyral based regions of interest.
NeuroImage, 31(3), 968-980.
https://doi.org/10.1016/j.neuroimage.2006.01.021

Available through FSL: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases""",
        "schaefer2018-100parcels-7networks": """Schaefer 2018 Atlas (100 parcels, 7 networks):
Schaefer, A., Kong, R., Gordon, E.M., et al. (2018).
Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic 
Functional Connectivity MRI.
Cerebral Cortex, 28(9), 3095-3114.
https://doi.org/10.1093/cercor/bhx179""",
        "schaefer2018-400parcels-7networks": """Schaefer 2018 Atlas (400 parcels, 7 networks):
Schaefer, A., Kong, R., Gordon, E.M., et al. (2018).
Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic 
Functional Connectivity MRI.
Cerebral Cortex, 28(9), 3095-3114.
https://doi.org/10.1093/cercor/bhx179""",
    }

    if name not in citations:
        available = list(citations.keys())
        return f"No citation available for '{name}'. Available: {', '.join(available)}"

    return citations[name]
