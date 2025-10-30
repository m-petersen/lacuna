"""
Regional damage analysis module.

Provides a convenient interface for computing lesion-atlas overlap.
This is a thin wrapper around AtlasAggregation configured for regional
damage analysis.

Examples
--------
>>> from ldk import LesionData
>>> from ldk.analysis import RegionalDamage
>>>
>>> # Load lesion data
>>> lesion = LesionData.from_nifti("lesion.nii.gz")
>>>
>>> # Compute regional damage
>>> analysis = RegionalDamage(atlas_dir="/data/atlases")
>>> result = analysis.run(lesion)
>>>
>>> # Access results (percent overlap per region)
>>> print(result.results["AtlasAggregation"])
"""

from pathlib import Path

from ldk.analysis.atlas_aggregation import AtlasAggregation


class RegionalDamage(AtlasAggregation):
    """
    Compute lesion overlap with atlas regions.

    This is a convenience wrapper around AtlasAggregation that:
    - Sets source="lesion_img" (analyze the lesion mask)
    - Sets aggregation="percent" (compute overlap percentages)

    This provides a simpler interface for the common use case of computing
    how much of each brain region is damaged by a lesion.

    Parameters
    ----------
    atlas_dir : str or Path
        Directory containing atlas files. Each atlas should have:
        - NIfTI file (.nii or .nii.gz)
        - Labels file with same base name + "_labels.txt" or ".txt"
    threshold : float, default=0.5
        For probabilistic atlases: minimum probability to consider a voxel
        as belonging to a region (0.0-1.0).

    Raises
    ------
    ValueError
        If atlas_dir doesn't exist or contains no valid atlas files.
    FileNotFoundError
        If specified atlas directory doesn't exist.

    Notes
    -----
    - Lesion mask must be in same space as atlases (typically MNI152)
    - Results show percentage of each region overlapping with lesion
    - For more control (e.g., computing volume instead of percent),
      use AtlasAggregation directly

    Examples
    --------
    >>> # Simple regional damage analysis
    >>> from ldk import LesionData
    >>> from ldk.analysis import RegionalDamage
    >>>
    >>> lesion = LesionData.from_nifti("lesion.nii.gz")
    >>> analysis = RegionalDamage(atlas_dir="/data/atlases")
    >>> result = analysis.run(lesion)
    >>>
    >>> # Results are in AtlasAggregation namespace
    >>> overlap_pcts = result.results["AtlasAggregation"]
    >>> for region, pct in overlap_pcts.items():
    ...     if pct > 10:  # Show regions with >10% damage
    ...         print(f"{region}: {pct:.1f}%")

    See Also
    --------
    AtlasAggregation : More flexible aggregation with custom sources/methods
    """

    def __init__(
        self,
        atlas_dir: str | Path,
        threshold: float = 0.5,
    ):
        """
        Initialize RegionalDamage analysis.

        This is equivalent to:
        AtlasAggregation(atlas_dir=atlas_dir, source="lesion_img",
                        aggregation="percent", threshold=threshold)
        """
        super().__init__(
            atlas_dir=atlas_dir,
            source="lesion_img",
            aggregation="percent",
            threshold=threshold,
        )
