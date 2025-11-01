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
    atlas_names : list of str or None, default=None
        If provided, only process atlases with these names (without file extensions).
        Atlas names should match the base filename (e.g., "HCP1065" for "HCP1065.nii.gz").
        If None, all atlases found in atlas_dir will be processed.
        Example: ["HCP1065", "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm"]

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
    >>>
    >>> # Process only specific atlases
    >>> analysis = RegionalDamage(
    ...     atlas_dir="/data/atlases",
    ...     atlas_names=["HCP1065"]  # Only white matter tracts
    ... )
    >>> result = analysis.run(lesion)

    See Also
    --------
    AtlasAggregation : More flexible aggregation with custom sources/methods
    """

    def __init__(
        self,
        atlas_dir: str | Path,
        threshold: float = 0.5,
        atlas_names: list[str] | None = None,
    ):
        """
        Initialize RegionalDamage analysis.

        This is equivalent to:
        AtlasAggregation(atlas_dir=atlas_dir, source="lesion_img",
                        aggregation="percent", threshold=threshold,
                        atlas_names=atlas_names)
        """
        super().__init__(
            atlas_dir=atlas_dir,
            source="lesion_img",
            aggregation="percent",
            threshold=threshold,
            atlas_names=atlas_names,
        )

    def _validate_inputs(self, lesion_data) -> None:
        """
        Validate inputs for regional damage analysis.

        Extends parent validation to ensure lesion mask is binary.

        Parameters
        ----------
        lesion_data : LesionData
            Lesion data to validate

        Raises
        ------
        ValueError
            If lesion mask is not binary (contains values other than 0 and 1)
        """
        # Run parent validation first
        super()._validate_inputs(lesion_data)

        # Check that lesion mask is binary
        import numpy as np

        lesion_data_arr = lesion_data.lesion_img.get_fdata()
        unique_vals = np.unique(lesion_data_arr)

        # Binary mask should only have 0 and 1 (or just 0, or just 1)
        if not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError(
                f"RegionalDamage requires binary lesion mask (0 and 1 only).\n"
                f"Found values: {unique_vals}\n"
                f"Use thresholding or binarization to convert continuous maps."
            )
