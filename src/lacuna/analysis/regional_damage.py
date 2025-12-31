"""
Regional damage analysis module.

Provides a convenient interface for computing lesion-atlas overlap.
This is a thin wrapper around ParcelAggregation configured for regional
damage analysis.

Examples
--------
>>> from lacuna import SubjectData
>>> from lacuna.analysis import RegionalDamage
>>>
>>> # Load lesion data
>>> lesion = SubjectData.from_nifti("lesion.nii.gz")
>>>
>>> # Compute regional damage
>>> analysis = RegionalDamage(atlas_dir="/data/atlases")
>>> result = analysis.run(lesion)
>>>
>>> # Access results (percent overlap per region)
>>> print(result.results["ParcelAggregation"])
"""

from lacuna.analysis.parcel_aggregation import ParcelAggregation


class RegionalDamage(ParcelAggregation):
    """
    Compute lesion overlap with atlas regions.

    This is a convenience wrapper around ParcelAggregation that:
    - Sets source="maskimg" (analyze the lesion mask)
    - Sets aggregation="percent" (compute overlap percentages)

    This provides a simpler interface for the common use case of computing
    how much of each brain region is damaged by a lesion.

    Attributes
    ----------
    batch_strategy : str
        Batch processing strategy. Set to "parallel" as regional damage
        analysis is independent per subject and benefits from parallel processing.

    Parameters
    ----------
    parcel_names : list of str or None, default=None
        Names of atlases from the registry to process (e.g., "Schaefer2018_100Parcels7Networks").
        If None, all registered atlases are processed.
        Use list_parcellations() to see available atlases.
    threshold : float | None, default=None
        Threshold for binary lesion conversion. If None, no thresholding is applied.
        For probabilistic atlases: minimum probability to consider a voxel
        as belonging to a region (0.0-1.0).

    Raises
    ------
    ValueError
        If parcel_names contains non-existent atlas names.

    Notes
    -----
    - Lesion mask must be in same space as atlases (typically MNI152)
    - Results show percentage of each region overlapping with lesion
    - For more control (e.g., computing volume instead of percent),
      use ParcelAggregation directly

    Examples
    --------
    >>> # Use all registered atlases
    >>> from lacuna import SubjectData
    >>> from lacuna.analysis import RegionalDamage
    >>>
    >>> lesion = SubjectData.from_nifti("lesion.nii.gz")
    >>> analysis = RegionalDamage()  # Uses all registered atlases
    >>> result = analysis.run(lesion)
    >>>
    >>> # Results are in ParcelAggregation namespace
    >>> overlap_pcts = result.results["ParcelAggregation"]
    >>> for region, pct in overlap_pcts.items():
    ...     if pct > 10:  # Show regions with >10% damage
    ...         print(f"{region}: {pct:.1f}%")
    >>>
    >>> # Process only specific atlases
    >>> analysis = RegionalDamage(
    ...     parcel_names=["Schaefer2018_100Parcels7Networks"]
    ... )
    >>> result = analysis.run(lesion)

    See Also
    --------
    ParcelAggregation : More flexible aggregation with custom sources/methods
    """

    #: Preferred batch processing strategy
    batch_strategy: str = "parallel"

    def __init__(
        self,
        parcel_names: list[str] | None = None,
        threshold: float | None = None,
        log_level: int = 1,
    ):
        """
        Initialize RegionalDamage analysis.

        This is equivalent to:
        ParcelAggregation(source="maskimg",
                        aggregation="percent", threshold=threshold,
                        parcel_names=parcel_names,
                        log_level=log_level)

        Parameters
        ----------
        threshold : float | None, default=None
            Threshold for binary lesion conversion. If None, no thresholding is applied.
        parcel_names : list[str] | None, optional
            List of specific parcellation names to use. If None, uses all available.
        log_level : int, default=1
            Logging verbosity level (0=silent, 1=normal, 2=verbose)
        """
        super().__init__(
            source="maskimg",
            aggregation="percent",
            threshold=threshold,
            parcel_names=parcel_names,
            log_level=log_level,
        )

    def _validate_inputs(self, mask_data) -> None:
        """
        Validate inputs for regional damage analysis.

        Extends parent validation to ensure lesion mask is binary.

        Parameters
        ----------
        mask_data : SubjectData
            Lesion data to validate

        Raises
        ------
        ValueError
            If lesion mask is not binary (contains values other than 0 and 1)
        """
        # Run parent validation first
        super()._validate_inputs(mask_data)

        # Check that lesion mask is binary
        import numpy as np

        mask_data_arr = mask_data.mask_img.get_fdata()
        unique_vals = np.unique(mask_data_arr)

        # Binary mask should only have 0 and 1 (or just 0, or just 1)
        if not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError(
                f"RegionalDamage requires binary lesion mask (0 and 1 only).\n"
                f"Found values: {unique_vals}\n"
                f"Use thresholding or binarization to convert continuous maps."
            )

    def _get_parameters(self) -> dict:
        """Get analysis parameters for provenance and display.

        Returns
        -------
        dict
            Dictionary of parameter names and values.
        """
        params = super()._get_parameters()
        # RegionalDamage is a specific configuration of ParcelAggregation
        # Override the source and aggregation to reflect the simplified API
        params.update(
            {
                "analysis_type": "RegionalDamage",
                "threshold": params.get("threshold", 0.5),
            }
        )
        return params
