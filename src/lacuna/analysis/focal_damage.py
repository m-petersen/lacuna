"""
Focal damage analysis module.

Provides a convenient interface for computing lesion-atlas overlap.
This is a thin wrapper around ParcelAggregation configured for regional
damage analysis.

Examples
--------
>>> from lacuna import SubjectData
>>> from lacuna.analysis import FocalDamage
>>>
>>> # Load mask data
>>> mask = SubjectData.from_nifti("mask.nii.gz")
>>>
>>> # Compute focal damage
>>> analysis = FocalDamage(atlas_dir="/data/atlases")
>>> result = analysis.run(mask)
>>>
>>> # Access results (percent overlap per region)
>>> print(result.results["FocalDamage"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lacuna.analysis.parcel_aggregation import ParcelAggregation
from lacuna.core.keys import build_result_key

if TYPE_CHECKING:
    from lacuna.core.data_types import DataContainer
    from lacuna.core.subject_data import SubjectData


class FocalDamage(ParcelAggregation):
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
        Batch processing strategy. Set to "sequential" to avoid race conditions
        with threading backends when accessing shared atlas resources.

    Parameters
    ----------
    parcel_names : list of str or None, default=None
        Names of atlases from the registry to process (e.g., "schaefer2018parcels100networks7").
        If None, all registered atlases are processed.
        Use list_parcellations() to see available atlases.

    Raises
    ------
    ValueError
        If parcel_names contains non-existent atlas names.

    Notes
    -----
    - Results show percentage of each region overlapping with mask
    - For more control (e.g., computing volume instead of percent),
      use ParcelAggregation directly

    Examples
    --------
    >>> # Use all registered atlases
    >>> from lacuna import SubjectData
    >>> from lacuna.analysis import FocalDamage
    >>>
    >>> mask = SubjectData.from_nifti("mask.nii.gz")
    >>> analysis = FocalDamage()  # Uses all registered atlases
    >>> result = analysis.run(mask)
    >>>
    >>> # Results are in FocalDamage namespace
    >>> overlap_pcts = result.results["FocalDamage"]
    >>> for region, pct in overlap_pcts.items():
    ...     if pct > 10:  # Show regions with >10% damage
    ...         print(f"{region}: {pct:.1f}%")
    >>>
    >>> # Process only specific atlases
    >>> analysis = FocalDamage(
    ...     parcel_names=["schaefer2018parcels100networks7"]
    ... )
    >>> result = analysis.run(mask)

    See Also
    --------
    ParcelAggregation : More flexible aggregation with custom sources/methods
    """

    #: Preferred batch processing strategy (sequential to avoid threading race conditions)
    batch_strategy: str = "sequential"

    def __init__(
        self,
        parcel_names: list[str] | None = None,
        verbose: bool = False,
        keep_intermediate: bool = False,
    ):
        """
        Initialize FocalDamage analysis.

        This is equivalent to:
        ParcelAggregation(source="maskimg",
                        aggregation="percent",
                        parcel_names=parcel_names,
                        verbose=verbose,
                        keep_intermediate=keep_intermediate)

        Parameters
        ----------
        parcel_names : list[str] | None, optional
            List of specific parcellation names to use. If None, uses all available.
        verbose : bool, default=False
            If True, print progress messages. If False, run silently.
        keep_intermediate : bool, default=False
            If True, include intermediate results (e.g., warped mask images)
            in the output. Useful for debugging and quality control.
        """
        super().__init__(
            source="maskimg",
            aggregation="percent",
            parcel_names=parcel_names,
            verbose=verbose,
            keep_intermediate=keep_intermediate,
        )

    def _validate_inputs(self, mask_data) -> None:
        """
        Validate inputs for focal damage analysis.

        Extends parent validation to ensure mask is binary.

        Parameters
        ----------
        mask_data : SubjectData
            Mask data to validate

        Raises
        ------
        ValueError
            If mask is not binary (contains values other than 0 and 1)
        """
        # Run parent validation first
        super()._validate_inputs(mask_data)

        # Check that mask is binary
        import numpy as np

        mask_data_arr = mask_data.mask_img.get_fdata()
        unique_vals = np.unique(mask_data_arr)

        # Binary mask should only have 0 and 1 (or just 0, or just 1)
        if not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError(
                f"FocalDamage requires binary mask (0 and 1 only).\n"
                f"Found values: {unique_vals}\n"
                f"Use thresholding or binarization to convert continuous maps."
            )

    def _run_analysis(self, mask_data: SubjectData) -> dict[str, DataContainer]:
        """
        Compute percentage and binary focal damage for all atlases.

        Extends parent to add binary damage results (1 if region has any
        overlap with the mask, 0 otherwise) alongside percentage results.

        Parameters
        ----------
        mask_data : SubjectData
            Validated lesion data.

        Returns
        -------
        dict[str, DataContainer]
            Dictionary with both percentage and binary ParcelData objects.
        """
        from lacuna.core.data_types import ParcelData

        # Get percentage results from parent (keyed as InputMask)
        parent_results = super()._run_analysis(mask_data)

        # Re-key percentage results and derive binary results
        results = {}
        for key, result in parent_results.items():
            if not isinstance(result, ParcelData):
                # Keep non-ParcelData results (e.g., intermediates) as-is
                results[key] = result
                continue

            # Re-key percentage result: atlas-{atlas}_source-FocalDamage_desc-damagepct
            pct_key = build_result_key(
                atlas=result.name,
                source="FocalDamage",
                desc="damagepct",
            )
            results[pct_key] = result

            # Derive binary result: >0 → 1, else → 0
            binary_data = {
                region: 1.0 if value > 0 else 0.0 for region, value in result.data.items()
            }

            binary_parcel = ParcelData(
                name=result.name,
                data=binary_data,
                parcel_names=result.parcel_names,
                aggregation_method="binary",
                metadata={
                    **result.metadata,
                    "derived_from": "percent",
                    "description": "Binary damage indicator (1 = any overlap, 0 = none)",
                },
            )

            binary_key = build_result_key(
                atlas=result.name,
                source="FocalDamage",
                desc="damagebin",
            )
            results[binary_key] = binary_parcel

        return results

    def _get_parameters(self) -> dict:
        """Get analysis parameters for provenance and display.

        Returns
        -------
        dict
            Dictionary of parameter names and values.
        """
        params = super()._get_parameters()
        # FocalDamage is a specific configuration of ParcelAggregation
        # Override the source and aggregation to reflect the simplified API
        params.update(
            {
                "analysis_type": "FocalDamage",
                "threshold": params.get("threshold"),
            }
        )
        return params
