"""
Atlas aggregation module.

Provides flexible ROI-level aggregation of voxel-level maps across multiple
atlases. This is the core tool for extracting region-of-interest statistics
from lesion masks, connectivity maps, or any other spatial maps.

Examples
--------
>>> from lacuna import LesionData
>>> from lacuna.analysis import AtlasAggregation
>>>
>>> # Use bundled atlases
>>> lesion = LesionData.from_nifti("lesion.nii.gz")
>>> analysis = AtlasAggregation(
...     source="lesion_img",
...     aggregation="percent"
... )
>>> result = analysis.run(lesion)
>>> print(result.results["AtlasAggregation"])
>>>
>>> # Use custom atlas directory
>>> analysis = AtlasAggregation(
...     atlas_dir="/data/atlases",
...     source="lesion_img",
...     aggregation="percent"
... )
>>> result = analysis.run(lesion)
>>>
>>> # Aggregate functional connectivity map by atlas regions
>>> from lacuna.analysis import FunctionalNetworkMapping
>>> fnm = FunctionalNetworkMapping(connectome_path="gsp1000.h5")
>>> result = fnm.run(lesion)
>>>
>>> # Now aggregate the network map to atlas ROIs
>>> agg = AtlasAggregation(
...     source="FunctionalNetworkMapping.network_map",
...     aggregation="mean"
... )
>>> final = agg.run(result)
"""

from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiLabelsMasker

from lacuna.analysis.base import BaseAnalysis
from lacuna.assets.atlases import load_atlas, list_atlases
from lacuna.core.spaces import detect_space_from_header
from lacuna.core.lesion_data import LesionData
from lacuna.core.output import ROIResult

if TYPE_CHECKING:
    from lacuna.core.output import AnalysisResult


class AtlasAggregation(BaseAnalysis):
    """Atlas aggregation analysis.
    
    Computations performed in input data space (atlases transformed to match input).
    """
    TARGET_SPACE = None  # Space determined from input data
    TARGET_RESOLUTION = None  # Resolution determined from input data
    """
    Aggregate voxel-level maps to ROI-level statistics using atlases.

    This is a composable analysis that can:
    1. Compute regional damage from lesion masks (percent overlap, volume)
    2. Aggregate connectivity maps from network analyses (mean, sum, etc.)
    3. Extract any voxel-level map to atlas ROI statistics

    The analysis discovers all atlases in the specified directory and computes
    the specified aggregation method for each region in each atlas.

    **Computation Space:**
    Atlases are automatically transformed to match the input data's coordinate space
    and resolution (parsed from metadata or BIDS-style filenames: tpl-{SPACE}_res-{RES}_...).
    If an atlas is already in the input space, no transformation is performed.
    After transformation, nilearn resamples the atlas to precisely match the input
    resolution for exact alignment.

    Attributes
    ----------
    TARGET_SPACE : None
        Space is determined from the input data. Atlases are transformed to match
        the input data's coordinate space automatically.
    TARGET_RESOLUTION : None
        Resolution is determined from the input data. Atlases are transformed to
        match the input resolution, then nilearn resamples for precise alignment.
    batch_strategy : str
        Batch processing strategy. Set to "parallel" as atlas aggregation
        is independent per subject and benefits from parallel processing.

    Parameters
    ----------
    source : str, default="lesion_img"
        Source of data to aggregate. Options:
        - "lesion_img": Use the lesion mask directly
        - "anatomical_img": Use the anatomical image
        - "{AnalysisName}.{result_key}": Use result from previous analysis
          Example: "FunctionalNetworkMapping.network_map"
    aggregation : str, default="mean"
        Aggregation method to use. Options:
        - "mean": Mean value across ROI voxels
        - "sum": Sum of values across ROI voxels
        - "percent": Percentage of ROI voxels that are non-zero (for binary masks)
        - "volume": Volume (in mm³) of non-zero voxels in ROI
        - "median": Median value across ROI voxels
        - "std": Standard deviation across ROI voxels
    threshold : float, default=0.5
        For probabilistic atlases: minimum probability to consider a voxel
        as belonging to a region (0.0-1.0).
    atlas_names : list of str or None, default=None
        Names of atlases from the registry to process (e.g., "Schaefer2018_100Parcels7Networks").
        If None, all registered atlases are processed.
        Use register_atlas() or register_atlases_from_directory() to add custom atlases.
        If None, all atlases found in atlas_dir will be processed.
        Example: ["HCP1065", "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm"]

    Raises
    ------
    ValueError
        If atlas_dir doesn't exist, is empty, or aggregation method is invalid.
    FileNotFoundError
        If specified atlas directory doesn't exist.

    Notes
    -----
    - Both 3D and 4D atlases support automatic resampling to match source data
      spatial resolution via nilearn
    - 3D atlases: integer labels, use NiftiLabelsMasker with nearest-neighbor
      interpolation to preserve labels
    - 4D atlases: automatically detect binary (0/1) vs probabilistic (0.0-1.0)
      * Binary: use nearest-neighbor interpolation to preserve binary masks
      * Probabilistic: use continuous interpolation for probability values
    - For 3D atlases: regions defined by integer labels (automatically rounded)
    - For 4D atlases: each volume is a binary or probability map for one region
    - 4D probabilistic maps are thresholded at `threshold` parameter (default 0.5)
    - Results stored in LesionData.results["AtlasAggregation"] as dict
      mapping atlas_name_region_name -> aggregated_value

    Examples
    --------
    >>> # Use all bundled/registered atlases
    >>> analysis = AtlasAggregation(
    ...     source="lesion_img",
    ...     aggregation="percent"
    ... )
    >>>
    >>> # Use specific registered atlases
    >>> analysis = AtlasAggregation(
    ...     source="lesion_img",
    ...     aggregation="percent",
    ...     atlas_names=["Schaefer2018_100Parcels7Networks", "TianSubcortex_3TS1"]
    ... )
    >>>
    >>> # Register custom atlases first, then use them
    >>> from lacuna.assets.atlases import register_atlases_from_directory
    >>> register_atlases_from_directory("/data/my_atlases")
    >>> analysis = AtlasAggregation(
    ...     source="lesion_img",
    ...     aggregation="percent"
    ... )
    >>>
    >>> # Average functional connectivity per ROI
    >>> analysis = AtlasAggregation(
    ...     source="FunctionalNetworkMapping.network_map",
    ...     aggregation="mean"
    ... )

    See Also
    --------
    RegionalDamage : Convenience wrapper for lesion overlap analysis
    BaseAnalysis : Parent class defining analysis interface
    """

    #: Preferred batch processing strategy
    batch_strategy: str = "parallel"

    VALID_AGGREGATIONS = ["mean", "sum", "percent", "volume", "median", "std"]
    VALID_SOURCES = ["lesion_img", "anatomical_img"]

    def __init__(
        self,
        source: str = "lesion_img",
        aggregation: str = "mean",
        threshold: float = 0.5,
        atlas_names: list[str] | None = None,
    ):
        """Initialize AtlasAggregation analysis."""
        super().__init__()

        self.source = source
        self.aggregation = aggregation
        self.threshold = threshold
        self.atlas_names = atlas_names

        # Validate aggregation method
        if aggregation not in self.VALID_AGGREGATIONS:
            raise ValueError(
                f"Invalid aggregation method: '{aggregation}'\n"
                f"Valid options: {', '.join(self.VALID_AGGREGATIONS)}"
            )

        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

        # Validate atlas_names if provided
        if atlas_names is not None:
            if not isinstance(atlas_names, list):
                raise TypeError(
                    f"atlas_names must be a list of strings or None, got {type(atlas_names).__name__}"
                )
            if not all(isinstance(name, str) for name in atlas_names):
                raise TypeError("All items in atlas_names must be strings")
            if not atlas_names:
                raise ValueError(
                    "atlas_names cannot be an empty list (use None to process all atlases)"
                )

        # Will be populated in _validate_inputs
        self.atlases = []

    def _validate_inputs(self, lesion_data: LesionData) -> None:
        """
        Validate lesion data and load atlases from registry.

        Parameters
        ----------
        lesion_data : LesionData
            Lesion data to validate

        Raises
        ------
        ValueError
            If lesion data is invalid or source data not found
        """
        # Check that source data exists
        source_img = self._get_source_image(lesion_data)

        if source_img is None:
            raise ValueError(
                f"Source data not found: {self.source}\n"
                "Check that the source exists in LesionData.\n"
                f"Available sources: lesion_img, anatomical_img, "
                f"or results from previous analyses."
            )

        # Load atlases from registry
        self.atlases = self._load_atlases_from_registry()

        if not self.atlases:
            if self.atlas_names is not None:
                raise ValueError(
                    f"No matching atlases found for specified names: {self.atlas_names}\n"
                    "Available atlases in registry: check list_atlases()\n"
                    "Use register_atlas() or register_atlases_from_directory() to add atlases"
                )
            else:
                raise ValueError(
                    "No valid atlases found in registry\n"
                    "Use register_atlas() or register_atlases_from_directory() to add atlases"
                )

        # Warn if some requested atlases weren't found
        if self.atlas_names is not None:
            found_names = {atlas["name"] for atlas in self.atlases}
            missing_names = set(self.atlas_names) - found_names
            if missing_names:
                import warnings

                warnings.warn(
                    f"Some requested atlases were not found: {sorted(missing_names)}\n"
                    f"Found atlases: {sorted(found_names)}",
                    UserWarning,
                    stacklevel=3,
                )

    def _load_atlases_from_registry(self) -> list[dict]:
        """
        Load atlases from the registry (bundled or user-registered).
        
        Returns
        -------
        list[dict]
            List of atlas dictionaries with keys: name, image, labels, space, resolution
        """
        from lacuna.assets.atlases.loader import BUNDLED_ATLASES_DIR
        
        # Get atlases from registry (filter by names if provided)
        if self.atlas_names is not None:
            # Load specific atlases by name
            atlases_data = []
            for name in self.atlas_names:
                try:
                    atlas = load_atlas(name)
                    
                    # Resolve paths (absolute or relative to bundled dir)
                    atlas_filename_path = Path(atlas.metadata.atlas_filename)
                    if atlas_filename_path.is_absolute():
                        atlas_path = atlas_filename_path
                    else:
                        atlas_path = BUNDLED_ATLASES_DIR / atlas.metadata.atlas_filename
                    
                    labels_filename_path = Path(atlas.metadata.labels_filename)
                    if labels_filename_path.is_absolute():
                        labels_path = labels_filename_path
                    else:
                        labels_path = BUNDLED_ATLASES_DIR / atlas.metadata.labels_filename
                    
                    atlases_data.append({
                        "name": name,
                        "atlas_path": atlas_path,
                        "labels_path": labels_path,
                        "labels": atlas.labels,
                        "space": atlas.metadata.space,
                        "resolution": atlas.metadata.resolution,
                        "is_4d": getattr(atlas.metadata, 'is_4d', False),
                    })
                except KeyError:
                    # Atlas not in registry - will be caught by validation
                    pass
        else:
            # Load all registered atlases
            atlas_metadatas = list_atlases()
            atlases_data = []
            for metadata in atlas_metadatas:
                atlas = load_atlas(metadata.name)
                
                # Resolve paths (absolute or relative to bundled dir)
                atlas_filename_path = Path(atlas.metadata.atlas_filename)
                if atlas_filename_path.is_absolute():
                    atlas_path = atlas_filename_path
                else:
                    atlas_path = BUNDLED_ATLASES_DIR / atlas.metadata.atlas_filename
                
                labels_filename_path = Path(atlas.metadata.labels_filename)
                if labels_filename_path.is_absolute():
                    labels_path = labels_filename_path
                else:
                    labels_path = BUNDLED_ATLASES_DIR / atlas.metadata.labels_filename
                
                atlases_data.append({
                    "name": metadata.name,
                    "atlas_path": atlas_path,
                    "labels_path": labels_path,
                    "labels": atlas.labels,
                    "space": metadata.space,
                    "resolution": metadata.resolution,
                    "is_4d": getattr(metadata, 'is_4d', False),
                })
        
        return atlases_data

    def _ensure_atlas_matches_input_space(
        self, 
        atlas_img: nib.Nifti1Image,
        atlas_space: str,
        atlas_resolution: int,
        input_space: str,
        input_resolution: int,
        input_affine: np.ndarray
    ) -> nib.Nifti1Image:
        """
        Transform atlas to match input data space if spaces don't match.
        
        This allows AtlasAggregation to work with any voxel-level image,
        not just lesion data, by transforming the atlas to the input space.

        Parameters
        ----------
        atlas_img : nib.Nifti1Image
            Atlas image to potentially transform
        atlas_space : str
            Atlas coordinate space (e.g., 'MNI152NLin6Asym')
        atlas_resolution : int
            Atlas resolution in mm (e.g., 1 or 2)
        input_space : str
            Input data coordinate space
        input_resolution : int
            Input data resolution in mm
        input_affine : np.ndarray
            Input data affine matrix

        Returns
        -------
        nib.Nifti1Image
            Atlas in input space (transformed if needed, original if already matching)
        """
        # If atlas doesn't specify space, assume it matches
        if atlas_space is None:
            return atlas_img
        
        # Check if spaces are equivalent (handles aliases like aAsym == cAsym)
        from lacuna.core.spaces import spaces_are_equivalent
        
        if spaces_are_equivalent(atlas_space, input_space):
            # Same space or equivalent alias - no coordinate transformation needed
            # (nilearn will handle resolution resampling during aggregation)
            return atlas_img
        
        # Need to transform atlas to input space
        from lacuna.core.spaces import CoordinateSpace
        from lacuna.spatial.transform import transform_image
        from lacuna.utils.logging import ConsoleLogger
        
        logger = ConsoleLogger()
        
        # Create target space matching input data
        target_space = CoordinateSpace(
            identifier=input_space,
            resolution=input_resolution,
            reference_affine=input_affine,
        )
        
        logger.info(
            f"Transforming atlas from {atlas_space}@{atlas_resolution}mm "
            f"to {input_space}@{input_resolution}mm to match input data"
        )
        
        # Transform atlas using nearest neighbor to preserve labels
        return transform_image(
            img=atlas_img,
            source_space=atlas_space,
            target_space=target_space,
            source_resolution=atlas_resolution,
            interpolation='nearest'  # Preserve integer labels
        )

    def _run_analysis(self, lesion_data: LesionData) -> list["AnalysisResult"]:
        """
        Compute ROI-level aggregation for all atlases.

        Parameters
        ----------
        lesion_data : LesionData
            Validated lesion data

        Returns
        -------
        list[AnalysisResult]
            List containing ROIResult with aggregated values per atlas region
        """
        results = {}

        # Get input data space/resolution once
        input_space = lesion_data.metadata.get('space')
        input_resolution = lesion_data.metadata.get('resolution')
        
        # Get source image (this is what we'll aggregate)
        source_img = self._get_source_image(lesion_data)
        
        # Calculate voxel volume from source data
        voxel_volume_mm3 = np.abs(np.linalg.det(source_img.affine[:3, :3]))

        # Collect atlas names for metadata
        atlas_names = []

        # Process each atlas
        for atlas_info in self.atlases:
            atlas_name = atlas_info["name"]
            atlas_names.append(atlas_name)
            atlas_space = atlas_info.get("space")
            atlas_resolution = atlas_info.get("resolution")
            
            # Load atlas image
            atlas_img = nib.load(atlas_info["atlas_path"])
            
            # Transform atlas to match input data space if needed
            atlas_img = self._ensure_atlas_matches_input_space(
                atlas_img=atlas_img,
                atlas_space=atlas_space,
                atlas_resolution=atlas_resolution,
                input_space=input_space,
                input_resolution=input_resolution,
                input_affine=source_img.affine
            )
            
            labels = atlas_info["labels"]
            atlas_data = atlas_img.get_fdata()

            # Warn if nilearn will resample atlas to match source resolution
            atlas_shape = atlas_data.shape[:3]  # Handle 4D atlases
            source_shape = source_img.get_fdata().shape
            if source_shape != atlas_shape:
                import warnings
                warnings.warn(
                    f"Atlas '{atlas_name}' will be resampled to match source data.\n"
                    f"Source shape: {source_shape}, Atlas shape: {atlas_shape}",
                    UserWarning,
                    stacklevel=2,
                )
            
            if atlas_data.ndim == 3:
                # 3D integer-labeled atlas - use nilearn NiftiLabelsMasker
                atlas_results = self._aggregate_3d_atlas(
                    source_img, atlas_img, labels, voxel_volume_mm3
                )
            elif atlas_data.ndim == 4:
                # 4D probabilistic atlas - use nilearn resampling
                atlas_results = self._aggregate_4d_atlas(
                    source_img, atlas_img, labels, voxel_volume_mm3
                )
            else:
                import warnings

                warnings.warn(
                    f"Skipping atlas '{atlas_name}': unexpected dimensions {atlas_data.ndim}D",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Add results with atlas name prefix
            for region_name, value in atlas_results.items():
                key = f"{atlas_name}_{region_name}"
                results[key] = value

        # Create ROIResult object
        roi_result = ROIResult(
            name="atlas_aggregation",
            data=results,
            atlas_names=atlas_names,
            aggregation_method=self.aggregation,
            metadata={
                "source": self.source,
                "threshold": self.threshold,
                "n_atlases": len(atlas_names),
                "n_regions": len(results)
            }
        )

        return [roi_result]

    def _aggregate_3d_atlas(
        self,
        source_img: nib.Nifti1Image,
        atlas_img: nib.Nifti1Image,
        labels: dict[int, str],
        voxel_volume_mm3: float,
    ) -> dict[str, float]:
        """
        Aggregate source data for 3D integer-labeled atlas using nilearn.

        Uses nilearn's NiftiLabelsMasker for robust extraction with automatic
        resampling, masking, and efficient computation.

        Parameters
        ----------
        source_img : nib.Nifti1Image
            Source image to aggregate
        atlas_img : nib.Nifti1Image
            3D atlas with integer labels
        labels : dict[int, str]
            Mapping from region ID to region name
        voxel_volume_mm3 : float
            Volume of one voxel in mm³ (for volume aggregation)

        Returns
        -------
        dict[str, float]
            Mapping from region name to aggregated value
        """
        # Map our aggregation methods to nilearn strategies
        strategy_map = {
            "mean": "mean",
            "sum": "sum",
            "median": "median",
            "std": "standard_deviation",
            "percent": "mean",  # Will multiply by 100
            "volume": "sum",  # Will multiply by voxel_volume_mm3
        }

        if self.aggregation not in strategy_map:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        strategy = strategy_map[self.aggregation]

        # Create label names list (NiftiLabelsMasker expects ordered list)
        # Background (0) should not be included
        atlas_data = atlas_img.get_fdata()

        # Round atlas values to ensure integer labels
        # This handles edge cases where resampling or data type conversion
        # might introduce small floating point values
        atlas_data_rounded = np.round(atlas_data).astype(int)

        region_ids = np.unique(atlas_data_rounded)
        region_ids = region_ids[
            region_ids > 0
        ]  # Exclude background        # Build ordered list of label names
        label_names = [labels.get(int(rid), f"Region{int(rid)}") for rid in sorted(region_ids)]

        # Initialize NiftiLabelsMasker with appropriate settings
        masker = NiftiLabelsMasker(
            labels_img=atlas_img,
            labels=label_names,
            background_label=0,
            strategy=strategy,
            resampling_target="data",  # Resample atlas to match source data
            standardize=False,  # Don't normalize for static maps
            detrend=False,  # No detrending for static maps
            memory=None,  # No caching for now
            verbose=0,
            keep_masked_labels=False,  # Remove empty region signals (future nilearn default)
        )

        # Extract values - nilearn expects 4D input (add time dimension if needed)
        if source_img.ndim == 3:
            # Add a dummy 4th dimension for time
            source_data_4d = source_img.get_fdata()[..., np.newaxis]
            source_img_4d = nib.Nifti1Image(source_data_4d, source_img.affine)
        else:
            source_img_4d = source_img

        # Transform: returns (n_timepoints, n_regions) array
        region_values = masker.fit_transform(source_img_4d)

        # Squeeze to get (n_regions,) for single timepoint
        if region_values.shape[0] == 1:
            region_values = region_values.squeeze(axis=0)

        # Apply post-processing based on aggregation type
        if self.aggregation == "percent":
            # Convert mean (0-1) to percentage (0-100)
            region_values = region_values * 100
        elif self.aggregation == "volume":
            # Convert count to volume (mm³)
            region_values = region_values * voxel_volume_mm3

        # Build results dict
        # Note: region_values length might not match label_names if regions are lost during resampling
        # We zip without strict=True and handle the mismatch
        results = {}
        for i, value in enumerate(region_values):
            if i < len(label_names):
                label_name = label_names[i]
            else:
                # Fallback if we get more regions than expected
                label_name = f"Region{i}"
            results[label_name] = float(value)

        return results

    def _aggregate_4d_atlas(
        self,
        source_img: nib.Nifti1Image,
        atlas_img: nib.Nifti1Image,
        labels: dict[int, str],
        voxel_volume_mm3: float,
    ) -> dict[str, float]:
        """
        Aggregate source data for 4D atlas with automatic resampling.

        Uses nilearn's resample_to_img for automatic spatial alignment of atlas
        to source data. Detects binary vs probabilistic atlases and uses appropriate
        interpolation method ('nearest' for binary, 'continuous' for probabilistic).

        Parameters
        ----------
        source_img : nib.Nifti1Image
            Source image to aggregate
        atlas_img : nib.Nifti1Image
            4D atlas (x, y, z, n_regions) with binary or probability maps
        labels : dict[int, str]
            Mapping from region ID to region name
        voxel_volume_mm3 : float
            Volume of one voxel in mm³ (for volume aggregation)

        Returns
        -------
        dict[str, float]
            Mapping from region name to aggregated value
        """
        # Detect if atlas is binary (only 0s and 1s) or probabilistic
        atlas_data_orig = atlas_img.get_fdata()
        unique_values = np.unique(atlas_data_orig)
        is_binary = np.all(np.isin(unique_values, [0.0, 1.0]))

        # Use appropriate interpolation based on atlas type
        # 'nearest' for binary to preserve 0/1 values
        # 'continuous' for probabilistic to interpolate between probability values
        interpolation = "nearest" if is_binary else "continuous"

        # Resample atlas to match source data spatial resolution
        atlas_resampled = resample_to_img(
            atlas_img,
            source_img,
            interpolation=interpolation,
            copy=True,
            force_resample=True,
            copy_header=True,
        )

        source_data = source_img.get_fdata()
        atlas_data = atlas_resampled.get_fdata()

        results = {}
        n_regions = atlas_data.shape[3]

        # Get sorted label IDs to map volume indices to label IDs
        # Volume index i corresponds to the i-th label ID in sorted order
        sorted_label_ids = sorted(labels.keys())

        # Validate that we have the right number of labels
        if len(sorted_label_ids) != n_regions:
            import warnings

            warnings.warn(
                f"Number of volumes ({n_regions}) does not match number of labels "
                f"({len(sorted_label_ids)}). Using available labels.",
                UserWarning,
                stacklevel=3,
            )

        for region_idx in range(n_regions):
            # Get probability map for this region
            prob_map = atlas_data[:, :, :, region_idx]

            # Threshold to create binary mask
            region_mask = prob_map >= self.threshold

            # Get values in this region
            region_values = source_data[region_mask]

            # Compute aggregation
            value = self._compute_aggregation(region_values, region_mask, voxel_volume_mm3)

            # Map volume index to label ID using sorted label IDs
            # Volume 0 → sorted_label_ids[0] (could be 0, 1, or any starting ID)
            # Volume 1 → sorted_label_ids[1], etc.
            if region_idx < len(sorted_label_ids):
                region_id = sorted_label_ids[region_idx]
                region_name = labels[region_id]
            else:
                # Fallback if more volumes than labels
                region_name = f"Region{region_idx}"

            results[region_name] = value

        return results

    def _compute_aggregation(
        self,
        region_values: np.ndarray,
        region_mask: np.ndarray,
        voxel_volume_mm3: float,
    ) -> float:
        """
        Compute specified aggregation method on region values.

        Parameters
        ----------
        region_values : np.ndarray
            Values within the region
        region_mask : np.ndarray
            Boolean mask for the region
        voxel_volume_mm3 : float
            Volume of one voxel in mm³

        Returns
        -------
        float
            Aggregated value
        """
        if len(region_values) == 0:
            return 0.0

        if self.aggregation == "mean":
            return float(np.mean(region_values))

        elif self.aggregation == "sum":
            return float(np.sum(region_values))

        elif self.aggregation == "median":
            return float(np.median(region_values))

        elif self.aggregation == "std":
            return float(np.std(region_values))

        elif self.aggregation == "percent":
            # Percentage of ROI voxels that are non-zero
            n_total = np.sum(region_mask)
            n_nonzero = np.sum(region_values > 0)
            return (n_nonzero / n_total * 100) if n_total > 0 else 0.0

        elif self.aggregation == "volume":
            # Volume of non-zero voxels in mm³
            n_nonzero = np.sum(region_values > 0)
            return n_nonzero * voxel_volume_mm3

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def _get_source_image(self, lesion_data: LesionData) -> nib.Nifti1Image | None:
        """
        Get source image from LesionData based on source parameter.

        Parameters
        ----------
        lesion_data : LesionData
            Lesion data containing source

        Returns
        -------
        nib.Nifti1Image or None
            Source image, or None if not found
        """
        # Direct image sources
        if self.source == "lesion_img":
            return lesion_data.lesion_img
        elif self.source == "anatomical_img":
            return lesion_data.anatomical_img

        # Result from previous analysis: "AnalysisName.result_key"
        if "." in self.source:
            analysis_name, result_key = self.source.split(".", 1)

            if analysis_name in lesion_data.results:
                analysis_results = lesion_data.results[analysis_name]

                if result_key in analysis_results:
                    result = analysis_results[result_key]

                    # If it's a NIfTI image, return it
                    if isinstance(result, nib.Nifti1Image):
                        return result

                    # If it's a path, load it
                    if isinstance(result, (str, Path)):
                        result_path = Path(result)
                        if result_path.exists():
                            return nib.load(result_path)

        return None


    def _get_parameters(self) -> dict:
        """Get analysis parameters for provenance and display.

        Returns
        -------
        dict
            Dictionary of parameter names and values.
        """
        return {
            "source": self.source,
            "aggregation": self.aggregation,
            "threshold": self.threshold,
            "atlas_names": self.atlas_names,
            "num_atlases": len(self.atlases) if hasattr(self, 'atlases') else None,
        }
