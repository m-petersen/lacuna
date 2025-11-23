"""
Atlas aggregation module.

Provides flexible ROI-level aggregation of voxel-level maps across multiple
atlases. This is the core tool for extracting region-of-interest statistics
from lesion masks, connectivity maps, or any other spatial maps.

Examples
--------
>>> from lacuna import MaskData
>>> from lacuna.analysis import ParcelAggregation
>>>
>>> # Use bundled atlases
>>> lesion = MaskData.from_nifti("lesion.nii.gz")
>>> analysis = ParcelAggregation(
...     source="mask_img",
...     aggregation="percent"
... )
>>> result = analysis.run(lesion)
>>> print(result.results["ParcelAggregation"])
>>>
>>> # Use custom atlas directory
>>> analysis = ParcelAggregation(
...     atlas_dir="/data/atlases",
...     source="mask_img",
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
>>> agg = ParcelAggregation(
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
from lacuna.assets.parcellations import list_parcellations, load_parcellation
from lacuna.core.data_types import ParcelData
from lacuna.core.mask_data import MaskData

if TYPE_CHECKING:
    from lacuna.core.data_types import DataContainer


class ParcelAggregation(BaseAnalysis):
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
    source : str, default="mask_img"
        Source of data to aggregate. Options:
        - "mask_img": Use the lesion mask directly
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
    parcel_names : list of str or None, default=None
        Names of atlases from the registry to process (e.g., "Schaefer2018_100Parcels7Networks").
        If None, all registered atlases are processed.
        Use register_parcellation() or register_parcellationes_from_directory() to add custom atlases.
        If None, all parcellations found in atlas_dir will be processed.
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
    - Results stored in MaskData.results["ParcelAggregation"] as dict
      mapping parcellation_name_region_name -> aggregated_value

    Examples
    --------
    >>> # Use all bundled/registered atlases
    >>> analysis = ParcelAggregation(
    ...     source="mask_img",
    ...     aggregation="percent"
    ... )
    >>>
    >>> # Use specific registered atlases
    >>> analysis = ParcelAggregation(
    ...     source="mask_img",
    ...     aggregation="percent",
    ...     parcel_names=["Schaefer2018_100Parcels7Networks", "TianSubcortex_3TS1"]
    ... )
    >>>
    >>> # Register custom atlases first, then use them
    >>> from lacuna.assets.parcellations import register_parcellations_from_directory
    >>> register_parcellationes_from_directory("/data/my_atlases")
    >>> analysis = ParcelAggregation(
    ...     source="mask_img",
    ...     aggregation="percent"
    ... )
    >>>
    >>> # Average functional connectivity per ROI
    >>> analysis = ParcelAggregation(
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
    VALID_SOURCES = ["mask_img"]

    def __init__(
        self,
        source: str = "mask_img",
        aggregation: str = "mean",
        threshold: float = 0.5,
        parcel_names: list[str] | None = None,
        log_level: int = 1,
    ):
        """Initialize ParcelAggregation analysis."""
        super().__init__(log_level=log_level)

        self.source = source
        self.aggregation = aggregation
        self.threshold = threshold
        self.parcel_names = parcel_names

        # Validate aggregation method
        if aggregation not in self.VALID_AGGREGATIONS:
            raise ValueError(
                f"Invalid aggregation method: '{aggregation}'\n"
                f"Valid options: {', '.join(self.VALID_AGGREGATIONS)}"
            )

        # Threshold validation removed - accepts any float value (T061)
        # This allows for flexible thresholding (e.g., negative z-scores, arbitrary cutoffs)

        # Validate parcel_names if provided
        if parcel_names is not None:
            if not isinstance(parcel_names, list):
                raise TypeError(
                    f"parcel_names must be a list of strings or None, got {type(parcel_names).__name__}"
                )
            if not all(isinstance(name, str) for name in parcel_names):
                raise TypeError("All items in parcel_names must be strings")
            if not parcel_names:
                raise ValueError(
                    "parcel_names cannot be an empty list (use None to process all atlases)"
                )

        # Will be populated in _validate_inputs
        self.atlases = []

    def run(
        self, data: "MaskData | nib.Nifti1Image | list[nib.Nifti1Image]"
    ) -> "MaskData | ParcelData | list[ParcelData]":
        """
        Execute atlas aggregation analysis on various input types.

        Supports flexible input types with matching return types:
        - MaskData -> MaskData (with results attached)
        - nibabel.Nifti1Image -> ParcelData
        - list[nibabel.Nifti1Image] -> list[ParcelData]

        Parameters
        ----------
        data : MaskData or nibabel.Nifti1Image or list[nibabel.Nifti1Image]
            Input data to aggregate:
            - MaskData: Standard workflow, returns MaskData with results
            - nibabel.Nifti1Image: Single image, returns ParcelData
            - list[nibabel.Nifti1Image]: Batch processing, returns list of results

        Returns
        -------
        MaskData or ParcelData or list[ParcelData]
            Results matching input type:
            - MaskData input: New MaskData instance with results in .results dict
            - nibabel input: Single ParcelData
            - list input: List of ParcelData objects (one per input image)

        Raises
        ------
        ValueError
            If input validation fails or source data not found.
        TypeError
            If input type is not supported.

        Notes
        -----
        This method overrides BaseAnalysis.run() to support flexible input types.
        The base class run() is designed for MaskData only.

        Examples
        --------
        >>> # MaskData input
        >>> mask_data = MaskData(mask_img, metadata={'space': 'MNI152NLin6Asym', 'resolution': 2})
        >>> analysis = ParcelAggregation(aggregation='percent')
        >>> result = analysis.run(mask_data)
        >>> isinstance(result, MaskData)
        True

        >>> # Nibabel image input
        >>> import nibabel as nib
        >>> img = nib.load('lesion.nii.gz')
        >>> result = analysis.run(img)
        >>> isinstance(result, ParcelData)
        True

        >>> # List of images
        >>> images = [nib.load(f'lesion_{i}.nii.gz') for i in range(5)]
        >>> results = analysis.run(images)
        >>> len(results) == 5
        True
        """
        from lacuna.core.data_types import VoxelMap

        # Detect input type and delegate to appropriate handler
        if isinstance(data, MaskData):
            # Standard MaskData workflow - use base class run()
            return super().run(data)

        elif isinstance(data, VoxelMap):
            # VoxelMap - run directly without MaskData wrapper
            return self._run_voxelmap(data)

        elif isinstance(data, nib.Nifti1Image):
            # Single nibabel image - return ParcelData
            return self._run_single_image(data)

        elif isinstance(data, list):
            # List of images or VoxelMaps - return list of results
            if not data:
                raise ValueError("Empty list provided - at least one image required")

            # Check if all are VoxelMaps or all are Images
            if all(isinstance(item, VoxelMap) for item in data):
                # Process VoxelMaps directly
                return [self._run_voxelmap(vm) for vm in data]

            elif all(isinstance(img, nib.Nifti1Image) for img in data):
                return self._run_batch_images(data)

            else:
                raise TypeError(
                    "When providing a list, all items must be of the same type: "
                    "either all VoxelMap or all nibabel.Nifti1Image objects"
                )

        else:
            raise TypeError(
                f"Unsupported input type: {type(data).__name__}\n"
                "Supported types: MaskData, VoxelMap, nibabel.Nifti1Image, "
                "list[VoxelMap], list[nibabel.Nifti1Image]"
            )

    def _run_single_image(self, img: nib.Nifti1Image) -> "ParcelData":
        """
        Run aggregation on a single nibabel image.

        Parameters
        ----------
        img : nibabel.Nifti1Image
            Input image to aggregate

        Returns
        -------
        ParcelData
            Aggregation result
        """
        # Create temporary MaskData wrapper to use existing infrastructure
        # Extract space and resolution from image if available in header
        # Default to MNI152NLin6Asym@2mm if not specified
        mask_data = MaskData(
            mask_img=img,
            metadata={
                "space": "MNI152NLin6Asym",  # TODO: Infer from image header if possible
                "resolution": 2.0,  # TODO: Infer from image header if possible
            },
        )

        # Run standard analysis
        result_mask_data = super().run(mask_data)

        # Extract and return just the aggregation results
        # Combine all atlas results into a single ParcelData
        atlas_results = result_mask_data.results.get(self.__class__.__name__, {})

        if not atlas_results:
            raise RuntimeError("No aggregation results generated")

        # If there's only one atlas, return its result directly
        if len(atlas_results) == 1:
            return list(atlas_results.values())[0]

        # If there are multiple atlases, we need to combine them somehow
        # For now, just return the first one
        # TODO: Consider returning a dict or combined result
        return list(atlas_results.values())[0]

    def _run_batch_images(self, images: list[nib.Nifti1Image]) -> list["ParcelData"]:
        """
        Run aggregation on a batch of nibabel images.

        Parameters
        ----------
        images : list[nibabel.Nifti1Image]
            List of images to aggregate

        Returns
        -------
        list[ParcelData]
            List of aggregation results (one per input image)
        """
        results = []
        for img in images:
            result = self._run_single_image(img)
            results.append(result)

        return results

    def _run_voxelmap(self, voxel_map: "VoxelMap") -> "ParcelData":
        """
        Run aggregation on a VoxelMap directly.

        This bypasses MaskData validation since VoxelMaps can contain
        continuous values (e.g., correlation maps, z-scores).

        Parameters
        ----------
        voxel_map : VoxelMap
            VoxelMap containing the data to aggregate

        Returns
        -------
        ParcelData
            Aggregation result combining all atlas aggregations
        """
        # Load atlases using same logic as _load_parcellations_from_registry
        if not hasattr(self, "atlases") or not self.atlases:
            self.atlases = self._load_parcellations_from_registry()

        # Get space and resolution from VoxelMap
        input_space = voxel_map.space
        input_resolution = voxel_map.resolution
        source_img = voxel_map.data

        # Calculate voxel volume from source data
        voxel_volume_mm3 = np.abs(np.linalg.det(source_img.affine[:3, :3]))

        # Collect all ROI results across atlases
        all_roi_data = {}

        # Process each atlas
        for atlas_info in self.atlases:
            parcellation_name = atlas_info["name"]
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
                input_affine=source_img.affine,
                parcellation_name=parcellation_name,
            )

            labels = atlas_info["labels"]
            atlas_data = atlas_img.get_fdata()

            if atlas_data.ndim == 3:
                # 3D integer-labeled atlas
                atlas_results = self._aggregate_3d_atlas(
                    source_img, atlas_img, labels, voxel_volume_mm3
                )
            elif atlas_data.ndim == 4:
                # 4D probabilistic atlas
                atlas_results = self._aggregate_4d_atlas(
                    source_img, atlas_img, labels, voxel_volume_mm3
                )
            else:
                continue

            # Merge results from this atlas
            all_roi_data.update(atlas_results)

        # Return single ParcelData with all ROI results
        from lacuna.core.data_types import ParcelData

        return ParcelData(
            name=f"{self.aggregation}_aggregation",
            data=all_roi_data,
            parcel_names=self.parcel_names if self.parcel_names else [a["name"] for a in self.atlases],
            aggregation_method=self.aggregation,
            metadata={
                "source": "VoxelMap",
                "source_name": voxel_map.name,
                "threshold": self.threshold,
                "n_regions": len(all_roi_data),
                "space": input_space,
                "resolution": input_resolution,
            },
        )

    def _validate_inputs(self, mask_data: MaskData) -> None:
        """
        Validate lesion data and load atlases from registry.

        Parameters
        ----------
        mask_data : MaskData
            Lesion data to validate

        Raises
        ------
        ValueError
            If lesion data is invalid or source data not found
        """
        # Check that source data exists
        source_img = self._get_source_image(mask_data)

        if source_img is None:
            # List available sources from MaskData results
            available = ["mask_img"]
            if mask_data.results:
                for analysis_name, analysis_results in mask_data.results.items():
                    for key in analysis_results.keys():
                        available.append(f"{analysis_name}.{key}")

            raise ValueError(
                f"Source data not found: {self.source}\n"
                "Check that the source exists in MaskData.\n"
                f"Available sources: {', '.join(available)}"
            )

        # Load atlases from registry
        self.atlases = self._load_parcellations_from_registry()

        if not self.atlases:
            if self.parcel_names is not None:
                raise ValueError(
                    f"No matching parcellations found for specified names: {self.parcel_names}\n"
                    "Available parcellations in registry: check list_parcellations()\n"
                    "Use register_parcellation() or register_parcellationes_from_directory() to add atlases"
                )
            else:
                raise ValueError(
                    "No valid parcellations found in registry\n"
                    "Use register_parcellation() or register_parcellationes_from_directory() to add atlases"
                )

        # Warn if some requested atlases weren't found
        if self.parcel_names is not None:
            found_names = {atlas["name"] for atlas in self.atlases}
            missing_names = set(self.parcel_names) - found_names
            if missing_names:
                import warnings

                warnings.warn(
                    f"Some requested atlases were not found: {sorted(missing_names)}\n"
                    f"Found atlases: {sorted(found_names)}",
                    UserWarning,
                    stacklevel=3,
                )

    def _load_parcellations_from_registry(self) -> list[dict]:
        """
        Load atlases from the registry (bundled or user-registered).

        Returns
        -------
        list[dict]
            List of atlas dictionaries with keys: name, image, labels, space, resolution
        """
        from lacuna.assets.parcellations.loader import BUNDLED_PARCELLATIONS_DIR

        # Get atlases from registry (filter by names if provided)
        if self.parcel_names is not None:
            # Load specific atlases by name
            atlases_data = []
            for name in self.parcel_names:
                try:
                    atlas = load_parcellation(name)

                    # Resolve paths (absolute or relative to bundled dir)
                    atlas_filename_path = Path(atlas.metadata.parcellation_filename)
                    if atlas_filename_path.is_absolute():
                        atlas_path = atlas_filename_path
                    else:
                        atlas_path = BUNDLED_PARCELLATIONS_DIR / atlas.metadata.parcellation_filename

                    labels_filename_path = Path(atlas.metadata.labels_filename)
                    if labels_filename_path.is_absolute():
                        labels_path = labels_filename_path
                    else:
                        labels_path = BUNDLED_PARCELLATIONS_DIR / atlas.metadata.labels_filename

                    atlases_data.append(
                        {
                            "name": name,
                            "atlas_path": atlas_path,
                            "labels_path": labels_path,
                            "labels": atlas.labels,
                            "space": atlas.metadata.space,
                            "resolution": atlas.metadata.resolution,
                            "is_4d": getattr(atlas.metadata, "is_4d", False),
                        }
                    )
                except KeyError:
                    # Atlas not in registry - will be caught by validation
                    pass
        else:
            # Load all registered atlases
            atlas_metadatas = list_parcellations()
            atlases_data = []
            for metadata in atlas_metadatas:
                atlas = load_parcellation(metadata.name)

                # Resolve paths (absolute or relative to bundled dir)
                atlas_filename_path = Path(atlas.metadata.parcellation_filename)
                if atlas_filename_path.is_absolute():
                    atlas_path = atlas_filename_path
                else:
                    atlas_path = BUNDLED_PARCELLATIONS_DIR / atlas.metadata.parcellation_filename

                labels_filename_path = Path(atlas.metadata.labels_filename)
                if labels_filename_path.is_absolute():
                    labels_path = labels_filename_path
                else:
                    labels_path = BUNDLED_PARCELLATIONS_DIR / atlas.metadata.labels_filename

                atlases_data.append(
                    {
                        "name": metadata.name,
                        "atlas_path": atlas_path,
                        "labels_path": labels_path,
                        "labels": atlas.labels,
                        "space": metadata.space,
                        "resolution": metadata.resolution,
                        "is_4d": getattr(metadata, "is_4d", False),
                    }
                )

        return atlases_data

    def _ensure_atlas_matches_input_space(
        self,
        atlas_img: nib.Nifti1Image,
        atlas_space: str,
        atlas_resolution: int,
        input_space: str,
        input_resolution: int,
        input_affine: np.ndarray,
        parcellation_name: str | None = None,
    ) -> nib.Nifti1Image:
        """
        Transform atlas to match input data space if spaces don't match.

        This allows ParcelAggregation to work with any voxel-level image,
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

        # Create target space matching input data
        target_space = CoordinateSpace(
            identifier=input_space,
            resolution=input_resolution,
            reference_affine=input_affine,
        )

        # Transform atlas using nearest neighbor to preserve labels
        # Logging is handled by transform_image
        return transform_image(
            img=atlas_img,
            source_space=atlas_space,
            target_space=target_space,
            source_resolution=atlas_resolution,
            interpolation="nearest",  # Preserve integer labels
            image_name=f"atlas '{parcellation_name}'" if parcellation_name else "atlas",
            log_level=self.log_level,
        )

    def _run_analysis(self, mask_data: MaskData) -> dict[str, "DataContainer"]:
        """
        Compute ROI-level aggregation for all atlases.

        Parameters
        ----------
        mask_data : MaskData
            Validated lesion data

        Returns
        -------
        dict[str, DataContainer]
            Dictionary mapping "atlas_{name}" to ParcelData objects
        """
        # Get input data space/resolution once
        input_space = mask_data.metadata.get("space")
        input_resolution = mask_data.metadata.get("resolution")

        # Get source image (this is what we'll aggregate)
        source_img = self._get_source_image(mask_data)

        # Calculate voxel volume from source data
        voxel_volume_mm3 = np.abs(np.linalg.det(source_img.affine[:3, :3]))

        # Collect results per atlas as dict with descriptive keys
        atlas_results_dict = {}

        # Process each atlas
        for atlas_info in self.atlases:
            parcellation_name = atlas_info["name"]
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
                input_affine=source_img.affine,
                parcellation_name=parcellation_name,
            )

            labels = atlas_info["labels"]
            atlas_data = atlas_img.get_fdata()

            # Warn if nilearn will resample atlas to match source resolution
            atlas_shape = atlas_data.shape[:3]  # Handle 4D atlases
            source_shape = source_img.get_fdata().shape
            if source_shape != atlas_shape:
                import warnings

                warnings.warn(
                    f"Atlas '{parcellation_name}' will be resampled to match source data.\n"
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
                    f"Skipping atlas '{parcellation_name}': unexpected dimensions {atlas_data.ndim}D",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Create one ParcelData per atlas
            roi_result = ParcelData(
                name=parcellation_name,
                data=atlas_results,
                parcel_names=[parcellation_name],
                aggregation_method=self.aggregation,
                metadata={
                    "source": self.source,
                    "threshold": self.threshold,
                    "n_regions": len(atlas_results),
                },
            )

            # Format: "atlas-{name}_desc-{source}"
            # Examples:
            #   - mask_img: "atlas-Schaefer2018_desc-maskImg"
            #   - Analysis.key: "atlas-Schaefer100_desc-disconnectionMap"

            # Shorten atlas name if it's very long
            short_parcellation_name = parcellation_name
            if len(parcellation_name) > 30:
                # Extract meaningful part (e.g., "Schaefer2018_100Parcels7Networks" -> "Schaefer100")
                if "Parcels" in parcellation_name:
                    # Extract number of parcels
                    import re

                    match = re.search(r"(\d+)Parcels", parcellation_name)
                    if match:
                        num_parcels = match.group(1)
                        if "Schaefer" in parcellation_name:
                            short_parcellation_name = f"Schaefer{num_parcels}"
                        elif "Tian" in parcellation_name:
                            short_parcellation_name = f"Tian{num_parcels}"
                        else:
                            short_parcellation_name = parcellation_name[:20]
                else:
                    short_parcellation_name = parcellation_name[:20]

            # Extract source key (remove "Analysis." prefix if present)
            if "." in self.source:
                # Cross-analysis source: "AnalysisName.result_key" -> "result_key"
                _, source_key = self.source.split(".", 1)
            else:
                # Direct source: "mask_img" -> "mask_img"
                source_key = self.source

            # Convert source_key to PascalCase for BIDS compliance
            # Examples: "mask_img" -> "MaskImg", "disconnection_map" -> "DisconnectionMap"
            source_pascal = "".join(
                word.capitalize()
                for word in source_key.split("_")
            )

            # Build BIDS-style result key
            result_key = f"atlas-{short_parcellation_name}_desc-{source_pascal}"

            atlas_results_dict[result_key] = roi_result

        return atlas_results_dict

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

    def _get_source_image(self, mask_data: MaskData) -> nib.Nifti1Image | None:
        """
        Get source image from MaskData based on source parameter.

        Parameters
        ----------
        mask_data : MaskData
            Lesion data containing source

        Returns
        -------
        nib.Nifti1Image or None
            Source image, or None if not found
        """
        # Direct image sources
        if self.source == "mask_img":
            return mask_data.mask_img

        # Result from previous analysis: "AnalysisName.result_key"
        if "." in self.source:
            analysis_name, result_key = self.source.split(".", 1)

            if analysis_name in mask_data.results:
                analysis_results = mask_data.results[analysis_name]

                if result_key in analysis_results:
                    result = analysis_results[result_key]

                    # If it's a NIfTI image, return it
                    if isinstance(result, nib.Nifti1Image):
                        return result

                    # If it's a VoxelMap, return the underlying image
                    from lacuna.core.data_types import VoxelMap
                    if isinstance(result, VoxelMap):
                        return result.data

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
            "parcel_names": self.parcel_names,
            "num_atlases": len(self.atlases) if hasattr(self, "atlases") else None,
            "log_level": self.log_level,
        }
